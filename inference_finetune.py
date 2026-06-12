import os
import json
import csv
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import SegformerForSemanticSegmentation, CLIPProcessor, CLIPModel

from train_clipheat_in4 import (
    resolve_local_first,
    clip_fire_heatmap,
    metrics_binary,
    hd95,
    hausdorff_distance,
    assd,
    tol_f1,
    SegFormer4Ch,
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_meta(model_path: str) -> dict:
    run_dir = os.path.dirname(model_path)
    for name in ("finetune_meta.json", "train_meta.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    print("⚠️  meta json 없음 → argparse 기본값 사용")
    return {}


def preprocess(image_rgb: np.ndarray, img_size: int):
    img = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return img, torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()


def heat_cache_path(cache_dir, img_name, grid):
    return os.path.join(cache_dir, f"{os.path.splitext(img_name)[0]}_g{grid}.npy")


def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    meta = load_meta(args.model_path)
    model_name     = meta.get("model_name",  args.model_name)
    img_size       = int(meta.get("img_size", args.img_size))
    clip_model_nm  = meta.get("clip_model",  args.clip_model)
    clip_grid      = int(meta.get("clip_grid", args.clip_grid))
    clip_prompts   = meta.get("clip_prompts", [p.strip() for p in args.clip_prompts.split(",") if p.strip()])
    local_base     = meta.get("local_models_base", args.local_models_base)

    print(f"model     : {model_name}")
    print(f"img_size  : {img_size}")
    print(f"clip_grid : {clip_grid}")
    print(f"prompts   : {clip_prompts}")
    print(f"device    : {device}")

    # CLIP
    clip_path, clip_local, clip_st, _ = resolve_local_first(clip_model_nm, base_dir=local_base)
    clip_kwargs = dict(local_files_only=clip_local)
    if clip_local and clip_st is not None:
        clip_kwargs["use_safetensors"] = clip_st
    clip_processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=clip_local)
    clip_model = CLIPModel.from_pretrained(clip_path, **clip_kwargs).to(args.clip_device)
    clip_model.eval()

    # SegFormer
    model = SegFormer4Ch(model_name, local_models_base=local_base).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print(f"✅ weights loaded: {args.model_path}")

    # 캐시 디렉토리
    os.makedirs(args.clip_cache_dir, exist_ok=True)

    # 데이터
    raw_dir = os.path.join(args.data_root, "raw")
    gt_dir  = os.path.join(args.data_root, "gt")
    has_gt  = os.path.exists(gt_dir)

    IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    images = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(IMG_EXT)])
    print(f"이미지 {len(images)}장 | GT {'있음' if has_gt else '없음 (추론만)'}")

    # 저장 폴더
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.out_dir, f"infer_ft_{ts}")
    overlay_dir  = os.path.join(save_dir, "overlays")
    mask_dir     = os.path.join(save_dir, "pred_masks")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(mask_dir,    exist_ok=True)

    rows = []
    sum_m = {k: 0.0 for k in ["IoU","Dice","Precision","Recall","F1","HD95","HD","ASSD","TolF1"]}
    cnt = 0

    for img_name in images:
        bgr = cv2.imread(os.path.join(raw_dir, img_name))
        if bgr is None:
            print(f"skip: {img_name}")
            continue
        rgb_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # CLIP heat (캐시)
        cp = heat_cache_path(args.clip_cache_dir, img_name, clip_grid)
        if os.path.exists(cp):
            heat = np.load(cp).astype(np.float32)
            if heat.shape != (img_size, img_size):
                heat = cv2.resize(heat, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        else:
            heat_full = clip_fire_heatmap(
                clip_model, clip_processor, rgb_orig,
                prompts=clip_prompts, grid=clip_grid,
                patch_px=224, device=args.clip_device,
            )
            heat = cv2.resize(heat_full, (img_size, img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            np.save(cp, heat)

        # 전처리
        rgb_resized, rgb_t = preprocess(rgb_orig, img_size)
        heat_t = torch.from_numpy(heat).unsqueeze(0).unsqueeze(0).float()
        x4 = torch.cat([rgb_t.to(device), heat_t.to(device)], dim=1)

        # 추론
        with torch.no_grad():
            logits = model(x4)
            logits = nn.functional.interpolate(logits, size=(img_size, img_size), mode="bilinear", align_corners=False)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (prob > args.thr).astype(np.uint8)

        # 저장
        stem = os.path.splitext(img_name)[0]
        cv2.imwrite(os.path.join(mask_dir, f"{stem}.png"), (pred * 255).astype(np.uint8))

        overlay = rgb_resized.copy()
        overlay[pred == 1] = (overlay[pred == 1] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgb_resized); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(overlay);     axes[1].set_title(f"Prediction (thr={args.thr})"); axes[1].axis("off")
        plt.suptitle(img_name); plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, f"{stem}.png"), dpi=120)
        plt.close()

        row = {"image": img_name}

        # 평가 (GT 있을 때)
        if has_gt:
            gt_name = img_name.replace("_Raw.png", "_Line.png")
            gt_path = os.path.join(gt_dir, gt_name)
            if not os.path.exists(gt_path):
                gt_path = os.path.join(gt_dir, img_name)

            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_img is not None:
                    gt_img = cv2.resize(gt_img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                    gt01 = (gt_img > 128).astype(np.uint8)

                    m = metrics_binary(pred, gt01)
                    row.update(m)
                    row["HD95"] = hd95(pred, gt01)
                    row["HD"]   = hausdorff_distance(pred, gt01)
                    row["ASSD"] = assd(pred, gt01)
                    row["TolF1"] = tol_f1(pred, gt01, tol_px=args.tol_px)

                    for k in sum_m:
                        sum_m[k] += float(row[k])
                    cnt += 1

                    print(f"{img_name}: Dice={row['Dice']:.4f} IoU={row['IoU']:.4f} TolF1={row['TolF1']:.4f}")
            else:
                print(f"{img_name}: GT 없음, 추론만")
        else:
            print(f"{img_name}: 저장 완료")

        rows.append(row)

    # CSV
    if rows or cnt > 0:
        csv_path = os.path.join(save_dir, "metrics.csv")
        all_keys = set()
        for r in rows:
            all_keys |= r.keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted(all_keys))
            w.writeheader(); w.writerows(rows)

    if cnt > 0:
        print("\n📊 평균 메트릭:")
        for k, v in sum_m.items():
            print(f"  {k}: {v/cnt:.4f}")

    print(f"\n✅ 완료: {save_dir}")


def build_parser():
    p = argparse.ArgumentParser("inference with finetuned SegFormer4Ch")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_root",  type=str, required=True, help="raw/ 와 gt/ 가 있는 폴더")
    p.add_argument("--out_dir",    type=str, default="./Inference_Result")
    p.add_argument("--clip_cache_dir", type=str, default="./clip_cache_real")
    p.add_argument("--thr",        type=float, default=0.5)
    p.add_argument("--tol_px",     type=int,   default=3)
    p.add_argument("--img_size",   type=int,   default=512)
    p.add_argument("--model_name", type=str,   default="nvidia/mit-b3")
    p.add_argument("--clip_model", type=str,   default="openai/clip-vit-base-patch32")
    p.add_argument("--clip_grid",  type=int,   default=16)
    p.add_argument("--clip_prompts", type=str, default="fire,flames,wildfire,burning")
    p.add_argument("--clip_device",  type=str, default="cpu", choices=["cpu","cuda"])
    p.add_argument("--local_models_base", type=str, default="./models")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)

import os
import json
import cv2
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation
from datetime import datetime

# ==========================================
# [1] Config
# ==========================================
DATA_ROOT = ""   # Structure: DATA_ROOT/raw, DATA_ROOT/gt  (set via --data-root)
MODEL_PATH = ""  # Path to trained model .pth checkpoint (set via --model-path)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join("./Inference_Result", f"Run_{current_time}")

# Must match training settings!
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 512  # automatically overridden if present in meta
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALPHA = 0.9
THRESH = 0.8
TOL_PX = 3  # Tol metrics tolerance

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# ==========================================
# [2] Utilities
# ==========================================
def find_train_meta(model_path: str) -> str:
    run_dir = os.path.dirname(model_path)
    cand = os.path.join(run_dir, "train_meta.json")
    return cand if os.path.exists(cand) else ""

def load_meta_if_exists(model_path: str) -> dict:
    meta_path = find_train_meta(model_path)
    if not meta_path:
        print("⚠️ train_meta.json not found next to MODEL_PATH. Fallback to hardcoded settings.")
        return {}

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"📝 Loaded metadata: {meta_path}")
        return meta
    except Exception as e:
        print(f"⚠️ Failed to read meta json: {meta_path} ({e}). Fallback to hardcoded settings.")
        return {}

def choose_local_weight_format(local_dir: str):
    st = os.path.join(local_dir, "model.safetensors")
    pt = os.path.join(local_dir, "pytorch_model.bin")

    if os.path.exists(st):
        try:
            if os.path.getsize(st) < 1024 * 1024:
                print(f"⚠️ suspicious model.safetensors too small -> ignore: {st} ({os.path.getsize(st)} bytes)")
            else:
                return "safetensors"
        except OSError:
            pass

    if os.path.exists(pt):
        return "bin"

    return None

def resolve_hf_local_first(model_name: str, base_dir: str = "./models"):
    local_dir = os.path.join(base_dir, model_name)
    local_has_config = os.path.exists(os.path.join(local_dir, "config.json"))
    local_weight_kind = choose_local_weight_format(local_dir)

    use_local = local_has_config and (local_weight_kind is not None)
    model_id_or_path = local_dir if use_local else model_name

    if use_local:
        use_safetensors = (local_weight_kind == "safetensors")
        return model_id_or_path, True, use_safetensors, local_weight_kind
    else:
        return model_id_or_path, False, None, None

def preprocess_to_tensor(input_rgb_uint8: np.ndarray) -> torch.Tensor:
    x = input_rgb_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
    return x

def apply_overlay(image_rgb, mask01, color_rgb=(255, 0, 0), alpha=0.8, dilate_vis=False, dilate_kernel=3):
    m = mask01.copy().astype(np.uint8)
    if dilate_vis:
        k = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        m = cv2.dilate(m, k, iterations=1)

    colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
    colored_mask[m == 1] = color_rgb
    return cv2.addWeighted(image_rgb, 1.0, colored_mask, alpha, 0)

# ==========================================
# [2.5] Metrics (including HD/ASSD) + TolP/TolR/TolF1
# ==========================================
def metrics_binary(pred01: np.ndarray, gt01: np.ndarray, eps: float = 1e-6):
    pred = pred01.astype(np.uint8)
    gt = gt01.astype(np.uint8)

    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    inter = tp
    union = (pred == 1).sum() + (gt == 1).sum() - tp
    iou = inter / (union + eps)

    dice = (2 * tp) / ((pred == 1).sum() + (gt == 1).sum() + eps)

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }

def mask_to_boundary(mask01: np.ndarray):
    m = (mask01 > 0).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)
    return (grad > 0).astype(np.uint8)

def hd95(pred01: np.ndarray, gt01: np.ndarray):
    pred = (pred01 > 0).astype(np.uint8)
    gt = (gt01 > 0).astype(np.uint8)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        h, w = pred.shape
        return float(np.sqrt(h * h + w * w))

    bp = mask_to_boundary(pred)
    bg = mask_to_boundary(gt)

    dp = cv2.distanceTransform((1 - bp).astype(np.uint8), cv2.DIST_L2, 3)
    dg = cv2.distanceTransform((1 - bg).astype(np.uint8), cv2.DIST_L2, 3)

    dist_p_to_g = dg[bp == 1]
    dist_g_to_p = dp[bg == 1]

    if dist_p_to_g.size == 0 or dist_g_to_p.size == 0:
        h, w = pred.shape
        return float(np.sqrt(h * h + w * w))

    all_d = np.concatenate([dist_p_to_g, dist_g_to_p])
    return float(np.percentile(all_d, 95))

def hausdorff_distance(pred01: np.ndarray, gt01: np.ndarray):
    pred = (pred01 > 0).astype(np.uint8)
    gt = (gt01 > 0).astype(np.uint8)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        h, w = pred.shape
        return float(np.sqrt(h * h + w * w))

    bp = mask_to_boundary(pred)
    bg = mask_to_boundary(gt)

    dp = cv2.distanceTransform((1 - bp).astype(np.uint8), cv2.DIST_L2, 3)
    dg = cv2.distanceTransform((1 - bg).astype(np.uint8), cv2.DIST_L2, 3)

    dist_p_to_g = dg[bp == 1]
    dist_g_to_p = dp[bg == 1]

    if dist_p_to_g.size == 0 or dist_g_to_p.size == 0:
        h, w = pred.shape
        return float(np.sqrt(h * h + w * w))

    return float(max(dist_p_to_g.max(), dist_g_to_p.max()))

def assd(pred01: np.ndarray, gt01: np.ndarray):
    pred = (pred01 > 0).astype(np.uint8)
    gt = (gt01 > 0).astype(np.uint8)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        h, w = pred.shape
        return float(np.sqrt(h * h + w * w))

    bp = mask_to_boundary(pred)
    bg = mask_to_boundary(gt)

    dp = cv2.distanceTransform((1 - bp).astype(np.uint8), cv2.DIST_L2, 3)
    dg = cv2.distanceTransform((1 - bg).astype(np.uint8), cv2.DIST_L2, 3)

    dist_p_to_g = dg[bp == 1]
    dist_g_to_p = dp[bg == 1]

    if dist_p_to_g.size == 0 or dist_g_to_p.size == 0:
        h, w = pred.shape
        return float(np.sqrt(h * h + w * w))

    all_d = np.concatenate([dist_p_to_g, dist_g_to_p])
    return float(all_d.mean())

def tol_prf(pred01: np.ndarray, gt01: np.ndarray, tol_px: int = 3, eps: float = 1e-6):
    """
    Returns tolerance-based Precision / Recall / F1
    - If tol_px=0, returns strict (exact pixel match) P/R/F1
    """
    pred = (pred01 > 0).astype(np.uint8)
    gt = (gt01 > 0).astype(np.uint8)

    if tol_px <= 0:
        m = metrics_binary(pred, gt, eps=eps)
        return float(m["Precision"]), float(m["Recall"]), float(m["F1"])

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
    gt_dil = cv2.dilate((gt * 255).astype(np.uint8), k, iterations=1) > 0
    pr_dil = cv2.dilate((pred * 255).astype(np.uint8), k, iterations=1) > 0

    # tol-precision: pred pixels inside gt_dil are counted as TP
    tp_p = np.logical_and(pred == 1, gt_dil).sum()
    fp_p = np.logical_and(pred == 1, np.logical_not(gt_dil)).sum()
    tol_prec = tp_p / (tp_p + fp_p + eps)

    # tol-recall: gt pixels inside pr_dil are counted as TP
    tp_r = np.logical_and(gt == 1, pr_dil).sum()
    fn_r = np.logical_and(gt == 1, np.logical_not(pr_dil)).sum()
    tol_rec = tp_r / (tp_r + fn_r + eps)

    tol_f1 = 2 * tol_prec * tol_rec / (tol_prec + tol_rec + eps)
    return float(tol_prec), float(tol_rec), float(tol_f1)

def find_gt_mask_path_same_name(gt_dir: str, img_name: str) -> str:
    """
    ✅ GT matching: use exact same filename only
    e.g.) test/raw/abc.png -> test/gt/abc.png
    """
    cand = os.path.join(gt_dir, img_name)
    return cand if os.path.exists(cand) else ""

# ==========================================
# [3] Inference Logic
# ==========================================
def run_inference():
    # Set up output folder structure
    os.makedirs(SAVE_DIR, exist_ok=True)
    overlay_dir = os.path.join(SAVE_DIR, "overlays")
    predmask_dir = os.path.join(SAVE_DIR, "pred_masks")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(predmask_dir, exist_ok=True)

    # Files to save evaluation results
    metrics_csv_path = os.path.join(SAVE_DIR, "metrics_per_image.csv")
    metrics_summary_path = os.path.join(SAVE_DIR, "metrics_summary.json")

    print(f"📂 Result save folder: '{SAVE_DIR}'")
    print("🚀 Starting inference!")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return

    # -------- Read meta to auto-apply base model name / img_size --------
    global IMG_SIZE
    meta = load_meta_if_exists(MODEL_PATH)

    base_model_name = meta.get("model_name", "nvidia/mit-b1")  # fallback if meta not found
    if "img_size" in meta:
        try:
            IMG_SIZE = int(meta["img_size"])
        except Exception:
            pass

    # ✅ prefer ./models + fallback to bin if safetensors not found
    model_id_or_path, local_files_only, use_safetensors, local_weight_kind = resolve_hf_local_first(
        base_model_name, base_dir="./models"
    )

    print(f"✅ Base model (from meta): {base_model_name}")
    print(f"✅ Base model source: {'LOCAL' if local_files_only else 'HF'} -> {model_id_or_path}")
    if local_files_only:
        print(f"✅ Local weight format: {local_weight_kind} (use_safetensors={use_safetensors})")
    print(f"✅ IMG_SIZE: {IMG_SIZE}")
    print(f"✅ DEVICE: {DEVICE}")
    print(f"✅ THRESH: {THRESH} | TOL_PX: {TOL_PX}")

    # -------- Build / load model --------
    kwargs = dict(
        num_labels=1,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )
    if local_files_only:
        kwargs["use_safetensors"] = bool(use_safetensors)

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_id_or_path,
        **kwargs
    ).to(DEVICE)

    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(sd, strict=True)
    print("✅ Loaded trained model weights. (strict=True)")

    model.eval()

    # List test files
    test_raw_dir = os.path.join(DATA_ROOT, "raw")
    test_gt_dir = os.path.join(DATA_ROOT, "gt")
    has_gt = os.path.exists(test_gt_dir)

    if not os.path.exists(test_raw_dir):
        print(f"❌ Test folder not found: {test_raw_dir}")
        return

    image_files = sorted([f for f in os.listdir(test_raw_dir) if f.lower().endswith(IMG_EXTENSIONS)])
    if len(image_files) == 0:
        print("⚠️ No image files found in the folder.")
        return

    if has_gt:
        print(f"✅ GT folder found: {test_gt_dir} -> will also perform evaluation. (same filename matching)")
    else:
        print(f"ℹ️ GT folder not found: {test_gt_dir} -> inference only (evaluation skipped).")

    print(f"🔍 Found {len(image_files)} image(s) in total.")

    # ---- Accumulate evaluation metrics ----
    metric_keys = [
        "IoU", "Dice", "Precision", "Recall", "F1",
        "HD95", "HD", "ASSD",
        "TolPrecision", "TolRecall", "TolF1"
    ]
    per_image_rows = []
    sum_metrics = {k: 0.0 for k in metric_keys}
    eval_count = 0

    # ---- Save / read counters (for missing file detection) ----
    read_ok_count = 0
    mask_save_ok_count = 0

    for idx, img_name in enumerate(image_files, 1):
        img_path = os.path.join(test_raw_dir, img_name)

        original_bgr = cv2.imread(img_path)
        if original_bgr is None:
            print(f"⚠️ Cannot read image (Skip): {img_name}")
            per_image_rows.append({
                "image": img_name,
                "mask_out": f"{os.path.splitext(img_name)[0]}_mask.png",
                "save_skip_reason": "image_read_failed"
            })
            continue

        read_ok_count += 1

        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        input_rgb = cv2.resize(original_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        input_tensor = preprocess_to_tensor(input_rgb).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(pixel_values=input_tensor)
            logits = nn.functional.interpolate(
                outputs.logits, size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear", align_corners=False
            )
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_mask01 = (prob > THRESH).astype(np.uint8)

        # --------------------------
        # ✅ Save predicted mask: original filename + "_mask.png"
        # + check for save failures / missing files
        # --------------------------
        pred_mask255 = (pred_mask01.astype(np.uint8) * 255)

        stem = os.path.splitext(img_name)[0]
        mask_save_name = f"{stem}_mask.png"
        mask_save_path = os.path.join(predmask_dir, mask_save_name)

        ok = cv2.imwrite(mask_save_path, pred_mask255)
        if (not ok) or (not os.path.exists(mask_save_path)) or (os.path.getsize(mask_save_path) == 0):
            print(f"❌ mask save failed: {mask_save_path} | imwrite_ok={ok}")
            per_image_rows.append({
                "image": img_name,
                "mask_out": mask_save_name,
                "save_skip_reason": "mask_write_failed"
            })
            continue

        mask_save_ok_count += 1

        # Save overlay
        overlay_rgb = apply_overlay(
            image_rgb=input_rgb.astype(np.uint8),
            mask01=pred_mask01,
            color_rgb=(0, 0, 255),
            alpha=ALPHA,
            dilate_vis=True,
            dilate_kernel=3
        )

        fig_save_path = os.path.join(overlay_dir, f"Result_{stem}.png")
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.title("Original"); plt.imshow(input_rgb); plt.axis("off")
        plt.subplot(1, 2, 2); plt.title("Prediction (Overlay)"); plt.imshow(overlay_rgb); plt.axis("off")
        plt.suptitle(f"File: {img_name}")
        plt.tight_layout()
        plt.savefig(fig_save_path, dpi=150)
        plt.close()

        # ---- Evaluation (only when GT is available) ----
        row = {"image": img_name, "mask_out": mask_save_name}
        if has_gt:
            gt_path = find_gt_mask_path_same_name(test_gt_dir, img_name)
            if gt_path:
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt is not None:
                    gt = cv2.resize(gt, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                    gt01 = (gt > 128).astype(np.uint8)

                    m = metrics_binary(pred_mask01, gt01)
                    row.update(m)

                    row["HD95"] = hd95(pred_mask01, gt01)
                    row["HD"]   = hausdorff_distance(pred_mask01, gt01)
                    row["ASSD"] = assd(pred_mask01, gt01)

                    tp, tr, tf = tol_prf(pred_mask01, gt01, tol_px=TOL_PX)
                    row["TolPrecision"] = tp
                    row["TolRecall"] = tr
                    row["TolF1"] = tf

                    for k in metric_keys:
                        sum_metrics[k] += float(row[k])
                    eval_count += 1
                else:
                    row["eval_skip_reason"] = "gt_read_failed"
            else:
                row["eval_skip_reason"] = "gt_not_found_same_name"

        per_image_rows.append(row)

        if has_gt and "IoU" in row:
            print(
                f"💾 [{idx}/{len(image_files)}] overlay: {fig_save_path} | mask: {mask_save_path} | "
                f"Dice={row['Dice']:.4f} IoU={row['IoU']:.4f} HD95={row['HD95']:.2f} "
                f"TolP={row['TolPrecision']:.4f} TolR={row['TolRecall']:.4f} TolF1={row['TolF1']:.4f}"
            )
        else:
            print(f"💾 [{idx}/{len(image_files)}] overlay: {fig_save_path} | mask: {mask_save_path}")

    # ---- Save results: CSV / JSON ----
    all_keys = set()
    for r in per_image_rows:
        all_keys |= set(r.keys())

    preferred = ["image", "mask_out"] + metric_keys + ["eval_skip_reason", "save_skip_reason"]
    header = [k for k in preferred if k in all_keys] + [k for k in sorted(all_keys) if k not in preferred]

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in per_image_rows:
            w.writerow(r)

    summary = {
        "data_root": DATA_ROOT,
        "model_path": MODEL_PATH,
        "img_size": IMG_SIZE,
        "device": DEVICE,
        "thresh": THRESH,
        "tol_px": TOL_PX,
        "count_total_images": len(image_files),
        "count_read_ok_images": int(read_ok_count),
        "count_mask_saved_ok": int(mask_save_ok_count),
        "count_eval_images": int(eval_count),
        "mean_metrics": None,
        "gt_matching": "same_filename_only"
    }

    if has_gt and eval_count > 0:
        mean_metrics = {k: (sum_metrics[k] / eval_count) for k in metric_keys}
        summary["mean_metrics"] = mean_metrics

        print("\n📊 [Evaluation Summary] (mean over evaluated images)")
        for k in metric_keys:
            if k in ("HD95", "HD", "ASSD"):
                print(f"- {k}: {mean_metrics[k]:.2f}")
            else:
                print(f"- {k}: {mean_metrics[k]:.4f}")
    else:
        print("\nℹ️ GT not found or eval_count=0 (matching failed); mean metrics not computed.")

    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- Check for missing saves (filesystem-based) ----
    saved_masks = sorted([f for f in os.listdir(predmask_dir) if f.lower().endswith(".png")])
    expected_mask_names = {f"{os.path.splitext(n)[0]}_mask.png" for n in image_files}
    saved_set = set(saved_masks)
    missing = sorted(list(expected_mask_names - saved_set))

    print("\n🧾 [Mask Save Check]")
    print(f"- raw images listed        : {len(image_files)}")
    print(f"- images read OK           : {read_ok_count}")
    print(f"- masks saved OK (tracked) : {mask_save_ok_count}")
    print(f"- mask pngs in folder      : {len(saved_masks)}")
    if missing:
        print(f"❗ missing masks detected: {len(missing)} (show up to 30)")
        for m in missing[:30]:
            print(" -", m)
        if len(missing) > 30:
            print(" ...")
    else:
        print("✅ no missing masks (by name check)")

    print("\n✅ Inference complete.")
    print(f"- overlays: {overlay_dir}")
    print(f"- pred masks: {predmask_dir}")
    print(f"- per-image metrics csv: {metrics_csv_path}")
    print(f"- summary json: {metrics_summary_path}")
    print(f"📂 Final output path: {SAVE_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference with baseline SegFormer (3ch RGB)")
    parser.add_argument("--data-root", required=True,
                        help="Test data root containing raw/ (and optionally gt/) subdirs")
    parser.add_argument("--model-path", required=True,
                        help="Path to trained model .pth checkpoint")
    parser.add_argument("--save-dir", default=None,
                        help="Output directory (default: ./outputs/Run_<timestamp>)")
    parser.add_argument("--thresh", type=float, default=THRESH,
                        help="Prediction threshold (default: %(default)s)")
    parser.add_argument("--tol-px", type=int, default=TOL_PX,
                        help="Tolerance radius in pixels for TolF1 (default: %(default)s)")
    parser.add_argument("--model-dir", default="./models",
                        help="Local HuggingFace model cache dir (default: %(default)s)")
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    MODEL_PATH = args.model_path
    THRESH = args.thresh
    TOL_PX = args.tol_px
    if args.save_dir:
        SAVE_DIR = args.save_dir

    run_inference()

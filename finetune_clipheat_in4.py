"""
finetune_clipheat_in4.py
====================================================
사전학습된 SegFormer4Ch 가중치를 실제 이미지 ~30장으로 fine-tuning.

freeze_mode:
  encoder   (기본) : 인코더 freeze, 디코더만 학습
  gradual         : 디코더 먼저 → 에폭별로 인코더 스테이지 점진 해동
  layerwise       : 레이어별 LR 감쇠 (디코더 > 인코더끝 > 인코더시작)
  full            : 전체 학습 (30장에선 오버피팅 주의)

+ 강한 augmentation (기하 + 색상, sim→real 도메인 갭 대응)
+ Early stopping (val Dice 기준)
+ CosineAnnealingLR
+ Gradient clipping
====================================================
"""

import os
import cv2
import json
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import CLIPProcessor, CLIPModel

from train_clipheat_in4 import (
    resolve_local_first,
    make_band_from_line,
    make_weight_map_from_band,
    clip_fire_heatmap,
    build_criterion,
    dice_score_from_logits,
    metrics_binary,
    hd95,
    hausdorff_distance,
    assd,
    tol_f1,
    run_eval,
    SegFormer4Ch,
)


# ============================================================
# 1) Augmentation transforms
# ============================================================
def get_geo_transform(img_size: int) -> A.Compose:
    """기하학적 변환: RGB / heat / mask 동시 적용"""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=20,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.25),
            A.Resize(img_size, img_size),
        ],
        additional_targets={"heat": "image"},
    )


def get_color_transform() -> A.Compose:
    """색상 변환: RGB에만 적용 (sim→real 도메인 갭 커버)"""
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=25, p=0.6),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.25),
        ]
    )


def get_normalize_transform() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# ============================================================
# 2) Dataset (augmentation 포함)
# ============================================================
class FineTuneDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 512,
        band_width_px: int = 5,
        max_weight: float = 6.0,
        weight_sigma: float = 4.0,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        clip_prompts: list = None,
        clip_grid: int = 16,
        clip_cache_dir: str = "./clip_cache_real",
        clip_device: str = "cpu",
        local_models_base: str = "./models",
        augment: bool = True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.band_width_px = band_width_px
        self.max_weight = max_weight
        self.weight_sigma = weight_sigma

        self.raw_dir = os.path.join(root_dir, split, "raw")
        self.gt_dir = os.path.join(root_dir, split, "gt")

        if not os.path.exists(self.raw_dir):
            self.images = []
        else:
            self.images = sorted(
                [f for f in os.listdir(self.raw_dir) if f.lower().endswith(".png")]
            )

        self.clip_prompts = clip_prompts or ["fire", "flames", "wildfire", "burning"]
        self.clip_grid = int(clip_grid)
        self.clip_cache_dir = os.path.join(clip_cache_dir, split)
        os.makedirs(self.clip_cache_dir, exist_ok=True)

        clip_path, clip_is_local, clip_use_st, _ = resolve_local_first(
            clip_model_name, base_dir=local_models_base
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            clip_path, local_files_only=clip_is_local
        )
        clip_kwargs = dict(local_files_only=clip_is_local)
        if clip_is_local and clip_use_st is not None:
            clip_kwargs["use_safetensors"] = clip_use_st
        self.clip_device = clip_device
        self.clip_model = CLIPModel.from_pretrained(clip_path, **clip_kwargs).to(clip_device)
        self.clip_model.eval()

        self.geo_tfm = get_geo_transform(img_size)
        self.color_tfm = get_color_transform()
        self.norm_tfm = get_normalize_transform()

    def __len__(self):
        return len(self.images)

    def _heat_cache_path(self, img_name: str) -> str:
        base = os.path.splitext(img_name)[0]
        return os.path.join(self.clip_cache_dir, f"{base}_g{self.clip_grid}.npy")

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.raw_dir, img_name)

        # GT 경로
        line_name = img_name.replace("_Raw.png", "_Line.png")
        line_path = os.path.join(self.gt_dir, line_name)
        if not os.path.exists(line_path):
            alt = os.path.join(self.gt_dir, img_name)
            if os.path.exists(alt):
                line_path = alt
        if not os.path.exists(line_path):
            raise RuntimeError(f"GT not found: {img_name}")

        # 이미지 로드
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # GT → resize → band
        line = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)
        if line is None:
            raise RuntimeError(f"Failed to read GT: {line_path}")
        line = cv2.resize(line, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        line01 = (line > 128).astype(np.float32)
        band01 = make_band_from_line(line01, self.band_width_px)

        # CLIP 히트맵 (캐시)
        cache_path = self._heat_cache_path(img_name)
        if os.path.exists(cache_path):
            heat = np.load(cache_path).astype(np.float32)
        else:
            heat_full = clip_fire_heatmap(
                self.clip_model, self.clip_processor, image_rgb,
                prompts=self.clip_prompts, grid=self.clip_grid,
                patch_px=224, device=self.clip_device,
            )
            heat = cv2.resize(
                heat_full, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            np.save(cache_path, heat)

        # 이미지도 img_size로 리사이즈
        image_rgb = cv2.resize(
            image_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )

        # ── Augmentation ─────────────────────────────────────
        if self.augment:
            # 기하학적: RGB + heat + band 동시 변환
            band_uint8 = (band01 * 255).astype(np.uint8)
            geo_out = self.geo_tfm(image=image_rgb, heat=heat, mask=band_uint8)
            image_rgb = geo_out["image"]
            heat = geo_out["heat"].astype(np.float32)
            band01 = (geo_out["mask"] > 128).astype(np.float32)

            # 색상: RGB만
            image_rgb = self.color_tfm(image=image_rgb)["image"]

        # augmented band 기반으로 weight map 재계산
        wmap = make_weight_map_from_band(band01, self.max_weight, self.weight_sigma)

        # Normalize + ToTensor (RGB)
        img_t = self.norm_tfm(image=image_rgb)["image"]          # (3,H,W)
        heat_t = torch.from_numpy(heat).unsqueeze(0).float()     # (1,H,W)
        x4 = torch.cat([img_t, heat_t], dim=0)                   # (4,H,W)

        band_t = torch.from_numpy(band01).unsqueeze(0).float()
        wmap_t = torch.from_numpy(wmap).unsqueeze(0).float()
        return x4, band_t, wmap_t


# ============================================================
# 3) Freeze / Unfreeze utilities
# ============================================================
def freeze_encoder(model: SegFormer4Ch):
    for param in model.base.segformer.parameters():
        param.requires_grad = False


def unfreeze_all(model: SegFormer4Ch):
    for param in model.parameters():
        param.requires_grad = True


def unfreeze_encoder_stage(model: SegFormer4Ch, stage: int):
    """stage 0(얕은) ~ 3(깊은) 중 특정 스테이지만 해동"""
    enc = model.base.segformer.encoder
    targets = (
        [enc.patch_embeddings[stage]]
        + list(enc.block[stage])
        + [enc.layer_norm[stage]]
    )
    for m in targets:
        for param in m.parameters():
            param.requires_grad = True


def print_trainable_params(model: SegFormer4Ch):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


# ============================================================
# 4) EarlyStopping
# ============================================================
class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best is None:
            self.best = score
            return False
        improved = (
            score > self.best + self.min_delta
            if self.mode == "max"
            else score < self.best - self.min_delta
        )
        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ============================================================
# 5) Optimizer (freeze mode별 파라미터 그룹)
# ============================================================
def build_finetune_optimizer(model: SegFormer4Ch, args):
    wd = args.weight_decay
    mode = args.freeze_mode

    if mode == "encoder":
        params = [p for p in model.base.decode_head.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=wd)

    if mode == "layerwise":
        enc = model.base.segformer.encoder
        s = args.encoder_lr_scale

        def stage_params(st):
            return (
                list(enc.patch_embeddings[st].parameters())
                + [p for blk in enc.block[st] for p in blk.parameters()]
                + list(enc.layer_norm[st].parameters())
            )

        param_groups = [
            {"params": [p for p in model.base.decode_head.parameters() if p.requires_grad],
             "lr": args.lr},
            {"params": [p for p in stage_params(3) if p.requires_grad],
             "lr": args.lr * s},
            {"params": [p for p in stage_params(2) if p.requires_grad],
             "lr": args.lr * s * 0.1},
            {"params": [p for p in stage_params(1) if p.requires_grad],
             "lr": args.lr * s * 0.01},
            {"params": [p for p in stage_params(0) if p.requires_grad],
             "lr": args.lr * s * 0.001},
        ]
        return torch.optim.AdamW(param_groups, weight_decay=wd)

    # gradual / full: 현재 requires_grad=True인 파라미터 전체
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=args.lr, weight_decay=wd)


# ============================================================
# 6) Fine-tuning main loop
# ============================================================
def finetune(args):
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    run_id = datetime.now().strftime(f"ft_{args.run_name}_%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))
    print(f"Run: {save_dir}")
    print(f"Device: {device} | freeze_mode: {args.freeze_mode} | LR: {args.lr}")

    clip_prompts = [p.strip() for p in args.clip_prompts.split(",") if p.strip()]

    # ── Dataset ──────────────────────────────────────────────
    ds_kwargs = dict(
        img_size=args.img_size,
        band_width_px=args.band_width_px,
        max_weight=args.max_weight,
        weight_sigma=args.weight_sigma,
        clip_model_name=args.clip_model,
        clip_prompts=clip_prompts,
        clip_grid=args.clip_grid,
        clip_cache_dir=args.clip_cache_dir,
        clip_device=args.clip_device,
        local_models_base=args.local_models_base,
    )
    train_ds = FineTuneDataset(args.data_root, "train", augment=True, **ds_kwargs)
    val_ds = FineTuneDataset(args.data_root, "val", augment=False, **ds_kwargs)

    if len(train_ds) == 0:
        print("❌ train 데이터가 없습니다. --data_root 경로를 확인하세요.")
        writer.close()
        return

    print(f"Train: {len(train_ds)}장 | Val: {len(val_ds)}장")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=(len(train_ds) > args.batch_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ── Model + load weights ──────────────────────────────────
    model = SegFormer4Ch(args.model_name, local_models_base=args.local_models_base).to(device)
    print(f"Loading weights: {args.pretrained_weights}")
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    try:
        model.load_state_dict(ckpt, strict=True)
        print("✅ Weights loaded (strict=True)")
    except RuntimeError as e:
        print(f"⚠️ strict=True 실패 → strict=False 재시도\n  {e}")
        model.load_state_dict(ckpt, strict=False)
        print("✅ Weights loaded (strict=False)")

    # ── Freeze 설정 ───────────────────────────────────────────
    if args.freeze_mode in ("encoder", "gradual"):
        freeze_encoder(model)
        print("🔒 Encoder frozen → decode_head만 학습")
    elif args.freeze_mode == "layerwise":
        unfreeze_all(model)
        print("🔓 전체 해동 (layer-wise LR 적용)")
    else:
        unfreeze_all(model)
        print("⚠️ 전체 학습 (30장에선 오버피팅 주의)")
    print_trainable_params(model)

    # ── Gradual unfreezing 스케줄 ─────────────────────────────
    # stage 3(깊은) 먼저 → 간격마다 더 얕은 스테이지 해동
    gradual_schedule: dict[int, int] = {}
    if args.freeze_mode == "gradual":
        for i, stage in enumerate(range(3, -1, -1)):
            ep = args.unfreeze_epoch + i * args.unfreeze_interval
            if ep <= args.epochs:
                gradual_schedule[ep] = stage
        print(f"Gradual schedule (epoch → stage): {gradual_schedule}")

    criterion = build_criterion(args).to(device)
    optimizer = build_finetune_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    early_stopping = EarlyStopping(patience=args.patience, mode="max")

    # ── Meta 저장 ─────────────────────────────────────────────
    meta = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pretrained_weights": args.pretrained_weights,
        "freeze_mode": args.freeze_mode,
        "data_root": args.data_root,
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "encoder_lr_scale": args.encoder_lr_scale,
        "patience": args.patience,
        "grad_clip": args.grad_clip,
        "model_name": args.model_name,
        "clip_model": args.clip_model,
        "clip_prompts": clip_prompts,
        "loss": args.loss,
        "gradual_schedule": {str(k): v for k, v in gradual_schedule.items()},
    }
    meta_path = os.path.join(save_dir, "finetune_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Metadata: {meta_path}")

    best_dice = -1.0
    best_path = os.path.join(save_dir, "best_finetuned.pth")
    global_step = 0
    unfrozen_stages: set[int] = set()

    for epoch in range(args.epochs):

        # ── Gradual unfreezing ──────────────────────────────
        if args.freeze_mode == "gradual" and (epoch + 1) in gradual_schedule:
            stage = gradual_schedule[epoch + 1]
            if stage not in unfrozen_stages:
                unfreeze_encoder_stage(model, stage)
                unfrozen_stages.add(stage)
                print(f"\n🔓 Epoch {epoch+1}: encoder stage {stage} 해동")
                print_trainable_params(model)
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=args.lr * args.encoder_lr_scale,
                    weight_decay=args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, args.epochs - epoch),
                    eta_min=args.lr * args.encoder_lr_scale * 0.01,
                )

        # ── Train ──────────────────────────────────────────
        model.train()
        t_loss, t_dice = 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [FT]", leave=False)
        for x4, band, wmap in pbar:
            x4 = x4.to(device, non_blocking=True)
            band = band.to(device, non_blocking=True)
            wmap = wmap.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x4)
            logits = F.interpolate(
                logits, size=(args.img_size, args.img_size),
                mode="bilinear", align_corners=False,
            )
            loss = criterion(logits, band, wmap)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
            optimizer.step()

            d = dice_score_from_logits(logits.detach(), band, thr=args.thr)
            t_loss += loss.item()
            t_dice += d

            writer.add_scalar("step/loss", loss.item(), global_step)
            writer.add_scalar("step/dice", d, global_step)
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

        t_loss /= max(1, len(train_loader))
        t_dice /= max(1, len(train_loader))

        # ── Validation ─────────────────────────────────────
        if len(val_ds) > 0:
            v_loss, v_metrics = run_eval(model, val_loader, criterion, args, device, name="Val")
        else:
            v_loss = t_loss
            v_metrics = {k: 0.0 for k in ["IoU","Dice","Precision","Recall","F1","HD95","HD","ASSD","TolF1"]}

        scheduler.step()

        # ── TensorBoard ────────────────────────────────────
        writer.add_scalar("loss/train", t_loss, epoch + 1)
        writer.add_scalar("dice/train", t_dice, epoch + 1)
        writer.add_scalar("loss/val", v_loss, epoch + 1)
        for k, v in v_metrics.items():
            writer.add_scalar(f"metrics/val_{k}", v, epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
        writer.flush()

        print(
            f"Epoch {epoch+1:3d}: "
            f"train loss={t_loss:.4f} dice={t_dice:.4f} | "
            f"val loss={v_loss:.4f} Dice={v_metrics['Dice']:.4f} "
            f"IoU={v_metrics['IoU']:.4f} TolF1={v_metrics['TolF1']:.4f} | "
            f"ES {early_stopping.counter}/{args.patience}"
        )

        # ── Best checkpoint (val Dice 기준) ─────────────────
        if v_metrics["Dice"] > best_dice:
            best_dice = v_metrics["Dice"]
            torch.save(model.state_dict(), best_path)
            print(f"  💾 Best saved (val Dice={best_dice:.4f}): {best_path}")
            meta["best"] = {
                "epoch": epoch + 1,
                "val_dice": float(best_dice),
                "val_metrics": {k: float(v) for k, v in v_metrics.items()},
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        # epoch별 체크포인트 저장
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

        # ── Early stopping ──────────────────────────────────
        if early_stopping.step(v_metrics["Dice"]):
            print(f"\n⏹ Early stopping (patience={args.patience})")
            break

    # Last
    last_path = os.path.join(save_dir, "last_finetuned.pth")
    torch.save(model.state_dict(), last_path)
    meta["last"] = {"epoch": epoch + 1, "path": os.path.basename(last_path)}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    writer.close()

    print(f"\n✅ Fine-tuning 완료")
    print(f"Best (Dice={best_dice:.4f}): {best_path}")
    print(f"Last: {last_path}")
    print(f"Run dir: {save_dir}")


# ============================================================
# 7) Argparse
# ============================================================
def build_parser():
    p = argparse.ArgumentParser("Fine-tune SegFormer4Ch on real images (~30장)")

    # 필수
    p.add_argument(
        "--pretrained_weights", type=str, required=True,
        help="사전학습 .pth 경로 (e.g., train_runs/.../best_fireline_model.pth)",
    )

    # 데이터
    p.add_argument("--data_root", type=str, default="./Dataset/real",
                   help="실제 이미지 루트 (train/raw, train/gt, val/raw, val/gt)")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2, help="30장 기준 2~4 권장")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)

    # 모델
    p.add_argument("--model_name", type=str, default="nvidia/mit-b3")
    p.add_argument("--local_models_base", type=str, default="./models")

    # freeze 전략
    p.add_argument(
        "--freeze_mode", type=str, default="encoder",
        choices=["encoder", "gradual", "layerwise", "full"],
        help=(
            "encoder: 인코더 고정(기본) | "
            "gradual: 점진적 해동 | "
            "layerwise: 레이어별 LR | "
            "full: 전체 학습"
        ),
    )
    p.add_argument("--unfreeze_epoch", type=int, default=10,
                   help="[gradual] 이 에폭부터 마지막 인코더 스테이지 해동")
    p.add_argument("--unfreeze_interval", type=int, default=10,
                   help="[gradual] 스테이지 해동 간격 (에폭)")
    p.add_argument("--encoder_lr_scale", type=float, default=0.1,
                   help="[layerwise/gradual] 인코더 LR = lr * scale")

    # band / weight map
    p.add_argument("--band_width_px", type=int, default=5)
    p.add_argument("--max_weight", type=float, default=6.0)
    p.add_argument("--weight_sigma", type=float, default=4.0)

    # loss
    p.add_argument("--loss", type=str, default="wft",
                   choices=["wft", "dice", "bce", "dicebce"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.33)
    p.add_argument("--dice_w", type=float, default=1.0)
    p.add_argument("--bce_w", type=float, default=1.0)

    # CLIP
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--clip_prompts", type=str, default="fire,flames,wildfire,burning")
    p.add_argument("--clip_grid", type=int, default=16)
    p.add_argument("--clip_cache_dir", type=str, default="./clip_cache_real",
                   help="실제 이미지 전용 캐시 (시뮬 캐시와 분리 권장)")
    p.add_argument("--clip_device", type=str, default="cpu", choices=["cpu", "cuda"])

    # 평가
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--tol_px", type=int, default=3)

    # 학습 옵션
    p.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping (0=off)")
    p.add_argument("--patience", type=int, default=20, help="early stopping patience")

    # 출력
    p.add_argument("--out_dir", type=str, default="./finetune_runs")
    p.add_argument("--run_name", type=str, default="real_finetune")
    p.add_argument("--cpu", action="store_true")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    finetune(args)

import os
import cv2
import json
import numpy as np
import argparse
import hashlib
from datetime import datetime
from typing import List, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import CLIPProcessor, CLIPModel
from transformers import SegformerForSemanticSegmentation


# ============================================================
# 0) Helpers: local-first (path or HF id) + safetensors/bin 선택
# ============================================================
def choose_local_weight_format(local_dir: str):
    st = os.path.join(local_dir, "model.safetensors")
    pt = os.path.join(local_dir, "pytorch_model.bin")

    if os.path.exists(st):
        try:
            if os.path.getsize(st) >= 1024 * 1024:
                return "safetensors"
        except OSError:
            pass

    if os.path.exists(pt):
        try:
            if os.path.getsize(pt) >= 1024 * 1024:
                return "bin"
        except OSError:
            pass

    return None


def resolve_local_first(model_id: str, base_dir: str = "./models"):
    """
    model_id can be:
      - HF repo id (e.g., "openai/clip-vit-base-patch32")
      - local dir path (absolute or relative) containing config.json + weights
      - local relative under base_dir (e.g., base_dir/model_id)
    """
    if os.path.isdir(model_id):
        local_dir = model_id
    else:
        local_dir = os.path.join(base_dir, model_id)

    local_has_config = os.path.exists(os.path.join(local_dir, "config.json"))
    local_weight_kind = choose_local_weight_format(local_dir) if local_has_config else None

    use_local = local_has_config and (local_weight_kind is not None)
    model_path_or_id = local_dir if use_local else model_id

    if use_local:
        use_safetensors = (local_weight_kind == "safetensors")
        print(f"Model source: LOCAL -> {model_path_or_id}")
        print(f"✅ Local weight format: {local_weight_kind} (use_safetensors={use_safetensors})")
        return model_path_or_id, True, use_safetensors, local_weight_kind
    else:
        print(f"Model source: HF -> {model_path_or_id}")
        return model_path_or_id, False, None, None


# ============================================================
# 1) Transforms (✅ augmentation 제거 버전)
# ============================================================
def get_rgb_transforms(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
    )


# ============================================================
# 2) CLIP Grid Heatmap (8x8) + (옵션) 주변 컨텍스트 + grid smoothing
# ============================================================
@torch.no_grad()
def clip_grid_heatmap(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    image_rgb_uint8: np.ndarray,
    prompts: List[str],
    grid: int = 8,
    patch_px: int = 336,
    device: str = "cpu",
    context_ratio: float = 0.25,        # ✅ 셀 주변을 얼마나 더 포함할지 (0.0이면 딱 셀만)
    smooth_scores: bool = True,         # ✅ 8x8 score를 이웃 고려로 스무딩
    smooth_ksize: int = 3,              # 3 권장 (홀수)
    gamma: float = 1.0,                 # heat 강조/억제 (1.0이면 그대로)
) -> np.ndarray:
    """
    Returns:
      heat_full: (H,W) float32 in [0,1] (absolute normalization: cosine -> (x+1)/2)
    """
    clip_model.eval()
    H, W, _ = image_rgb_uint8.shape

    # ---- text embedding (avg prompts) ----
    text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(device)
    text_feats = clip_model.get_text_features(**text_inputs)  # (P, D)
    text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-6)
    text_feat = text_feats.mean(dim=0, keepdim=True)          # (1, D)
    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)

    ys = np.linspace(0, H, grid + 1, dtype=np.int32)
    xs = np.linspace(0, W, grid + 1, dtype=np.int32)
    scores01 = np.zeros((grid, grid), dtype=np.float32)

    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = ys[gy], ys[gy + 1]
            x0, x1 = xs[gx], xs[gx + 1]

            # ---- ✅ 주변 컨텍스트 포함: 셀 박스를 약간 확장 ----
            ch = max(1, y1 - y0)
            cw = max(1, x1 - x0)
            pad_y = int(round(ch * context_ratio))
            pad_x = int(round(cw * context_ratio))

            yy0 = max(0, y0 - pad_y)
            yy1 = min(H, y1 + pad_y)
            xx0 = max(0, x0 - pad_x)
            xx1 = min(W, x1 + pad_x)

            patch = image_rgb_uint8[yy0:yy1, xx0:xx1]
            if patch.size == 0:
                continue

            # CLIP 입력 크기로 맞춤 (large-336에 맞춰 기본 336)
            patch = cv2.resize(patch, (patch_px, patch_px), interpolation=cv2.INTER_LINEAR)

            img_inputs = clip_processor(images=patch, return_tensors="pt").to(device)
            img_feat = clip_model.get_image_features(**img_inputs)  # (1, D)
            img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-6)

            # cosine similarity in [-1,1] (정규화된 상태)
            sim = (img_feat * text_feat).sum(dim=-1).item()
            sim = float(np.clip(sim, -1.0, 1.0))

            # ---- ✅ absolute normalization: [-1,1] -> [0,1] ----
            s01 = (sim + 1.0) * 0.5
            s01 = float(np.clip(s01, 0.0, 1.0))

            scores01[gy, gx] = s01

    # ---- ✅ grid smoothing (이웃 고려) ----
    if smooth_scores and smooth_ksize and (smooth_ksize >= 3) and (smooth_ksize % 2 == 1):
        # 작은 grid라서 가우시안도 괜찮음. (평균필터 원하면 blur 대신 boxFilter 써도 됨)
        scores01 = cv2.GaussianBlur(scores01, (smooth_ksize, smooth_ksize), 0).astype(np.float32)

    # gamma
    if gamma is not None and float(gamma) != 1.0:
        scores01 = np.clip(scores01, 0.0, 1.0) ** float(gamma)

    # upsample to full res
    heat_full = cv2.resize(scores01, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    heat_full = np.clip(heat_full, 0.0, 1.0)
    return heat_full


def save_heat_overlay(image_rgb_uint8: np.ndarray, heat01_fullres: np.ndarray, out_path: str, alpha: float = 0.45):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    heat_u8 = np.clip(heat01_fullres * 255.0, 0, 255).astype(np.uint8)
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, heat_color_bgr, alpha, 0.0)
    cv2.imwrite(out_path, overlay)


# ============================================================
# 3) Dataset: (4ch input, gt_mask) with CLIP cache
# ============================================================
class FireLineClipDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 512,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        clip_prompts: Optional[List[str]] = None,
        clip_grid: int = 8,
        clip_patch_px: int = 336,
        clip_context_ratio: float = 0.25,
        clip_smooth_scores: bool = True,
        clip_smooth_ksize: int = 3,
        clip_cache_dir: str = "./clip_cache",
        clip_device: str = "cpu",
        local_models_base: str = "./models",
        save_heat_vis: bool = False,
        heat_gamma: float = 1.0,
        heat_blur_ksize: int = 0,
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = int(img_size)

        self.raw_dir = os.path.join(root_dir, split, "raw")
        self.gt_dir = os.path.join(root_dir, split, "gt")

        if not os.path.exists(self.raw_dir):
            self.images = []
        else:
            self.images = sorted([f for f in os.listdir(self.raw_dir) if f.lower().endswith(".png")])

        self.rgb_tfm = get_rgb_transforms(self.img_size)

        self.clip_model_name = clip_model_name
        self.clip_prompts = clip_prompts or [
            "leading edge of a wildfire",
            "wildfire front",
            "line of flames",
            "flame front",
            "flames at the ground",
            "flame base touching the ground",
            "front of a forest fire",
            "edge of the flames",
        ]

        self.clip_grid = int(clip_grid)
        self.clip_patch_px = int(clip_patch_px)
        self.clip_context_ratio = float(clip_context_ratio)
        self.clip_smooth_scores = bool(clip_smooth_scores)
        self.clip_smooth_ksize = int(clip_smooth_ksize)

        self.clip_cache_dir = os.path.join(clip_cache_dir, split)
        os.makedirs(self.clip_cache_dir, exist_ok=True)

        self.save_heat_vis = bool(save_heat_vis)
        self.vis_dir = os.path.join(clip_cache_dir, "vis", split)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.clip_device = clip_device
        self.heat_gamma = float(heat_gamma)
        self.heat_blur_ksize = int(heat_blur_ksize)

        # ---- CLIP local-first ----
        clip_path_or_id, clip_is_local, clip_use_safetensors, _ = resolve_local_first(
            clip_model_name, base_dir=local_models_base
        )

        self.clip_processor = CLIPProcessor.from_pretrained(
            clip_path_or_id,
            local_files_only=clip_is_local
        )

        clip_kwargs = dict(local_files_only=clip_is_local)
        if clip_is_local and (clip_use_safetensors is not None):
            clip_kwargs["use_safetensors"] = clip_use_safetensors

        self.clip_model = CLIPModel.from_pretrained(
            clip_path_or_id,
            **clip_kwargs
        ).to(self.clip_device)
        self.clip_model.eval()

    def __len__(self):
        return len(self.images)

    def _cache_key(self) -> str:
        prompts_norm = [p.strip().lower() for p in self.clip_prompts]
        payload = (
            f"grid|{self.clip_model_name}|{prompts_norm}"
            f"|g{self.clip_grid}|px{self.clip_patch_px}"
            f"|ctx{self.clip_context_ratio}|sm{int(self.clip_smooth_scores)}k{self.clip_smooth_ksize}"
            f"|img{self.img_size}|gamma{self.heat_gamma}|blur{self.heat_blur_ksize}"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def _heat_cache_path(self, img_name: str) -> str:
        base = os.path.splitext(img_name)[0]
        k = self._cache_key()
        return os.path.join(self.clip_cache_dir, f"{base}_grid_{k}.npy")

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.raw_dir, img_name)

        # GT: "동일한 파일명" 우선, 없으면 _Raw-> _Line fallback
        gt_path = os.path.join(self.gt_dir, img_name)
        if not os.path.exists(gt_path):
            alt = os.path.join(self.gt_dir, img_name.replace("_Raw.png", "_Line.png"))
            if os.path.exists(alt):
                gt_path = alt

        if not os.path.exists(gt_path):
            raise RuntimeError(f"GT not found for {img_name}")

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise RuntimeError(f"Failed to read gt: {gt_path}")
        gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        gt01 = (gt > 128).astype(np.float32)

        cache_path = self._heat_cache_path(img_name)
        if os.path.exists(cache_path):
            heat = np.load(cache_path).astype(np.float32)  # (img_size,img_size) in [0,1]
        else:
            heat_full = clip_grid_heatmap(
                self.clip_model,
                self.clip_processor,
                image_rgb_uint8=image_rgb,
                prompts=self.clip_prompts,
                grid=self.clip_grid,
                patch_px=self.clip_patch_px,
                device=self.clip_device,
                context_ratio=self.clip_context_ratio,
                smooth_scores=self.clip_smooth_scores,
                smooth_ksize=self.clip_smooth_ksize,
                gamma=self.heat_gamma,
            )

            # 학습 해상도에 맞춰 저장
            heat = cv2.resize(heat_full, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_LINEAR).astype(np.float32)

            if self.heat_blur_ksize and self.heat_blur_ksize >= 3 and (self.heat_blur_ksize % 2 == 1):
                heat = cv2.GaussianBlur(heat, (self.heat_blur_ksize, self.heat_blur_ksize), 0)

            heat = np.clip(heat, 0.0, 1.0).astype(np.float32)

            tmp_path = cache_path + f".tmp_{os.getpid()}"
            with open(tmp_path, "wb") as f:
                np.save(f, heat)
            os.replace(tmp_path, cache_path)

            if self.save_heat_vis:
                base = os.path.splitext(img_name)[0]
                overlay_path = os.path.join(self.vis_dir, f"{base}_grid_overlay.png")
                save_heat_overlay(image_rgb, heat_full, overlay_path, alpha=0.45)

        # transform (no aug)
        out = self.rgb_tfm(image=image_rgb)
        img_t = out["image"]  # (3,H,W)

        # heat normalize: [0,1] -> [-1,1]
        heat_norm = (heat - 0.5) / 0.5
        heat_t = torch.from_numpy(heat_norm).unsqueeze(0).float()  # (1,H,W)

        x4 = torch.cat([img_t, heat_t], dim=0)  # (4,H,W)
        gt_t = torch.from_numpy(gt01).unsqueeze(0).float()
        return x4, gt_t


# ============================================================
# 4) Loss
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return (1.0 - dice).mean()


@torch.no_grad()
def dice_score_from_logits(logits, targets, thr=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    pred = (probs > thr).float()
    t = targets.float()
    inter = (pred * t).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    return ((2.0 * inter + eps) / (union + eps)).mean().item()


# ============================================================
# 5) Metrics
# ============================================================
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


def tol_f1(pred01: np.ndarray, gt01: np.ndarray, tol_px: int = 3, eps: float = 1e-6):
    pred = (pred01 > 0).astype(np.uint8)
    gt = (gt01 > 0).astype(np.uint8)

    if tol_px <= 0:
        return metrics_binary(pred, gt, eps=eps)["F1"]

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
    gt_dil = cv2.dilate((gt * 255).astype(np.uint8), k, iterations=1) > 0
    pr_dil = cv2.dilate((pred * 255).astype(np.uint8), k, iterations=1) > 0

    tp_p = np.logical_and(pred == 1, gt_dil).sum()
    fp_p = np.logical_and(pred == 1, np.logical_not(gt_dil)).sum()
    prec = tp_p / (tp_p + fp_p + eps)

    tp_r = np.logical_and(gt == 1, pr_dil).sum()
    fn_r = np.logical_and(gt == 1, np.logical_not(pr_dil)).sum()
    rec = tp_r / (tp_r + fn_r + eps)

    f1 = 2 * prec * rec / (prec + rec + eps)
    return float(f1)


# ============================================================
# 6) Model: 4ch adapter -> SegFormer(3ch)
# ============================================================
class SegFormerWithInputAdapter(nn.Module):
    def __init__(self, seg_model_name: str, local_models_base: str = "./models"):
        super().__init__()
        self.adapter = nn.Conv2d(4, 3, kernel_size=1, bias=False)

        seg_path_or_id, seg_is_local, seg_use_safetensors, _ = resolve_local_first(
            seg_model_name, base_dir=local_models_base
        )

        seg_kwargs = dict(
            num_labels=1,
            ignore_mismatched_sizes=True,
            local_files_only=seg_is_local
        )
        if seg_is_local and (seg_use_safetensors is not None):
            seg_kwargs["use_safetensors"] = seg_use_safetensors

        self.base = SegformerForSemanticSegmentation.from_pretrained(
            seg_path_or_id,
            **seg_kwargs
        )

    def forward(self, x4):
        x3 = self.adapter(x4)
        out = self.base(pixel_values=x3)
        return out.logits


# ============================================================
# 7) Evaluation helper
# ============================================================
@torch.no_grad()
def evaluate_one_loader(model, loader, criterion, args, device, name="Val"):
    model.eval()

    total_loss = 0.0
    keys = ["IoU", "Dice", "Precision", "Recall", "F1", "HD95", "HD", "ASSD", "TolF1"]
    sum_metrics = {k: 0.0 for k in keys}
    count = 0

    pbar = tqdm(loader, desc=f"[Eval {name}]", leave=False)
    for x4, gt in pbar:
        x4 = x4.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        logits = model(x4)
        logits = F.interpolate(logits, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)

        loss = criterion(logits, gt)
        total_loss += loss.item()

        prob = torch.sigmoid(logits).detach().cpu().numpy()
        pred_np = (prob > args.thr).astype(np.uint8)
        gt_np = (gt.detach().cpu().numpy() > 0.5).astype(np.uint8)

        B = pred_np.shape[0]
        for b in range(B):
            pred = pred_np[b, 0]
            gtb = gt_np[b, 0]

            m = metrics_binary(pred, gtb)
            sum_metrics["IoU"] += m["IoU"]
            sum_metrics["Dice"] += m["Dice"]
            sum_metrics["Precision"] += m["Precision"]
            sum_metrics["Recall"] += m["Recall"]
            sum_metrics["F1"] += m["F1"]
            sum_metrics["HD95"] += hd95(pred, gtb)
            sum_metrics["HD"] += hausdorff_distance(pred, gtb)
            sum_metrics["ASSD"] += assd(pred, gtb)
            sum_metrics["TolF1"] += tol_f1(pred, gtb, tol_px=args.tol_px)
            count += 1

        avg_dice_tmp = sum_metrics["Dice"] / max(1, count)
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{avg_dice_tmp:.4f}")

    avg_loss = total_loss / max(1, len(loader))
    avg_metrics = {k: (sum_metrics[k] / max(1, count)) for k in keys}
    return avg_loss, avg_metrics


# ============================================================
# 8) Train
# ============================================================
def train(args):
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    run_id = datetime.now().strftime(f"{args.run_name}_%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)

    tb_dir = os.path.join(save_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"📈 TensorBoard logdir: {tb_dir}")

    clip_prompts = [p.strip() for p in args.clip_prompts.split(",") if p.strip()]

    print(f"Training start: {save_dir}")
    print(f"Device: {device}")
    print(f"DATA_ROOT={args.data_root}")
    print(f"SegFormer={args.model_name} | IMG={args.img_size} | BS={args.batch_size} | LR={args.lr} | EP={args.epochs}")
    print(f"CLIP={args.clip_model} | clip_device={args.clip_device}")
    print(f"Heat(grid): grid={args.clip_grid} patch_px={args.clip_patch_px} ctx={args.clip_context_ratio} "
          f"smooth={args.clip_smooth_scores} k={args.clip_smooth_ksize} gamma={args.heat_gamma} blur={args.heat_blur_ksize}")
    print(f"LOSS=dice")
    print(f"VAL metrics: IoU/Dice/Precision/Recall/F1/HD95/HD/ASSD/TolF1 (thr={args.thr}, tol_px={args.tol_px})")
    if args.clip_device == "cuda" and args.num_workers > 0:
        print("⚠️ WARNING: clip_device=cuda with num_workers>0 can crash. Use --num_workers 0 for safety.")

    train_ds = FireLineClipDataset(
        args.data_root, "train",
        img_size=args.img_size,
        clip_model_name=args.clip_model,
        clip_prompts=clip_prompts,
        clip_grid=args.clip_grid,
        clip_patch_px=args.clip_patch_px,
        clip_context_ratio=args.clip_context_ratio,
        clip_smooth_scores=args.clip_smooth_scores,
        clip_smooth_ksize=args.clip_smooth_ksize,
        clip_cache_dir=args.clip_cache_dir,
        clip_device=args.clip_device,
        local_models_base=args.local_models_base,
        save_heat_vis=args.save_heat_vis,
        heat_gamma=args.heat_gamma,
        heat_blur_ksize=args.heat_blur_ksize,
    )
    val_ds = FireLineClipDataset(
        args.data_root, "val",
        img_size=args.img_size,
        clip_model_name=args.clip_model,
        clip_prompts=clip_prompts,
        clip_grid=args.clip_grid,
        clip_patch_px=args.clip_patch_px,
        clip_context_ratio=args.clip_context_ratio,
        clip_smooth_scores=args.clip_smooth_scores,
        clip_smooth_ksize=args.clip_smooth_ksize,
        clip_cache_dir=args.clip_cache_dir,
        clip_device=args.clip_device,
        local_models_base=args.local_models_base,
        save_heat_vis=args.save_heat_vis,
        heat_gamma=args.heat_gamma,
        heat_blur_ksize=args.heat_blur_ksize,
    )

    real_val_dir = os.path.join(args.data_root, "real_val", "raw")
    real_val_ds = None
    if os.path.exists(real_val_dir):
        real_val_ds = FireLineClipDataset(
            args.data_root, "real_val",
            img_size=args.img_size,
            clip_model_name=args.clip_model,
            clip_prompts=clip_prompts,
            clip_grid=args.clip_grid,
            clip_patch_px=args.clip_patch_px,
            clip_context_ratio=args.clip_context_ratio,
            clip_smooth_scores=args.clip_smooth_scores,
            clip_smooth_ksize=args.clip_smooth_ksize,
            clip_cache_dir=args.clip_cache_dir,
            clip_device=args.clip_device,
            local_models_base=args.local_models_base,
            save_heat_vis=args.save_heat_vis,
            heat_gamma=args.heat_gamma,
            heat_blur_ksize=args.heat_blur_ksize,
        )
        if len(real_val_ds) == 0:
            real_val_ds = None

    if len(train_ds) == 0:
        print("❌ train 데이터가 없습니다.")
        writer.close()
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )

    real_val_loader = None
    if real_val_ds is not None:
        real_val_loader = DataLoader(
            real_val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device == "cuda")
        )
        print(f"✅ real_val enabled: {len(real_val_ds)} images")
    else:
        print("ℹ️ real_val not found or empty -> skip real_val evaluation.")

    model = SegFormerWithInputAdapter(args.model_name, local_models_base=args.local_models_base).to(device)
    criterion = DiceLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    meta = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "data_root": args.data_root,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "model_name": args.model_name,
        "clip_model": args.clip_model,
        "clip_device": args.clip_device,
        "clip_prompts": clip_prompts,
        "clip_cache_dir": args.clip_cache_dir,
        "save_heat_vis": bool(args.save_heat_vis),
        "augmentation": "OFF",
        "heatmap": {
            "type": "CLIP_grid_patch",
            "grid": int(args.clip_grid),
            "patch_px": int(args.clip_patch_px),
            "context_ratio": float(args.clip_context_ratio),
            "smooth_scores": bool(args.clip_smooth_scores),
            "smooth_ksize": int(args.clip_smooth_ksize),
            "gamma": float(args.heat_gamma),
            "blur_ksize": int(args.heat_blur_ksize),
            "norm": "cosine(-1..1)->(x+1)/2 (no per-image minmax)",
        },
        "loss": {"name": "DiceLoss"},
        "val_metrics": ["IoU", "Dice", "Precision", "Recall", "F1", "HD95", "HD", "ASSD", "TolF1"],
        "extra_validation": ["real_val"] if real_val_loader is not None else []
    }

    meta_path = os.path.join(save_dir, "train_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"📝 Saved metadata: {meta_path}")

    best_val = float("inf")
    best_path = os.path.join(save_dir, "best_fireline_model.pth")

    best_real = float("inf")
    real_best_path = os.path.join(save_dir, "real_best_fireline_model.pth")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        t_loss = 0.0
        t_dice = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for x4, gt in pbar:
            x4 = x4.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x4)
            logits = F.interpolate(logits, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)

            loss = criterion(logits, gt)
            loss.backward()
            optimizer.step()

            d = dice_score_from_logits(logits.detach(), gt, thr=args.thr)
            t_loss += loss.item()
            t_dice += d

            writer.add_scalar("step/loss", loss.item(), global_step)
            writer.add_scalar("step/dice", d, global_step)
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

        t_loss /= max(1, len(train_loader))
        t_dice /= max(1, len(train_loader))

        v_loss, v_metrics = evaluate_one_loader(model, val_loader, criterion, args, device, name="Val")

        rv_loss, rv_metrics = None, None
        if real_val_loader is not None:
            rv_loss, rv_metrics = evaluate_one_loader(model, real_val_loader, criterion, args, device, name="RealVal")

        writer.add_scalar("epoch/loss_train", t_loss, epoch + 1)
        writer.add_scalar("epoch/dice_train", t_dice, epoch + 1)

        writer.add_scalar("epoch/loss_val", v_loss, epoch + 1)
        for k, v in v_metrics.items():
            writer.add_scalar(f"metrics/val_{k}", v, epoch + 1)

        if rv_loss is not None:
            writer.add_scalar("epoch/loss_real_val", rv_loss, epoch + 1)
            for k, v in rv_metrics.items():
                writer.add_scalar(f"metrics/real_val_{k}", v, epoch + 1)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
        writer.flush()

        msg = (
            f"Epoch {epoch+1}: "
            f"Train loss={t_loss:.4f} dice={t_dice:.4f} | "
            f"Val loss={v_loss:.4f} Dice={v_metrics['Dice']:.4f} IoU={v_metrics['IoU']:.4f} "
            f"F1={v_metrics['F1']:.4f} HD95={v_metrics['HD95']:.2f} HD={v_metrics['HD']:.2f} "
            f"ASSD={v_metrics['ASSD']:.2f} TolF1={v_metrics['TolF1']:.4f}"
        )
        if rv_loss is not None:
            msg += (
                f" | RealVal loss={rv_loss:.4f} Dice={rv_metrics['Dice']:.4f} IoU={rv_metrics['IoU']:.4f} "
                f"F1={rv_metrics['F1']:.4f} HD95={rv_metrics['HD95']:.2f} HD={rv_metrics['HD']:.2f} "
                f"ASSD={rv_metrics['ASSD']:.2f} TolF1={rv_metrics['TolF1']:.4f}"
            )
        print(msg)

        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), best_path)
            print(f"  💾 Best(Val) saved: {best_path} (val_loss={best_val:.4f})")
            meta["best_val"] = {
                "epoch": epoch + 1,
                "val_loss": float(v_loss),
                "val_metrics": {k: float(v_metrics[k]) for k in v_metrics.keys()},
                "path": os.path.basename(best_path),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        if rv_loss is not None and rv_loss < best_real:
            best_real = rv_loss
            torch.save(model.state_dict(), real_best_path)
            print(f"  💾 Best(RealVal) saved: {real_best_path} (real_val_loss={best_real:.4f})")
            meta["best_real_val"] = {
                "epoch": epoch + 1,
                "real_val_loss": float(rv_loss),
                "real_val_metrics": {k: float(rv_metrics[k]) for k in rv_metrics.keys()},
                "path": os.path.basename(real_best_path),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    last_path = os.path.join(save_dir, "last_fireline_model.pth")
    torch.save(model.state_dict(), last_path)
    meta["last"] = {"epoch": args.epochs, "path": os.path.basename(last_path)}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    writer.close()

    print(f"\n✅ Training done.")
    print(f"Best(Val): {best_path}")
    if real_val_loader is not None:
        print(f"Best(RealVal): {real_best_path}")
    print(f"Last: {last_path}")
    print(f"Run dir: {save_dir}")
    print(f"Metadata: {meta_path}")
    print(f"TensorBoard: {tb_dir}")


# ============================================================
# 9) Argparse
# ============================================================
def str2bool(v: str) -> bool:
    v = v.strip().lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("bool value expected")


def build_parser():
    p = argparse.ArgumentParser(
        "Train SegFormer with CLIP Grid Heatmap (RGB+1) + DiceLoss + Val/RealVal metrics"
    )

    p.add_argument("--data_root", type=str, required=True,
                   help="Dataset root (expects train/raw, train/gt, val/raw, val/gt subdirs)")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--model_name", type=str, default="./models/nvidia/segformer-b3-finetuned-ade-512-512")

    # ✅ CLIP: large-336 기본
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14-336")
    p.add_argument("--clip_prompts", type=str,
                   default="leading edge of a wildfire, wildfire front, line of flames, flame front, flames at the ground, flame base touching the ground, front of a forest fire, edge of the flames")
    p.add_argument("--clip_cache_dir", type=str, default="./clip_cache")
    p.add_argument("--clip_device", type=str, default="cpu", choices=["cpu", "cuda"])

    # ✅ grid heat 옵션
    p.add_argument("--clip_grid", type=int, default=8)
    p.add_argument("--clip_patch_px", type=int, default=336)
    p.add_argument("--clip_context_ratio", type=float, default=0.25,
                   help="each grid cell expanded by this ratio to include neighbor context (0.0 = no expansion)")
    p.add_argument("--clip_smooth_scores", type=str2bool, default=True,
                   help="smooth 8x8 score grid before upsampling")
    p.add_argument("--clip_smooth_ksize", type=int, default=3)

    # heat postprocess
    p.add_argument("--heat_gamma", type=float, default=1.0)
    p.add_argument("--heat_blur_ksize", type=int, default=0)

    p.add_argument("--save_heat_vis", action="store_true")

    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--tol_px", type=int, default=3)

    p.add_argument("--out_dir", type=str, default="./train_runs")
    p.add_argument("--run_name", type=str, default="rgb_clipheat_grid_dice")

    p.add_argument("--local_models_base", type=str, default="./models")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)

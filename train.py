# Unified training script: CLIP-Heat mode (default) + Baseline RGB-only mode (--no-clip)
#
#   Default (no flag): 4-channel SegFormer4Ch with CLIP heatmap  (train_clipheat.py behaviour)
#   --no-clip flag:    standard 3-channel SegFormer, RGB only      (train_baseline.py behaviour)

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

from transformers import SegformerForSemanticSegmentation


# ============================================================
# 0) Helpers: local-first + safetensors/bin selection
# ============================================================
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
        try:
            if os.path.getsize(pt) < 1024 * 1024:
                print(f"⚠️ suspicious pytorch_model.bin too small -> ignore: {pt} ({os.path.getsize(pt)} bytes)")
            else:
                return "bin"
        except OSError:
            pass

    return None


def resolve_local_first(model_id: str, base_dir: str = "./models"):
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
# 1) Transforms
# ============================================================
def get_rgb_transforms(img_size: int):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# ============================================================
# 2) GT utilities (band/weightmap)
# ============================================================
def make_band_from_line(mask01: np.ndarray, band_width_px: int) -> np.ndarray:
    if band_width_px <= 1:
        return (mask01 > 0.5).astype(np.float32)

    m = (mask01 > 0.5).astype(np.uint8) * 255
    k = max(1, int(band_width_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(m, kernel, iterations=1)
    return (dil > 0).astype(np.float32)


def make_weight_map_from_band(band01: np.ndarray, max_weight: float = 6.0, sigma: float = 4.0) -> np.ndarray:
    band = (band01 > 0.5).astype(np.uint8)
    inv = (1 - band).astype(np.uint8)
    dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)

    sigma = max(1e-3, float(sigma))
    w = 1.0 + (float(max_weight) - 1.0) * np.exp(-dist / sigma)
    return w.astype(np.float32)


# ============================================================
# 3) CLIP heatmap generator (grid patches) — only used in CLIP mode
# ============================================================
@torch.no_grad()
def clip_fire_heatmap(
    clip_model,
    clip_processor,
    image_rgb_uint8: np.ndarray,
    prompts: list,
    grid: int = 16,
    patch_px: int = 224,
    device: str = "cpu"
) -> np.ndarray:
    """
    Returns heatmap (H,W) float32 in [0,1], same size as input image.
    grid x grid patches, CLIP similarity to text prompts.
    """
    H, W, _ = image_rgb_uint8.shape

    # text embedding (avg)
    text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(device)
    text_feats = clip_model.get_text_features(**text_inputs)  # (P, D)
    text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-6)
    text_feat = text_feats.mean(dim=0, keepdim=True)
    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)

    ys = np.linspace(0, H, grid + 1, dtype=np.int32)
    xs = np.linspace(0, W, grid + 1, dtype=np.int32)
    scores = np.zeros((grid, grid), dtype=np.float32)

    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = ys[gy], ys[gy + 1]
            x0, x1 = xs[gx], xs[gx + 1]
            patch = image_rgb_uint8[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            patch = cv2.resize(patch, (patch_px, patch_px), interpolation=cv2.INTER_LINEAR)
            img_inputs = clip_processor(images=patch, return_tensors="pt").to(device)
            img_feat = clip_model.get_image_features(**img_inputs)
            img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-6)

            sim = (img_feat * text_feat).sum(dim=-1).item()
            scores[gy, gx] = sim

    smin, smax = float(scores.min()), float(scores.max())
    if smax - smin < 1e-6:
        norm = np.zeros_like(scores, dtype=np.float32)
    else:
        norm = (scores - smin) / (smax - smin)

    heat = cv2.resize(norm, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return heat


# ============================================================
# 4a) Dataset for CLIP-Heat mode: (4ch input, band_mask, wmap)
# ============================================================
class FireLineClipDataset(Dataset):
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
        clip_cache_dir: str = "./clip_cache",
        clip_device: str = "cpu",
        local_models_base: str = "./models",
    ):
        from transformers import CLIPProcessor, CLIPModel

        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        self.raw_dir = os.path.join(root_dir, split, "raw")
        self.gt_dir = os.path.join(root_dir, split, "gt")

        if not os.path.exists(self.raw_dir):
            self.images = []
        else:
            self.images = sorted([f for f in os.listdir(self.raw_dir) if f.lower().endswith(".png")])

        self.rgb_tfm = get_rgb_transforms(img_size)
        self.band_width_px = band_width_px
        self.max_weight = max_weight
        self.weight_sigma = weight_sigma

        self.clip_model_name = clip_model_name
        self.clip_prompts = clip_prompts or ["fire", "flames", "wildfire", "burning"]
        self.clip_grid = int(clip_grid)

        self.clip_cache_dir = os.path.join(clip_cache_dir, split)
        os.makedirs(self.clip_cache_dir, exist_ok=True)

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

        self.clip_device = clip_device
        self.clip_model = CLIPModel.from_pretrained(
            clip_path_or_id,
            **clip_kwargs
        ).to(self.clip_device)
        self.clip_model.eval()

    def __len__(self):
        return len(self.images)

    def _heat_cache_path(self, img_name: str) -> str:
        base = os.path.splitext(img_name)[0]
        return os.path.join(self.clip_cache_dir, f"{base}_g{self.clip_grid}.npy")

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.raw_dir, img_name)

        # GT path
        line_name = img_name.replace("_Raw.png", "_Line.png")
        line_path = os.path.join(self.gt_dir, line_name)
        if not os.path.exists(line_path):
            alt = os.path.join(self.gt_dir, img_name)
            if os.path.exists(alt):
                line_path = alt
        if not os.path.exists(line_path):
            raise RuntimeError(f"GT not found for {img_name}")

        # read image RGB
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # read GT line
        line = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)
        if line is None:
            raise RuntimeError(f"Failed to read gt: {line_path}")
        line = cv2.resize(line, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        line01 = (line > 128).astype(np.float32)

        # band + weightmap
        band01 = make_band_from_line(line01, self.band_width_px)
        wmap = make_weight_map_from_band(band01, self.max_weight, self.weight_sigma)

        # CLIP heatmap (cache)
        cache_path = self._heat_cache_path(img_name)
        if os.path.exists(cache_path):
            heat = np.load(cache_path).astype(np.float32)
        else:
            heat_full = clip_fire_heatmap(
                self.clip_model,
                self.clip_processor,
                image_rgb,
                prompts=self.clip_prompts,
                grid=self.clip_grid,
                patch_px=224,
                device=self.clip_device
            )
            heat = cv2.resize(heat_full, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            np.save(cache_path, heat)

        # RGB transforms -> (3,H,W)
        out = self.rgb_tfm(image=image_rgb)
        img_t = out["image"]

        # heat -> (1,H,W)
        heat_t = torch.from_numpy(heat).unsqueeze(0).float()

        # concat -> (4,H,W)
        x4 = torch.cat([img_t, heat_t], dim=0)

        band_t = torch.from_numpy(band01).unsqueeze(0).float()
        wmap_t = torch.from_numpy(wmap).unsqueeze(0).float()
        return x4, band_t, wmap_t


# ============================================================
# 4b) Dataset for baseline (RGB-only) mode: (3ch input, mask)
# ============================================================
class FireLineDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.raw_dir = os.path.join(root_dir, split, "raw")
        self.gt_dir  = os.path.join(root_dir, split, "gt")

        if not os.path.exists(self.raw_dir):
            self.images = []
        else:
            self.images = sorted([f for f in os.listdir(self.raw_dir) if f.lower().endswith(".png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.raw_dir, img_name)

        # Rule: _Raw.png -> _Line.png
        mask_name = img_name.replace("_Raw.png", "_Line.png")
        mask_path = os.path.join(self.gt_dir, mask_name)

        # fallback: in case GT has the same name as raw
        if not os.path.exists(mask_path):
            alt = os.path.join(self.gt_dir, img_name)
            if os.path.exists(alt):
                mask_path = alt

        if not os.path.exists(mask_path):
            raise RuntimeError(f"Mask not found for image: {img_name}\nTried: {mask_path}")

        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        mask = (mask > 128).astype(np.float32)  # 0/1 float

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image = out["image"]                 # (3,H,W)
            mask  = out["mask"].unsqueeze(0)     # (1,H,W)
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            mask  = torch.tensor(mask).unsqueeze(0).float()

        return image, mask


# ============================================================
# 5) Losses (dice / bce / dice+bce / wft)
#    forward(logits, targets, weight_map=None)
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, logits, targets, weight_map=None):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return (1.0 - dice).mean()


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, weight_map=None):
        return self.bce(logits, targets)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_w=1.0, bce_w=1.0, eps=1e-6):
        super().__init__()
        self.dice = DiceLoss(eps=eps)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_w = float(dice_w)
        self.bce_w = float(bce_w)

    def forward(self, logits, targets, weight_map=None):
        dl = self.dice(logits, targets)
        bl = self.bce(logits, targets)
        return self.dice_w * dl + self.bce_w * bl


class WeightedFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.33, eps=1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, logits, targets, weight_map=None):
        probs = torch.sigmoid(logits)
        if weight_map is None:
            weight_map = torch.ones_like(targets)

        w = weight_map
        tp = (w * probs * targets).sum(dim=(1, 2, 3))
        fp = (w * probs * (1.0 - targets)).sum(dim=(1, 2, 3))
        fn = (w * (1.0 - probs) * targets).sum(dim=(1, 2, 3))

        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return ((1.0 - ti) ** self.gamma).mean()


def build_criterion(args):
    name = args.loss.lower()

    if name == "dice":
        return DiceLoss()

    if name == "bce":
        return BCELoss()

    if name in ("dicebce", "dice_bce", "dice+bce"):
        return DiceBCELoss(dice_w=args.dice_w, bce_w=args.bce_w)

    if name in ("wft", "weightedfocaltversky", "weighted_focal_tversky"):
        return WeightedFocalTverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    raise ValueError(f"Unknown --loss {args.loss}. Use one of: wft, dice, bce, dicebce")


@torch.no_grad()
def dice_score_from_logits(logits, targets, thr=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    pred = (probs > thr).float()
    t = targets.float()
    inter = (pred * t).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    return ((2.0 * inter + eps) / (union + eps)).mean().item()


# ============================================================
# 6) Validation metrics (including HD/ASSD)
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
# 7) Model: SegFormer4Ch (CLIP mode) — first input conv patched to 4ch
# ============================================================
def _replace_first_conv_in_segformer_to_4ch(seg_model: SegformerForSemanticSegmentation):
    """
    Replace SegFormer's first input Conv (in_channels=3) with in_channels=4,
    copy pretrained RGB weights, and initialize heat channel weights to 0.
    """
    proj = None
    try:
        proj = seg_model.segformer.encoder.patch_embeddings[0].proj
    except Exception:
        proj = None

    if isinstance(proj, nn.Conv2d) and proj.in_channels == 3:
        old = proj
        new = nn.Conv2d(
            in_channels=4,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=(old.bias is not None),
            padding_mode=old.padding_mode,
        )
        with torch.no_grad():
            new.weight[:, :3, :, :].copy_(old.weight)   # copy RGB weights
            new.weight[:, 3:4, :, :].zero_()            # heat channel init to 0
            if old.bias is not None:
                new.bias.copy_(old.bias)

        seg_model.segformer.encoder.patch_embeddings[0].proj = new
        return seg_model

    first_name = None
    first_conv = None
    for name, m in seg_model.segformer.named_modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            first_name = name
            first_conv = m
            break

    if first_conv is None:
        raise RuntimeError("Could not find first Conv2d with in_channels=3 inside segformer.")

    old = first_conv
    new = nn.Conv2d(
        in_channels=4,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    )
    with torch.no_grad():
        new.weight[:, :3, :, :].copy_(old.weight)
        new.weight[:, 3:4, :, :].zero_()
        if old.bias is not None:
            new.bias.copy_(old.bias)

    parent = seg_model.segformer
    parts = first_name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)

    return seg_model


class SegFormer4Ch(nn.Module):
    """
    (B,4,H,W) -> SegFormer(first input conv patched to 4ch) -> logits
    Used in CLIP-Heat mode (default).
    """
    def __init__(self, seg_model_name: str, local_models_base: str = "./models"):
        super().__init__()

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

        self.base = _replace_first_conv_in_segformer_to_4ch(self.base)

    def forward(self, x4):
        out = self.base(pixel_values=x4)
        return out.logits


# ============================================================
# 8) Eval helper (shared for Val/RealVal)
#    Works for both modes: CLIP batches yield (x, band, wmap),
#    baseline batches yield (imgs, masks) — wmap is None in baseline.
# ============================================================
@torch.no_grad()
def run_eval(model, loader, criterion, args, device, name="Val", use_clip=True):
    model.eval()
    total_loss = 0.0

    keys = ["IoU", "Dice", "Precision", "Recall", "F1", "HD95", "HD", "ASSD", "TolF1"]
    sum_metrics = {k: 0.0 for k in keys}
    count = 0

    pbar = tqdm(loader, desc=f"Epoch [Eval {name}]", leave=False)
    for batch in pbar:
        if use_clip:
            x, band, wmap = batch
            x = x.to(device, non_blocking=True)
            band = band.to(device, non_blocking=True)
            wmap = wmap.to(device, non_blocking=True)
            logits = model(x)
            logits = F.interpolate(logits, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)
            loss = criterion(logits, band, wmap)
            gt_band = band
        else:
            imgs, masks = batch
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            outputs = model(pixel_values=imgs)
            logits = F.interpolate(outputs.logits, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            gt_band = masks

        total_loss += loss.item()

        prob = torch.sigmoid(logits).detach().cpu().numpy()
        pred_np = (prob > args.thr).astype(np.uint8)
        gt_np = (gt_band.detach().cpu().numpy() > 0.5).astype(np.uint8)

        B = pred_np.shape[0]
        for b in range(B):
            pred = pred_np[b, 0]
            gt = gt_np[b, 0]

            m = metrics_binary(pred, gt)
            sum_metrics["IoU"] += m["IoU"]
            sum_metrics["Dice"] += m["Dice"]
            sum_metrics["Precision"] += m["Precision"]
            sum_metrics["Recall"] += m["Recall"]
            sum_metrics["F1"] += m["F1"]
            sum_metrics["HD95"] += hd95(pred, gt)
            sum_metrics["HD"] += hausdorff_distance(pred, gt)
            sum_metrics["ASSD"] += assd(pred, gt)
            sum_metrics["TolF1"] += tol_f1(pred, gt, tol_px=args.tol_px)
            count += 1

        avg_dice_tmp = sum_metrics["Dice"] / max(1, count)
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{avg_dice_tmp:.4f}")

    avg_loss = total_loss / max(1, len(loader))
    avg_metrics = {k: (sum_metrics[k] / max(1, count)) for k in keys}
    return avg_loss, avg_metrics


# ============================================================
# 9) Train (with TensorBoard) + RealVal evaluation + checkpoint saving
# ============================================================
def train(args):
    use_clip = not args.no_clip

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    run_id = datetime.now().strftime(f"{args.run_name}_%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)

    # TensorBoard
    tb_dir = os.path.join(save_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"📈 TensorBoard logdir: {tb_dir}")

    print(f"Training start: {save_dir}")
    print(f"Device: {device}")
    print(f"Mode: {'CLIP-Heat (4ch)' if use_clip else 'Baseline RGB (3ch)'}")
    print(f"DATA_ROOT={args.data_root}")
    print(f"SegFormer={args.model_name} | IMG={args.img_size} | BS={args.batch_size} | LR={args.lr} | EP={args.epochs}")
    print(f"BAND={args.band_width_px} | W_MAX={args.max_weight} | W_SIGMA={args.weight_sigma}")
    print(f"LOSS={args.loss} (dicebce: dice_w={args.dice_w}, bce_w={args.bce_w})")
    print(f"VAL metrics: IoU/Dice/Precision/Recall/F1/HD95/HD/ASSD/TolF1 (thr={args.thr}, tol_px={args.tol_px})")

    if use_clip:
        clip_prompts = [p.strip() for p in args.clip_prompts.split(",") if p.strip()]
        print(f"CLIP={args.clip_model} | grid={args.clip_grid} | clip_device={args.clip_device}")
        print(f"Prompts={clip_prompts}")
        if args.clip_device == "cuda" and args.num_workers > 0:
            print("⚠️ WARNING: clip_device=cuda with num_workers>0 can crash. Use --num_workers 0 for safety.")

    # ---- Build datasets ----
    if use_clip:
        train_ds = FireLineClipDataset(
            args.data_root, "train",
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
        val_ds = FireLineClipDataset(
            args.data_root, "val",
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
        real_val_raw = os.path.join(args.data_root, "real_val", "raw")
        real_val_ds = None
        if os.path.exists(real_val_raw):
            real_val_ds = FireLineClipDataset(
                args.data_root, "real_val",
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
            if len(real_val_ds) == 0:
                real_val_ds = None
    else:
        tfm = get_rgb_transforms(args.img_size)
        train_ds = FireLineDataset(args.data_root, split="train", transform=tfm)
        val_ds   = FireLineDataset(args.data_root, split="val",   transform=tfm)
        real_val_raw = os.path.join(args.data_root, "real_val", "raw")
        real_val_ds = None
        if os.path.exists(real_val_raw):
            real_val_ds = FireLineDataset(args.data_root, split="real_val", transform=tfm)
            if len(real_val_ds) == 0:
                real_val_ds = None

    if len(train_ds) == 0:
        print("❌ No training data found.")
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

    # ---- Build model ----
    if use_clip:
        model = SegFormer4Ch(args.model_name, local_models_base=args.local_models_base).to(device)
    else:
        seg_path_or_id, seg_is_local, seg_use_safetensors, _ = resolve_local_first(
            args.model_name, base_dir=args.local_models_base
        )
        seg_kwargs = dict(
            num_labels=1,
            ignore_mismatched_sizes=True,
            local_files_only=seg_is_local
        )
        if seg_is_local and (seg_use_safetensors is not None):
            seg_kwargs["use_safetensors"] = seg_use_safetensors
        model = SegformerForSemanticSegmentation.from_pretrained(
            seg_path_or_id, **seg_kwargs
        ).to(device)

    criterion = build_criterion(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- Save meta ----
    meta = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "use_clip": use_clip,
        "data_root": args.data_root,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "model_name": args.model_name,
        "band_width_px": args.band_width_px,
        "max_weight": args.max_weight,
        "weight_sigma": args.weight_sigma,
        "loss": {
            "name": args.loss,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "dice_w": args.dice_w,
            "bce_w": args.bce_w,
        },
        "local_models_base": args.local_models_base,
        "notes": {
            "thr_for_metrics": args.thr,
            "tol_px_for_TolF1": args.tol_px,
            "val_metrics": ["IoU", "Dice", "Precision", "Recall", "F1", "HD95", "HD", "ASSD", "TolF1"],
            "extra_validation": ["real_val"] if real_val_loader is not None else []
        }
    }

    if use_clip:
        meta["clip_model"] = args.clip_model
        meta["clip_grid"] = args.clip_grid
        meta["clip_device"] = args.clip_device
        meta["clip_prompts"] = clip_prompts
        meta["clip_cache_dir"] = args.clip_cache_dir
        meta["notes"]["input_channels"] = 4
        meta["notes"]["adapter"] = "None (SegFormer first conv patched 3->4; RGB weights copied; heat weights init=0)"
    else:
        meta["notes"]["input_channels"] = 3
        meta["notes"]["adapter"] = "None (standard 3ch SegFormer)"

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
        # ---- Train ----
        model.train()
        t_loss = 0.0
        t_dice = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for batch in pbar:
            if use_clip:
                x, band, wmap = batch
                x = x.to(device, non_blocking=True)
                band = band.to(device, non_blocking=True)
                wmap = wmap.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                logits = F.interpolate(logits, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)

                loss = criterion(logits, band, wmap)
                loss.backward()
                optimizer.step()

                d = dice_score_from_logits(logits.detach(), band, thr=args.thr)
            else:
                imgs, masks = batch
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(pixel_values=imgs)
                logits = F.interpolate(outputs.logits, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)

                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                d = dice_score_from_logits(logits.detach(), masks, thr=args.thr)

            t_loss += loss.item()
            t_dice += d

            writer.add_scalar("step/loss", loss.item(), global_step)
            writer.add_scalar("step/dice", d, global_step)
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

        t_loss /= max(1, len(train_loader))
        t_dice /= max(1, len(train_loader))

        # ---- val + real_val eval ----
        v_loss, v_metrics = run_eval(model, val_loader, criterion, args, device, name="Val", use_clip=use_clip)

        rv_loss, rv_metrics = None, None
        if real_val_loader is not None:
            rv_loss, rv_metrics = run_eval(model, real_val_loader, criterion, args, device, name="RealVal", use_clip=use_clip)

        # ---- TensorBoard epoch logging ----
        writer.add_scalar("loss/train", t_loss, epoch + 1)
        writer.add_scalar("dice/train", t_dice, epoch + 1)

        writer.add_scalar("loss/val", v_loss, epoch + 1)
        for k, v in v_metrics.items():
            writer.add_scalar(f"metrics/val_{k}", v, epoch + 1)

        if rv_loss is not None:
            writer.add_scalar("loss/real_val", rv_loss, epoch + 1)
            for k, v in rv_metrics.items():
                writer.add_scalar(f"metrics/real_val_{k}", v, epoch + 1)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
        writer.flush()

        msg = (
            f"Epoch {epoch+1}: "
            f"Train loss={t_loss:.4f} dice={t_dice:.4f} | "
            f"Val loss={v_loss:.4f} "
            f"Dice={v_metrics['Dice']:.4f} IoU={v_metrics['IoU']:.4f} F1={v_metrics['F1']:.4f} "
            f"HD95={v_metrics['HD95']:.2f} HD={v_metrics['HD']:.2f} ASSD={v_metrics['ASSD']:.2f} "
            f"TolF1={v_metrics['TolF1']:.4f}"
        )
        if rv_loss is not None:
            msg += (
                f" | RealVal loss={rv_loss:.4f} "
                f"Dice={rv_metrics['Dice']:.4f} IoU={rv_metrics['IoU']:.4f} F1={rv_metrics['F1']:.4f} "
                f"HD95={rv_metrics['HD95']:.2f} HD={rv_metrics['HD']:.2f} ASSD={rv_metrics['ASSD']:.2f} "
                f"TolF1={rv_metrics['TolF1']:.4f}"
            )
        print(msg)

        # save epoch checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

        # best (val)
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

        # best (real_val)
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

    # save last checkpoint
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
# 10) Argparse
# ============================================================
def build_parser():
    p = argparse.ArgumentParser(
        "Train SegFormer — CLIP-Heat 4ch (default) or Baseline RGB 3ch (--no-clip)"
    )

    p.add_argument("--data_root", type=str, required=True,
                   help="Dataset root (expects train/raw, train/gt, val/raw, val/gt subdirs)")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--model_name", type=str, default="nvidia/mit-b3",
                   help="SegFormer checkpoint (e.g., nvidia/mit-b1)")

    # band/weight
    p.add_argument("--band_width_px", type=int, default=1)
    p.add_argument("--max_weight", type=float, default=1.0)
    p.add_argument("--weight_sigma", type=float, default=1.0)

    # loss selection
    p.add_argument("--loss", type=str, default="wft",
                   choices=["wft", "dice", "bce", "dicebce"],
                   help="Loss: wft | dice | bce | dicebce")

    # wft params
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.33)

    # dicebce weights
    p.add_argument("--dice_w", type=float, default=1.0)
    p.add_argument("--bce_w", type=float, default=1.0)

    # CLIP args (ignored when --no-clip)
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--clip_prompts", type=str, default="fire,flames,wildfire,burning")
    p.add_argument("--clip_grid", type=int, default=16)
    p.add_argument("--clip_cache_dir", type=str, default="./clip_cache")
    p.add_argument("--clip_device", type=str, default="cpu", choices=["cpu", "cuda"])

    # mode switch
    p.add_argument("--no-clip", dest="no_clip", action="store_true",
                   help="Disable CLIP-Heat and train a standard 3ch RGB SegFormer (baseline mode)")

    # threshold & tol
    p.add_argument("--thr", type=float, default=0.5, help="threshold for val binarization metrics")
    p.add_argument("--tol_px", type=int, default=3, help="tolerance (pixels) for TolF1")

    # output
    p.add_argument("--out_dir", type=str, default="./train_runs")
    p.add_argument("--run_name", type=str, default="rgb_clipheat_line")

    # local models base
    p.add_argument("--local_models_base", type=str, default="./models",
                   help="Local HF model base dir. e.g., ./models/<repo_id>/config.json")

    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)

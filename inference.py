# Unified inference script: CLIP-Heat mode (default) + Baseline RGB-only mode (--no-clip)
#
#   Default (no flag): 4-channel SegFormer4Ch + CLIP heatmap  (inference_clipheat.py behaviour)
#   --no-clip flag:    standard 3-channel SegFormer, RGB only  (original inference.py behaviour)
#
#   If train_meta.json is found next to the checkpoint and contains "use_clip": false,
#   --no-clip is inferred automatically.

import os
import json
import csv
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation
from datetime import datetime


# ==========================================
# [1] Config (overridden by CLI args)
# ==========================================
DATA_ROOT = ""   # Structure: DATA_ROOT/raw, DATA_ROOT/gt  (set via --data-root)
MODEL_PATH = ""  # Path to trained model .pth checkpoint (set via --model-path)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join("./Inference_Result", f"Run_{current_time}")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALPHA = 1.0
THRESH = 0.8
TOL_PX = 3

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# Heat channel normalization option (must match training settings)
# - "none": keep as-is in 0~1
# - "center": (h-0.5)/0.5  -> [-1,1]
HEAT_NORM_MODE = "center"


# ==========================================
# [2] Utilities: meta / local-first
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
    cand = os.path.join(gt_dir, img_name)
    return cand if os.path.exists(cand) else ""


# ==========================================
# [3] Preprocessing / Overlay
# ==========================================
def preprocess_rgb_to_tensor(input_rgb_uint8: np.ndarray) -> torch.Tensor:
    x = input_rgb_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
    return x

def normalize_heat(heat01: np.ndarray, mode: str) -> np.ndarray:
    if mode == "center":
        return (heat01.astype(np.float32) - 0.5) / 0.5
    return heat01.astype(np.float32)

def apply_overlay(image_rgb, mask01, color_rgb=(0, 0, 255), alpha=0.8, dilate_vis=False, dilate_kernel=3):
    m = mask01.copy().astype(np.uint8)
    if dilate_vis:
        k = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        m = cv2.dilate(m, k, iterations=1)

    colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
    colored_mask[m == 1] = color_rgb
    return cv2.addWeighted(image_rgb, 1.0, colored_mask, alpha, 0)


# ==========================================
# [4] CLIP heatmap + grid scores (CLIP mode only)
# ==========================================
@torch.no_grad()
def clip_fire_heatmap_and_grid(
    clip_model,
    clip_processor,
    image_rgb_uint8: np.ndarray,
    prompts: list,
    grid: int = 16,
    patch_px: int = 224,
    device: str = "cpu"
):
    """
    Returns:
      - heat_full: (H,W) float32 in [0,1]  (heat upsampled to original resolution)
      - norm_grid: (grid,grid) float32 in [0,1] (per-patch score normalized)
    """
    H, W, _ = image_rgb_uint8.shape

    text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(device)
    text_feats = clip_model.get_text_features(**text_inputs)  # (P,D)
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
        norm_grid = np.zeros_like(scores, dtype=np.float32)
    else:
        norm_grid = (scores - smin) / (smax - smin)

    heat_full = cv2.resize(norm_grid, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return heat_full, norm_grid

def heat_cache_path(cache_dir: str, img_name: str, grid: int) -> str:
    base = os.path.splitext(img_name)[0]
    return os.path.join(cache_dir, f"{base}_g{grid}.npy")

def grid_cache_path(cache_dir: str, img_name: str, grid: int) -> str:
    base = os.path.splitext(img_name)[0]
    return os.path.join(cache_dir, f"{base}_grid_g{grid}.npy")


# ==========================================
# [5] Visualization saving (CLIP mode only)
# ==========================================
def save_heat_colormap_overlay(image_rgb_uint8: np.ndarray, heat01: np.ndarray, out_path: str, alpha: float = 0.45):
    heat_u8 = np.clip(heat01 * 255.0, 0, 255).astype(np.uint8)
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb_uint8, 1.0, heat_color_rgb, float(alpha), 0)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def save_grid_numbers_overlay(norm_grid: np.ndarray, image_rgb_uint8: np.ndarray, out_path: str):
    g = int(norm_grid.shape[0])
    H, W, _ = image_rgb_uint8.shape
    vis = image_rgb_uint8.copy()

    cell_w = W / g
    cell_h = H / g
    cell_min = min(cell_w, cell_h)

    font_scale = float(np.clip(cell_min / 220.0, 0.25, 0.9))
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_th = 1 if cell_min < 120 else 2

    for k in range(1, g):
        x = int(round(W * k / g))
        y = int(round(H * k / g))
        cv2.line(vis, (x, 0), (x, H - 1), (255, 255, 255), line_th, cv2.LINE_AA)
        cv2.line(vis, (0, y), (W - 1, y), (255, 255, 255), line_th, cv2.LINE_AA)

    for gy in range(g):
        for gx in range(g):
            val = float(norm_grid[gy, gx])
            txt = f"{val:.2f}"

            cx = int((gx + 0.5) * cell_w)
            cy = int((gy + 0.5) * cell_h)

            (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness)
            tx = cx - tw // 2
            ty = cy + th // 2

            cv2.putText(vis, txt, (tx, ty), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(vis, txt, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# ==========================================
# [6] Model: SegFormer4Ch (CLIP mode)
# ==========================================
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

    # fallback: find the first Conv with in_channels==3
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


class SegFormerWithInputAdapter(nn.Module):
    """
    (B,4,H,W) -> 1x1 Conv(4->3) adapter -> SegFormer -> logits
    (Fallback for older CLIP-heat checkpoints trained with adapter method)
    """
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


# ==========================================
# [7] Inference Logic
# ==========================================
def run_inference(use_clip: bool):
    os.makedirs(SAVE_DIR, exist_ok=True)

    overlay_dir = os.path.join(SAVE_DIR, "overlays")
    predmask_dir = os.path.join(SAVE_DIR, "pred_masks")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(predmask_dir, exist_ok=True)

    metrics_csv_path = os.path.join(SAVE_DIR, "metrics_per_image.csv")
    metrics_summary_path = os.path.join(SAVE_DIR, "metrics_summary.json")

    print(f"📂 Result save folder: '{SAVE_DIR}'")
    print(f"🚀 Starting inference! Mode: {'CLIP-Heat (4ch)' if use_clip else 'Baseline RGB (3ch)'}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return

    # -------- Read meta --------
    global IMG_SIZE
    meta = load_meta_if_exists(MODEL_PATH)

    base_model_name = meta.get("model_name", "nvidia/mit-b1")
    IMG_SIZE = int(meta.get("img_size", IMG_SIZE))
    local_models_base = meta.get("local_models_base", "./models")

    print(f"✅ Base model: {base_model_name}")
    print(f"✅ IMG_SIZE: {IMG_SIZE}")
    print(f"✅ DEVICE: {DEVICE}")
    print(f"✅ THRESH={THRESH} | TOL_PX={TOL_PX}")

    # -------- CLIP-specific setup --------
    clip_model = None
    clip_processor = None
    clip_model_name = None
    clip_grid = 16
    clip_prompts = ["fire", "flames", "wildfire", "burning"]
    clip_device = "cpu"
    clip_cache_dir = "./clip_cache"

    if use_clip:
        clip_model_name = meta.get("clip_model", "openai/clip-vit-base-patch32")
        clip_grid = int(meta.get("clip_grid", 16))
        clip_prompts = meta.get("clip_prompts", ["fire", "flames", "wildfire", "burning"])
        if isinstance(clip_prompts, str):
            clip_prompts = [p.strip() for p in clip_prompts.split(",") if p.strip()]

        base_clip_cache = meta.get("clip_cache_dir", "./clip_cache")
        split_name = os.path.basename(os.path.normpath(DATA_ROOT)) or "infer"
        clip_cache_dir = os.path.join(base_clip_cache, split_name)
        os.makedirs(clip_cache_dir, exist_ok=True)

        clip_device = "cpu"
        if meta.get("clip_device", "cpu") == "cuda" and torch.cuda.is_available():
            clip_device = "cuda"

        print(f"✅ CLIP model: {clip_model_name} | grid={clip_grid} | device={clip_device}")
        print(f"✅ CLIP prompts: {clip_prompts}")
        print(f"✅ CLIP cache: {clip_cache_dir}")
        print(f"✅ HEAT_NORM_MODE={HEAT_NORM_MODE}")

        from transformers import CLIPProcessor, CLIPModel

        clip_path_or_id, clip_is_local, clip_use_safetensors, _ = resolve_local_first(
            clip_model_name, base_dir=local_models_base
        )

        clip_processor = CLIPProcessor.from_pretrained(
            clip_path_or_id,
            local_files_only=clip_is_local
        )

        clip_kwargs = dict(local_files_only=clip_is_local)
        if clip_is_local and (clip_use_safetensors is not None):
            clip_kwargs["use_safetensors"] = clip_use_safetensors

        clip_model = CLIPModel.from_pretrained(
            clip_path_or_id,
            **clip_kwargs
        ).to(clip_device)
        clip_model.eval()

        # Extra visualisation dirs only in CLIP mode
        heat_vis_dir = os.path.join(SAVE_DIR, "heat_vis")
        heat_overlay_dir = os.path.join(heat_vis_dir, "heat_overlay")
        grid_numbers_dir = os.path.join(heat_vis_dir, "grid_numbers")
        for d in [heat_vis_dir, heat_overlay_dir, grid_numbers_dir]:
            os.makedirs(d, exist_ok=True)
    else:
        heat_overlay_dir = None
        grid_numbers_dir = None

    # -------- Load SegFormer + weights --------
    sd = torch.load(MODEL_PATH, map_location=DEVICE)

    if use_clip:
        # Method A first, fallback to adapter
        model = None
        loaded_kind = None

        try:
            model_try = SegFormer4Ch(base_model_name, local_models_base=local_models_base).to(DEVICE)
            model_try.load_state_dict(sd, strict=True)
            model = model_try
            loaded_kind = "A(4ch first-conv patched)"
            print("✅ Checkpoint loaded successfully with Method A (4ch first-conv), strict=True")
        except Exception as e_a:
            print(f"⚠️ Method A strict load failed: {e_a}")

            try:
                model_try = SegFormerWithInputAdapter(base_model_name, local_models_base=local_models_base).to(DEVICE)
                model_try.load_state_dict(sd, strict=True)
                model = model_try
                loaded_kind = "Adapter(4->3 1x1 conv)"
                print("✅ Checkpoint loaded successfully with Adapter method, strict=True")
            except Exception as e_b:
                print(f"❌ Adapter strict load also failed: {e_b}")
                raise

        print(f"✅ Loaded model kind: {loaded_kind}")
    else:
        # Baseline: standard 3ch SegFormer
        model_id_or_path, local_files_only, use_safetensors, local_weight_kind = resolve_local_first(
            base_model_name, base_dir=local_models_base
        )
        print(f"✅ Base model source: {'LOCAL' if local_files_only else 'HF'} -> {model_id_or_path}")
        if local_files_only:
            print(f"✅ Local weight format: {local_weight_kind} (use_safetensors={use_safetensors})")

        kwargs = dict(
            num_labels=1,
            ignore_mismatched_sizes=True,
            local_files_only=local_files_only,
        )
        if local_files_only:
            kwargs["use_safetensors"] = bool(use_safetensors)

        model = SegformerForSemanticSegmentation.from_pretrained(
            model_id_or_path, **kwargs
        ).to(DEVICE)
        model.load_state_dict(sd, strict=True)
        loaded_kind = "3ch baseline SegFormer"
        print("✅ Loaded trained model weights. (strict=True)")

    model.eval()

    # -------- Data paths --------
    raw_dir = os.path.join(DATA_ROOT, "raw")
    gt_dir = os.path.join(DATA_ROOT, "gt")
    has_gt = os.path.exists(gt_dir)

    if not os.path.exists(raw_dir):
        print(f"❌ raw folder not found: {raw_dir}")
        return

    image_files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(IMG_EXTENSIONS)])
    if len(image_files) == 0:
        print("⚠️ No image files found in the folder.")
        return

    if has_gt:
        print(f"✅ GT folder found: {gt_dir} -> will also perform evaluation. (same filename matching)")
    else:
        print(f"ℹ️ GT folder not found: {gt_dir} -> inference only (evaluation skipped).")

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

    # counters (baseline mode)
    read_ok_count = 0
    mask_save_ok_count = 0

    for idx, img_name in enumerate(image_files, 1):
        img_path = os.path.join(raw_dir, img_name)

        original_bgr = cv2.imread(img_path)
        if original_bgr is None:
            print(f"⚠️ Cannot read image (Skip): {img_name}")
            if not use_clip:
                per_image_rows.append({
                    "image": img_name,
                    "mask_out": f"{os.path.splitext(img_name)[0]}_mask.png",
                    "save_skip_reason": "image_read_failed"
                })
            continue

        read_ok_count += 1
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        if use_clip:
            # ---- CLIP heat / grid (cache first) ----
            h_path = heat_cache_path(clip_cache_dir, img_name, clip_grid)
            g_path = grid_cache_path(clip_cache_dir, img_name, clip_grid)

            heat = None
            norm_grid = None

            if os.path.exists(h_path):
                heat = np.load(h_path).astype(np.float32)
            if os.path.exists(g_path):
                norm_grid = np.load(g_path).astype(np.float32)

            need_recompute = False
            if heat is None or heat.shape != (IMG_SIZE, IMG_SIZE):
                need_recompute = True
            if norm_grid is None or norm_grid.shape != (clip_grid, clip_grid):
                need_recompute = True

            if need_recompute:
                heat_full, norm_grid = clip_fire_heatmap_and_grid(
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    image_rgb_uint8=original_rgb,
                    prompts=clip_prompts,
                    grid=clip_grid,
                    patch_px=224,
                    device=clip_device
                )
                heat = cv2.resize(heat_full, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                np.save(h_path, heat)
                np.save(g_path, norm_grid)

            input_rgb = cv2.resize(original_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

            # Save heat visualization
            heat_overlay_path = os.path.join(heat_overlay_dir, f"HeatOverlay_{os.path.splitext(img_name)[0]}.png")
            save_heat_colormap_overlay(input_rgb.astype(np.uint8), heat, heat_overlay_path, alpha=0.45)

            grid_numbers_path = os.path.join(grid_numbers_dir, f"GridNumbers_{os.path.splitext(img_name)[0]}.png")
            save_grid_numbers_overlay(norm_grid, input_rgb.astype(np.uint8), grid_numbers_path)

            # Build 4ch tensor
            rgb_t = preprocess_rgb_to_tensor(input_rgb).to(DEVICE)
            heat_in = normalize_heat(heat, HEAT_NORM_MODE)
            heat_t = torch.from_numpy(heat_in).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            x_in = torch.cat([rgb_t, heat_t], dim=1)  # (1,4,H,W)

            # Inference
            with torch.no_grad():
                logits = model(x_in)
                logits = nn.functional.interpolate(
                    logits, size=(IMG_SIZE, IMG_SIZE),
                    mode="bilinear", align_corners=False
                )
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                pred_mask01 = (prob > THRESH).astype(np.uint8)

            # Save predicted mask (stem only, no _mask suffix — matches clipheat behaviour)
            pred_mask255 = (pred_mask01 * 255).astype(np.uint8)
            stem = os.path.splitext(img_name)[0]
            mask_save_name = f"{stem}.png"
            mask_save_path = os.path.join(predmask_dir, mask_save_name)
            cv2.imwrite(mask_save_path, pred_mask255)
            mask_save_ok_count += 1

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

            row = {
                "image": img_name,
                "mask_out": mask_save_name,
                "overlay_out": os.path.basename(fig_save_path),
                "heat_overlay_out": os.path.basename(heat_overlay_path),
                "grid_numbers_out": os.path.basename(grid_numbers_path),
            }

        else:
            # ---- Baseline 3ch path ----
            input_rgb = cv2.resize(original_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            input_tensor = preprocess_rgb_to_tensor(input_rgb).to(DEVICE)

            with torch.no_grad():
                outputs = model(pixel_values=input_tensor)
                logits = nn.functional.interpolate(
                    outputs.logits, size=(IMG_SIZE, IMG_SIZE),
                    mode="bilinear", align_corners=False
                )
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                pred_mask01 = (prob > THRESH).astype(np.uint8)

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

            row = {"image": img_name, "mask_out": mask_save_name}

        # ---- Evaluation (only when GT is available) ----
        if has_gt:
            gt_path = find_gt_mask_path_same_name(gt_dir, img_name)
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

        if has_gt and ("IoU" in row):
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

    if use_clip:
        preferred = ["image", "mask_out", "overlay_out", "heat_overlay_out", "grid_numbers_out"] + metric_keys + ["eval_skip_reason"]
    else:
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
        "use_clip": use_clip,
        "img_size": IMG_SIZE,
        "device": DEVICE,
        "thresh": THRESH,
        "tol_px": TOL_PX,
        "count_total_images": len(image_files),
        "count_read_ok_images": int(read_ok_count),
        "count_mask_saved_ok": int(mask_save_ok_count),
        "count_eval_images": int(eval_count),
        "mean_metrics": None,
        "gt_matching": "same_filename_only",
        "model_loaded_kind": loaded_kind,
    }

    if use_clip:
        summary["heat_norm_mode"] = HEAT_NORM_MODE
        summary["clip"] = {
            "model": clip_model_name,
            "grid": clip_grid,
            "device": clip_device,
            "prompts": clip_prompts,
            "cache_dir": clip_cache_dir,
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

    # ---- Mask save check (baseline mode) ----
    if not use_clip:
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
    if use_clip:
        print(f"- heat overlay: {os.path.join(SAVE_DIR, 'heat_vis', 'heat_overlay')}")
        print(f"- grid numbers: {os.path.join(SAVE_DIR, 'heat_vis', 'grid_numbers')}")
    print(f"- per-image metrics csv: {metrics_csv_path}")
    print(f"- summary json: {metrics_summary_path}")
    print(f"📂 Final output path: {SAVE_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Inference — CLIP-Heat 4ch (default) or Baseline RGB 3ch (--no-clip)"
    )
    parser.add_argument("--data-root", required=True,
                        help="Test data root containing raw/ (and optionally gt/) subdirs")
    parser.add_argument("--model-path", required=True,
                        help="Path to trained model .pth checkpoint")
    parser.add_argument("--save-dir", default=None,
                        help="Output directory (default: ./Inference_Result/Run_<timestamp>)")
    parser.add_argument("--thresh", type=float, default=THRESH,
                        help="Prediction threshold (default: %(default)s)")
    parser.add_argument("--tol-px", type=int, default=TOL_PX,
                        help="Tolerance radius in pixels (default: %(default)s)")
    parser.add_argument("--model-dir", default="./models",
                        help="Local HuggingFace model cache dir (default: %(default)s)")
    parser.add_argument("--no-clip", dest="no_clip", action="store_true",
                        help="Disable CLIP-Heat and run standard 3ch RGB SegFormer (baseline mode)")
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    MODEL_PATH = args.model_path
    THRESH = args.thresh
    TOL_PX = args.tol_px
    if args.save_dir:
        SAVE_DIR = args.save_dir

    # Auto-detect use_clip from train_meta.json if available
    use_clip = not args.no_clip
    if use_clip:
        _meta = load_meta_if_exists(MODEL_PATH)
        if "use_clip" in _meta and not _meta["use_clip"]:
            print("ℹ️ train_meta.json has use_clip=false -> switching to --no-clip mode automatically.")
            use_clip = False

    run_inference(use_clip=use_clip)

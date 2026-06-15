import os
import cv2
import json
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import SegformerForSemanticSegmentation
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# =========================
# Transform
# =========================
def get_baseline_transforms(img_size: int):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# =========================
# Dataset
# =========================
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

# =========================
# Losses
# =========================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1 - dice

class DiceBCELoss(nn.Module):
    """
    Dice + BCEWithLogitsLoss
    Weights are fixed at 1:1 by default.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice_w = 1.0
        self.bce_w  = 1.0

    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        bce_loss  = self.bce(logits, targets)
        total = self.dice_w * dice_loss + self.bce_w * bce_loss
        return total, dice_loss.detach(), bce_loss.detach()

class WeightedFocalTverskyLoss(nn.Module):
    """
    Weighted Focal Tversky Loss
    - If weight_map is None, treated as ones (can also be used with baseline dataset)
    """
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

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = (1.0 - tversky) ** self.gamma
        return loss.mean()

def build_criterion(args):
    """
    To avoid modifying the training loop:
      - all loss types (Dice, BCE, DiceBCE, WFT) return a unified (total, dice, bce) tuple.
    """
    loss_name = args.loss.lower()

    if loss_name == "dice":
        dice = DiceLoss()

        class _DiceWrapper(nn.Module):
            def forward(self, logits, targets):
                d = dice(logits, targets)
                z = torch.tensor(0.0, device=logits.device)  # no bce
                return d, d.detach(), z

        return _DiceWrapper()

    if loss_name == "bce":
        bce = nn.BCEWithLogitsLoss()

        class _BCEWrapper(nn.Module):
            def forward(self, logits, targets):
                b = bce(logits, targets)
                z = torch.tensor(0.0, device=logits.device)  # no dice
                return b, z, b.detach()

        return _BCEWrapper()

    if loss_name in ("dicebce", "dice_bce", "dice+bce"):
        return DiceBCELoss()

    if loss_name in ("wft", "weightedfocaltversky", "weighted_focal_tversky"):
        wft = WeightedFocalTverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        dice_for_log = DiceLoss()

        class _WFTWrapper(nn.Module):
            def forward(self, logits, targets):
                total = wft(logits, targets, weight_map=None)
                d = dice_for_log(logits, targets)  # dice loss for logging
                z = torch.tensor(0.0, device=logits.device)  # no bce
                return total, d.detach(), z

        return _WFTWrapper()

    raise ValueError(f"Unknown --loss {loss_name}. Use one of: dice, bce, dicebce, wft")

# =========================
# Validation Metrics
# =========================
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
    """
    Hausdorff Distance (max, symmetric). Unit: pixel
    """
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
    """
    Average Symmetric Surface Distance (ASSD). Unit: pixel
    """
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

# =========================
# Helpers: local checkpoint preference
# =========================
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

# =========================
# Train
# =========================
def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    run_id = datetime.now().strftime(f"{args.run_name}_%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)

    # ✅ TensorBoard
    tb_dir = os.path.join(save_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"📈 TensorBoard logdir: {tb_dir}")

    print(f"Training start: {save_dir}")
    print(f"Device: {device}")
    print(f"DATA_ROOT={args.data_root}")
    print(f"IMG_SIZE={args.img_size} | BATCH={args.batch_size} | LR={args.lr} | EPOCHS={args.epochs}")
    print(f"LOSS={args.loss}")
    if args.loss.lower() in ("wft", "weightedfocaltversky", "weighted_focal_tversky"):
        print(f"WFT params: alpha={args.alpha} beta={args.beta} gamma={args.gamma}")
    print(f"Model: {args.model_name}")
    print(f"VAL metrics thr={args.thr} | TolF1 tol_px={args.tol_px}")

    tfm = get_baseline_transforms(args.img_size)
    train_ds = FireLineDataset(args.data_root, split="train", transform=tfm)
    valid_ds = FireLineDataset(args.data_root, split="val", transform=tfm)

    # ✅ add real_val
    real_val_ds = None
    real_val_raw = os.path.join(args.data_root, "real_val", "raw")
    has_real_val = os.path.exists(real_val_raw)
    if has_real_val:
        real_val_ds = FireLineDataset(args.data_root, split="real_val", transform=tfm)
        if len(real_val_ds) == 0:
            real_val_ds = None

    if len(train_ds) == 0:
        print("❌ Error: No training data found.")
        writer.close()
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )

    # ✅ real_val loader setup
    real_val_loader = None
    if real_val_ds is not None:
        real_val_loader = DataLoader(
            real_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda")
        )
        print(f"✅ real_val enabled: {len(real_val_ds)} images")
    else:
        print("ℹ️ real_val not found (or empty) -> skip real_val evaluation.")

    # =========================
    # Model load (LOCAL-first -> fallback to HF)
    # =========================
    local_dir = os.path.join("./models", args.model_name)

    local_has_config = os.path.exists(os.path.join(local_dir, "config.json"))
    local_weight_kind = choose_local_weight_format(local_dir)

    use_local = local_has_config and (local_weight_kind is not None)
    model_id_or_path = local_dir if use_local else args.model_name

    if use_local:
        print(f"Model source: LOCAL -> {model_id_or_path}")
        print(f"✅ Local weight format: {local_weight_kind}")
        use_safetensors_arg = True if (local_weight_kind == "safetensors") else False
    else:
        print(f"Model source: HF -> {model_id_or_path}")
        use_safetensors_arg = None

    # =========================
    # Save run metadata (JSON) - early
    # =========================
    metric_keys = ["IoU", "Dice", "Precision", "Recall", "F1", "HD95", "HD", "ASSD", "TolF1"]
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
        "loss": args.loss,
        "loss_params": {
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
        } if args.loss.lower() in ("wft", "weightedfocaltversky", "weighted_focal_tversky") else None,
        "val_metrics": {
            "thr": args.thr,
            "tol_px": args.tol_px,
            "keys": metric_keys,
        },
        "model_name": args.model_name,
        "model_source": "LOCAL" if use_local else "HF",
        "model_load_path_or_id": model_id_or_path,
        "local_hf_models_base": "./models",
        "tb_dir": tb_dir,
        "extra_validation": {
            "real_val": bool(real_val_loader is not None),
            "real_val_count": int(len(real_val_ds)) if real_val_ds is not None else 0
        }
    }

    meta_path = os.path.join(save_dir, "train_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"📝 Saved metadata: {meta_path}")

    # =========================
    # Load model (force safetensors/bin if local)
    # =========================
    kwargs = dict(
        num_labels=1,
        ignore_mismatched_sizes=True,
        local_files_only=use_local,
    )
    if use_safetensors_arg is not None:
        kwargs["use_safetensors"] = use_safetensors_arg

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_id_or_path,
        **kwargs
    ).to(device)

    criterion = build_criterion(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- best / last ----
    best_val = float("inf")
    best_path = os.path.join(save_dir, "best_fireline_model.pth")

    best_real = float("inf")
    real_best_path = os.path.join(save_dir, "real_best_fireline_model.pth")

    last_path = os.path.join(save_dir, "last_fireline_model.pth")

    global_step = 0  # ✅ TensorBoard global step counter

    for epoch in range(args.epochs):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        total_diceL = 0.0
        total_bceL  = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=imgs)

            logits = nn.functional.interpolate(
                outputs.logits, size=(args.img_size, args.img_size),
                mode="bilinear", align_corners=False
            )

            loss, dice_l, bce_l = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_diceL += dice_l.item()
            total_bceL  += bce_l.item()

            # ✅ TensorBoard (per step)
            writer.add_scalar("step/loss", loss.item(), global_step)
            writer.add_scalar("step/dice_loss", dice_l.item(), global_step)
            writer.add_scalar("step/bce_loss", bce_l.item(), global_step)
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", diceL=f"{dice_l.item():.4f}", bceL=f"{bce_l.item():.4f}")

        avg_train = total_loss / max(1, len(train_loader))
        avg_train_diceL = total_diceL / max(1, len(train_loader))
        avg_train_bceL  = total_bceL  / max(1, len(train_loader))

        # ---- Eval helper ----
        model.eval()

        def run_eval(loader, name: str):
            total_loss = 0.0
            total_diceL = 0.0
            total_bceL  = 0.0

            keys = metric_keys
            sum_metrics = {k: 0.0 for k in keys}
            count = 0

            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs} [{name}]", leave=False)
            with torch.no_grad():
                for imgs, masks in pbar:
                    imgs = imgs.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)

                    outputs = model(pixel_values=imgs)
                    logits = nn.functional.interpolate(
                        outputs.logits, size=(args.img_size, args.img_size),
                        mode="bilinear", align_corners=False
                    )

                    loss, dice_l, bce_l = criterion(logits, masks)
                    total_loss += loss.item()
                    total_diceL += dice_l.item()
                    total_bceL  += bce_l.item()

                    prob = torch.sigmoid(logits).detach().cpu().numpy()
                    pred_np = (prob > args.thr).astype(np.uint8)
                    gt_np = (masks.detach().cpu().numpy() > 0.5).astype(np.uint8)

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
            avg_diceL = total_diceL / max(1, len(loader))
            avg_bceL  = total_bceL  / max(1, len(loader))
            avg_metrics = {k: (sum_metrics[k] / max(1, count)) for k in keys}
            return avg_loss, avg_diceL, avg_bceL, avg_metrics

        # ---- Val ----
        avg_val, avg_val_diceL, avg_val_bceL, avg_metrics = run_eval(valid_loader, "Val")

        # ---- RealVal (optional) ----
        avg_rval = None
        avg_rval_diceL = None
        avg_rval_bceL = None
        avg_rmetrics = None
        if real_val_loader is not None:
            avg_rval, avg_rval_diceL, avg_rval_bceL, avg_rmetrics = run_eval(real_val_loader, "RealVal")

        # ✅ TensorBoard (per epoch)
        writer.add_scalar("epoch/loss_train", avg_train, epoch + 1)
        writer.add_scalar("epoch/dice_loss_train", avg_train_diceL, epoch + 1)
        writer.add_scalar("epoch/bce_loss_train", avg_train_bceL, epoch + 1)

        writer.add_scalar("epoch/loss_val", avg_val, epoch + 1)
        writer.add_scalar("epoch/dice_loss_val", avg_val_diceL, epoch + 1)
        writer.add_scalar("epoch/bce_loss_val", avg_val_bceL, epoch + 1)
        for k, v in avg_metrics.items():
            writer.add_scalar(f"metrics/val_{k}", v, epoch + 1)

        # ✅ real_val TensorBoard logging
        if avg_rval is not None:
            writer.add_scalar("epoch/loss_real_val", avg_rval, epoch + 1)
            writer.add_scalar("epoch/dice_loss_real_val", avg_rval_diceL, epoch + 1)
            writer.add_scalar("epoch/bce_loss_real_val", avg_rval_bceL, epoch + 1)
            for k, v in avg_rmetrics.items():
                writer.add_scalar(f"metrics/real_val_{k}", v, epoch + 1)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
        writer.flush()

        # print
        msg = (
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train: loss={avg_train:.4f} (diceL={avg_train_diceL:.4f}, bceL={avg_train_bceL:.4f}) | "
            f"Val: loss={avg_val:.4f} (diceL={avg_val_diceL:.4f}, bceL={avg_val_bceL:.4f}) | "
            f"ValMetrics: Dice={avg_metrics['Dice']:.4f} IoU={avg_metrics['IoU']:.4f} "
            f"F1={avg_metrics['F1']:.4f} HD95={avg_metrics['HD95']:.2f} "
            f"HD={avg_metrics['HD']:.2f} ASSD={avg_metrics['ASSD']:.2f} TolF1={avg_metrics['TolF1']:.4f}"
        )
        if avg_rval is not None:
            msg += (
                f" | RealVal: loss={avg_rval:.4f} (diceL={avg_rval_diceL:.4f}, bceL={avg_rval_bceL:.4f}) "
                f"RealValMetrics: Dice={avg_rmetrics['Dice']:.4f} IoU={avg_rmetrics['IoU']:.4f} "
                f"F1={avg_rmetrics['F1']:.4f} HD95={avg_rmetrics['HD95']:.2f} "
                f"HD={avg_rmetrics['HD']:.2f} ASSD={avg_rmetrics['ASSD']:.2f} TolF1={avg_rmetrics['TolF1']:.4f}"
            )
        print(msg)

        # ---- Save best on VAL ----
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_path)
            print(f"  💾 Best(VAL) Model Updated: {best_path} (val_loss={best_val:.4f})")

            meta["best"] = {
                "epoch": epoch + 1,
                "val_loss": float(avg_val),
                "val_dice_loss": float(avg_val_diceL),
                "val_bce_loss": float(avg_val_bceL),
                "val_metrics": {k: float(avg_metrics[k]) for k in avg_metrics.keys()},
                "path": os.path.basename(best_path)
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        # ---- Save best on REAL_VAL (separate) ----
        if avg_rval is not None and avg_rval < best_real:
            best_real = avg_rval
            torch.save(model.state_dict(), real_best_path)
            print(f"  💾 Best(REAL_VAL) Model Updated: {real_best_path} (real_val_loss={best_real:.4f})")

            meta["real_best"] = {
                "epoch": epoch + 1,
                "real_val_loss": float(avg_rval),
                "real_val_dice_loss": float(avg_rval_diceL),
                "real_val_bce_loss": float(avg_rval_bceL),
                "real_val_metrics": {k: float(avg_rmetrics[k]) for k in avg_rmetrics.keys()},
                "path": os.path.basename(real_best_path)
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- Save last ----
    torch.save(model.state_dict(), last_path)
    meta["last"] = {"epoch": args.epochs, "path": os.path.basename(last_path)}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    writer.close()

    print(f"\n✅ Training done.")
    print(f"Best(VAL) saved at: {best_path}")
    if real_val_loader is not None:
        print(f"Best(REAL_VAL) saved at: {real_best_path}")
    print(f"Last saved at: {last_path}")
    print(f"Run dir: {save_dir}")
    print(f"Metadata: {meta_path}")
    print(f"TensorBoard: {tb_dir}")

# =========================
# Argparse
# =========================
def build_parser():
    p = argparse.ArgumentParser(description="Fireline SegFormer Training (RGB baseline + val metrics + TensorBoard)")

    p.add_argument("--data_root", type=str, required=True,
                   help="Dataset root (train/raw, train/gt, val/raw, val/gt, real_val/raw, real_val/gt)")
    p.add_argument("--img_size", type=int, default=512, help="Input image size (square)")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size")
    p.add_argument("--epochs", type=int, default=15, help="Epochs")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    p.add_argument("--loss", type=str, default="dicebce",
                   choices=["dice", "bce", "dicebce", "wft"],
                   help="Loss type: dice | bce | dicebce | wft")

    p.add_argument("--alpha", type=float, default=0.5, help="WFT alpha (FP weight)")
    p.add_argument("--beta", type=float, default=0.5, help="WFT beta (FN weight)")
    p.add_argument("--gamma", type=float, default=1.33, help="WFT gamma (focal exponent)")

    p.add_argument("--thr", type=float, default=0.5, help="threshold for binarization metrics")
    p.add_argument("--tol_px", type=int, default=3, help="tolerance pixels for TolF1")

    p.add_argument("--model_name", type=str, default="nvidia/mit-b3",
                   help="HF checkpoint (e.g., nvidia/mit-b1)")
    p.add_argument("--out_dir", type=str, default="./train_runs", help="Output root directory")
    p.add_argument("--run_name", type=str, default="run", help="Run prefix")
    p.add_argument("--cpu", action="store_true", help="Force CPU")

    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)




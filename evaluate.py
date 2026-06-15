import os
import cv2
import numpy as np
import csv
from glob import glob
from datetime import datetime

# =========================
# Config
# =========================
PRED_DIR = ""   # Predicted mask folder (set via --pred-dir)
GT_DIR = ""     # GT folder (set via --gt-dir)
IMG_SIZE = 250

# Tolerance radius (pixels) for tolerance F1
TOLERANCE_R = 3

EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

# ✅ (1) Root folder to save evaluation results
EVAL_ROOT = "./evaluate"


# =========================
# Utils
# =========================
def read_mask_01(path, target_size=None):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if target_size is not None:
        m = cv2.resize(m, target_size, interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)

# def find_gt_for_pred(pred_path, gt_dir):
#     base = os.path.splitext(os.path.basename(pred_path))[0]
#     for ext in EXTS:
#         cand = os.path.join(gt_dir, base + ext)
#         if os.path.exists(cand):
#             return cand
#     return None
def find_gt_for_pred(pred_path, gt_dir):
    base = os.path.splitext(os.path.basename(pred_path))[0]  # e.g. "0000"
    gt_name = f"Wildfire_{base}_Line.png"
    cand = os.path.join(gt_dir, gt_name)
    return cand if os.path.exists(cand) else None

def seg_metrics(pred01, gt01):
    pred = pred01.astype(bool)
    gt = gt01.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    if gt.sum() == 0 and pred.sum() == 0:
        iou = dice = precision = recall = f1 = 1.0

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }

def boundary_points(mask01):
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros((0, 2), dtype=np.int32)
    k = np.ones((3,3), np.uint8)
    er = cv2.erode(m, k, iterations=1)
    edge = (m - er)
    ys, xs = np.where(edge > 0)
    return np.stack([ys, xs], axis=1).astype(np.int32)

def chamfer_and_hd95(pred01, gt01):
    pred_pts = boundary_points(pred01)
    gt_pts = boundary_points(gt01)

    h, w = pred01.shape
    diag = float(np.sqrt(h*h + w*w))

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return {"Chamfer": 0.0, "HD95": 0.0}
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return {"Chamfer": diag, "HD95": diag}

    gt_edge_img = np.ones_like(gt01, dtype=np.uint8)
    gt_edge_img[gt_pts[:, 0], gt_pts[:, 1]] = 0
    dt_gt = cv2.distanceTransform(gt_edge_img, cv2.DIST_L2, 3)

    pred_edge_img = np.ones_like(pred01, dtype=np.uint8)
    pred_edge_img[pred_pts[:, 0], pred_pts[:, 1]] = 0
    dt_pred = cv2.distanceTransform(pred_edge_img, cv2.DIST_L2, 3)

    d_pred_to_gt = dt_gt[pred_pts[:, 0], pred_pts[:, 1]]
    d_gt_to_pred = dt_pred[gt_pts[:, 0], gt_pts[:, 1]]

    chamfer = float(d_pred_to_gt.mean() + d_gt_to_pred.mean())
    hd95 = float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))
    return {"Chamfer": chamfer, "HD95": hd95}

def tolerance_f1(pred01, gt01, r=3):
    pred = (pred01 > 0).astype(np.uint8)
    gt = (gt01 > 0).astype(np.uint8)

    if pred.sum() == 0 and gt.sum() == 0:
        return {"TolPrecision": 1.0, "TolRecall": 1.0, "TolF1": 1.0}

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    gt_d = cv2.dilate(gt, k, iterations=1)
    pred_d = cv2.dilate(pred, k, iterations=1)

    tp_p = (pred & gt_d).sum()
    tp_r = (gt & pred_d).sum()

    eps = 1e-8
    tol_prec = float(tp_p / (pred.sum() + eps))
    tol_rec = float(tp_r / (gt.sum() + eps))
    tol_f1 = float((2 * tol_prec * tol_rec) / (tol_prec + tol_rec + eps))
    return {"TolPrecision": tol_prec, "TolRecall": tol_rec, "TolF1": tol_f1}


# =========================
# Main
# =========================
def main():
    # ✅ (1) Create a new evaluate/Run_<timestamp> folder for each run
    run_id = datetime.now().strftime("Run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(EVAL_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "metrics.csv")
    out_txt = os.path.join(out_dir, "summary.txt")

    pred_files = sorted([p for p in glob(os.path.join(PRED_DIR, "*")) if os.path.splitext(p)[1].lower() in EXTS])
    if len(pred_files) == 0:
        print(f"❌ No predicted masks found: {PRED_DIR}")
        return

    rows = []
    missing_gt = 0

    for i, pred_path in enumerate(pred_files, 1):
        gt_path = find_gt_for_pred(pred_path, GT_DIR)
        if gt_path is None:
            missing_gt += 1
            continue

        pred01 = read_mask_01(pred_path, target_size=(IMG_SIZE, IMG_SIZE))
        gt01 = read_mask_01(gt_path, target_size=(IMG_SIZE, IMG_SIZE))
        if pred01 is None or gt01 is None:
            continue

        m = seg_metrics(pred01, gt01)
        d = chamfer_and_hd95(pred01, gt01)
        t = tolerance_f1(pred01, gt01, r=TOLERANCE_R)

        row = {
            "file": os.path.basename(pred_path),
            "gt_file": os.path.basename(gt_path),
            **m,
            **d,
            **t,
        }
        rows.append(row)

        if i % 50 == 0 or i == len(pred_files):
            print(f"✅ {i}/{len(pred_files)} processed... (matched {len(rows)}, missing_gt {missing_gt})")

    if len(rows) == 0:
        print("❌ No matched GT found. Please check the filename convention.")
        print(f"- PRED_DIR: {PRED_DIR}")
        print(f"- GT_DIR: {GT_DIR}")
        return

    # Save CSV
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Compute summary statistics
    keys = ["IoU", "Dice", "Precision", "Recall", "F1", "Chamfer", "HD95", "TolF1"]
    summary_lines = []
    summary_lines.append(f"Total pred files: {len(pred_files)}")
    summary_lines.append(f"Total matched: {len(rows)}")
    summary_lines.append(f"Missing GT: {missing_gt}")
    summary_lines.append(f"Tolerance r: {TOLERANCE_R}px\n")

    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float32)
        summary_lines.append(f"{k}: mean={vals.mean():.6f}, std={vals.std():.6f}")

    # Save summary.txt
    with open(out_txt, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    # ✅ (2) Print summary to console after evaluation
    print("\n" + "="*50)
    print("📌 Evaluation Summary (dataset-level)")
    print("="*50)
    for line in summary_lines:
        print(line)
    print("="*50)
    print(f"💾 Saved metrics.csv: {out_csv}")
    print(f"💾 Saved summary.txt: {out_txt}")
    print(f"📂 Run folder: {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions against GT masks")
    parser.add_argument("--pred-dir", required=True, help="Directory containing predicted mask PNGs")
    parser.add_argument("--gt-dir", required=True, help="Directory containing GT mask PNGs")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="Resize masks to this size (default: %(default)s)")
    parser.add_argument("--tol-r", type=int, default=TOLERANCE_R, help="Tolerance radius in pixels (default: %(default)s)")
    parser.add_argument("--out-root", default="./evaluate", help="Root folder to save results (default: %(default)s)")
    args = parser.parse_args()

    PRED_DIR = args.pred_dir
    GT_DIR = args.gt_dir
    IMG_SIZE = args.img_size
    TOLERANCE_R = args.tol_r
    EVAL_ROOT = args.out_root

    main()

import os
import cv2
import numpy as np
from glob import glob
from collections import defaultdict

# =========================
# Config
# =========================
RAW_DIR = ""              # Raw image folder (set via --raw-dir)
GT_DIR  = ""              # GT mask folder (set via --gt-dir)
OUT_DIR = "./outputs/overlay_gt"  # Output save folder (can be changed via --out-dir)

COLOR = (255, 0, 0)      # 🔵 Blue color (BGR)
LINE_THICKNESS = 3       # GT line thickness (1=original, 3~5 recommended)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Filename suffix rules
RAW_TAG = "_Raw"         # Tag appended to raw filenames
GT_TAG  = "_Line"        # Tag appended to GT filenames

# =========================
# Utils
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_images(dir_path):
    return sorted([
        p for p in glob(os.path.join(dir_path, "*"))
        if os.path.splitext(p)[1].lower() in IMG_EXTS
    ])

def strip_tag(stem: str, tag: str) -> str:
    """Remove tag from the end of stem if present"""
    if not tag:
        return stem
    if stem.lower().endswith(tag.lower()):
        return stem[:-len(tag)]
    return stem

def normalize_pair_stem(stem: str) -> str:
    """
    Build a common key for comparing raw/gt filenames.
    - Strip _Raw if present
    - Strip _Line if present
    Either input normalizes to the same common stem.
    """
    s = strip_tag(stem, RAW_TAG)
    s = strip_tag(s, GT_TAG)
    return s

def build_gt_index(gt_dir):
    """
    Index GT files by their common stem
    """
    gt_files = list_images(gt_dir)
    gt_by_stem = defaultdict(list)

    for p in gt_files:
        fname = os.path.basename(p)
        stem, _ = os.path.splitext(fname)
        key = normalize_pair_stem(stem).lower()
        gt_by_stem[key].append(p)

    return gt_by_stem

def pick_best_candidate(raw_path, cands):
    """
    If multiple candidates exist, prefer one with the same extension as raw.
    If still multiple, pick the first alphabetically.
    """
    if not cands:
        return None
    raw_ext = os.path.splitext(raw_path)[1].lower()
    same_ext = [p for p in cands if os.path.splitext(p)[1].lower() == raw_ext]
    return sorted(same_ext if same_ext else cands)[0]

def read_mask01(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 0).astype(np.uint8)

def thicken_mask(mask01, thickness):
    if thickness <= 1:
        return mask01
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    return cv2.dilate(mask01, kernel, iterations=1)

# =========================
# Main
# =========================
def main():
    ensure_dir(OUT_DIR)

    raw_files = list_images(RAW_DIR)
    if not raw_files:
        print("❌ No RAW images found")
        return

    gt_by_stem = build_gt_index(GT_DIR)

    saved = missing = unread = mismatch = 0

    for raw_path in raw_files:
        raw_fname = os.path.basename(raw_path)
        raw_stem, _ = os.path.splitext(raw_fname)

        key = normalize_pair_stem(raw_stem).lower()
        gt_cands = gt_by_stem.get(key, [])
        gt_path = pick_best_candidate(raw_path, gt_cands)

        if gt_path is None:
            missing += 1
            continue

        img = cv2.imread(raw_path)
        if img is None:
            unread += 1
            continue

        mask = read_mask01(gt_path)
        if mask is None:
            unread += 1
            continue

        if img.shape[:2] != mask.shape:
            mismatch += 1
            continue

        # 🔥 Apply thickness
        mask = thicken_mask(mask, LINE_THICKNESS)

        # 🔵 Direct pixel overwrite
        out = img.copy()
        out[mask == 1] = COLOR

        out_name = f"{raw_stem}_gt_overlay.png"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), out)
        saved += 1

    print(f"✅ Saved: {saved}")
    print(f"⚠️ GT not found: {missing}")
    print(f"⚠️ Size mismatch: {mismatch}")
    print(f"⚠️ Read failed: {unread}")
    print(f"📂 OUT_DIR: {OUT_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Overlay GT masks on raw images for visual inspection")
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw images")
    parser.add_argument("--gt-dir", required=True, help="Directory containing GT mask images")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory (default: %(default)s)")
    parser.add_argument("--thickness", type=int, default=LINE_THICKNESS,
                        help="GT line thickness for visualization (default: %(default)s)")
    args = parser.parse_args()

    RAW_DIR = args.raw_dir
    GT_DIR = args.gt_dir
    OUT_DIR = args.out_dir
    LINE_THICKNESS = args.thickness

    main()

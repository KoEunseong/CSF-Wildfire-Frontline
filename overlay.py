import os
import cv2
import numpy as np
from glob import glob
from collections import defaultdict

# =========================
# 설정
# =========================
RAW_DIR = ""              # raw 이미지 폴더 (--raw-dir 로 지정)
GT_DIR  = ""              # GT 마스크 폴더 (--gt-dir 로 지정)
OUT_DIR = "./outputs/overlay_gt"  # 결과 저장 폴더 (--out-dir 로 변경 가능)

COLOR = (255, 0, 0)      # 🔵 파란색 (BGR)
LINE_THICKNESS = 3       # GT 선 두께 (1=원본, 3~5 추천)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# 파일명 suffix 규칙
RAW_TAG = "_Raw"         # raw 파일명 끝에 붙는 태그
GT_TAG  = "_Line"        # gt  파일명 끝에 붙는 태그

# =========================
# 유틸
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_images(dir_path):
    return sorted([
        p for p in glob(os.path.join(dir_path, "*"))
        if os.path.splitext(p)[1].lower() in IMG_EXTS
    ])

def strip_tag(stem: str, tag: str) -> str:
    """stem 끝에 tag가 붙어있으면 제거"""
    if not tag:
        return stem
    if stem.lower().endswith(tag.lower()):
        return stem[:-len(tag)]
    return stem

def normalize_pair_stem(stem: str) -> str:
    """
    raw/gt 모두에서 공통으로 비교할 key 만들기.
    - raw면 _Raw 제거
    - gt면 _Line 제거
    둘 중 뭐든 들어오면 공통 stem으로 맞춰짐.
    """
    s = strip_tag(stem, RAW_TAG)
    s = strip_tag(s, GT_TAG)
    return s

def build_gt_index(gt_dir):
    """
    GT 파일들을 공통 stem 기준으로 인덱싱
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
    후보가 여러 개면 raw와 확장자가 같은 것을 우선.
    그래도 여러 개면 알파벳 순으로 하나 선택.
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
# 메인
# =========================
def main():
    ensure_dir(OUT_DIR)

    raw_files = list_images(RAW_DIR)
    if not raw_files:
        print("❌ RAW 이미지 없음")
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

        # 🔥 두께 적용
        mask = thicken_mask(mask, LINE_THICKNESS)

        # 🔵 그냥 덮어쓰기
        out = img.copy()
        out[mask == 1] = COLOR

        out_name = f"{raw_stem}_gt_overlay.png"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), out)
        saved += 1

    print(f"✅ 저장: {saved}")
    print(f"⚠️ GT 없음: {missing}")
    print(f"⚠️ 크기 불일치: {mismatch}")
    print(f"⚠️ 읽기 실패: {unread}")
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

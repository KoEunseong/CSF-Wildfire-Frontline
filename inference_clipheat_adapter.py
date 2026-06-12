import os
import json
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, CLIPProcessor, CLIPModel
from datetime import datetime

# ==========================================
# [1] 설정
# ==========================================
DATA_ROOT = ""   # 구조: DATA_ROOT/test/raw  (--data-root 로 지정)
MODEL_PATH = ""  # 학습된 모델 .pth 경로 (--model-path 로 지정)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join("./Inference_Result", f"Run_{current_time}")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALPHA = 1.0
THRESH = 0.3

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# ==========================================
# [2] 유틸리티: meta / local-first
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
# [3] 전처리 / overlay
# ==========================================
def preprocess_rgb_to_tensor(input_rgb_uint8: np.ndarray) -> torch.Tensor:
    x = input_rgb_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
    return x

def apply_overlay(image_rgb, mask01, color_rgb=(0, 0, 255), alpha=0.8, dilate_vis=False, dilate_kernel=3):
    m = mask01.copy().astype(np.uint8)
    if dilate_vis:
        k = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        m = cv2.dilate(m, k, iterations=1)

    colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
    colored_mask[m == 1] = color_rgb
    return cv2.addWeighted(image_rgb, 1.0, colored_mask, alpha, 0)

# ==========================================
# [4] CLIP heatmap + grid scores
# ==========================================
@torch.no_grad()
def clip_fire_heatmap_and_grid(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    image_rgb_uint8: np.ndarray,
    prompts: list[str],
    grid: int = 4,
    patch_px: int = 224,
    device: str = "cpu"
):
    """
    Returns:
      - heat_full: (H,W) float32 in [0,1]  (원본 해상도 기준 업샘플된 heat)
      - norm_grid: (grid,grid) float32 in [0,1] (패치 단위 점수 정규화)
    """
    H, W, _ = image_rgb_uint8.shape

    # text embedding (avg)
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
# [5] 시각화 저장: (A) 컬러맵 오버레이 (B) 그리드 숫자
# ==========================================
def save_heat_colormap_overlay(image_rgb_uint8: np.ndarray, heat01: np.ndarray, out_path: str, alpha: float = 0.45):
    """
    image_rgb_uint8: (H,W,3) uint8
    heat01: (H,W) float32 in [0,1]
    out_path: png 저장 경로
    """
    heat_u8 = np.clip(heat01 * 255.0, 0, 255).astype(np.uint8)  # (H,W)
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # (H,W,3) BGR
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_rgb_uint8, 1.0, heat_color_rgb, float(alpha), 0)

    # 저장은 BGR
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def save_grid_numbers_overlay(norm_grid: np.ndarray, image_rgb_uint8: np.ndarray, out_path: str):
    """
    norm_grid(g,g) 값을 원본 이미지 위에 그리드 + 중앙 숫자만 표시해서 저장 (색 오버레이 없음)
    """
    g = int(norm_grid.shape[0])
    H, W, _ = image_rgb_uint8.shape
    vis = image_rgb_uint8.copy()

    cell_w = W / g
    cell_h = H / g
    cell_min = min(cell_w, cell_h)

    # 패치 크기에 맞춰 폰트 스케일 자동 조절
    font_scale = float(np.clip(cell_min / 220.0, 0.25, 0.9))
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_th = 1 if cell_min < 120 else 2

    # grid 라인
    for k in range(1, g):
        x = int(round(W * k / g))
        y = int(round(H * k / g))
        cv2.line(vis, (x, 0), (x, H - 1), (255, 255, 255), line_th, cv2.LINE_AA)
        cv2.line(vis, (0, y), (W - 1, y), (255, 255, 255), line_th, cv2.LINE_AA)

    # 중앙 숫자
    for gy in range(g):
        for gx in range(g):
            val = float(norm_grid[gy, gx])
            txt = f"{val:.2f}"

            cx = int((gx + 0.5) * cell_w)
            cy = int((gy + 0.5) * cell_h)

            (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness)
            tx = cx - tw // 2
            ty = cy + th // 2

            # 검정 테두리 + 흰색 글자
            cv2.putText(vis, txt, (tx, ty), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(vis, txt, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

# ==========================================
# [6] 모델: 4ch adapter -> SegFormer(3ch)
# ==========================================
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

# ==========================================
# [7] 실행 로직
# ==========================================
def run_inference():
    os.makedirs(SAVE_DIR, exist_ok=True)
    overlay_dir = os.path.join(SAVE_DIR, "overlays")
    predmask_dir = os.path.join(SAVE_DIR, "pred_masks")

    # ✅ 추가 저장 폴더
    heat_vis_dir = os.path.join(SAVE_DIR, "heat_vis")
    heat_overlay_dir = os.path.join(heat_vis_dir, "heat_overlay")      # 컬러맵 오버레이(숫자 없음)
    grid_numbers_dir = os.path.join(heat_vis_dir, "grid_numbers")      # 그리드+숫자

    for d in [overlay_dir, predmask_dir, heat_vis_dir, heat_overlay_dir, grid_numbers_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"📂 결과 저장 폴더: '{SAVE_DIR}'")
    print("🚀 추론 시작!")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일이 없습니다: {MODEL_PATH}")
        return

    # -------- meta 읽기 --------
    global IMG_SIZE
    meta = load_meta_if_exists(MODEL_PATH)

    base_model_name = meta.get("model_name", "nvidia/mit-b3")
    IMG_SIZE = int(meta.get("img_size", IMG_SIZE))

    clip_model_name = meta.get("clip_model", "openai/clip-vit-base-patch32")
    clip_grid = int(meta.get("clip_grid", 4))
    clip_prompts = meta.get("clip_prompts", ["the boundary line where fire meets the ground", "flame base touching the soil", "the frontline of a forest fire"])
    if isinstance(clip_prompts, str):
        clip_prompts = [p.strip() for p in clip_prompts.split(",") if p.strip()]

    local_models_base = meta.get("local_models_base", "./models")
    clip_cache_dir = meta.get("clip_cache_dir", "./clip_cache")
    clip_cache_dir = os.path.join(clip_cache_dir, "test")
    os.makedirs(clip_cache_dir, exist_ok=True)

    clip_device = "cpu"
    if meta.get("clip_device", "cpu") == "cuda" and torch.cuda.is_available():
        # 원하면 여기서 cuda로 바꿔도 되지만 안전하게 cpu 유지
        clip_device = "cpu"

    print(f"✅ Base model: {base_model_name}")
    print(f"✅ IMG_SIZE: {IMG_SIZE}")
    print(f"✅ DEVICE: {DEVICE}")
    print(f"✅ CLIP model: {clip_model_name} | grid={clip_grid} | device={clip_device}")
    print(f"✅ CLIP prompts: {clip_prompts}")
    print(f"✅ CLIP cache: {clip_cache_dir}")
    print(f"✅ local_models_base: {local_models_base}")

    # -------- CLIP 로드 (local-first) --------
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

    # -------- SegFormer(4ch adapter) 로드 + weight 로드 --------
    model = SegFormerWithInputAdapter(base_model_name, local_models_base=local_models_base).to(DEVICE)

    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(sd, strict=True)
    print("✅ 학습된 모델 가중치를 불러왔습니다. (SegFormerWithInputAdapter, strict=True)")
    model.eval()

    # 테스트 파일 목록
    test_dir = os.path.join(DATA_ROOT, "test", "raw")
    if not os.path.exists(test_dir):
        print(f"❌ 테스트 폴더가 없습니다: {test_dir}")
        return

    image_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(IMG_EXTENSIONS)])
    if len(image_files) == 0:
        print("⚠️ 해당 폴더에 이미지 파일이 없습니다.")
        return

    print(f"🔍 총 {len(image_files)}장의 이미지를 발견했습니다.")

    for idx, img_name in enumerate(image_files, 1):
        img_path = os.path.join(test_dir, img_name)

        original_bgr = cv2.imread(img_path)
        if original_bgr is None:
            print(f"⚠️ 이미지를 읽을 수 없습니다 (Skip): {img_name}")
            continue

        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        # ---------- CLIP heat / grid (cache 우선) ----------
        h_path = heat_cache_path(clip_cache_dir, img_name, clip_grid)
        g_path = grid_cache_path(clip_cache_dir, img_name, clip_grid)

        heat = None
        norm_grid = None

        if os.path.exists(h_path):
            heat = np.load(h_path).astype(np.float32)
        if os.path.exists(g_path):
            norm_grid = np.load(g_path).astype(np.float32)

        # 캐시가 없거나, shape가 이상하면 다시 계산
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

        # input resize (SegFormer 입력 크기)
        input_rgb = cv2.resize(original_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # ✅ 추가 저장 1) 컬러맵 heat 오버레이(숫자 없음)
        heat_overlay_path = os.path.join(heat_overlay_dir, f"HeatOverlay_{os.path.splitext(img_name)[0]}.png")
        save_heat_colormap_overlay(input_rgb.astype(np.uint8), heat, heat_overlay_path, alpha=0.45)

        # ✅ 추가 저장 2) 그리드 + 중앙 숫자 (색 없음)
        grid_numbers_path = os.path.join(grid_numbers_dir, f"GridNumbers_{os.path.splitext(img_name)[0]}.png")
        save_grid_numbers_overlay(norm_grid, input_rgb.astype(np.uint8), grid_numbers_path)

        # ---------- 모델 입력 ----------
        rgb_t = preprocess_rgb_to_tensor(input_rgb).to(DEVICE)  # (1,3,H,W)
        heat_t = torch.from_numpy(heat).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # (1,1,H,W)
        x4 = torch.cat([rgb_t, heat_t], dim=1)  # (1,4,H,W)

        # ---------- 추론 ----------
        with torch.no_grad():
            logits = model(x4)
            logits = nn.functional.interpolate(
                logits, size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear", align_corners=False
            )
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_mask01 = (prob > THRESH).astype(np.uint8)

        # 예측 마스크 저장: 0/255 PNG (기존 기능 유지)
        pred_mask255 = (pred_mask01 * 255).astype(np.uint8)
        mask_save_name = f"{idx-1:04d}.png"
        mask_save_path = os.path.join(predmask_dir, mask_save_name)
        cv2.imwrite(mask_save_path, pred_mask255)

        # prediction overlay 저장 (기존 기능 유지)
        overlay_rgb = apply_overlay(
            image_rgb=input_rgb.astype(np.uint8),
            mask01=pred_mask01,
            color_rgb=(0, 0, 255),
            alpha=ALPHA,
            dilate_vis=True,
            dilate_kernel=3
        )

        fig_save_path = os.path.join(overlay_dir, f"Result_{os.path.splitext(img_name)[0]}.png")
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.title("Original"); plt.imshow(input_rgb); plt.axis("off")
        plt.subplot(1, 2, 2); plt.title("Prediction (Overlay)"); plt.imshow(overlay_rgb); plt.axis("off")
        plt.suptitle(f"File: {img_name}")
        plt.tight_layout()
        plt.savefig(fig_save_path, dpi=150)
        plt.close()

        print(f"💾 [{idx}/{len(image_files)}] "
              f"pred_overlay: {fig_save_path} | mask: {mask_save_path} | "
              f"heat_overlay: {heat_overlay_path} | grid_nums: {grid_numbers_path}")

    print("\n✅ inference 완료")
    print(f"- overlays: {overlay_dir}")
    print(f"- pred masks: {predmask_dir}")
    print(f"- heat overlay: {heat_overlay_dir}")
    print(f"- grid numbers: {grid_numbers_dir}")
    print(f"📂 최종 저장 경로: {SAVE_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference with CLIP-Heat Adapter SegFormer (4ch)")
    parser.add_argument("--data-root", required=True,
                        help="Data root; expects <data-root>/test/raw/ images")
    parser.add_argument("--model-path", required=True,
                        help="Path to trained model .pth checkpoint")
    parser.add_argument("--save-dir", default=None,
                        help="Output directory (default: ./outputs/Run_<timestamp>)")
    parser.add_argument("--thresh", type=float, default=THRESH,
                        help="Prediction threshold (default: %(default)s)")
    parser.add_argument("--model-dir", default="./models",
                        help="Local HuggingFace model cache dir (default: %(default)s)")
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    MODEL_PATH = args.model_path
    THRESH = args.thresh
    if args.save_dir:
        SAVE_DIR = args.save_dir

    run_inference()

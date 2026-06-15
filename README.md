# Frontline Model — Wildfire Frontline Segmentation

Segmentation model for detecting wildfire frontlines in aerial/UAV imagery.  
Architecture: **CLIP-Heat + SegFormer** — a 4-channel input SegFormer that fuses RGB with a CLIP-based fire-likelihood heatmap.

---

## Models

| Script | Architecture | Input |
|---|---|---|
| `train_clipheat_in4.py` | SegFormer 4ch direct | RGB + CLIP heatmap (4ch tensor) |
| `train_baseline.py` | SegFormer 3ch | RGB only |
| `finetune_clipheat_in4.py` | Fine-tune on real images | RGB + CLIP heatmap |
| `comparison_experiments/train_deeplabv3plus.py` | DeepLabV3+ | RGB only |

---

## Installation

See [install.md](install.md) for full setup instructions.

**Quick start:**

```bash
conda create -n frontline python=3.11 -y
conda activate frontline

# PyTorch (CUDA 12.8)
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

---

## Dataset Structure

```
Dataset/
├── sim/            # Simulation-generated data
│   ├── train/
│   │   ├── raw/    # RGB images  (e.g., Wildfire_0000_Raw.png)
│   │   └── gt/     # GT masks    (e.g., Wildfire_0000_Line.png)
│   └── test/
│       ├── raw/
│       └── gt/
└── real/           # Real UAV imagery
    └── real_data/
        ├── raw/
        └── gt/
```

---

## HuggingFace Models

Scripts automatically download from HuggingFace if not cached locally.  
To use offline, place model files under `models/<hf-repo-id>/`:

```
models/
├── nvidia/mit-b3/
│   ├── config.json
│   └── model.safetensors
└── openai/clip-vit-base-patch32/
    ├── config.json
    └── model.safetensors
```

---

## Training

```bash
# CLIP-Heat In4 (main model)
python train_clipheat_in4.py \
    --data-root ./Dataset/sim \
    --model-name nvidia/mit-b3 \
    --clip-model openai/clip-vit-base-patch32

# Baseline (RGB only)
python train_baseline.py \
    --data-root ./Dataset/sim \
    --model-name nvidia/mit-b1

# Fine-tune on real images
python finetune_clipheat_in4.py \
    --pretrained-path ./train_runs/<run_id>/best_model.pth \
    --data-root ./Dataset/real/real_data
```

Training checkpoints and `train_meta.json` are saved to `./train_runs/<run_id>/`.

---

## Inference

```bash
# Baseline
python inference.py \
    --data-root ./Dataset/real/real_data \
    --model-path ./train_runs/<run_id>/best_model.pth

# CLIP-Heat In4
python inference_clipheat_in4.py \
    --data-root ./Dataset/real/real_data \
    --model-path ./train_runs/<run_id>/best_model.pth

# After fine-tuning
python inference_finetune.py \
    --model-path ./finetune_runs/<run_id>/best_model.pth \
    --data-root ./Dataset/real/real_data
```

Results are saved to `./outputs/Run_<timestamp>/`.

---

## Evaluation

```bash
python evaluate.py \
    --pred-dir ./outputs/Run_<timestamp>/pred_masks \
    --gt-dir ./Dataset/real/real_data/gt
```

Metrics: IoU, Dice, Precision, Recall, F1, Chamfer, HD95, TolF1.

---

## GT Overlay Visualization

```bash
python overlay.py \
    --raw-dir ./Dataset/sim/train/raw \
    --gt-dir ./Dataset/sim/train/gt \
    --out-dir ./outputs/overlay_gt
```

---

## Project Structure

```
frontline_model_public/
├── train_clipheat_in4.py
├── train_baseline.py
├── finetune_clipheat_in4.py
├── inference.py
├── inference_clipheat_in4.py
├── inference_finetune.py
├── evaluate.py
├── overlay.py
├── comparison_experiments/
│   └── train_deeplabv3plus.py
├── models/             # HuggingFace model cache (not tracked)
├── clip_cache/         # CLIP feature cache (not tracked)
├── outputs/            # Inference / eval results (not tracked)
├── requirements.txt
└── install.md
```

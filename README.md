# Frontline Model — Wildfire Frontline Segmentation

Segmentation model for detecting wildfire frontlines in aerial/UAV imagery.  
Architecture: **CLIP-Heat + SegFormer** — a 4-channel input SegFormer that fuses RGB with a CLIP-based fire-likelihood heatmap.

---

## Models

| Script | Mode | Architecture | Input |
|---|---|---|---|
| `train.py` (default) | CLIP-Heat | SegFormer 4ch direct | RGB + CLIP heatmap (4ch tensor) |
| `train.py --no-clip` | Baseline | SegFormer 3ch | RGB only |

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
# CLIP-Heat (default — 4ch SegFormer + CLIP heatmap)
python train.py \
    --data_root ./Dataset/sim \
    --model_name nvidia/mit-b3 \
    --clip_model openai/clip-vit-base-patch32

# Baseline (RGB only — standard 3ch SegFormer)
python train.py --no-clip \
    --data_root ./Dataset/sim \
    --model_name nvidia/mit-b3
```

Training checkpoints and `train_meta.json` are saved to `./train_runs/<run_id>/`.

---

## Inference

```bash
# CLIP-Heat (default — 4ch model + CLIP heatmap)
python inference.py \
    --data-root ./Dataset/real/test \
    --model-path ./train_runs/<run_id>/best_fireline_model.pth

# Baseline (RGB only — standard 3ch model)
python inference.py --no-clip \
    --data-root ./Dataset/real/test \
    --model-path ./train_runs/<run_id>/best_fireline_model.pth
```

If `train_meta.json` is found next to the checkpoint and contains `"use_clip": false`,
`--no-clip` is inferred automatically.

Results are saved to `./Inference_Result/Run_<timestamp>/`.

---

## Evaluation

```bash
python evaluate.py \
    --pred-dir ./Inference_Result/Run_<timestamp>/pred_masks \
    --gt-dir ./Dataset/real/test/gt
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
├── train.py
├── inference.py
├── evaluate.py
├── overlay.py
├── models/             # HuggingFace model cache (not tracked)
├── clip_cache/         # CLIP feature cache (not tracked)
├── outputs/            # Inference / eval results (not tracked)
├── requirements.txt
└── install.md
```

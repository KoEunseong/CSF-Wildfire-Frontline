# Dataset Structure

Download the dataset from HuggingFace — see the main [README](../README.md) for instructions.

After download, the structure will be:

```
Dataset/
├── checkpoints/
│   └── real_best_fireline_model.pth   # pretrained checkpoint
├── synthetic/                          # Simulation-generated data
│   ├── train/
│   │   └── gt/     # GT line masks (e.g., Wildfire_0000_Line.png)
│   └── test/
│       ├── raw/    # RGB images    (e.g., Wildfire_0000_Raw.png)
│       └── gt/     # GT line masks
└── real/                               # Real UAV imagery
    ├── raw/
    └── gt/
```

## Naming Convention

| Split | Raw image | GT mask |
|---|---|---|
| synthetic | `Wildfire_XXXX_Raw.png` | `Wildfire_XXXX_Line.png` |
| real | `<filename>.png` | `<filename>.png` (same name) |

## Notes

- GT masks are **binary** images: foreground (fire frontline) = 255, background = 0
- Images are resized to the `--img_size` value (default: 512) at training/inference time
- `synthetic/` data is used for initial training; `real/` data is used for evaluation

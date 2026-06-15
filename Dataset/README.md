# Dataset Structure

Place your dataset files following the structure below.

```
Dataset/
├── sim/                        # Simulation-generated data
│   ├── train/
│   │   ├── raw/                # RGB images    (e.g., Wildfire_0000_Raw.png)
│   │   └── gt/                 # GT line masks (e.g., Wildfire_0000_Line.png)
│   ├── val/
│   │   ├── raw/
│   │   └── gt/
│   └── test/
│       ├── raw/
│       └── gt/
└── real/                       # Real UAV imagery
    └── real_data/
        ├── raw/
        └── gt/
```

## Naming Convention

| Split | Raw image | GT mask |
|---|---|---|
| sim | `Wildfire_XXXX_Raw.png` | `Wildfire_XXXX_Line.png` |
| real | `<filename>.png` | `<filename>.png` (same name) |

## Notes

- GT masks are **binary** images: foreground (fire frontline) = 255, background = 0
- Images are resized to the `--img_size` value (default: 512) at training/inference time
- `sim/` data is used for initial training; `real/` data is used for evaluation and fine-tuning

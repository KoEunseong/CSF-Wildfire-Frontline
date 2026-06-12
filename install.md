# Installation

## Requirements

- **OS**: Ubuntu 22.04
- **CUDA**: 12.8
- **Python**: 3.11

## Setup

### 1. Create conda environment

```bash
conda create -n frontline python=3.11 -y
conda activate frontline
```

### 2. Install PyTorch (CUDA 12.8)

```bash
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `requirements.txt` does not include torch/torchvision — install them first via step 2 to ensure the correct CUDA build.

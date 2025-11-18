# Reproducibility Guide for ZAC2025 Training

## Overview

Để đảm bảo reproducibility (khả năng tái tạo kết quả) trong training và inference, cần cố định **random seeds** cho tất cả các thành phần.

## Required Seed Parameters

### 1. `--seed` (Model/Optimizer Seed)
Controls random initialization for:
- Model weight initialization
- Optimizer state
- Dropout layers
- PyTorch, NumPy, Python random modules

**Default value**: `42`

### 2. `--data_seed` (Data Shuffling Seed)
Controls:
- Data shuffling order
- Train/validation splits
- DataLoader sampling

**Default value**: If not set, uses `seed + epoch` (NOT reproducible across runs!)

## How to Enable Full Reproducibility

### Training Scripts
All training scripts now include both seed parameters:

```bash
deepspeed src/train/train_sft.py \
    --seed 42 \
    --data_seed 42 \
    # ... other args
```

### Resume Training
When resuming from checkpoint with `--resume_from_checkpoint`:
- RNG states (random number generator) are automatically saved and restored
- Data order is preserved with `--data_seed`
- Model/optimizer states are restored from checkpoint

**Important**: Always use the same `--seed` and `--data_seed` values when resuming!

## Verified Scripts

All training scripts have been updated with reproducibility support:

✅ `scripts/train_zac_kaggle.sh` - Phase 1 training
✅ `scripts/train_zac_kaggle_2gpu.sh` - 2-GPU training
✅ `scripts/train_zac_kaggle_phase2.sh` - Phase 2 (unfreeze LLM)
✅ `scripts/train_zac_kaggle_phase2a.sh` - Phase 2a (unfreeze top-k layers)
✅ `scripts/train_zac_lora.sh` - LoRA training

## Resume from Checkpoint (Fixed)

### Fixed Issues
1. **Resume logic**: Now respects `--resume_from_checkpoint` argument (priority over auto-detection)
2. **Checkpoint structure**: No longer copies checkpoint incorrectly
3. **Seed consistency**: Both `seed` and `data_seed` are set for resumed training

### How Resume Works

**Priority order**:
1. If `--resume_from_checkpoint /path/to/checkpoint` is provided → use that path
2. Otherwise, auto-detect latest `checkpoint-*` in `--output_dir`
3. If no checkpoint found → train from scratch

**Example** (Phase 2a):
```bash
RESUME_FROM="/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest"
OUTPUT_DIR="/kaggle/working/checkpoints/zac_qwen2vl_phase2a"

# Training will:
# 1. Resume from RESUME_FROM checkpoint
# 2. Save new checkpoints to OUTPUT_DIR
# 3. Maintain reproducibility with same seeds
deepspeed src/train/train_sft.py \
    --resume_from_checkpoint "$RESUME_FROM" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    --data_seed 42 \
    # ...
```

## Technical Details

### What Gets Seeded?

The `set_seed()` function in `src/train/train_sft.py` sets:

```python
# Python standard library
random.seed(seed)

# NumPy
np.random.seed(seed)

# PyTorch CPU & CUDA
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# CuDNN deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Saved RNG States

When saving checkpoints, the following are automatically saved:
- PyTorch RNG state
- CUDA RNG state (all GPUs)
- Python random state
- NumPy random state

These are restored when using `--resume_from_checkpoint`.

## Inference Reproducibility

For inference, set seeds in your inference script:

```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call before inference
set_seed(42)
```

## Important Notes

⚠️ **Performance Impact**: Setting `cudnn.deterministic=True` may reduce training speed slightly (~5-10%) but ensures reproducibility.

⚠️ **Multi-GPU**: With DeepSpeed/DDP, each GPU rank may see different data samples (as expected), but the overall training is reproducible.

⚠️ **DataLoader Workers**: When using `--dataloader_num_workers > 0`, worker processes are automatically seeded by Transformers Trainer.

## Validation

To verify reproducibility:

1. Train for N steps with `--seed 42 --data_seed 42`
2. Resume from same checkpoint with same seeds
3. Results should be identical

## Competition Requirements

> Để bảo đảm reproducibility, các đội phải:
> - Set seed cố định trong training và inference.

✅ **Compliance**: All scripts now set fixed seeds (`42`) for both training and can be used for inference.

## References

- HuggingFace TrainingArguments: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
- PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html

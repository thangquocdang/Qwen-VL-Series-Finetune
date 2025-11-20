# RoadBuddy - Video Question Answering Solution
## Zalo AI Challenge 2025

### Team Information
- **Team Name**: [Your Team Name]
- **Members**: [Team Members]

---

## Solution Overview

Our solution tackles the video question answering task using a fine-tuned Qwen2-VL-2B-Instruct model with incremental unfreezing strategy.

### Key Components

1. **Base Model**: Qwen2-VL-2B-Instruct (vision-language model)
2. **Training Strategy**: 2-phase incremental unfreezing
3. **Optimization**: LoRA (rank=128) + Merger fine-tuning
4. **Inference**: Zero-shot reasoning with Chain-of-Thought

---

## Training Approach

### Phase 1: LoRA + Merger Training

**Trainable Components:**
- LoRA adapters (all 28 LLM layers)
- Visual-to-text merger layer

**Frozen Components:**
- Vision tower (visual encoder)
- LLM base weights

**Configuration:**
```bash
# Hyperparameters
- Learning rate: 1e-4
- LoRA rank: 128
- LoRA alpha: 256
- Batch size: 8 (global)
- Epochs: 3
- Optimizer: AdamW
- Scheduler: Cosine with warmup
```

**Training Data:**
- Format: LLaVA-style conversation format
- Videos: Traffic dashcam footage
- Annotations: Multiple-choice Q&A with reasoning

### Phase 2a: Incremental Unfreezing (Top-6 LLM Layers)

**Additional Trainable:**
- Top 6 LLM layers (layers 22-27)

**Why incremental unfreezing?**
- Preserve knowledge from Phase 1
- Fine-tune deeper layers for better reasoning
- Memory efficient (compared to full unfreezing)

**Configuration:**
```bash
# Hyperparameters
- Learning rate: 5e-5 (lower than Phase 1)
- Unfreeze: Top 6 layers
- Epochs: 8
- Resume from: Phase 1 checkpoint
```

**Key Innovation:**
Our approach uses **incremental unfreezing** instead of full fine-tuning, which:
- Reduces overfitting risk
- Maintains stability from Phase 1
- Improves reasoning on complex traffic scenarios

---

## Training Data

### Data Format

Training data follows this format:

```json
{
  "id": "sample_001",
  "video": "videos/sample.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "<video>\n[Question]\n\nA. [Choice A]\nB. [Choice B]\nC. [Choice C]\nD. [Choice D]"
    },
    {
      "from": "gpt",
      "value": "1. QUAN SÁT: [Observation]\n2. KẾT LUẬN: [Reasoning]\n4. Đáp án: [A/B/C/D]"
    }
  ]
}
```

### Data Sources

**Training Data:**
- **Source**: Provided by organizers + augmentation
- **Size**: ~600 samples (training set)
- **Augmentation**: Paraphrased questions with same video
- **Location**: Upload to HuggingFace (see below)

**Download Links:**
```bash
# Training data
wget https://huggingface.co/datasets/[YOUR-USERNAME]/zac2025-training-data/training_data.json

# Phase 1 checkpoint (LoRA + Merger)
wget https://huggingface.co/[YOUR-USERNAME]/zac2025-phase1/checkpoint-latest.tar.gz

# Phase 2a checkpoint (LoRA + Merger + Top-6 layers)
wget https://huggingface.co/[YOUR-USERNAME]/zac2025-phase2a/checkpoint-latest.tar.gz
```

**Note**: Replace `[YOUR-USERNAME]` with your HuggingFace username.

---

## Reproducibility

### Seed Configuration

All experiments use **fixed seed = 42**:

```python
import random, numpy as np, torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Training Scripts

**Phase 1:**
```bash
cd /path/to/Qwen-VL-Series-Finetune
bash scripts/train_zac_kaggle.sh
```

**Phase 2a:**
```bash
bash scripts/train_zac_kaggle_phase2a.sh
```

### Environment

**Hardware:**
- GPU: 2x T4 (Kaggle)
- RAM: 30GB
- Storage: 40GB

**Software:**
- Python: 3.10
- PyTorch: 2.1.2
- Transformers: 4.47.1
- CUDA: 12.1

---

## Inference Approach

### Prompt Format

We use the **exact training format** for inference:

```
<video>
[Question text]

A. [Choice A]
B. [Choice B]
C. [Choice C]
D. [Choice D]
```

**No additional instructions** - model learned format from training.

### Response Format

Model outputs reasoning before answer:

```
1. QUAN SÁT: [What the model observes in the video]
2. KẾT LUẬN: [Reasoning about the answer]
4. Đáp án: [A/B/C/D]. [Explanation]
```

### Answer Extraction

We extract final answer with priority:
1. "Đáp án: X" pattern (training format)
2. Standalone A/B/C/D
3. First character if matches A/B/C/D

---

## Model Architecture Details

### Trainable Parameters

**Phase 1:**
- LoRA adapters: 187M params (37.3%)
- Merger layers: 34M params (6.8%)
- **Total trainable: 221M params (9.2%)**

**Phase 2a:**
- LoRA adapters: 187M params (37.3%)
- Merger layers: 34M params (6.8%)
- Top-6 LLM layers: 281M params (55.9%)
- **Total trainable: 502M params (20.9%)**

### Memory Optimization

- FP16 mixed precision training
- Gradient checkpointing
- DeepSpeed ZeRO-2
- Liger kernel for memory efficiency

---

## Validation & Results

### Validation Strategy

- Split: 90/10 train/val
- Metric: Multiple-choice accuracy
- Early stopping: patience=3

### Training Loss Curve

**Phase 1:**
- Initial loss: ~2.0
- Final loss: ~0.002
- Converged after ~900 steps

**Phase 2a:**
- Starts from Phase 1 weights
- Additional 8 epochs
- Final loss: ~0.001

---

## Code Structure

```
Qwen-VL-Series-Finetune/
├── src/
│   ├── train/
│   │   └── train_sft.py          # Main training script
│   ├── dataset.py                 # Data loading
│   ├── trainer/
│   │   └── sft_trainer.py        # Custom trainer
│   └── params.py                  # Training arguments
├── scripts/
│   ├── train_zac_kaggle.sh       # Phase 1 training
│   └── train_zac_kaggle_phase2a.sh # Phase 2a training
├── submission/
│   ├── predict.py                 # Inference script
│   ├── predict.sh                 # Bash wrapper
│   └── predict_notebook.ipynb    # Time measurement
└── docs/
    ├── REPRODUCIBILITY.md         # Reproducibility guide
    └── VIDEO_QA_INFERENCE.md      # Inference guide
```

---

## Reproduce Training

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download data:
```bash
# From HuggingFace
wget https://huggingface.co/datasets/[YOUR-USERNAME]/zac2025-training-data/training_data.json

# Or upload your data to HuggingFace first
```

3. Set up environment:
```bash
export PYTHONPATH=/path/to/Qwen-VL-Series-Finetune:$PYTHONPATH
```

### Phase 1 Training

```bash
cd Qwen-VL-Series-Finetune
bash scripts/train_zac_kaggle.sh
```

**Expected output:**
- Checkpoint: `/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest`
- Training time: ~2-3 hours (2x T4)
- Final loss: ~0.002

### Phase 2a Training

```bash
bash scripts/train_zac_kaggle_phase2a.sh
```

**Expected output:**
- Checkpoint: `/kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest`
- Training time: ~4-5 hours (2x T4)
- Final loss: ~0.001

### Verify Checkpoint

```bash
ls -lh /kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest/
```

Expected files:
- `adapter_model.safetensors` (LoRA weights)
- `adapter_config.json`
- `merger_weights.bin` (merger weights)
- `config.json`, `tokenizer.json`, etc.

---

## Performance Metrics

### Inference Speed

**Hardware**: Single T4 GPU

- Model load time: ~15-20 seconds
- Per-sample inference: ~1.8 seconds average
- Total for 405 samples: ~12-13 minutes

### Memory Usage

- Model memory (FP16): ~4.5GB
- Peak memory during inference: ~6GB

---

## Key Technical Decisions

### 1. Why Qwen2-VL?

- State-of-art vision-language model
- Strong video understanding
- Efficient architecture (2B params)

### 2. Why Incremental Unfreezing?

- Better than full fine-tuning (less overfitting)
- Better than LoRA-only (more capacity)
- Memory efficient

### 3. Why Chain-of-Thought Format?

- Improves reasoning quality
- Explainable predictions
- Robust answer extraction

---

## Known Limitations

1. **Long videos**: Model may miss details in very long clips (>30s)
2. **Rare scenarios**: Limited training data for uncommon traffic situations
3. **Ambiguous questions**: May struggle with poorly worded questions

---

## Future Improvements

1. **Data augmentation**: More diverse question paraphrasing
2. **Video sampling**: Smart frame selection instead of uniform sampling
3. **Ensemble**: Combine multiple checkpoints
4. **Post-processing**: Confidence-based answer refinement

---

## References

- **Qwen2-VL Paper**: https://arxiv.org/abs/2409.12191
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Repository**: https://github.com/QwenLM/Qwen2-VL

---

## Contact

For questions about this solution, please contact:
- Email: [your-email@example.com]
- GitHub: [your-github-username]

---

## Acknowledgments

- Zalo AI Challenge 2025 organizers
- Qwen2-VL team for the amazing base model
- Kaggle for compute resources

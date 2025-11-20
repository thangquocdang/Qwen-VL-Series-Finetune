# Zalo AI Challenge 2025 - Submission Package

Docker submission package for video question answering task.

## 📦 Package Contents

```
submission/
├── predict.py                      # Main inference script
├── predict.sh                      # Bash wrapper for predict.py
├── predict_notebook.ipynb          # Jupyter notebook for time measurement
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker container definition
├── start_jupyter.sh               # Script to start Jupyter server
├── setup_submission.sh            # Helper script to copy checkpoint
├── SUBMISSION_CHECKLIST.md        # Detailed checklist and instructions
├── training_code/
│   └── README.md                  # Training methodology and reproducibility
└── saved_models/                   # Your trained model checkpoint (add here)
    └── checkpoint-latest/
```

## ⚠️ Prerequisites: NVIDIA Docker Setup

**QUAN TRỌNG**: Để sử dụng GPU trong Docker, bạn cần cài đặt **NVIDIA Docker** (nvidia-docker2).

### Yêu cầu môi trường (matching BTC server):
- **Driver Version**: ≥ 535.86.10
- **CUDA Version**: 12.2

### Cài đặt nhanh NVIDIA Docker:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Kiểm tra môi trường tự động:

```bash
cd submission
bash check_gpu_setup.sh
```

Script sẽ kiểm tra:
- ✅ NVIDIA Driver (≥ 535.86.10)
- ✅ CUDA Version (12.2)
- ✅ Docker installed
- ✅ NVIDIA Docker working
- ✅ GPU accessible in Docker

**📖 Xem chi tiết**: [NVIDIA_DOCKER_SETUP.md](NVIDIA_DOCKER_SETUP.md) - Hướng dẫn đầy đủ và troubleshooting

---

## 🚀 Quick Start

### 1. Copy Your Checkpoint

Use the provided setup script:

```bash
cd submission
bash setup_submission.sh /path/to/your/checkpoint-latest
```

Or manually:

```bash
mkdir -p submission/saved_models
cp -r /kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest \
      submission/saved_models/
```

### 2. Build Docker Image

```bash
cd submission
docker build -t zac2025-submission .
```

### 3. Test Locally

```bash
# Prepare test data
mkdir -p test_data/videos test_results

# Create test.json (see format below)
# Copy test videos to test_data/videos/

# Run inference
docker run --gpus all \
    -v $(pwd)/test_data:/data \
    -v $(pwd)/test_results:/result \
    zac2025-submission

# Check output
cat test_results/submission.csv
```

### 4. Export for Submission

```bash
docker save zac2025-submission -o zac2025-submission.tar
gzip zac2025-submission.tar
```

## 📋 Input Format

The Docker container expects `/data/test.json` with this format:

```json
{
  "data": [
    {
      "id": "test_0001",
      "question": "Theo trong video, nếu ô tô đi hướng chếch sang phải là hướng vào đường nào?",
      "choices": [
        "A. Không có thông tin",
        "B. Dầu Giây Long Thành",
        "C. Đường Đỗ Xuân Hợp",
        "D. Xa Lộ Hà Nội"
      ],
      "video_path": "videos/sample.mp4"
    }
  ]
}
```

Videos should be in `/data/videos/` (or path specified in `video_path`).

## 📊 Output Format

The container produces `/result/submission.csv`:

```csv
id,answer
test_0001,D
test_0002,A
```

## 📓 Jupyter Notebook

To run the time measurement notebook:

```bash
docker run --gpus all \
    -v $(pwd)/test_data:/data \
    -v $(pwd)/test_results:/result \
    -p 8888:8888 \
    zac2025-submission \
    /bin/bash start_jupyter.sh
```

Open http://localhost:8888 and run `predict_notebook.ipynb`.

The notebook produces:
- `jupyter_submission.csv` - Same format as submission.csv
- `time_submission.csv` - Columns: id, time (ms), answer

## 🔧 Customization

### Update Team Info

Edit `training_code/README.md`:

```markdown
### Team Information
- **Team Name**: [Your Team Name]
- **Members**: [Team Members]
```

### Modify Inference Parameters

Edit `predict.py` to change:
- `max_new_tokens` (default: 256 for reasoning)
- `do_sample` (default: False for deterministic output)
- Seed (default: 42)

## 📚 Documentation

- **SUBMISSION_CHECKLIST.md** - Complete checklist and troubleshooting guide
- **training_code/README.md** - Training methodology and reproducibility
- **../docs/VIDEO_QA_INFERENCE.md** - Detailed inference guide

## 🧪 Testing

### Minimal Test

```bash
# Build image
docker build -t zac2025-submission .

# Test with single sample (create test.json first)
docker run --gpus all \
    -v $(pwd)/test_data:/data \
    -v $(pwd)/test_results:/result \
    zac2025-submission

# Verify output
cat test_results/submission.csv
```

### Full Test

See SUBMISSION_CHECKLIST.md for detailed testing instructions.

## 📏 Requirements

### Hardware
- NVIDIA GPU with 8GB+ VRAM (T4, V100, A100, etc.)
- 16GB+ system RAM

### Software
- Docker with GPU support (nvidia-docker2)
- NVIDIA drivers compatible with CUDA 12.1

## 🔍 Troubleshooting

### Build fails
- Check internet connection (downloads base model ~4.5GB)
- Verify Docker has sufficient disk space (~15GB)

### Inference fails
- Check GPU is available: `docker run --gpus all nvidia-smi`
- Verify checkpoint files exist in `saved_models/checkpoint-latest/`
- Check video files are accessible at `/data/videos/`

### Wrong predictions
- Verify using Phase 2a checkpoint (not Phase 1 or base model)
- Check checkpoint files:
  - `adapter_model.safetensors` (~380MB)
  - `merger_weights.bin` (~130MB)
  - `adapter_config.json`

## 📦 Package Size

- Docker image: ~12-15 GB
- Compressed (tar.gz): ~6-8 GB
- Checkpoint alone: ~500-600 MB

## ⏱️ Performance

- Model load time: ~15-20 seconds
- Inference per sample: ~1.8 seconds
- Total for 405 samples: ~12-13 minutes

## 🎯 Solution Overview

This submission uses:

- **Base Model**: Qwen2-VL-2B-Instruct
- **Training**: 2-phase incremental unfreezing
  - Phase 1: LoRA (rank=128) + Merger
  - Phase 2a: + Top-6 LLM layers
- **Inference**: Zero-shot with Chain-of-Thought reasoning
- **Format**: Matches training format exactly

See `training_code/README.md` for detailed methodology.

## 📞 Support

For questions or issues:

1. Check SUBMISSION_CHECKLIST.md for troubleshooting
2. Review training_code/README.md for methodology
3. Examine ../docs/VIDEO_QA_INFERENCE.md for inference details

## ✅ Final Checklist

Before submission:

- [ ] Checkpoint copied to `saved_models/checkpoint-latest/`
- [ ] Team info updated in `training_code/README.md`
- [ ] Docker image builds successfully
- [ ] Tested with sample data
- [ ] Output format verified
- [ ] Image exported as .tar.gz

## 🎉 Ready to Submit!

Once all steps are complete, your submission package is ready for upload.

Good luck with the competition!

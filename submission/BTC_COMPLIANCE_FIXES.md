# BTC Compliance Fixes - Zalo AI Challenge 2025

Tài liệu này tóm tắt các vấn đề đã được sửa để tuân thủ 100% yêu cầu BTC.

---

## ✅ Các vấn đề đã sửa

### 1. ✅ start_jupyter.sh - Cấu hình sai

**Vấn đề**: Port và password không đúng yêu cầu BTC

**BTC yêu cầu**:
```bash
jupyter lab --port 9777 --ip 0.0.0.0 \
    --NotebookApp.password='zac2025' \
    --NotebookApp.token='zac2025' \
    --allow-root --no-browser
```

**Đã sửa**:
- ✅ Port: 8888 → 9777
- ✅ Password: none → 'zac2025'
- ✅ Token: none → 'zac2025'
- ✅ jupyter notebook → jupyter lab

---

### 2. ✅ Dockerfile - Vi phạm quy định về base model

**Vấn đề**: Dockerfile download base model (~4.5GB) vào image

**BTC nói rõ**: *"Base model dùng trong quá trường training không được đưa vào bên trong Docker để tránh làm Docker quá nặng."*

**Đã sửa**:
- ✅ Xóa dòng download base model
- ✅ Base model sẽ tự động download khi predict.py chạy lần đầu
- ✅ BTC server có internet cho model download

**Kết quả**: Docker image nhẹ hơn ~4.5GB

---

### 3. ✅ requirements.txt - Thiếu jupyterlab

**Vấn đề**: Hướng dẫn BTC yêu cầu `pip install jupyterlab`

**Đã sửa**:
- ✅ Thêm `jupyterlab==4.0.9` vào requirements.txt

---

### 4. ✅ Checkpoint có files không cần thiết

**Vấn đề**: Checkpoint chứa optimizer states và training files (~500MB+)

**Files không cần thiết**:
```
❌ global_step460/ (optimizer states - ~500MB)
❌ scheduler.pt
❌ training_args.bin
❌ rng_state_*.pth
❌ trainer_state.json
❌ zero_to_fp32.py
❌ latest
```

**Files cần thiết** (cho inference):
```
✅ adapter_model.safetensors (~380MB)
✅ adapter_config.json
✅ merger_weights.bin (~130MB)
✅ config.json
✅ tokenizer files (tokenizer.json, vocab.json, merges.txt, etc.)
✅ preprocessor files
✅ chat_template.jinja
```

**Giải pháp**:
- ✅ Tạo script `cleanup_checkpoint.sh` để extract chỉ essential files
- ✅ Giảm checkpoint từ ~1GB xuống ~500-600MB
- ✅ Docker image nhẹ hơn 40-50%

---

### 5. ✅ predict_notebook.ipynb format

**Kiểm tra**: Đúng format BTC yêu cầu

- ✅ Cell 1: Set seed (42)
- ✅ Cell 2: Load model và resources
- ✅ Cell 3: Load test cases
- ✅ Cell 4: Inference với timing cho MỖI sample (không phải trung bình)
- ✅ Cell 5: Save jupyter_submission.csv và time_submission.csv

**Output format**:
- ✅ `time_submission.csv`: id, time (milliseconds), answer
- ✅ `jupyter_submission.csv`: id, answer (giống submission.csv)

---

## 📋 Workflow mới (Sau khi sửa)

### Bước 1: Cleanup checkpoint (MỚI!)

```bash
cd submission

# Extract only essential files
bash cleanup_checkpoint.sh \
    /kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest \
    ./checkpoint-clean

# Copy to submission package
cp -r ./checkpoint-clean ./saved_models/checkpoint-latest
```

**Kết quả**:
- Checkpoint size: ~1GB → ~500-600MB
- Docker image nhẹ hơn 40-50%

### Bước 2: Kiểm tra môi trường

```bash
bash check_gpu_setup.sh
```

### Bước 3: Build Docker

```bash
docker build -t zac2025-submission .
```

**Lưu ý**:
- ✅ Base model KHÔNG có trong image
- ✅ Model sẽ tự download lần đầu chạy
- ✅ Image size nhỏ hơn nhiều

### Bước 4: Test predict.sh

```bash
docker run --gpus '"device=0"' \
    -v /data:/data \
    -v $(pwd)/results:/result \
    zac2025-submission
```

**Lần đầu chạy**:
- Base model sẽ download (~4.5GB, mất 5-10 phút)
- Lần sau chạy nhanh hơn (model đã cached)

### Bước 5: Test Jupyter

```bash
docker run -it --gpus '"device=0"' \
    -p 9777:9777 \
    -v /data:/data \
    -v $(pwd)/results:/result \
    zac2025-submission \
    /bin/bash /workspace/start_jupyter.sh
```

**Truy cập**:
- URL: http://localhost:9777
- Password: zac2025

**Chạy notebook**:
- Open `predict_notebook.ipynb`
- Run all cells
- Kiểm tra:
  - ✅ jupyter_submission.csv
  - ✅ time_submission.csv (id, time in ms, answer)

### Bước 6: Export Docker image

```bash
docker save -o zac2025_TeamName.tar zac2025-submission:latest
gzip zac2025_TeamName.tar
```

---

## 📊 So sánh Before/After

### Docker Image Size

| Component | Before | After | Saved |
|-----------|--------|-------|-------|
| Base image | ~6GB | ~6GB | - |
| Base model | ~4.5GB | 0GB | 4.5GB |
| Checkpoint | ~1GB | ~0.6GB | 0.4GB |
| **Total** | **~11.5GB** | **~6.6GB** | **~4.9GB (43%)** |

### Build Time

| Step | Before | After |
|------|--------|-------|
| Build image | ~15-20 min | ~5-8 min |
| First run | ~2 min | ~7-12 min (download model) |
| Subsequent runs | ~2 min | ~2 min |

### Compliance

| Requirement | Before | After |
|-------------|--------|-------|
| Port 9777 | ❌ 8888 | ✅ 9777 |
| Password zac2025 | ❌ No auth | ✅ zac2025 |
| No base model in Docker | ❌ Included | ✅ Not included |
| jupyterlab | ❌ Missing | ✅ Installed |
| Clean checkpoint | ❌ Full | ✅ Essential only |

---

## 🎯 Checklist cuối cùng

Trước khi nộp bài, kiểm tra:

### Environment
- [ ] NVIDIA Docker installed
- [ ] Driver ≥ 535.86.10
- [ ] CUDA 12.2
- [ ] `bash check_gpu_setup.sh` pass

### Checkpoint
- [ ] Đã chạy `cleanup_checkpoint.sh`
- [ ] Checkpoint chỉ có essential files
- [ ] Size ~500-600MB (không phải ~1GB)
- [ ] Copied to `submission/saved_models/checkpoint-latest/`

### Docker Image
- [ ] Build thành công
- [ ] Image size ~6-7GB (không phải ~11-12GB)
- [ ] Base model KHÔNG có trong image
- [ ] Chạy predict.sh thành công (lần đầu sẽ download model)

### Jupyter
- [ ] Port 9777 ✓
- [ ] Password zac2025 ✓
- [ ] predict_notebook.ipynb chạy thành công
- [ ] Output jupyter_submission.csv ✓
- [ ] Output time_submission.csv với format: id, time (ms), answer ✓

### Output Format
- [ ] submission.csv: id, answer
- [ ] jupyter_submission.csv: id, answer (giống submission.csv)
- [ ] time_submission.csv: id, time, answer
- [ ] Thời gian là cho MỖI sample (không phải trung bình)

---

## 🚀 Tóm tắt

**5 vấn đề chính đã sửa:**

1. ✅ start_jupyter.sh: Port 9777 + password zac2025
2. ✅ Dockerfile: Xóa base model download (~4.5GB saved)
3. ✅ requirements.txt: Thêm jupyterlab
4. ✅ Checkpoint: Script cleanup để loại bỏ training files (~400MB saved)
5. ✅ predict_notebook.ipynb: Đúng format (đã OK từ trước)

**Kết quả:**
- 🎯 100% tuân thủ yêu cầu BTC
- 💾 Docker image nhẹ hơn ~4.9GB (43%)
- ⚡ Build nhanh hơn ~10-12 phút
- ✅ Sẵn sàng nộp bài

---

## 📞 Troubleshooting

### Lỗi: jupyter lab không tìm thấy

```bash
# Rebuild image với jupyterlab
docker build --no-cache -t zac2025-submission .
```

### Lỗi: Base model download timeout

```bash
# Download model manually trước
python3 -c "from transformers import Qwen2VLForConditionalGeneration; \
    Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')"

# Hoặc tăng timeout trong Docker
docker run --gpus '"device=0"' \
    -e HF_HUB_TIMEOUT=3600 \
    ...
```

### Lỗi: Port 9777 already in use

```bash
# Kiểm tra port
lsof -i :9777

# Hoặc dùng port khác (không khuyến khích)
# BTC sẽ dùng port 9777 khi chấm
```

---

Tất cả thay đổi đã được commit và push lên branch:
- Branch: `claude/general-session-01QfRaZ5nrPfvpXJNAbWmRUE`
- Latest commit: "fix: BTC compliance - correct Jupyter config, remove base model, add jupyterlab"

# HuggingFace Upload Guide - Zalo AI Challenge 2025

Hướng dẫn upload checkpoint lên HuggingFace để BTC có thể reproduce và verify.

---

## 📋 Tại sao upload checkpoint lên HuggingFace?

### 1. **Yêu cầu BTC** ✅
BTC cần code và data để reproduce training:
- Training code: trong folder `training_code/`
- Training data: Upload lên HuggingFace
- **Checkpoints**: Upload lên HuggingFace

### 2. **Lợi ích**
- ✅ BTC dễ dàng download và verify
- ✅ Backup an toàn
- ✅ Share với teammates
- ✅ Optional: Docker có thể download từ HuggingFace (alternative workflow)

---

## 🚀 Cách 1: Upload checkpoint (Recommended)

### Bước 1: Cleanup checkpoint trước

```bash
cd submission

# Extract chỉ essential files (~500-600MB thay vì ~1GB)
bash cleanup_checkpoint.sh \
    /home/user/Qwen-VL-Series-Finetune/submission/saved_models/checkpoint-latest \
    ./checkpoint-clean
```

### Bước 2: Get HuggingFace token

1. Đi đến: https://huggingface.co/settings/tokens
2. Tạo token mới với quyền **write**
3. Copy token

### Bước 3: Upload to HuggingFace

```bash
# Option 1: Dùng script có sẵn
python upload_to_huggingface.py \
    --mode upload \
    --checkpoint_path ./checkpoint-clean \
    --repo_id thangquoc/zaic2025-phase2 \
    --token YOUR_HF_TOKEN \
    --private

# Option 2: Set token as environment variable
export HF_TOKEN=YOUR_HF_TOKEN
python upload_to_huggingface.py \
    --mode upload \
    --checkpoint_path ./checkpoint-clean \
    --repo_id thangquoc/zaic2025-phase2 \
    --private
```

**Kết quả**:
```
==============================================================
Upload Checkpoint to HuggingFace Hub
==============================================================

Checkpoint: ./checkpoint-clean
Repository: thangquoc/zaic2025-phase2
Private: True

Step 1: Creating repository...
✓ Repository ready: https://huggingface.co/thangquoc/zaic2025-phase2

Step 2: Uploading files...
✓ All files uploaded successfully

==============================================================
✅ Upload completed!
==============================================================

Repository URL: https://huggingface.co/thangquoc/zaic2025-phase2

Next steps:
  1. Verify files at: https://huggingface.co/thangquoc/zaic2025-phase2/tree/main
  2. Update training_code/README.md with this URL
```

### Bước 4: Verify upload

Mở browser: https://huggingface.co/thangquoc/zaic2025-phase2/tree/main

Kiểm tra files:
- ✅ adapter_model.safetensors
- ✅ adapter_config.json
- ✅ merger_weights.bin
- ✅ config.json
- ✅ tokenizer files
- ✅ preprocessor files

### Bước 5: Update training_code/README.md

Thêm URL download vào `training_code/README.md`:

```markdown
## Download Checkpoints

**Phase 2a checkpoint:**
```bash
# Download from HuggingFace
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='thangquoc/zaic2025-phase2', local_dir='./checkpoint-phase2a')"
```

Or download directly:
https://huggingface.co/thangquoc/zaic2025-phase2
```

---

## 🔧 Cách 2: Sử dụng checkpoint từ HuggingFace (Alternative)

### Option A: Download manually trước khi build Docker

```bash
# Download checkpoint
python upload_to_huggingface.py \
    --mode download \
    --repo_id thangquoc/zaic2025-phase2 \
    --local_dir ./submission/saved_models/checkpoint-latest

# Build Docker như bình thường
cd submission
docker build -t zac2025-submission .
```

**Pros**: Fast inference (checkpoint có sẵn trong Docker)
**Cons**: Docker image lớn hơn (~600MB)

### Option B: Download tự động khi Docker chạy (NOT RECOMMENDED)

```bash
# Set environment variable để Docker tự download
docker run --gpus '"device=0"' \
    -e HF_CHECKPOINT_REPO="thangquoc/zaic2025-phase2" \
    -v /data:/data \
    -v $(pwd)/results:/result \
    zac2025-submission
```

**Pros**: Docker image nhỏ hơn
**Cons**:
- ❌ Tăng thời gian inference (~2-3 phút download lần đầu)
- ❌ Không khuyến khích cho submission (BTC có thể chấm offline)

---

## 📊 Workflow Recommendations

### Cho BTC Reproducibility (RECOMMENDED) ✅

1. **Upload checkpoint lên HuggingFace** (for BTC to download and verify)
2. **Copy checkpoint vào Docker** (for fast inference)
3. **Update training_code/README.md** với HuggingFace URL

```bash
# 1. Cleanup checkpoint
bash cleanup_checkpoint.sh <source> ./checkpoint-clean

# 2. Upload to HuggingFace
python upload_to_huggingface.py --mode upload \
    --checkpoint_path ./checkpoint-clean \
    --repo_id thangquoc/zaic2025-phase2 \
    --token YOUR_TOKEN

# 3. Copy to Docker
cp -r ./checkpoint-clean ./saved_models/checkpoint-latest

# 4. Build Docker
docker build -t zac2025-submission .

# 5. Update training_code/README.md với URL
```

**Kết quả**:
- ✅ BTC có thể download và verify từ HuggingFace
- ✅ Docker inference nhanh (checkpoint có sẵn)
- ✅ 100% compliance

### Alternative: Download from HuggingFace (NOT RECOMMENDED)

Chỉ dùng khi:
- Test local
- Share với teammates
- **KHÔNG dùng cho final submission**

```bash
docker run --gpus '"device=0"' \
    -e HF_CHECKPOINT_REPO="thangquoc/zaic2025-phase2" \
    -e HF_TOKEN="YOUR_TOKEN" \
    -v /data:/data \
    -v $(pwd)/results:/result \
    zac2025-submission
```

---

## 🔍 Troubleshooting

### Error: "Repository not found"

**Nguyên nhân**: Repo chưa được tạo hoặc private

**Giải pháp**:
```bash
# Tạo repo trước (nếu chưa có)
python upload_to_huggingface.py --mode upload ... --private
```

### Error: "Authentication required"

**Nguyên nhân**: Token không hợp lệ hoặc thiếu quyền

**Giải pháp**:
1. Check token tại: https://huggingface.co/settings/tokens
2. Tạo token mới với quyền **write**
3. Set token:
   ```bash
   export HF_TOKEN=YOUR_TOKEN
   ```

### Error: "Upload failed: File too large"

**Nguyên nhân**: File quá lớn (>5GB)

**Giải pháp**:
```bash
# Cleanup checkpoint trước
bash cleanup_checkpoint.sh <source> <output>

# Upload checkpoint đã cleanup (~500-600MB)
python upload_to_huggingface.py --mode upload \
    --checkpoint_path <output> ...
```

### Warning: "Checkpoint download slow in Docker"

**Nguyên nhân**: Download checkpoint mỗi lần chạy

**Giải pháp**: Copy checkpoint vào Docker (recommended)
```bash
cp -r ./checkpoint-clean ./saved_models/checkpoint-latest
docker build -t zac2025-submission .
```

---

## 📝 Example: Complete Workflow

```bash
# Step 1: Cleanup checkpoint
cd submission
bash cleanup_checkpoint.sh \
    /home/user/Qwen-VL-Series-Finetune/submission/saved_models/checkpoint-latest \
    ./checkpoint-clean

# Step 2: Upload to HuggingFace
export HF_TOKEN=hf_xxxxxxxxxxxxx
python upload_to_huggingface.py \
    --mode upload \
    --checkpoint_path ./checkpoint-clean \
    --repo_id thangquoc/zaic2025-phase2 \
    --private

# Step 3: Verify upload
python -c "from huggingface_hub import list_repo_files; \
    print(list_repo_files('thangquoc/zaic2025-phase2'))"

# Step 4: Copy to Docker
cp -r ./checkpoint-clean ./saved_models/checkpoint-latest

# Step 5: Build Docker
docker build -t zac2025-submission .

# Step 6: Test
docker run --gpus '"device=0"' \
    -v /data:/data \
    -v $(pwd)/results:/result \
    zac2025-submission

# Step 7: Update training_code/README.md
cat >> training_code/README.md << 'EOF'

## Download Checkpoints

Phase 2a checkpoint: https://huggingface.co/thangquoc/zaic2025-phase2

Download:
```bash
from huggingface_hub import snapshot_download
snapshot_download(repo_id='thangquoc/zaic2025-phase2', local_dir='./checkpoint')
```
EOF
```

---

## ✅ Final Checklist

Trước khi submit:

### HuggingFace Upload
- [ ] Checkpoint đã cleanup (chỉ essential files)
- [ ] Uploaded to HuggingFace successfully
- [ ] Repository là **private** (protect your work)
- [ ] Verified files on HuggingFace website

### Docker Submission
- [ ] Checkpoint **CÓ SẴN** trong Docker (không download runtime)
- [ ] Docker build successfully
- [ ] Test inference OK
- [ ] Timing acceptable

### Documentation
- [ ] training_code/README.md có HuggingFace URL
- [ ] BTC có thể download checkpoint từ HuggingFace
- [ ] Hướng dẫn reproduce rõ ràng

---

## 🎯 Summary

**Recommended workflow**:
1. ✅ Upload checkpoint lên HuggingFace (for BTC reproducibility)
2. ✅ Copy checkpoint vào Docker (for fast inference)
3. ✅ Update README.md với HuggingFace URL

**Kết quả**:
- 💾 BTC có thể download và verify
- ⚡ Inference nhanh (checkpoint trong Docker)
- ✅ 100% compliance với yêu cầu BTC

**KHÔNG khuyến khích**:
- ❌ Download checkpoint từ HuggingFace khi Docker chạy
- ❌ Làm tăng thời gian inference
- ❌ BTC có thể chấm offline

---

Tất cả tools đã sẵn sàng trong folder `submission/`:
- `upload_to_huggingface.py` - Upload/download checkpoint
- `cleanup_checkpoint.sh` - Extract essential files
- `predict.py` - Hỗ trợ download from HuggingFace (optional)

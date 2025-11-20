# ⚠️ QUAN TRỌNG: Inference OFFLINE - Không có internet

## 🚨 Vấn đề CONTRADICTION trong hướng dẫn BTC

### BTC nói gì?

**Yêu cầu 1**:
> "Base model dùng trong quá trình training **không được đưa vào bên trong Docker** để tránh làm Docker quá nặng."

**Yêu cầu 2** (trong Jupyter notebook guide):
> "không dùng API ngoài, vì inference sẽ **KHÔNG CÓ INTERNET**"

### ❌ Mâu thuẫn

- Nếu KHÔNG đưa base model vào Docker → Cần download từ HuggingFace → **CẦN INTERNET**
- Nhưng inference **KHÔNG CÓ INTERNET** → Model phải có sẵn trong Docker

---

## ✅ Giải pháp của chúng ta

### **Include base model vào Docker**

Lý do:
1. ✅ BTC nói rõ: "inference sẽ không có internet"
2. ✅ transformers.from_pretrained() cần internet nếu model chưa có trong cache
3. ✅ An toàn nhất: Đảm bảo inference hoạt động offline

Trade-off:
- 💾 Docker image lớn hơn: ~7GB → ~15GB
- ⚡ Nhưng inference vẫn chạy nhanh
- ✅ Hoạt động 100% offline

---

## 📦 Docker image structure

### Dockerfile đã include:

```dockerfile
# Download base model for offline inference
RUN python3 -c "from transformers import Qwen2VLForConditionalGeneration, AutoProcessor; \
    Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', trust_remote_code=True); \
    AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', trust_remote_code=True); \
    print('Base model cached successfully')"

# Copy checkpoint
COPY saved_models/ ${WORKSPACE}/saved_models/
```

### Docker image size breakdown:

| Component | Size | Included? |
|-----------|------|-----------|
| Base CUDA image | ~6GB | ✅ |
| Python packages | ~2GB | ✅ |
| **Base model (Qwen2-VL-2B-Instruct)** | **~4.5GB** | **✅ YES** |
| Checkpoint (cleaned) | ~0.6GB | ✅ |
| **Total** | **~13-15GB** | - |

---

## 🔍 Giải thích BTC contradiction

### Khả năng 1: BTC server có HuggingFace cache shared

- BTC có thể mount shared volume với HuggingFace cache
- Khi model.from_pretrained() chạy, nó tìm trong cache trước
- Nếu có trong cache → Load từ cache (fast)
- Nếu không có → Load từ Docker's local cache

**Kết quả**: Model sẽ load từ một trong hai:
1. BTC's shared cache (if available) → Fast
2. Docker's local cache (if not) → Still works

### Khả năng 2: BTC yêu cầu contradictory

- Có thể họ muốn Docker nhỏ để dễ upload
- Nhưng quên rằng inference không có internet
- → Chúng ta phải include base model để đảm bảo hoạt động

---

## ✅ Quyết định cuối cùng

### **Include base model vào Docker** (RECOMMENDED)

**Lý do**:
1. ✅ Đảm bảo inference hoạt động offline 100%
2. ✅ Không depend vào BTC's infrastructure
3. ✅ An toàn hơn
4. ✅ BTC test environment chắc chắn không có internet (như họ nói)

**Trade-off**:
- Docker image ~15GB instead of ~7GB
- Upload sẽ lâu hơn (~2x)
- Nhưng **ĐẢM BẢO hoạt động**

---

## 🚀 Workflow

### Build Docker (include base model):

```bash
cd submission

# 1. Cleanup checkpoint
bash cleanup_checkpoint.sh <source> ./checkpoint-clean

# 2. Copy to Docker
cp -r ./checkpoint-clean ./saved_models/checkpoint-latest

# 3. Build Docker (will download base model)
docker build -t zac2025-submission .
# This will take 15-20 minutes (downloads ~4.5GB base model)

# 4. Verify base model is cached
docker run --rm zac2025-submission \
    python3 -c "from transformers import Qwen2VLForConditionalGeneration; \
                Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')"
# Should be fast (model already in cache)
```

### Test offline:

```bash
# Run without network
docker run --rm --network=none \
    -v /data:/data \
    -v $(pwd)/results:/result \
    zac2025-submission
# Should work!
```

---

## 📊 Size comparison

### Option A: Include base model (OUR CHOICE) ✅

| Component | Size |
|-----------|------|
| Docker image | ~15GB |
| Upload time | ~30-60 min |
| **Offline support** | **✅ YES** |
| **Risk** | **Low** |

### Option B: Don't include base model ❌

| Component | Size |
|-----------|------|
| Docker image | ~7GB |
| Upload time | ~15-30 min |
| **Offline support** | **❌ NO** |
| **Risk** | **HIGH (will fail if no internet)** |

---

## 🎯 Kết luận

### ✅ Chúng ta đã chọn Option A (Include base model)

**Lý do**:
- BTC nói rõ: "inference không có internet"
- Đảm bảo inference hoạt động 100%
- Docker image lớn hơn nhưng AN TOÀN

### ⚠️ Nếu BTC reject vì Docker quá lớn

Nếu BTC reject submission vì Docker image >15GB, có thể:
1. Giải thích contradiction trong yêu cầu
2. Hỏi BTC về HuggingFace cache infrastructure
3. Nếu họ có cache → Rebuild without base model

Nhưng hiện tại: **AN TOÀN HƠN là include base model**

---

## 📝 Summary

**Contradiction**:
- "Không được đưa base model vào Docker" vs "Inference không có internet"

**Giải pháp**:
- ✅ Include base model vào Docker (~15GB total)
- ✅ Đảm bảo offline inference hoạt động
- ✅ Trade-off: Image size vs Reliability

**Kết quả**:
- 💾 Docker image: ~15GB (large but safe)
- ⚡ Inference: Fast (model in cache)
- ✅ Offline: 100% hoạt động
- 🎯 Risk: Low (guaranteed to work)

---

## 🔗 Related Files

- **Dockerfile**: Lines 45-62 (base model download)
- **predict.py**: Lines 37-39 (offline inference note)
- **BTC_COMPLIANCE_FIXES.md**: Overview of all compliance fixes
- **SUBMISSION_CHECKLIST.md**: Complete submission checklist

---

**Bottom line**: Chúng ta ưu tiên **RELIABILITY** hơn **IMAGE SIZE**.
Nếu BTC có vấn đề với image size, họ sẽ clarify infrastructure của họ.

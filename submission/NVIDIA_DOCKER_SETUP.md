# NVIDIA Docker Setup Guide

## Yêu cầu môi trường BTC

Server của BTC sử dụng:
- **Driver Version**: 535.86.10
- **CUDA Version**: 12.2

## Cài đặt NVIDIA Docker

### Prerequisites

1. **NVIDIA GPU** với driver tương thích
2. **Docker Engine** đã được cài đặt
3. **NVIDIA GPU driver** phiên bản ≥ 535.86.10

### Bước 1: Cài đặt NVIDIA Container Toolkit

#### Ubuntu/Debian

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list
sudo apt-get update

# Install nvidia-docker2
sudo apt-get install -y nvidia-docker2

# Restart Docker daemon
sudo systemctl restart docker
```

#### CentOS/RHEL

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.repo | \
    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install nvidia-docker2
sudo yum install -y nvidia-docker2

# Restart Docker daemon
sudo systemctl restart docker
```

### Bước 2: Kiểm tra cài đặt

```bash
# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

**Kết quả mong đợi:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.10              Driver Version: 535.86.10      CUDA Version: 12.2     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
...
```

### Bước 3: Xác nhận CUDA version

```bash
# Check CUDA version
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvcc --version
```

**Kết quả mong đợi:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
```

---

## Xác nhận môi trường phù hợp với BTC

### 1. Kiểm tra NVIDIA Driver

```bash
nvidia-smi
```

**Yêu cầu**: Driver version ≥ 535.86.10

### 2. Kiểm tra CUDA version trong Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 \
    bash -c "nvidia-smi && nvcc --version"
```

**Yêu cầu**: CUDA 12.2

### 3. Test với submission Docker image

```bash
# Build submission image
cd submission
docker build -t zac2025-submission .

# Test GPU access
docker run --rm --gpus all zac2025-submission nvidia-smi
```

---

## Dockerfile Configuration

Submission package đã được cấu hình để match với môi trường BTC:

```dockerfile
# Base image with CUDA 12.2 (matching BTC server)
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04
```

### Các thư viện quan trọng

```
torch==2.1.2+cu121  # PyTorch with CUDA 12.1 support
transformers==4.47.1
```

**Lưu ý**: PyTorch 2.1.2 với CUDA 12.1 **tương thích** với CUDA 12.2 runtime. NVIDIA CUDA có backward compatibility.

---

## Chạy Docker với GPU

### Cú pháp cơ bản

```bash
docker run --gpus all [IMAGE_NAME]
```

### Chạy với volume mapping

```bash
docker run --gpus all \
    -v /path/to/test_data:/data \
    -v /path/to/results:/result \
    [IMAGE_NAME]
```

### Chạy interactive mode (debug)

```bash
docker run --gpus all -it [IMAGE_NAME] /bin/bash
```

### Giới hạn GPU cụ thể

```bash
# Use GPU 0 only
docker run --gpus '"device=0"' [IMAGE_NAME]

# Use GPU 0 and 1
docker run --gpus '"device=0,1"' [IMAGE_NAME]
```

---

## Troubleshooting

### Error: "could not select device driver"

**Nguyên nhân**: nvidia-docker2 chưa được cài đặt hoặc Docker daemon chưa restart

**Giải pháp**:
```bash
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Error: "Failed to initialize NVML: Driver/library version mismatch"

**Nguyên nhân**: NVIDIA driver và kernel module không khớp

**Giải pháp**:
```bash
# Reboot system
sudo reboot

# Or reload kernel module
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
```

### Error: "CUDA version mismatch"

**Nguyên nhân**: Docker image CUDA version không tương thích với host driver

**Giải pháp**:
- Đảm bảo driver ≥ 535.86.10 (hỗ trợ CUDA 12.2)
- Update driver nếu cần:
  ```bash
  # Ubuntu
  sudo ubuntu-drivers autoinstall
  sudo reboot
  ```

### Warning: "Using deprecated NumPy API"

**Không quan trọng**: Đây là warning từ thư viện, không ảnh hưởng inference.

---

## Kiểm tra trước khi submit

### Checklist

- [ ] NVIDIA driver version ≥ 535.86.10
- [ ] nvidia-docker2 installed
- [ ] Docker có thể chạy với `--gpus all`
- [ ] Dockerfile sử dụng `nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04`
- [ ] Test inference thành công với GPU

### Script kiểm tra nhanh

```bash
#!/bin/bash
echo "=== Checking NVIDIA Docker Setup ==="
echo ""

# Check 1: NVIDIA driver
echo "1. NVIDIA Driver:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo ""

# Check 2: CUDA version
echo "2. CUDA Version:"
nvidia-smi --query-gpu=cuda_version --format=csv,noheader
echo ""

# Check 3: Docker + GPU
echo "3. Docker GPU access:"
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Docker can access GPU"
else
    echo "❌ Docker cannot access GPU - install nvidia-docker2"
fi
echo ""

# Check 4: Submission image
echo "4. Submission image:"
if docker images | grep -q "zac2025-submission"; then
    echo "✅ Submission image built"
    docker run --rm --gpus all zac2025-submission nvidia-smi > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Submission image can access GPU"
    else
        echo "❌ Submission image cannot access GPU"
    fi
else
    echo "⚠️  Submission image not built yet"
fi

echo ""
echo "=== Setup Check Complete ==="
```

Lưu script này thành `check_gpu_setup.sh` và chạy:

```bash
chmod +x check_gpu_setup.sh
./check_gpu_setup.sh
```

---

## Resources

- **NVIDIA Docker GitHub**: https://github.com/NVIDIA/nvidia-docker
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- **CUDA Compatibility**: https://docs.nvidia.com/deploy/cuda-compatibility/

---

## Liên hệ

Nếu gặp vấn đề với NVIDIA Docker setup, tham khảo:
1. Official docs: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
2. GitHub issues: https://github.com/NVIDIA/nvidia-docker/issues
3. NVIDIA Developer Forums: https://forums.developer.nvidia.com/

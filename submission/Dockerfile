# Dockerfile for Zalo AI Challenge 2025 - RoadBuddy
#
# IMPORTANT: This Dockerfile matches BTC server environment:
# - Driver Version: 535.86.10
# - CUDA Version: 12.2
#
# Prerequisites:
# - NVIDIA Docker (nvidia-docker2) must be installed on host
# - See NVIDIA_DOCKER_SETUP.md for installation instructions
# - Run check_gpu_setup.sh to verify your environment
#
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV WORKSPACE=/workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create workspace
WORKDIR ${WORKSPACE}

# Copy requirements first (for better caching)
COPY requirements.txt ${WORKSPACE}/
RUN pip3 install --no-cache-dir -r requirements.txt

# IMPORTANT: Base model handling for OFFLINE inference
# BTC documentation states: "không dùng API ngoài, vì inference sẽ không có internet"
#
# CONTRADICTION:
# - BTC says: "Base model không được đưa vào Docker để tránh quá nặng"
# - But also: "Inference sẽ KHÔNG CÓ INTERNET"
#
# SOLUTION: Download base model to Docker (required for offline inference)
# Docker image will be larger (~15GB total) but will work offline
#
# If BTC server has HuggingFace cache mounted, model will load from cache
# If not, model will load from Docker's local cache
#
RUN python3 -c "from transformers import Qwen2VLForConditionalGeneration, AutoProcessor; \
    print('Downloading base model for offline inference...'); \
    Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', trust_remote_code=True); \
    AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', trust_remote_code=True); \
    print('Base model cached successfully')"

# Copy inference scripts
COPY predict.py ${WORKSPACE}/
COPY predict.sh ${WORKSPACE}/
COPY predict_notebook.ipynb ${WORKSPACE}/

# Copy model checkpoint (LoRA + Merger weights)
# Note: saved_models/ should contain your Phase 2a checkpoint
COPY saved_models/ ${WORKSPACE}/saved_models/

# Make scripts executable
RUN chmod +x ${WORKSPACE}/predict.sh

# Create directories for input/output
RUN mkdir -p /data /result

# Set working directory
WORKDIR ${WORKSPACE}

# Default command (run inference)
CMD ["/bin/bash", "./predict.sh"]

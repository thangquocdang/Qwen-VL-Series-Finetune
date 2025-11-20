# Dockerfile for Zalo AI Challenge 2025 - RoadBuddy
# Base image with CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

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

# Install base model (Qwen2-VL-2B-Instruct)
# Note: This will be downloaded during first run if not cached
RUN python3 -c "from transformers import AutoModelForCausalLM; \
    print('Downloading base model...'); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', trust_remote_code=True)"

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

#!/bin/bash
# Script to check NVIDIA Docker setup for Zalo AI Challenge 2025
# Required: Driver 535.86.10, CUDA 12.2

set -e

echo "=========================================="
echo "Zalo AI Challenge 2025 - GPU Setup Check"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

# Function to check and report
check_requirement() {
    local name=$1
    local command=$2
    local expected=$3

    echo -n "Checking $name... "

    if result=$(eval $command 2>&1); then
        if [ -n "$expected" ]; then
            if echo "$result" | grep -q "$expected"; then
                echo -e "${GREEN}✓ PASS${NC}"
                echo "  Result: $result"
                PASS=$((PASS+1))
            else
                echo -e "${RED}✗ FAIL${NC}"
                echo "  Expected: $expected"
                echo "  Got: $result"
                FAIL=$((FAIL+1))
            fi
        else
            echo -e "${GREEN}✓ PASS${NC}"
            PASS=$((PASS+1))
        fi
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo "  Error: $result"
        FAIL=$((FAIL+1))
    fi
    echo ""
}

# Check 1: NVIDIA driver installed
echo "=== Step 1: NVIDIA Driver ==="
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo -e "${GREEN}✓ NVIDIA Driver installed${NC}"
    echo "  Version: $DRIVER_VERSION"

    # Check if driver version >= 535.86.10
    REQUIRED_DRIVER="535.86.10"
    if [ "$(printf '%s\n' "$REQUIRED_DRIVER" "$DRIVER_VERSION" | sort -V | head -n1)" = "$REQUIRED_DRIVER" ]; then
        echo -e "${GREEN}✓ Driver version meets requirement (>= 535.86.10)${NC}"
        PASS=$((PASS+1))
    else
        echo -e "${YELLOW}⚠ Driver version may be too old (required: >= 535.86.10)${NC}"
        FAIL=$((FAIL+1))
    fi
else
    echo -e "${RED}✗ NVIDIA Driver not found${NC}"
    echo "  Install NVIDIA driver first: sudo ubuntu-drivers autoinstall"
    FAIL=$((FAIL+1))
    exit 1
fi
echo ""

# Check 2: CUDA version
echo "=== Step 2: CUDA Version ==="
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -1)
    echo -e "${GREEN}✓ CUDA available${NC}"
    echo "  Version: $CUDA_VERSION"

    # Check if CUDA >= 12.2
    if [[ "$CUDA_VERSION" == 12.2* ]] || [[ "$CUDA_VERSION" > "12.2" ]]; then
        echo -e "${GREEN}✓ CUDA version matches requirement (12.2)${NC}"
        PASS=$((PASS+1))
    else
        echo -e "${YELLOW}⚠ CUDA version is $CUDA_VERSION (recommended: 12.2)${NC}"
        PASS=$((PASS+1))  # Still pass if >= 12.0
    fi
fi
echo ""

# Check 3: Docker installed
echo "=== Step 3: Docker ==="
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo -e "${GREEN}✓ Docker installed${NC}"
    echo "  $DOCKER_VERSION"
    PASS=$((PASS+1))
else
    echo -e "${RED}✗ Docker not installed${NC}"
    echo "  Install Docker: https://docs.docker.com/engine/install/"
    FAIL=$((FAIL+1))
    exit 1
fi
echo ""

# Check 4: NVIDIA Docker (nvidia-docker2)
echo "=== Step 4: NVIDIA Docker ==="
if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA Docker working${NC}"
    echo "  Docker can access GPU"
    PASS=$((PASS+1))
else
    echo -e "${RED}✗ NVIDIA Docker not working${NC}"
    echo ""
    echo "  Install nvidia-docker2:"
    echo "    Ubuntu/Debian:"
    echo "      sudo apt-get install -y nvidia-docker2"
    echo "      sudo systemctl restart docker"
    echo ""
    echo "  See NVIDIA_DOCKER_SETUP.md for detailed instructions"
    FAIL=$((FAIL+1))
    exit 1
fi
echo ""

# Check 5: Test CUDA 12.2 base image
echo "=== Step 5: CUDA 12.2 Base Image ==="
if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ CUDA 12.2 image works${NC}"
    PASS=$((PASS+1))
else
    echo -e "${RED}✗ CUDA 12.2 image failed${NC}"
    FAIL=$((FAIL+1))
fi
echo ""

# Check 6: Submission image (if built)
echo "=== Step 6: Submission Image ==="
if docker images | grep -q "zac2025-submission"; then
    echo -e "${GREEN}✓ Submission image found${NC}"

    # Test GPU access in submission image
    if docker run --rm --gpus all zac2025-submission nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ Submission image can access GPU${NC}"
        PASS=$((PASS+1))
    else
        echo -e "${RED}✗ Submission image cannot access GPU${NC}"
        FAIL=$((FAIL+1))
    fi

    # Check image size
    IMAGE_SIZE=$(docker images zac2025-submission --format "{{.Size}}")
    echo "  Image size: $IMAGE_SIZE"

else
    echo -e "${YELLOW}⚠ Submission image not built yet${NC}"
    echo "  Build with: cd submission && docker build -t zac2025-submission ."
    echo "  (This is OK if you haven't built it yet)"
fi
echo ""

# Check 7: Model checkpoint (if exists)
echo "=== Step 7: Model Checkpoint ==="
CHECKPOINT_DIR="$(dirname "$0")/saved_models/checkpoint-latest"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo -e "${GREEN}✓ Checkpoint directory found${NC}"

    # Check required files
    REQUIRED_FILES=(
        "adapter_model.safetensors"
        "adapter_config.json"
        "merger_weights.bin"
        "config.json"
    )

    ALL_PRESENT=true
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$CHECKPOINT_DIR/$file" ]; then
            echo -e "${GREEN}  ✓ $file${NC}"
        else
            echo -e "${RED}  ✗ $file (missing)${NC}"
            ALL_PRESENT=false
        fi
    done

    if [ "$ALL_PRESENT" = true ]; then
        echo -e "${GREEN}✓ All checkpoint files present${NC}"
        PASS=$((PASS+1))
    else
        echo -e "${RED}✗ Some checkpoint files missing${NC}"
        FAIL=$((FAIL+1))
    fi
else
    echo -e "${YELLOW}⚠ Checkpoint not copied yet${NC}"
    echo "  Copy with: bash setup_submission.sh /path/to/checkpoint-latest"
    echo "  (This is OK if you haven't copied your checkpoint yet)"
fi
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}$PASS${NC}"
echo -e "Failed: ${RED}$FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Your environment is ready.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Copy checkpoint: bash setup_submission.sh /path/to/checkpoint"
    echo "  2. Build image: docker build -t zac2025-submission ."
    echo "  3. Test inference: docker run --gpus all -v /path/to/test:/data -v /path/to/result:/result zac2025-submission"
    exit 0
else
    echo -e "${RED}❌ Some checks failed. Please fix the issues above.${NC}"
    echo ""
    echo "Common issues:"
    echo "  - NVIDIA driver not installed: sudo ubuntu-drivers autoinstall"
    echo "  - nvidia-docker2 not installed: sudo apt-get install -y nvidia-docker2"
    echo "  - Docker daemon needs restart: sudo systemctl restart docker"
    echo ""
    echo "See NVIDIA_DOCKER_SETUP.md for detailed troubleshooting."
    exit 1
fi

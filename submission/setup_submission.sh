#!/bin/bash
# Setup script for Zalo AI Challenge 2025 submission package
# This script helps you prepare the submission by copying your checkpoint

set -e  # Exit on error

echo "============================================"
echo "Zalo AI Challenge 2025 - Submission Setup"
echo "============================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Checkpoint path not provided${NC}"
    echo ""
    echo "Usage:"
    echo "  bash setup_submission.sh /path/to/checkpoint-latest"
    echo ""
    echo "Example:"
    echo "  bash setup_submission.sh /kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest"
    echo ""
    exit 1
fi

CHECKPOINT_PATH="$1"
SUBMISSION_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$SUBMISSION_DIR/saved_models"

echo "Configuration:"
echo "  Checkpoint source: $CHECKPOINT_PATH"
echo "  Submission dir: $SUBMISSION_DIR"
echo "  Target dir: $TARGET_DIR"
echo ""

# Step 1: Verify checkpoint exists
echo -e "${YELLOW}Step 1: Verifying checkpoint...${NC}"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}Error: Checkpoint directory not found: $CHECKPOINT_PATH${NC}"
    exit 1
fi

# Check required files
REQUIRED_FILES=(
    "adapter_model.safetensors"
    "adapter_config.json"
    "merger_weights.bin"
    "config.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$CHECKPOINT_PATH/$file" ]; then
        echo -e "${RED}Error: Required file not found: $file${NC}"
        echo "Your checkpoint may be incomplete or corrupted."
        exit 1
    fi
done

echo -e "${GREEN}✓ Checkpoint verified${NC}"
echo ""

# Step 2: Create target directory
echo -e "${YELLOW}Step 2: Creating target directory...${NC}"
mkdir -p "$TARGET_DIR"
echo -e "${GREEN}✓ Target directory ready${NC}"
echo ""

# Step 3: Copy checkpoint
echo -e "${YELLOW}Step 3: Copying checkpoint (this may take a few minutes)...${NC}"
echo "  Source: $CHECKPOINT_PATH"
echo "  Target: $TARGET_DIR/checkpoint-latest"

# Remove existing checkpoint if present
if [ -d "$TARGET_DIR/checkpoint-latest" ]; then
    echo "  Removing existing checkpoint..."
    rm -rf "$TARGET_DIR/checkpoint-latest"
fi

# Copy checkpoint
cp -r "$CHECKPOINT_PATH" "$TARGET_DIR/checkpoint-latest"

echo -e "${GREEN}✓ Checkpoint copied${NC}"
echo ""

# Step 4: Verify copied files
echo -e "${YELLOW}Step 4: Verifying copied files...${NC}"
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$TARGET_DIR/checkpoint-latest/$file" ]; then
        echo -e "${RED}Error: File not copied correctly: $file${NC}"
        exit 1
    fi
done

# Show file sizes
echo "Checkpoint files:"
ls -lh "$TARGET_DIR/checkpoint-latest/" | grep -E "(adapter_model|merger_weights|config.json)"

echo -e "${GREEN}✓ All files copied successfully${NC}"
echo ""

# Step 5: Calculate total size
echo -e "${YELLOW}Step 5: Checking total size...${NC}"
CHECKPOINT_SIZE=$(du -sh "$TARGET_DIR/checkpoint-latest" | cut -f1)
echo "  Checkpoint size: $CHECKPOINT_SIZE"
echo ""

# Step 6: Verify submission structure
echo -e "${YELLOW}Step 6: Verifying submission structure...${NC}"

REQUIRED_SUBMISSION_FILES=(
    "predict.py"
    "predict.sh"
    "predict_notebook.ipynb"
    "requirements.txt"
    "Dockerfile"
    "start_jupyter.sh"
    "training_code/README.md"
)

ALL_PRESENT=true
for file in "${REQUIRED_SUBMISSION_FILES[@]}"; do
    if [ ! -f "$SUBMISSION_DIR/$file" ]; then
        echo -e "${RED}✗ Missing: $file${NC}"
        ALL_PRESENT=false
    else
        echo -e "${GREEN}✓ Found: $file${NC}"
    fi
done

if [ "$ALL_PRESENT" = false ]; then
    echo ""
    echo -e "${RED}Error: Some required files are missing${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All submission files present${NC}"
echo ""

# Summary
echo "============================================"
echo -e "${GREEN}Setup Complete! ✅${NC}"
echo "============================================"
echo ""
echo "Your submission package is ready at:"
echo "  $SUBMISSION_DIR"
echo ""
echo "Next steps:"
echo "  1. Update team info in training_code/README.md"
echo "  2. Build Docker image:"
echo "     cd $SUBMISSION_DIR"
echo "     docker build -t zac2025-submission ."
echo ""
echo "  3. Test locally:"
echo "     docker run --gpus all \\"
echo "       -v /path/to/test_data:/data \\"
echo "       -v /path/to/results:/result \\"
echo "       zac2025-submission"
echo ""
echo "  4. Export image:"
echo "     docker save zac2025-submission -o zac2025-submission.tar"
echo "     gzip zac2025-submission.tar"
echo ""
echo "See SUBMISSION_CHECKLIST.md for detailed instructions."
echo ""

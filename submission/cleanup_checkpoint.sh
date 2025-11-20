#!/bin/bash
# cleanup_checkpoint.sh - Extract only essential files for inference
# This reduces checkpoint size significantly by removing training-only files
#
# Usage: bash cleanup_checkpoint.sh /path/to/checkpoint-latest /path/to/output-clean

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "============================================"
echo "Checkpoint Cleanup - Extract Essential Files"
echo "============================================"
echo ""

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Missing arguments${NC}"
    echo ""
    echo "Usage:"
    echo "  bash cleanup_checkpoint.sh <source_checkpoint> <output_directory>"
    echo ""
    echo "Example:"
    echo "  bash cleanup_checkpoint.sh /kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest ./checkpoint-clean"
    echo ""
    exit 1
fi

SOURCE_DIR="$1"
OUTPUT_DIR="$2"

echo "Configuration:"
echo "  Source: $SOURCE_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

# Check source exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}Error: Source checkpoint not found: $SOURCE_DIR${NC}"
    exit 1
fi

# Essential files for inference (ONLY these are needed)
ESSENTIAL_FILES=(
    # LoRA adapter files
    "adapter_model.safetensors"
    "adapter_config.json"

    # Merger weights
    "merger_weights.bin"

    # Model config
    "config.json"
    "preprocessor_config.json"
    "video_preprocessor_config.json"

    # Tokenizer files
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
    "merges.txt"
    "special_tokens_map.json"
    "added_tokens.json"

    # Chat template
    "chat_template.jinja"

    # README (optional)
    "README.md"
)

# Files to SKIP (not needed for inference, waste space)
SKIP_PATTERNS=(
    "global_step*"           # Optimizer states
    "scheduler.pt"           # Scheduler state
    "training_args.bin"      # Training arguments
    "trainer_state.json"     # Trainer state
    "rng_state*.pth"         # Random state
    "zero_to_fp32.py"        # DeepSpeed utility
    "*.pt"                   # PyTorch checkpoints
    "optimizer.pt"           # Optimizer state
)

echo -e "${YELLOW}Step 1: Creating output directory...${NC}"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ Created: $OUTPUT_DIR${NC}"
echo ""

echo -e "${YELLOW}Step 2: Copying essential files...${NC}"

COPIED_COUNT=0
MISSING_COUNT=0
TOTAL_SIZE=0

for file in "${ESSENTIAL_FILES[@]}"; do
    SOURCE_FILE="$SOURCE_DIR/$file"

    if [ -f "$SOURCE_FILE" ]; then
        # Copy file
        cp "$SOURCE_FILE" "$OUTPUT_DIR/"

        # Get file size
        SIZE=$(du -sh "$SOURCE_FILE" | cut -f1)

        echo -e "${GREEN}  ✓ $file${NC} ($SIZE)"
        COPIED_COUNT=$((COPIED_COUNT + 1))

    else
        echo -e "${YELLOW}  ⚠ $file (not found, skipping)${NC}"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

echo ""
echo -e "${GREEN}✓ Copied $COPIED_COUNT files${NC}"
if [ $MISSING_COUNT -gt 0 ]; then
    echo -e "${YELLOW}⚠ $MISSING_COUNT files missing (optional)${NC}"
fi
echo ""

# Show what was skipped
echo -e "${YELLOW}Step 3: Files/directories skipped (not needed for inference):${NC}"

SKIPPED_ITEMS=$(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 \( \
    -name "global_step*" -o \
    -name "scheduler.pt" -o \
    -name "training_args.bin" -o \
    -name "trainer_state.json" -o \
    -name "rng_state*.pth" -o \
    -name "zero_to_fp32.py" -o \
    -name "*.pt" -o \
    -name "optimizer.pt" \
\))

if [ -n "$SKIPPED_ITEMS" ]; then
    while IFS= read -r item; do
        ITEM_NAME=$(basename "$item")
        ITEM_SIZE=$(du -sh "$item" | cut -f1)
        echo "  ✗ $ITEM_NAME ($ITEM_SIZE)"
    done <<< "$SKIPPED_ITEMS"
else
    echo "  (none)"
fi

echo ""

# Compare sizes
echo -e "${YELLOW}Step 4: Size comparison...${NC}"

SOURCE_SIZE=$(du -sh "$SOURCE_DIR" | cut -f1)
OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)

echo "  Original checkpoint: $SOURCE_SIZE"
echo "  Cleaned checkpoint:  $OUTPUT_SIZE"
echo ""

# Calculate space saved
SOURCE_BYTES=$(du -sb "$SOURCE_DIR" | cut -f1)
OUTPUT_BYTES=$(du -sb "$OUTPUT_DIR" | cut -f1)
SAVED_BYTES=$((SOURCE_BYTES - OUTPUT_BYTES))
SAVED_MB=$((SAVED_BYTES / 1024 / 1024))
REDUCTION_PCT=$((100 * SAVED_BYTES / SOURCE_BYTES))

if [ $SAVED_BYTES -gt 0 ]; then
    echo -e "${GREEN}✓ Space saved: ${SAVED_MB}MB (${REDUCTION_PCT}% reduction)${NC}"
else
    echo "  No significant space saved"
fi

echo ""

# Verify essential files
echo -e "${YELLOW}Step 5: Verifying cleaned checkpoint...${NC}"

REQUIRED_CORE_FILES=(
    "adapter_model.safetensors"
    "adapter_config.json"
    "merger_weights.bin"
    "config.json"
)

ALL_PRESENT=true
for file in "${REQUIRED_CORE_FILES[@]}"; do
    if [ -f "$OUTPUT_DIR/$file" ]; then
        echo -e "${GREEN}  ✓ $file${NC}"
    else
        echo -e "${RED}  ✗ $file (MISSING - CRITICAL)${NC}"
        ALL_PRESENT=false
    fi
done

echo ""

if [ "$ALL_PRESENT" = true ]; then
    echo "============================================"
    echo -e "${GREEN}✅ Checkpoint cleaned successfully!${NC}"
    echo "============================================"
    echo ""
    echo "Cleaned checkpoint ready at:"
    echo "  $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Copy to submission package:"
    echo "     cp -r $OUTPUT_DIR submission/saved_models/checkpoint-latest"
    echo ""
    echo "  2. Build Docker image:"
    echo "     cd submission && docker build -t zac2025-submission ."
    echo ""
    echo "Docker image will be significantly smaller!"
    exit 0
else
    echo "============================================"
    echo -e "${RED}❌ Error: Some critical files are missing${NC}"
    echo "============================================"
    echo ""
    echo "Please check your source checkpoint."
    exit 1
fi

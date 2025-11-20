#!/bin/bash

# predict.sh - Inference script for Zalo AI Challenge 2025
# Usage: ./predict.sh
# Input: /data/test.json
# Output: /result/submission.csv

echo "=========================================="
echo "Zalo AI Challenge 2025 - Inference Script"
echo "=========================================="
echo ""

# Set environment variables
export PYTHONPATH=/workspace:$PYTHONPATH

# Check input data exists
if [ ! -f "/data/test.json" ]; then
    echo "❌ ERROR: Input file /data/test.json not found!"
    exit 1
fi

echo "✓ Input file found: /data/test.json"

# Create output directory
mkdir -p /result
echo "✓ Output directory created: /result"
echo ""

# Run inference
echo "Running inference..."
echo ""
python3 /workspace/predict.py

# Check if output was created
if [ -f "/result/submission.csv" ]; then
    echo ""
    echo "=========================================="
    echo "✅ Inference completed successfully!"
    echo "Output file: /result/submission.csv"

    # Show number of predictions
    lines=$(wc -l < /result/submission.csv)
    predictions=$((lines - 1))
    echo "Number of predictions: $predictions"

    # Show first few lines
    echo ""
    echo "First 5 predictions:"
    head -6 /result/submission.csv

    echo "=========================================="
    exit 0
else
    echo ""
    echo "=========================================="
    echo "❌ ERROR: Output file not created!"
    echo "=========================================="
    exit 1
fi

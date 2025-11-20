#!/bin/bash

# start_jupyter.sh - Start Jupyter Notebook server for time measurement
# Usage: ./start_jupyter.sh

echo "=========================================="
echo "Starting Jupyter Notebook Server"
echo "=========================================="
echo ""

# Set environment
export PYTHONPATH=/workspace:$PYTHONPATH

# Check if notebook exists
if [ ! -f "/workspace/predict_notebook.ipynb" ]; then
    echo "❌ ERROR: predict_notebook.ipynb not found!"
    exit 1
fi

echo "✓ Notebook found: /workspace/predict_notebook.ipynb"
echo ""

# Start Jupyter with no authentication (for competition evaluation)
echo "Starting Jupyter Notebook server..."
echo ""

jupyter notebook \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --notebook-dir=/workspace

echo ""
echo "=========================================="
echo "Jupyter server stopped"
echo "=========================================="

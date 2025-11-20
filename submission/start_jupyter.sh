#!/bin/bash

# start_jupyter.sh - Start Jupyter Lab server for time measurement
# Usage: ./start_jupyter.sh
#
# Competition requirements (BTC):
# - Port: 9777
# - Password: zac2025
# - Token: zac2025

echo "=========================================="
echo "Starting Jupyter Lab Server"
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

# Start Jupyter Lab with BTC-required configuration
echo "Starting Jupyter Lab server..."
echo "  Port: 9777"
echo "  Password: zac2025"
echo "  Token: zac2025"
echo ""

jupyter lab \
    --port 9777 \
    --ip 0.0.0.0 \
    --NotebookApp.password='zac2025' \
    --NotebookApp.token='zac2025' \
    --allow-root \
    --no-browser

echo ""
echo "=========================================="
echo "Jupyter server stopped"
echo "=========================================="

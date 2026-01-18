#!/bin/bash
# Fix numpy/numba compatibility issue
# This script reinstalls numpy and numba to fix import errors

echo "Fixing numpy/numba compatibility issue..."

# Check if using conda
if command -v conda &> /dev/null; then
    echo "Using conda environment..."
    conda install -y numpy numba -c conda-forge
else
    echo "Using pip..."
    python3 -m pip install --upgrade --force-reinstall numpy numba
fi

echo "Done! Please restart your Python environment."

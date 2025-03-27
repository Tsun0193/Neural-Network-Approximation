#!/usr/bin/env bash

# Exit on error
set -e

# Create environment
conda create -n nna python=3.12 -y

# Load conda into the script's shell (no terminal restarts, no infinite loops)
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate nna

# Install Python dependencies
pip install --upgrade -r requirements.txt --quiet
pip install -e . --quiet

echo -e "\n✅ Setup complete! Environment 'nna' is ready."
echo "👉 Run: conda activate nna"

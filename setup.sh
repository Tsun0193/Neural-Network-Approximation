#!/usr/bin/env bash

set -e
conda create -n nna python=3.12 -y
eval "$(conda shell.bash hook)"
conda activate nna

pip install --upgrade -r requirements.txt --quiet
pip install -e . --quiet

echo -e "\n✅ Setup complete! Environment 'nna' is ready."
echo "👉 Run: conda activate nna"
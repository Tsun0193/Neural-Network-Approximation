@echo off
REM Create conda environment "nna" with Python 3.12
conda create -n nna python=3.12 -y

REM Activate the newly created environment
call conda activate nna

REM Upgrade packages from requirements.txt and install the package in editable mode
pip install --upgrade -r requirements.txt --quiet
pip install -e . --quiet

echo.
echo ✅ Setup complete! Environment 'nna' is ready.
echo 👉 Run: conda activate nna
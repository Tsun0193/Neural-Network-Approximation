import os
import platform
import subprocess

PYTHON = "python3.12"  # <--- Update here

def run(command):
    subprocess.run(command, shell=True, check=True)

is_windows = platform.system() == "Windows"
venv_dir = ".venv"

# Create venv
run(f"{PYTHON} -m venv {venv_dir}")
activate = f"{venv_dir}\\Scripts\\activate.bat" if is_windows else f"source {venv_dir}/bin/activate"
print(f"Activate with: {activate}")

# Install deps
pip = f"{venv_dir}\\Scripts\\pip" if is_windows else f"{venv_dir}/bin/pip"
run(f"{pip} install --upgrade pip")
run(f"{pip} install -r requirements.txt")

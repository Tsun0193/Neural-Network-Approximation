conda create -n nna python=3.11
conda activate nna

pip install -r requirements.txt --q --upgrade
pip install -e . --q
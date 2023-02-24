
rm -rf .venv

python -m venv .venv

source .venv/bin/activate

pip install stable-baselines3==1.7.0 gym==0.21.0 --no-deps

pip install -r requirements.txt 


export PYTHONPATH=$PWD

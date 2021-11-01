rm handy bash cmd
conda env create -f environment.yml
conda activate Wine_env

pip install -r requirements.txt

cd GIT 
pip install .

Black . 
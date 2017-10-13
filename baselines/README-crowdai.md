Notes
-----

This repository has been modified to work with Python2 (at least the PPO implementation).

It has been modified to work with osim-rl for CrowdAI's Learning to Run Competition.

Installation
------------

# with pyenv
pyenv virtualenv anaconda2-4.4.0 baselines
# Set pyenv's PYTHON_VERSION to 'baselines' or put 'baselines' in .python-version

# without pyenv
conda env create baselines
source activate baselines

conda install libgcc
conda install --channel kidzik opensim
conda install --channel conda-forge lapack
pip install git+https://github.com/stanfordnmbl/osim-rl.git
pip install -e .
#!/usr/bin/bash
# Get the location of this script
SCRIPT_PATH="$(dirname -- "${BASH_SOURCE[0]}")"
# Name of the virtual environment
VENV=.venv
# Name of the models directory
MODELS=src/aimnet2exttool/models

# Switch to the source directory
cd $SCRIPT_PATH
# Remove any previously existing virtual environment
rm -rf $VENV 2>/dev/null
# Create a new virtual environment and activate it
python3 -m venv $VENV
source $VENV/bin/activate

# Install build tools
pip install -U pip wheel setuptools
# Install the CPU-only version of torch
pip install "torch==2.5.0" --index-url https://download.pytorch.org/whl/cpu
# Install the matching version of torch-cluster
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
# TODO: install aimnet2 without the ASE dependency
pip install -e .

# remove any previously existing models directory
rm -rf $MODELS 2>/dev/null
mkdir $MODELS
# download the aimnet2 models from the aimnet-model-zoo repo
wget -O- https://github.com/zubatyuk/aimnet-model-zoo/archive/refs/heads/main.tar.gz | \
tar xz -C $MODELS/ --wildcards 'aimnet-model-zoo-main/aimnet2/*' --strip-components 2

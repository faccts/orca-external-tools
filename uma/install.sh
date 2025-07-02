#!/usr/bin/bash
# Get the location of this script
SCRIPT_PATH="$(dirname -- "${BASH_SOURCE[0]}")"
# Name of the virtual environment
VENV=.venv
# Name of the models directory
MODELS=src/umaexttool/models

# Switch to the source directory
cd $SCRIPT_PATH
# Remove any previously existing virtual environment
rm -rf $VENV 2>/dev/null
# Create a new virtual environment and activate it
python3 -m venv $VENV
source $VENV/bin/activate

# Install relevant packages and create fairchem core
pip install -e .

# remove any previously existing models directory
rm -rf $MODELS 2>/dev/null
mkdir $MODELS
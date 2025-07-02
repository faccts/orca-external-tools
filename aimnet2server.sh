#!/bin/bash

SCRIPT_PATH="$(dirname -- "${BASH_SOURCE[0]}")"
src="${SCRIPT_PATH}/aimnet2"
venv="$src/.venv"

if [ ! -d $venv ]; then
  # create a venv and install the wrapper
  $src/install.sh
fi

# activate an existing venv
source $venv/bin/activate

# Run the aimnet2server script with all passed arguments.
aimnet2server "$@"

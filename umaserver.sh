#!/bin/bash

SCRIPT_PATH="$(dirname -- "${BASH_SOURCE[0]}")"
src="${SCRIPT_PATH}/uma"
venv="$src/.venv"

if [ ! -d $venv ]; then
  # create a venv and install the wrapper
  $src/install.sh
fi

# activate an existing venv
source $venv/bin/activate

# Run the umaserver script with all passed arguments.
umaserver "$@" &
PID=$!
echo "UMASERVER_PID: $PID"

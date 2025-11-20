#!/bin/bash

SCRIPT_PATH="$(dirname -- "${BASH_SOURCE[0]}")"
src="${SCRIPT_PATH}/mace"
venv="$src/.venv"

need_setup=0
if [ ! -d "$venv" ]; then
  need_setup=1
elif [ ! -x "$venv/bin/maceexttool" ]; then
  need_setup=1
fi

if [ $need_setup -eq 1 ]; then
  "$src"/install.sh
fi

source "$venv"/bin/activate
maceexttool "$@"

#!/usr/bin/env bash
#!/usr/bin/env bash
# Get the location of this script
SCRIPT_PATH="$(dirname -- "${BASH_SOURCE[0]}")"
# Name of the virtual environment
VENV=.venv

set -e

cd "$SCRIPT_PATH"

choose_python() {
  for exe in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$exe" >/dev/null 2>&1; then
      if "$exe" -c 'import sys; raise SystemExit(int(sys.version_info< (3,10)))'; then
        echo "$exe"; return 0
      fi
    fi
  done
  echo ""; return 1
}

PYEXE="$(choose_python)" || {
  echo "No suitable Python >=3.10 found to create venv. Please install Python 3.10+." >&2
  exit 1
}
echo "Using Python interpreter: $PYEXE"

# Reset venv
rm -rf "$VENV" 2>/dev/null || true
"$PYEXE" -m venv "$VENV"
source "$VENV/bin/activate"

# Ensure modern pip/setuptools for editable installs with pyproject
python -m pip install --upgrade pip setuptools wheel

# Install wrapper
pip install -e .

# Try to install local MACE if available; otherwise rely on dependency
LOCAL_MACE_DIR="$(cd "$SCRIPT_PATH/../.." && pwd)/mace"
if [ -d "$LOCAL_MACE_DIR" ]; then
  echo "Installing local MACE from $LOCAL_MACE_DIR"
  pip install -e "$LOCAL_MACE_DIR" || echo "Warning: failed to install local MACE; ensure mace-torch is available."
else
  echo "Local MACE repo not found. Using mace-torch from PyPI if available."
fi

echo "Setup complete. Activate venv with: source $SCRIPT_PATH/$VENV/bin/activate"

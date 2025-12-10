import os
import sys
from pathlib import Path


# Ensure local src is importable as maceexttool
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# No dummy calculator: use real MACE in integration tests

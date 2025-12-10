from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("oet")
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent

# Assets (model/parameter files, etc.)
ASSETS_DIR = ROOT_DIR / "assets"

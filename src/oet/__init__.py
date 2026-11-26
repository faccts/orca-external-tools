from pathlib import Path

try:
    # Written at build time by [tool.setuptools_scm]. Importing it is cheaper than resolving the version through importlib.metadata.
    # "as __version__" marks it as re-exported for mypy (no_implicit_reexport)
    from oet._version import __version__ as __version__
except ImportError:
    __version__ = "unknown"

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent

# Assets (model/parameter files, etc.)
ASSETS_DIR = ROOT_DIR / "assets"

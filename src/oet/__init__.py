from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('oet')
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version
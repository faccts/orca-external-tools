from __future__ import annotations

import warnings

with warnings.catch_warnings():
    # suppress the missing PySisiphus warning
    warnings.simplefilter("ignore")
    from aimnet2calc import AIMNet2Calculator


def init(*args, **kwargs) -> AIMNet2Calculator:
    """Initialize an AIMNet2Calculator instance."""
    return AIMNet2Calculator(*args, **kwargs)
from __future__ import annotations

from typing import Optional

from ase import Atoms


def init(
    suite: str,
    model: Optional[str] = None,
    *,
    device: str = "",
    default_dtype: Optional[str] = None,
    dispersion: bool = False,
    damping: str = "bj",
    dispersion_xc: str = "pbe",
    dispersion_cutoff: Optional[float] = None,
    head: Optional[str] = None,
):
    """Initialize a MACE calculator based on suite and options.

    Returns an ASE calculator compatible object which can be attached to Atoms.
    """
    # Lazy import to avoid heavy deps at import-time
    from mace.calculators.foundations_models import mace_mp, mace_omol
    from ase import units

    if suite == "mp":
        kwargs = dict(
            model=model,
            device=device or ("cuda" if _torch_cuda_available() else "cpu"),
            default_dtype=default_dtype or "float32",
            dispersion=dispersion,
            damping=damping,
            dispersion_xc=dispersion_xc,
            dispersion_cutoff=(dispersion_cutoff if dispersion_cutoff is not None else 40.0 * units.Bohr),
        )
        if head:
            kwargs["head"] = head
        calc = mace_mp(**kwargs)
        return calc
    elif suite == "omol":
        calc = mace_omol(
            model=model,
            device=device or ("cuda" if _torch_cuda_available() else "cpu"),
            default_dtype=default_dtype or "float64",
        )
        return calc
    else:
        raise ValueError(f"Unknown suite: {suite}. Expected 'mp' or 'omol'.")


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return torch.cuda.is_available()
    except Exception:
        return False

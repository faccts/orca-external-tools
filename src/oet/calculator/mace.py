#!/usr/bin/env python3
"""
MACE wrapper for ORCA's ExtTool interface.
"""

import sys
import warnings
from argparse import ArgumentParser, Namespace
from typing import Any

import numpy as np

from oet.core.base_calc import BaseCalc, CalculationData, CalculationResult
from oet.core.definitions import ENERGY_CONVERSION, LENGTH_CONVERSION, Property
from oet.core.misc import xyzfile_to_at_coord

try:
    with warnings.catch_warnings():
        # Catch the warning that the `weights_only` bool is not set during `torch.load`. This will not affect the calculation.
        warnings.filterwarnings(
            "ignore",
            message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected",
        )
        from ase.calculators.calculator import PropertyNotImplementedError
        from mace.calculators.foundations_models import mace_mp, mace_omol
        from mace.calculators.mace import MACECalculator
except ImportError as e:
    print(
        f"[MISSING] Required module mace not found: {e}.\n"
        "Please install the packages in the virtual environment.\n"
        "Therefore, activate the venv, got to the orca-external-tools "
        "main directory and use pip install -r ./requirements/mace.txt"
    )
    sys.exit(1)

try:
    import torch
except ImportError as e:
    print("[MISSING] torch not found:", e)
    sys.exit(1)

try:
    from ase import Atoms
except ImportError as e:
    print("[MISSING] ase not found:", e)
    sys.exit(1)


class MaceCalc(BaseCalc):
    # MACE calculator used for computations
    _calc: MACECalculator | None = None
    # For MACE, in addition to the default properties, also Hessians are available.
    DEFAULT_PROPERTIES = BaseCalc.DEFAULT_PROPERTIES | {Property.HESSIAN}

    def set_calculator(
        self,
        suite: str,
        model: str,
        dispersion: bool,
        damping: str,
        dispersion_xc: str,
        dispersion_cutoff: float,
        device: str,
        default_dtype: str,
        head: str,
        force: bool = False,
    ) -> None:
        """
        Prepare a MACE calculator object that is compatible with the ASE calculator object to compute energy and gradient, if not done already.

        Parameters
        ----------
        suite: str
            Suite for the calculator.
        model: str
            Model/model path to be used.
        dispersion: bool
            Whether to use a dispersion model or not.
        damping: str
            Dispersion damping.
        dispersion_xc: str
            XC functional for the dispersion model.
        dispersion_cutoff: str
            Cutoff radius for the dispersion model.
        device: str
            Device (cuda or cpu).
        default_dtype: str
            Precision type.
        head: str
            The head of the MACE model.
        force: bool, default = False
            Force re-initialization of the calculator, even if already initialized.
        """
        if self._calc and not force:
            return
        match suite:
            case "mp":
                kwargs: dict[str, object] = {
                    "model": model,
                    "device": device,
                    "default_dtype": default_dtype or "float32",
                    "dispersion": dispersion,
                    "damping": damping,
                    "dispersion_xc": dispersion_xc,
                    "dispersion_cutoff": (dispersion_cutoff * LENGTH_CONVERSION["Ang"]),
                }
                if head:
                    kwargs["head"] = head
                calc = mace_mp(**kwargs)
            # No dispersion applicable for omol
            case "omol":
                calc = mace_omol(
                    model=model,
                    device=device,
                    default_dtype=default_dtype or "float64",
                )
            case _:
                raise RuntimeError(f"The suite {suite} is not supported.")
        self._calc = calc

    def get_calculator(self) -> MACECalculator | None:
        """
        Returns the current MACE calculator
        """
        return self._calc

    @classmethod
    def extend_parser(cls, parser: ArgumentParser) -> None:
        """
        Add Mace parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        parser.add_argument(
            "-s",
            "--suite",
            dest="suite",
            choices=["mp", "omol", "mace-mp", "mace-omol"],
            default="omol",
            help="Select MACE suite: mp/mace-mp or omol/mace-omol. Default: omol",
        )
        # Settings the default model to None uses the MACE default.
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default=None,
            help="Model spec or local path. MP: small/medium/large/medium-mpa-0/... OMOL: extra_large or path.",
        )
        parser.add_argument(
            "--default-dtype",
            choices=["float32", "float64"],
            default="float64",
            help="Default float precision (recommended: use float32 only for MD).",
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="cpu",
            metavar="DEVICE",
            dest="device",
            choices=(device_choices := ["cpu", "cuda"]),
            help="Device to perform the calculation on. "
            "Options: " + ", ".join(device_choices) + ". "
            "Default: cpu. ",
        )
        # MP-specific extras
        parser.add_argument(
            "--dispersion",
            action="store_true",
            help=(
                "Enable D3 dispersion. Only applicable for the MP suite. "
                "This requires the installation of torch_dftd to the virtual "
                "environment and that liblzma was installed when setting up "
                "the environment."
            ),
        )
        parser.add_argument(
            "--damping",
            type=str,
            default="bj",
            choices=(damp_choices := ["zero", "bj", "zerom", "bjm"]),
            help="D3 damping. "
            "Options: " + ", ".join(damp_choices) + "."
            "Only applicable for the MP suite. Default BJ for MP suite. ",
        )
        parser.add_argument(
            "--dispersion-xc",
            type=str,
            default="pbe",
            help="XC functional for D3. Only applicable for the MP suite. Default PBE for MP suite.",
        )
        parser.add_argument(
            "--dispersion-cutoff",
            type=float,
            default=40.0,
            help="Cutoff radius for D3 in Bohr. "
            "Only applicable for the MP suite. "
            "Default: 40 Bohr for MP suite.",
        )
        parser.add_argument(
            "--head",
            type=str,
            default=None,
            help="Advanced option: Select MACE head. Only applicable for the MP suite.",
        )

    @classmethod
    def process_input_keywords(
        cls, args_parsed: Namespace, calc_data: CalculationData
    ) -> tuple[Any, ...]:
        """
        Extract and sanitize the calculation settings from the input Namespace.

        Parameters
        ----------
        args_parsed: Namespace
            The parsed namespace.
        calc_data: CalculationData
            The calculation data.

        Returns
        -------
        tuple[Any, ...]: The settings used for setting up the calculator.
        """

        suite = args_parsed.suite
        model = args_parsed.model
        dispersion = args_parsed.dispersion
        damping = args_parsed.damping
        dispersion_xc = args_parsed.dispersion_xc
        dispersion_cutoff = args_parsed.dispersion_cutoff
        device = args_parsed.device
        default_dtype = args_parsed.default_dtype
        head = args_parsed.head

        if suite.startswith("mace"):
            suite = suite.split("-", 1)[1]

        if dispersion and suite != "mp":
            print(
                "WARNING: Dispersion flag recognized, but MP suite not used. Ignoring all options related to dispersion."
            )

        return (
            suite,
            model,
            dispersion,
            damping,
            dispersion_xc,
            dispersion_cutoff,
            device,
            default_dtype,
            head,
        )

    def _get_hessian(self, atoms: Atoms) -> list[float]:
        """
        Compute a Hessian with MACE.

        Parameters
        ----------
        atoms: Atoms
            The ASE Atoms type object for computing the Hessian.

        Returns
        -------
        list[float]
            Flattened Hessian vector (Eh/Bohr^2).
        """

        # Safety check
        if self._calc is None:
            raise RuntimeError("MACE calculator has not been initialized.")

        # Get the raw Hessian
        raw = self._calc.get_hessian(atoms=atoms)
        # Flatten the MACE Hessian layout (3N, N, 3) or (3N, 3N).
        natoms = len(atoms)
        n = 3 * natoms
        arr = np.asarray(raw, dtype=np.float64)
        # Sanity check
        if arr.shape not in {(n, natoms, 3), (n, n)}:
            raise ValueError(
                f"Unexpected MACE Hessian shape {arr.shape}; expected {(n, natoms, 3)} or {(n, n)}."
            )
        if not np.isfinite(arr).all():
            raise ValueError("MACE Hessian contains NaN or infinity.")
        matrix = arr.reshape(n, n, order="C")

        # `get_hessian` returns raw model derivatives, unlike the
        # energy/forces in calc.results, which are already converted to ASE units.
        # Foundation MP/OMOL models use energy_units_to_eV = length_units_to_A = 1.
        # Take the constant as implemented in MACE.
        e_to_ev = float(self._calc.energy_units_to_eV)
        l_to_ang = float(self._calc.length_units_to_A)
        if not np.isfinite([e_to_ev, l_to_ang]).all() or min(e_to_ev, l_to_ang) <= 0:
            raise ValueError("Invalid MACE model unit conversion factors.")
        factor = e_to_ev / l_to_ang**2 * LENGTH_CONVERSION["Ang"] ** 2 / ENERGY_CONVERSION["eV"]
        converted = matrix * factor
        if not np.isfinite(converted).all():
            raise ValueError("Converted MACE Hessian contains NaN or infinity.")
        return converted.ravel(order="C").tolist()

    def run_mace(
        self,
        atom_types: list[str],
        coordinates: list[tuple[float, float, float]],
        calc_data: CalculationData,
    ) -> tuple[float, list[float], list[float]]:
        """
        Runs a Mace calculation.

        Parameters
        ----------
        atom_types : list[str]
            List of element symbols (e.g., ["O", "H", "H"])
        coordinates : list[tuple[float, float, float]]
            List of (x, y, z) coordinates
        calc_data: CalculationData
            Object with calculation data for the run

        Returns
        -------
        float
            The computed energy (Eh)
        list[float]
            Flattened gradient vector (Eh/Bohr), if computed, otherwise empty
        list[float]
            Flattened Hessian vector (Eh/Bohr^2), if computed, otherwise empty
        """

        # set the number of threads
        torch.set_num_threads(calc_data.ncores)

        # make ase atoms object for calculation
        atoms = Atoms(symbols=atom_types, positions=coordinates)
        atoms.info = {"charge": calc_data.charge, "spin": calc_data.mult}
        atoms.calc = self._calc

        # Get the energy and gradients
        energy = atoms.get_potential_energy() / ENERGY_CONVERSION["eV"]
        gradient = []
        try:
            forces = atoms.get_forces()
            # Convert forces to gradient (-1) and unit conversion
            fac = -LENGTH_CONVERSION["Ang"] / ENERGY_CONVERSION["eV"]
            gradient = (fac * forces).flatten().tolist()
        except PropertyNotImplementedError:
            # forces may not be available
            pass

        # Compute Hessians
        hessian = self._get_hessian(atoms) if calc_data.dohess else []

        return energy, gradient, hessian

    def calc(
        self,
        calc_data: CalculationData,
        args_parsed: Namespace,
        args_not_parsed: list[str],
    ) -> CalculationResult:
        """
        Routine for calculating energy and optional gradient.

        Parameters
        ----------
        calc_data: CalculationData
            Object with calculation data for the run
        args_parsed: Namespace
            Arguments parsed as defined in extend_parser
        args_not_parsed: list[str]
            Arguments not parsed so far

        Returns
        -------
        CalculationResult
            Energy (Eh) and, if requested, gradient (Eh/Bohr) and Hessian (Eh/Bohr^2)
        """

        (
            suite,
            model,
            dispersion,
            damping,
            dispersion_xc,
            dispersion_cutoff,
            device,
            default_dtype,
            head,
        ) = self.process_input_keywords(args_parsed, calc_data)

        # Setup calculator if not already set.
        # This is important as usage on a server would otherwise cause an initialization with every call so that nothing is gained.
        # Catch the warning that the `weights_only` bool is not set during `torch.load`. This will not affect the calculation.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected",
            )
            self.set_calculator(
                suite=suite,
                model=model,
                dispersion=dispersion,
                damping=damping,
                dispersion_xc=dispersion_xc,
                dispersion_cutoff=dispersion_cutoff,
                device=device,
                default_dtype=default_dtype,
                head=head,
            )

        # process the XYZ file
        atom_types, coordinates = xyzfile_to_at_coord(calc_data.xyzfile)

        # run mace
        energy, gradient, hessian = self.run_mace(
            atom_types=atom_types, coordinates=coordinates, calc_data=calc_data
        )

        return CalculationResult(energy=energy, gradient=gradient, hessian=hessian)


def main() -> None:
    """
    Main routine for execution.
    """
    calculator = MaceCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Calculator for using AIMNet2 (https://github.com/isayevlab/AIMNet2),
compatible with ORCA's ExtTool interface.

Provides
--------
class: Aimnet2Calc(CalcServer)
    Class for performing a AIMNet2 calculation together with ORCA
main: function
    Main function
"""

import shutil
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from requests.exceptions import HTTPError

from oet import ASSETS_DIR
from oet.core.base_calc import BaseCalc, CalculationData
from oet.core.misc import ENERGY_CONVERSION, LENGTH_CONVERSION, xyzfile_to_at_coord

try:
    # Suppress PySisyphus missing warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from aimnet2calc import AIMNet2Calculator
        from aimnet2calc.models import get_model_path, model_registry_aliases
except ImportError:
    print(
        "[MISSING] Required module aimnet2calc not found.\n"
        "Please install the packages in the virtual environment.\n"
        "Therefore, activate the venv, got to the orca-external-tools "
        "main directory and use pip install -r requirements/aimnet2.txt"
    )
    sys.exit(1)

try:
    import torch
except ImportError as e:
    print("[MISSING] torch not found:", e)
    sys.exit(1)


DEFAULT_MODEL_PATH = ASSETS_DIR / "aimnet2"


class Aimnet2Calc(BaseCalc):
    # Elements covered by AIMNet2
    ELEMENT_TO_ATOMIC_NUMBER = {
        "H": 1,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "As": 33,
        "Se": 34,
        "Br": 35,
        "I": 53,
    }

    # Supported devices
    supported_devices = ("cpu", "cuda", "auto")

    # AIMNet2 calculator used to compute energy and grad
    _calc: AIMNet2Calculator | None = None

    def get_calculator(self) -> AIMNet2Calculator:
        """
        Returns AIMNet2 calculator

        Returns
        -------
        AIMNet2Calculator: AIMNet2 calculator
        """
        return self._calc

    def set_calculator(self, model: str, device: str) -> None:
        """
        Set the calculator

        Parameters
        ----------
        model: str
            Model of the calculator
        device: str
            device to use
        """
        if device == "cpu":
            # Monkey-patch torch.cuda.is_available() to return False
            # Another way to do it would be to set CUDA_VISIBLE_DEVICES="" *before* the initial import of torch
            # Ideally, AIMNet2Calculator would just have an extra argument for this.
            torch.cuda.is_available = lambda: False
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        self._calc = AIMNet2Calculator(model=model)

    @staticmethod
    def get_model_file(model: str, model_dir: str) -> Path:
        """
        Make sure model file exists in the correct location.
        If `model` is an absolute path, it must already exist.
        Otherwise, let AIMNet2 download it, then move it to `model_dir`.

        Parameters
        ----------
        model
            Model name, e.g. "aimnet2_wb97m", or filename, e.g. "aimnet2_wb97m_0.jpt", or absolute path
        model_dir
            directory to look for or store model file

        Returns
        -------
        model_path: Path
            Full path to the model file

        Raises
        ------
        FileNotFoundError
            If the model file is given by absolute path and does not exist
        FileExistsError
            If `model_dir` exists but is not a directory or `model_path` exists but is not a file
        """
        # Check if `model` is already an absolute path
        if (model_path := Path(model)).is_absolute():
            if not model_path.exists():
                raise FileNotFoundError(f'Model file "{model_path}" not found')
            return model_path
        # `model` must be the name of a model
        else:
            # check aliases
            model_file = model_registry_aliases.get(model, model)
            # add jpt extension if not already present
            if not model_file.endswith('.jpt'):
                model_file += '.jpt'
            # strip any directories
            model_file = Path(model_file).name
            # make sure the directory exists
            model_dir = Path(model_dir)
            if model_dir.exists() and not model_dir.is_dir():
                raise FileExistsError(f'Path "{model_dir}" exists but is not a directory')
            model_dir.mkdir(parents=True, exist_ok=True)
            # construct the full path
            model_path = model_dir / model_file
            # if the file exists, pass the path to `get_model_path`
            if model_path.exists():
                if model_path.is_file():
                    model = str(model_path)
                else:
                    raise FileExistsError(f'Path "{model_path}" exists but is not a file')
            # obtain the file from AIMNet2
            try:
                actual_path = Path(get_model_path(model))
            except HTTPError as e:
                # If the URL is not found, it's possible the user requested, e.g. "aimnet2_wb97m_1.jpt"
                # This is actually under "aimnet2/aimnet2_wb97m_1.jpt" and also not in the `model_registry_aliases`
                if not "/" in model:
                    # look for "aimnet2_..." under "aimnet2/aimnet2_..."
                    model_subdir = model.split("_")[0] + "/" + model
                    print(f'Failed to find model "{model}" at URL: {e.response.url}\n'
                          f'Trying again with model name "model_subdir"', file=sys.stderr)
                    actual_path = Path(get_model_path(model_subdir))
                else:
                    raise e
            # move it to the correct destination for subsequent runs
            if not (model_path.exists() and model_path.samefile(actual_path)):
                shutil.move(actual_path, model_path)
            # finally return the path
            return model_path

    def setup(self, model: str, model_dir: str, device: str) -> None:
        """
        Sets the calculator. Does nothing, if it is already set.

        Parameters
        ----------
        model: str
            Model that the calculator should have
        model_dir: str
            Path to the model files
        device: str
            device to use

        Returns
        -------
        dict: Arguments where all entries are removed that were processed
        """
        if not self._calc:
            # Sanitize the model path and fetch the actual file
            model_path = str(self.get_model_file(model, model_dir))
            self.set_calculator(model=model_path, device=device)

    @classmethod
    def extend_parser(cls, parser: ArgumentParser) -> None:
        """Add AimNet2 parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            dest="model",
            default="aimnet2_wb97m",
            help='The AIMNet2 model name or file name or absolute path. '
                 'If an absolute path is given, the file must exist. '
                 'Otherwise, it will be downloaded to DIR if necessary. '
                 'Default: "aimnet2_wb97m".',
        )
        parser.add_argument(
            "-p",
            "--model-path",
            metavar="DIR",
            dest="model_dir",
            type=str,
            default=str(DEFAULT_MODEL_PATH),
            help=f'The directory to look for and store AIMNet2 model files. Default: "{DEFAULT_MODEL_PATH}". ',
        )
        parser.add_argument(
            "-d",
            "--device",
            metavar="DEVICE",
            dest="device",
            type=str,
            choices=["cpu", "cuda", "auto"],
            default="cpu",
            help="Device to perform the calculation on. "
            "Options: cpu, cuda, or auto (i.e. use cuda if available, otherwise cpu). "
            "Default: cpu. ",
        )

    def atomic_symbol_to_number(self, symbol: str) -> int:
        """Convert an element symbol to an atomic number.

        Parameters
        ----------
        symbol
            Element symbol, e.g. "Cl"

        Returns
        -------
        int
            atomic number of the element

        Raises
        ------
        ValueError
            if the element is not in `ELEMENT_TO_ATOMIC_NUMBER`
        """
        try:
            return self.ELEMENT_TO_ATOMIC_NUMBER[symbol.title()]
        except KeyError:
            raise ValueError(f"Unknown element symbol: {symbol}")

    def serialize_input(
        self,
        atom_types: list[str],
        coordinates: list[tuple[float, float, float]],
        charge: int,
        mult: int,
        dograd: bool,
    ) -> dict[str, Any]:
        """Serialize the input data into kwargs for AIMNet2Calculator

        Parameters
        ----------
        atom_types : list[str]
            List of element symbols (e.g., ["O", "H", "H"])
        coordinates : list[tuple[float, float, float]]
            List of (x, y, z) coordinates
        charge : int
            Molecular charge
        mult : int
            Spin multiplicity
        dograd : bool
            Whether to compute the gradient

        Returns
        -------
        dict[str, Any]
            kwargs for AIMNet2Calculator.eval()
        """
        numbers = [self.atomic_symbol_to_number(sym) for sym in atom_types]
        return {
            "data": {
                "coord": [coordinates],
                "numbers": [numbers],
                "charge": [charge],
                "mult": [mult],
            },
            "forces": dograd,
            "stress": False,
            "hessian": False,
        }

    def run_aimnet2(
        self,
        atom_types: list[str],
        coordinates: list[tuple[float, float, float]],
        calc_data: CalculationData,
    ) -> tuple[float, list[float]]:
        """
        Runs an AimNet2 calculation.

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
        tuple[float, list[float]]
            energy : float
                The computed energy (Eh)
            gradient : list[float]
                Flattened gradient vector (Eh/Bohr), if computed, otherwise empty.
        """

        # set the number of threads
        torch.set_num_threads(calc_data.ncores)

        # make ase atoms object for calculation
        aimnet2_input = self.serialize_input(
            atom_types=atom_types,
            coordinates=coordinates,
            mult=calc_data.mult,
            charge=calc_data.charge,
            dograd=calc_data.dograd,
        )

        if not self._calc:
            raise RuntimeError("Calculator could not be initialized.")
        results = self._calc(**aimnet2_input)

        energy = float(results["energy"].detach()) / ENERGY_CONVERSION["eV"]
        gradient = []
        if (forces := results.get("forces", None)) is not None:
            # unit conversion & factor of -1 to convert from forces to gradient
            fac = -LENGTH_CONVERSION["Ang"] / ENERGY_CONVERSION["eV"]
            gradient = (forces * fac).flatten().tolist()

        return energy, gradient

    def calc(
        self,
        calc_data: CalculationData,
        args_parsed: dict[str, Any],
        args_not_parsed: list[str],
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient.
        Writes ORCA output


        Parameters
        ----------
        calc_data: CalculationData
            Object with calculation data for the run
        args_parsed: dict[str, Any]
            Arguments parsed as defined in extend_parser
        args_not_parsed: list[str]
            Arguments not parsed so far
        """
        # Get the arguments parsed as defined in extend_parser
        model = args_parsed.get("model")
        model_dir = args_parsed.get("model_dir")
        device = str(args_parsed.get("device"))
        # Check if device is allowed
        if device not in tuple(self.supported_devices):
            raise RuntimeError(
                f"Device {device} not applicable for AIMNet2 calculation."
                "Please use `--device` to choose between {supported_devices}."
            )
        if not isinstance(model, str) or not isinstance(model_dir, str):
            raise RuntimeError("Problems detecting model parameters.")
        # setup calculator if not already set
        # this is important as usage on a server would otherwise cause
        # initialization with every call so that nothing is gained
        self.setup(model=model, model_dir=model_dir, device=device)
        # process the XYZ file
        atom_types, coordinates = xyzfile_to_at_coord(calc_data.xyzfile)

        # run uma
        energy, gradient = self.run_aimnet2(
            atom_types=atom_types, coordinates=coordinates, calc_data=calc_data
        )

        return energy, gradient


def main() -> None:
    """
    Main routine for execution
    """
    calculator = Aimnet2Calc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()

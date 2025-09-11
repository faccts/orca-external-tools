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
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import sys
import warnings
from oet.core.base_calc import BasicSettings, CalcServer
from oet.core.misc import xyzfile_to_at_coord, ENERGY_CONVERSION, LENGTH_CONVERSION

try:
    # Suppress PySisiphus missing warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from aimnet2calc import AIMNet2Calculator
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


class Aimnet2Calc(CalcServer):

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

    # AIMNet2 calculator used to compute energy and grad
    _calc: AIMNet2Calculator

    def set_calculator(self, model: str) -> None:
        """
        Set the AimNet2 calculator used by the Aimnet2Calc object to compute energy and grad

        Parameters
        ----------
        model: str
            Path to the model files
        """
        self._calc = AIMNet2Calculator(model=model)

    def get_calculator(self) -> AIMNet2Calculator:
        """
        Returns AIMNet2 calculator

        Returns
        -------
        AIMNet2Calculator: AIMNet2 calculator
        """
        return self._calc

    def setup(self, args: dict) -> dict:
        """
        Filters the command line arguments for setup arguments
        Returns the respective dict where the processed arguments are removed

        Parameters
        ----------
        args: dict
            Arguments provided via cmd

        Returns
        -------
        dict: Arguments where all entries are removed that were processed
        """
        # Check whether models are present
        model = args.pop("model")
        model_path = str(args.pop("model_dir") / Path(model))
        if os.path.isfile(model_path):
            self.set_calculator(model=model_path)
        # If not, aimnet will download them automatically
        else:
            self.set_calculator(model=model)
        return args

    def extend_parser(self, parser: ArgumentParser):
        """Add AimNet2 parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        self.extend_parser_setup(parser=parser)

    def extend_parser_setup(self, parser: ArgumentParser):
        """
        Add AimNet2 parsing options that are used for setting up the calculator.

        Parameters
        ----------
        parser: ArgumentParser
            Argument parser to extend
        """
        default_model_path = Path(__file__).resolve().parent / "models"
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            dest="model",
            default="aimnet2_wb97m",
            help='The AIMNet2 model file name (must be in MODEL_DIR) or absolute path. Default: "aimnet2_wb97m".',
        )
        parser.add_argument(
            "-d",
            "--model-dir",
            metavar="MODEL_DIR",
            dest="model_dir",
            type=Path,
            default=default_model_path,
            help=f'The directory to look for AIMNet2 model files. Default: "{default_model_path}".',
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

    def extend_parser_settings(self, parser: ArgumentParser):
        """
        Add Uma parsing options that are used for calculation specific settings.

        Parameters
        ----------
        parser: ArgumentParser
            Argument parser to extend
        """
        pass

    def run_aimnet2(
        self,
        atom_types: list[str],
        coordinates: list[tuple[float, float, float]],
        settings: BasicSettings,
    ) -> tuple[float, list[float]]:
        """
        Runs an AimNet2 calculation.

        Parameters
        ----------
        atom_types : list[str]
            List of element symbols (e.g., ["O", "H", "H"])
        coordinates : list[tuple[float, float, float]]
            List of (x, y, z) coordinates
        settings: BasicSettings
            Object with basic settings for the run

        Returns
        -------
        tuple[float, list[float]]
            energy : float
                The computed energy (Eh)
            gradient : list[float]
                Flattened gradient vector (Eh/Bohr), if computed, otherwise empty.
        """

        # set the number of threads
        torch.set_num_threads(settings.ncores)

        # make ase atoms object for calculation
        aimnet2_input = self.serialize_input(
            atom_types=atom_types,
            coordinates=coordinates,
            mult=settings.mult,
            charge=settings.charge,
            dograd=settings.dograd,
        )

        results = self._calc(**aimnet2_input)

        energy = float(results["energy"]) / ENERGY_CONVERSION["eV"]
        gradient = []
        if (forces := results.get("forces", None)) is not None:
            # unit conversion & factor of -1 to convert from forces to gradient
            fac = -LENGTH_CONVERSION["Ang"] / ENERGY_CONVERSION["eV"]
            gradient = (forces * fac).flatten().tolist()

        return energy, gradient

    def calc(
        self,
        settings: BasicSettings,
        args_not_parsed: list[str],
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient.
        Writes ORCA output


        Parameters
        ----------
        settings: BasicSettings
            Object with basic settings for the run
        directory: Path
            Directory where to work in
        args_not_parsed: list[str]
            Arguments not parsed so far
        """

        # process the XYZ file
        atom_types, coordinates = xyzfile_to_at_coord(settings.xyzfile)

        # run uma
        energy, gradient = self.run_aimnet2(
            atom_types=atom_types, coordinates=coordinates, settings=settings
        )

        return energy, gradient


def main():
    """
    Main routine for execution
    """
    calculator = Aimnet2Calc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.setup(args)
    calculator.run(
        inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed
    )


# Python entry point
if __name__ == "__main__":
    main()

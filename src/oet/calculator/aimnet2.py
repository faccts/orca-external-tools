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
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from oet.core.base_calc import BaseCalc, CalculationData
from oet.core.misc import ENERGY_CONVERSION, LENGTH_CONVERSION, xyzfile_to_at_coord

try:
    # Suppress PySisyphus missing warning
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

    def set_calculator(self, model: str) -> None:
        """
        Set the calculator

        Parameters
        ----------
        model: str
            Model of the calculator
        """
        self._calc = AIMNet2Calculator(model=model)

    def setup(self, model: str, model_dir: str) -> None:
        """
        Sets the calculator. Does nothing, if it is already set.

        Parameters
        ----------
        model: str
            Model that the calculator should have
        model_dir: str
            Path to the model files

        Returns
        -------
        dict: Arguments where all entries are removed that were processed
        """
        if not self._calc:
            # Check whether models are present
            model_path = str(Path(model_dir) / Path(model))
            if os.path.isfile(model_path):
                self.set_calculator(model=model_path)
            # If not, aimnet will download them automatically
            else:
                self.set_calculator(model=model)

    @classmethod
    def extend_parser(cls, parser: ArgumentParser) -> None:
        """Add AimNet2 parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
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
            type=str,
            default=str(default_model_path),
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

        energy = float(results["energy"]) / ENERGY_CONVERSION["eV"]
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
        if not isinstance(model, str) or not isinstance(model_dir, str):
            raise RuntimeError("Problems detecting model parameters.")
        # setup calculator if not already set
        # this is important as usage on a server would otherwise cause
        # initialization with every call so that nothing is gained
        self.setup(model=model, model_dir=model_dir)
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

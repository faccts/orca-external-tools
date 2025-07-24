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
from oet.core.base_calc import CalcServer
import torch

try:
    from aimnet2calc import AIMNet2Calculator
    import numpy as np
    import torch
except ImportError:
    print(
        "[MISSING] Required module aimnet2calc not found.\n"
        "Please install the packages in the virtual environment.\n"
        "Therefore, activate the venv, got to the orca-external-tools "
        "main directory and use pip install -r requirements/aimnet2.txt"
    )


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

    # Fairchem calculator used to compute energy and grad
    _calc: AIMNet2Calculator

    @property
    def PROGRAM_KEYS(self) -> set[str]:
        """Program keys to search for in PATH"""
        return {""}

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
        #    raise FileNotFoundError(
        #        "Please provide a valid path to the models.\n"
        #        "For convenience, you can use "
        #    )
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
        charge: int,
        mult: int,
        dograd: bool,
        nthreads: int,
    ) -> tuple[float, list[float]]:
        """
        Runs an AimNet2 calculation.

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
        basemodel: str
            The UMA base model to use
        param: str
            The param to use
        dograd : bool
            Whether to compute the gradient (currently always computed)
        nthreads : int
            Number of threads to use for the calculation

        Returns
        -------
        tuple[float, list[float]]
            energy : float
                The computed energy (Eh)
            gradient : list[float]
                Flattened gradient vector (Eh/Bohr), if computed, otherwise empty.
        """
        # Conversion factors
        ENERGY_CONVERSION = {"eV": 27.21138625}
        LENGTH_CONVERSION = {"Ang": 0.529177210903}

        # set the number of threads
        torch.set_num_threads(nthreads)

        # make ase atoms object for calculation
        aimnet2_input = self.serialize_input(
            atom_types=atom_types,
            coordinates=coordinates,
            mult=mult,
            charge=charge,
            dograd=dograd,
        )

        results = self._calc(**aimnet2_input)

        energy = float(results["energy"]) / ENERGY_CONVERSION["eV"]
        gradient = []
        if (forces := results.get("forces", None)) is not None:
            # unit conversion & factor of -1 to convert from forces to gradient
            fac = -LENGTH_CONVERSION["Ang"] / ENERGY_CONVERSION["eV"]
            gradient = (np.asarray(forces) * fac).flatten().tolist()

        return energy, gradient

    def calc(
        self,
        orca_input: dict,
        directory: Path,
        clear_args: list[str],
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient.
        Writes ORCA output


        Parameters
        ----------
        orca_input: dict
            Input parameters
        directory: Path
            Directory where to work in
        clear_args: list[str]
            Arguments not parsed so far
        """
        # Get the information needed
        xyz_file = orca_input["xyz_file"]
        chrg = orca_input["chrg"]
        mult = orca_input["mult"]
        ncores = orca_input["ncores"]
        dograd = orca_input["dograd"]

        xyz_file = directory / Path(xyz_file)

        # process the XYZ file
        atom_types, coordinates = self.xyzfile_to_at_coord(xyz_file)

        # run uma
        energy, gradient = self.run_aimnet2(
            atom_types=atom_types,
            coordinates=coordinates,
            charge=chrg,
            mult=mult,
            dograd=dograd,
            nthreads=ncores,
        )

        # Delete files
        self.clean_files()

        return energy, gradient


def main():
    """
    Main routine for execution
    """
    calculator = Aimnet2Calc()
    inputfile, args, clear_args = calculator.parse_args()
    calculator.setup(args)
    calculator.run(inputfile=inputfile, settings=args, clear_args=clear_args)


# Python entry point
if __name__ == "__main__":
    main()

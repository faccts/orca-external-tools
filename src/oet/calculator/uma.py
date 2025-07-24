#!/usr/bin/env python3
"""
Calculator for using UMA (https://github.com/facebookresearch/fairchem),
compatible with ORCA's ExtTool interface.

Provides
--------
class: UmaCalc(CalcServer)
    Class for performing a UMA calculation together with ORCA
main: function
    Main function
"""
from argparse import ArgumentParser
from pathlib import Path
from oet.core.base_calc import CalcServer

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    import torch
    import numpy as np
    from ase import Atoms
except ImportError:
    print(
        "[MISSING] Required module umacalc not found.\n"
        "Please install the packages in the virtual environment.\n"
        "Therefore, activate the venv, got to the orca-external-tools "
        "main directory and use pip install -r ./requirements/uma.txt\n"
        "Also, make sure you are logged in with your Hugging Face account:\n"
        "https://fair-chem.github.io/core/install.html#access-to-gated-models-on-huggingface"
    )


class UmaCalc(CalcServer):

    # Fairchem calculator used to compute energy and grad
    _calc: FAIRChemCalculator

    @property
    def PROGRAM_KEYS(self) -> set[str]:
        """Program keys to search for in PATH"""
        return {""}

    def set_calculator(self, param: str, basemodel: str) -> None:
        """
        Set the UMA calculator used by the UmaCalc object to compute energy and grad

        Parameters
        ----------
        param: str
            parameter set to use
        basemodel: str
            UMA basemodel
        """
        predictor = pretrained_mlip.get_predict_unit(basemodel, device="cpu")
        self._calc = FAIRChemCalculator(predictor, task_name=param)

    def get_calculator(self) -> FAIRChemCalculator:
        """
        Returns UMA calculator
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
        self.set_calculator(param=args.pop("param"), basemodel=args.pop("basemodel"))
        return args

    def extend_parser(self, parser: ArgumentParser) -> None:
        """Add Uma parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        self.extend_parser_setup(parser=parser)

    def extend_parser_setup(self, parser: ArgumentParser) -> None:
        """
        Add Uma parsing options that are used for setting up the calculator.

        Parameters
        ----------
        parser: ArgumentParser
            Argument parser to extend
        """
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default="omol",
            dest="param",
            help="The uma param.",
        )
        parser.add_argument(
            "-bm",
            "--basemodel",
            type=str,
            default="uma-s-1",
            dest="basemodel",
            help="The uma basemodel.",
        )

    def extend_parser_settings(self, parser: ArgumentParser) -> None:
        """
        Add Uma parsing options that are used for calculation specific settings.

        Parameters
        ----------
        parser: ArgumentParser
            Argument parser to extend
        """
        pass

    def run_uma(
        self,
        atom_types: list[str],
        coordinates: list[tuple[float, float, float]],
        charge: int,
        mult: int,
        dograd: bool,
        nthreads: int,
    ) -> tuple[float, list[float]]:
        """
        Runs an UMA calculation.

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
            Whether to compute the gradient (currently always computed)
        nthreads : int
            Number of threads to use for the calculation

        Returns
        -------
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
        atoms = Atoms(symbols=atom_types, positions=coordinates)
        atoms.info = {"charge": charge, "spin": mult}
        atoms.calc = self._calc

        energy = atoms.get_potential_energy() / ENERGY_CONVERSION["eV"]
        gradient = []
        try:
            forces = atoms.get_forces()
            # Convert forces to gradient (-1) and unit conversion
            fac = -LENGTH_CONVERSION["Ang"] / ENERGY_CONVERSION["eV"]
            gradient = (fac * np.asarray(forces)).flatten().tolist()
        except Exception:
            # forces may not be available
            pass

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

        Returns
        -------
        float: energy
        list[float]: gradients
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
        energy, gradient = self.run_uma(
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
    calculator = UmaCalc()
    inputfile, args, clear_args = calculator.parse_args()
    calculator.setup(args)
    calculator.run(inputfile=inputfile, settings=args, clear_args=clear_args)


# Python entry point
if __name__ == "__main__":
    main()

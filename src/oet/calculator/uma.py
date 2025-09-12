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
import sys
import warnings
from oet.core.base_calc import BasicSettings, BaseCalc
from oet.core.misc import xyzfile_to_at_coord, ENERGY_CONVERSION, LENGTH_CONVERSION

try:
    # Suppress pkg_resources deprecated warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
except ImportError as e:
    print(
        f"[MISSING] Required module umacalc not found: {e}.\n"
        "Please install the packages in the virtual environment.\n"
        "Therefore, activate the venv, got to the orca-external-tools "
        "main directory and use pip install -r ./requirements/uma.txt\n"
        "Also, make sure you are logged in with your Hugging Face account:\n"
        "https://fair-chem.github.io/core/install.html#access-to-gated-models-on-huggingface"
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


class UmaCalc(BaseCalc):

    # Fairchem calculator used to compute energy and grad
    _calc: FAIRChemCalculator | None = None

    def set_calculator(self, param: str, basemodel: str, device: str) -> None:
        """
        Set the UMA calculator used by the UmaCalc object to compute energy and grad

        Parameters
        ----------
        param: str
            parameter set to use
        basemodel: str
            UMA basemodel
        device: str, default: "cpu"
            Device that should be used, e.g., cpu or cuda
        """
        predictor = pretrained_mlip.get_predict_unit(basemodel, device=device)
        self._calc = FAIRChemCalculator(predictor, task_name=param)

    def get_calculator(self) -> FAIRChemCalculator:
        """
        Returns UMA calculator
        """
        return self._calc

    def setup(self, param: str, basemodel: str, device: str) -> None:
        """
        Filters the command line arguments for setup arguments
        Returns the respective dict where the processed arguments are removed

        Parameters
        ----------
        args: dict
            Arguments provided via cmd
        param: str
            Parameter set used by fairchem
        basemode: str
            Basemodel
        device:
            device to run the calculation on

        Returns
        -------
        dict: Arguments where all entries are removed that were processed
        """
        if not self._calc:
            self.set_calculator(
                param=param,
                basemodel=basemodel,
                device=device,
            )

    def extend_parser(self, parser: ArgumentParser) -> None:
        """Add Uma parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
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
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="cpu",
            dest="device",
            help="Device to perform the calculation on.",
        )

    def run_uma(
        self,
        atom_types: list[str],
        coordinates: list[tuple[float, float, float]],
        settings: BasicSettings,
    ) -> tuple[float, list[float]]:
        """
        Runs an UMA calculation.

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
        energy : float
            The computed energy (Eh)
        gradient : list[float]
            Flattened gradient vector (Eh/Bohr), if computed, otherwise empty.
        """

        # set the number of threads
        torch.set_num_threads(settings.ncores)

        # make ase atoms object for calculation
        atoms = Atoms(symbols=atom_types, positions=coordinates)
        atoms.info = {"charge": settings.charge, "spin": settings.mult}
        atoms.calc = self._calc

        energy = atoms.get_potential_energy() / ENERGY_CONVERSION["eV"]
        gradient = []
        try:
            forces = atoms.get_forces()
            # Convert forces to gradient (-1) and unit conversion
            fac = -LENGTH_CONVERSION["Ang"] / ENERGY_CONVERSION["eV"]
            gradient = (fac * forces).flatten().tolist()
        except Exception:
            # forces may not be available
            pass

        return energy, gradient

    def calc(
        self,
        settings: BasicSettings,
        args_not_parsed: list[str],
        param: str,
        basemodel: str,
        device: str,
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient.
        Writes ORCA output


        Parameters
        ----------
        settings: BasicSettings
            Object with basic settings for the run
        args_not_parsed: list[str]
            Arguments not parsed so far
        param: str
            Parameter set used by fairchem
        basemode: str
            Basemodel
        device:
            device to run the calculation on

        Returns
        -------
        float: energy
        list[float]: gradients
        """

        # setup calculator if not already set
        # this is important as usage on a server would otherwise cause
        # initialization with every call so that nothing is gained
        self.setup(param=param, basemodel=basemodel, device=device)

        # process the XYZ file
        atom_types, coordinates = xyzfile_to_at_coord(settings.xyzfile)

        # run uma
        energy, gradient = self.run_uma(
            atom_types=atom_types, coordinates=coordinates, settings=settings
        )

        return energy, gradient


def main():
    """
    Main routine for execution
    """
    calculator = UmaCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(
        inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed
    )


# Python entry point
if __name__ == "__main__":
    main()

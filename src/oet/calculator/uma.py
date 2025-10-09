#!/usr/bin/env python3
"""
This package provides [fairchem](https://github.com/facebookresearch/fairchem) wrappers for ORCA's ExtTool interface.
Before starting to use this module, please make sure your have access to the [fairchem repository](https://huggingface.co/facebook/UMA) and logged in with your
huggingface account. For details, please see the GitHub repository or the [respective tutorials](https://fair-chem.github.io/).

Provides
--------
class: UmaCalc(CalcServer)
    Class for performing a UMA calculation together with ORCA
main: function
    Main function
"""

import sys
import warnings
from argparse import ArgumentParser
from typing import Any

from oet.core.base_calc import BaseCalc, CalculationData
from oet.core.misc import ENERGY_CONVERSION, LENGTH_CONVERSION, xyzfile_to_at_coord

try:
    # Suppress pkg_resources deprecated warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        from fairchem.core.calculate.pretrained_mlip import available_models
        from fairchem.core.units.mlip_unit.api.inference import UMATask
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
        Set the UMA calculator used by the UmaCalc object to compute energy and gradient

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

    @classmethod
    def extend_parser(cls, parser: ArgumentParser) -> None:
        """Add Uma parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        parser.add_argument(
            "-t",
            "--task",
            type=UMATask,
            choices=list(UMATask),
            default=(default_task := UMATask.OMOL),
            metavar="TASK",
            dest="param",
            help="The UMA task/parameter set name. "
            "Options: " + ", ".join(UMATask) + ". "
            f"Default: {default_task}. ",
        )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default=(default_model := "uma-s-1"),
            metavar="MODEL",
            dest="basemodel",
            choices=available_models,
            help="The UMA base model. "
            "Options: " + ", ".join(available_models) + ". "
            f"Default: {default_model}. ",
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

    def run_uma(
        self,
        atom_types: list[str],
        coordinates: list[tuple[float, float, float]],
        calc_data: CalculationData,
    ) -> tuple[float, list[float]]:
        """
        Runs an UMA calculation.

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
        energy : float
            The computed energy (Eh)
        gradient : list[float]
            Flattened gradient vector (Eh/Bohr), if computed, otherwise empty.
        """

        # set the number of threads
        torch.set_num_threads(calc_data.ncores)

        # make ase atoms object for calculation
        atoms = Atoms(symbols=atom_types, positions=coordinates)
        atoms.info = {"charge": calc_data.charge, "spin": calc_data.mult}
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

        Returns
        -------
        float: energy
        list[float]: gradients
        """
        # Get the arguments parsed as defined in extend_parser
        param = args_parsed.get("param")
        basemodel = args_parsed.get("basemodel")
        device = args_parsed.get("device")
        if (
            not isinstance(param, str)
            or not isinstance(basemodel, str)
            or not isinstance(device, str)
        ):
            raise RuntimeError("Problems handling input parameters.")
        # setup calculator if not already set
        # this is important as usage on a server would otherwise cause
        # initialization with every call so that nothing is gained
        self.setup(param=param, basemodel=basemodel, device=device)

        # process the XYZ file
        atom_types, coordinates = xyzfile_to_at_coord(calc_data.xyzfile)

        # run uma
        energy, gradient = self.run_uma(
            atom_types=atom_types, coordinates=coordinates, calc_data=calc_data
        )

        return energy, gradient


def main() -> None:
    """
    Main routine for execution
    """
    calculator = UmaCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()

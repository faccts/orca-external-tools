"""
Classes with basic functionalities provided by the oet.

Provides
--------
class: BaseCalc(ABC)
    Abstract base class for all calculators, which must be derived from it.
    Takes care of communication with ORCA and can be used in both standalone and server/client mode.
class: CalculationData
    Holds data and settings related to a specific calculation: file names, coordinates, etc.
"""

import os
import shutil
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Sequence

from oet.core.misc import (
    check_multi_progs,
    check_path,
    copy_files_to_tmpdir,
    nat_from_xyzfile,
    read_input,
    search_path,
    write_output,
)

# Full list of all available calculator types.
# Used, e.g., by otool and server.py
CALCULATOR_CLASSES = {
    "aenet": ("oet.calculator.aenet", "AenetCalc"),
    "aimnet2": ("oet.calculator.aimnet2", "Aimnet2Calc"),
    "client": ("oet.server_client.client", "client"),
    "gxtb": ("oet.calculator.gxtb", "GxtbCalc"),
    "mlatom": ("oet.calculator.mlatom", "MlatomCalc"),
    "mopac": ("oet.calculator.mopac", "MopacCalc"),
    "uma": ("oet.calculator.uma", "UmaCalc"),
    "xtb": ("oet.calculator.xtb", "XtbCalc"),
}


class CalculationData:
    """
    Holds the data for a given calculation. Performs file handling and input reading
    """

    def __init__(self, inputfile: str, program_names: set[str] | None) -> None:
        """
        Initialize the calculation data. Also, create tmp dir

        Parameters
        ----------
        inputfile: str
            Name of the input file
        program_names: list[str] | None
            List of program names that are tried to detect for settings path to program
        """
        # Path and filenames
        # ------------------
        # Original input file location
        self.orig_inputfile_path = check_path(Path(inputfile).resolve())
        # Path to the orca input if it's in a different directory, e.g.,
        # orca /path/to/input
        self.path_to_input_file = Path(inputfile).parent.resolve()
        # Basname of the calculation
        self.basename = Path(inputfile.removesuffix(".extinp.tmp")).name
        # Outputfile read by ORCA
        self.orca_engrad = Path(self.basename + ".engrad")
        # Outputfile of the program
        self.prog_out = Path(self.basename + ".tmp.out")
        # Directory the calculation was called from
        self.start_dir = Path.cwd()
        # Tmp directory to run the calculation in
        self.tmp_dir = self.start_dir / self.path_to_input_file / Path(self.basename)
        # Prog Path that is used to call external binaries (and python)
        self.prog_path: Path | None = None
        # If program_names is None or if they are not found, self.prog_path remains None
        self.set_program_path(program_names)
        # Read input file
        xyzfile, charge, mult, ncores, dograd, pointcharges = read_input(
            inputfile=self.orig_inputfile_path
        )
        # Original structure file location
        self.orig_xyzfile = check_path(Path(xyzfile).resolve())
        # Molecular charge
        self.charge = charge
        # Multiplicity
        self.mult = mult
        # Number of cores
        self.ncores = ncores
        # Do gradient?
        self.dograd = dograd
        # File with pointcharges
        self.pointcharges = pointcharges

        # Make tmp directory so that every calculation is performed in its own directory
        inp_tmp, xyz_tmp = copy_files_to_tmpdir(
            files_to_copy=[self.orig_inputfile_path, self.orig_xyzfile],
            tmp_dir=self.tmp_dir,
        )
        # Input file in tmp directory
        self.inputfile_path = inp_tmp
        # Structure file in tmp directory (should be used for calculation)
        self.xyzfile = xyz_tmp

    def remove_tmp(self) -> None:
        """
        Removes tmp dir. Goes back to starting dir, if necessary
        """
        # Go to starting directory (if not done before)
        if Path.cwd() == self.tmp_dir:
            os.chdir(self.start_dir)
        shutil.rmtree(self.tmp_dir)

    def set_program_path(self, exe_path_or_name: set[str] | str | Path | None) -> bool:
        """
        Checks for executable of program and sets self.prog_path to it.
        If nothing was found, False is returned and self.prog_path is None

        Params
        ------
        exe_path_or_name: set[str] | str | Path | None
            Full path to an executable, just its name or list of names.
            If a list is provided, the first match will set the program path.
            If just a name is provided, it is searched for and set to the name.

        Returns
        -------
        bool: True if an executable was found, False otherwise
        """
        # If exe_path_or_name is None, return False
        if not exe_path_or_name:
            return False
        # Set handling
        if isinstance(exe_path_or_name, set):
            self.prog_path = check_multi_progs(exe_path_or_name)
            return self.prog_path is not None
        # str | Path handling
        else:
            try:
                self.prog_path = search_path(exe_path_or_name)
                return True
            # The error could also be raised to the caller, but as
            # it should be standardized, it is printed here.
            except Exception as e:
                print(f"Warning: Provided executable not valid: {e}")
                return False


class BaseCalc(ABC):
    """
    Abstract base class with basic functionality.
    Wrapper classes should inherit.

    Things that must be overwritten:
    calc:
        Main routine for calculation. Receives relevant settings
        and should return and energy in Hartree and a gradient in
        Hartree/Bohr. Gradient can be empty if dograd = False.

    Things that might be overwritten:
    minimal_python_version:
        Minimal python version required
    PROGRAM_NAMES:
        Names of executable that is searched for in PATH
    extend_parser:
        Here, the parser can be extended with individual arguments
        They are provided to the calc routine unpacked
    """

    # Minimal Python version required;
    # Overwrite in wrapper if you require newer versions
    minimal_python_version: tuple[int, int] = (3, 10)

    @property
    def PROGRAM_NAMES(self) -> set[str] | None:
        """Set of executables to search for in Path."""
        return None

    @abstractmethod
    def calc(
        self,
        calc_data: CalculationData,
        args_parsed: dict[str, Any],
        args_not_parsed: list[str],
    ) -> tuple[float, list[float]]:
        """
        Main routing for calculating engrad.
        Gets information from input file and should return energy
        and optional gradient.

        Parameters
        ----------
        calc_data: CalculationData
            Object with all information about the calculation
        args_parsed: dict[str, Any]
            All arguments parsed with options defined by the subclasses
        args_not_parsed: list[str]
            Additional arguments not parsed so far

        Returns
        -------
        float: electronic energy in Hartree
        list[float]: gradient in Hartree/Bohr, leave empty if dograd=False
        """
        pass

    def run(
        self,
        inputfile: str,
        args_parsed: dict[str, Any],
        args_not_parsed: Sequence[str] = (),
        directory: Path | str | None = None,
    ) -> None:
        """
        Main routine that computes energy and gradient based on inputfile

        Parameters
        ----------
        inputfile: str
            Name of the inputfile generated by ORCA
        args_parsed: dict[str, Any]
            Arguments already parsed
        args_not_parsed: list[str], default: []
            Arguments not parsed. Might be provided directly to executing program
        directory: Path | str | None
            Where to run the calculation

        Raises
        ------
        RuntimeError: If parsing or energy/gradient calculation failed
        """
        # Check if python version matches requirements if redefined by subclass
        self._check_python_version()
        # Check if calculation directory is located somewhere else (important for server)
        start_dir = Path.cwd()
        if directory:
            directory = check_path(directory)
            os.chdir(directory)
        # Set filenames and paths according to inputfile name. Also make tmpdir
        calc_data = CalculationData(inputfile=inputfile, program_names=self.PROGRAM_NAMES)
        # Run the routine performing actual calculation
        try:
            # Go to tmp dir where the calculation should be performed
            os.chdir(calc_data.tmp_dir)
            # Perform calculation
            energy, gradient = self.calc(
                calc_data=calc_data,
                args_parsed=args_parsed,
                args_not_parsed=list(args_not_parsed),
            )
            # Go back to directory where input file was located
            os.chdir(calc_data.path_to_input_file)
        except Exception as e:
            raise RuntimeError(f"Failed to compute energy and/or gradient") from e
        # Get number of atoms
        nat = nat_from_xyzfile(calc_data.xyzfile)
        # Write ORCA input
        write_output(filename=calc_data.orca_engrad, nat=nat, etot=energy, grad=gradient)
        # Remove tmp dir
        calc_data.remove_tmp()
        # Go back to start directory
        if directory:
            os.chdir(start_dir)

    def parse_args(self, args: list[str] | None = None) -> tuple[str, dict[str, Any], list[str]]:
        """
        Main parser
        Can be extended by the subclasses with extend_parser routine

        Parameters
        ----------
        args: list[str] | None, default: None
            Optional list of arguments to parse. If not given, command line arguments from `sys.argv` are used

        Returns
        -------
        str: Name of inputfile
        dict: Dictionary of parsed arguments
        list[str]: Remaining arguments parsed by the subclasses
        """
        # Setup new parser
        parser = ArgumentParser(
            prog="ORCA otool",
            description="ORCA external tools wrapper.",
        )
        parser.add_argument("inputfile")
        # Extend parser with in subclass defined arguments
        self.extend_parser(parser)

        # Argument parsing
        try:
            args, args_not_parsed = parser.parse_known_args(args)
        except Exception as e:
            raise RuntimeError(f"Failed to parse arguments: {e}")
        # Get inputfile
        inputfile = args.inputfile
        # Remove the inputfile from aragparser as it is already parsed
        delattr(args, "inputfile")

        return inputfile, vars(args), args_not_parsed

    @classmethod
    def extend_parser(self, parser: ArgumentParser) -> None:
        """
        Subclasses override this to add custom arguments.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        pass

    def _check_python_version(self) -> None:
        """
        Checks whether the Python version matches the minimum requirement

        Raises
        ------
        RuntimeError: If minimum requirement is not satisfied
        """
        if sys.version_info < self.minimal_python_version:
            raise RuntimeError(
                f"Python version must be higher than {self.minimal_python_version[0]}.{self.minimal_python_version[1]}"
            )

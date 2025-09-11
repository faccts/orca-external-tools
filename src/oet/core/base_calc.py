"""
Classes with basic functionalities provided by the oet.

Provides
--------
class: BaseCalc(ABC)
    Class for basic calculators. Derive from this class if
    you rely purely on executables
class: CalcServer(BaseCalc)
    Extends the BaseCalc for necessary routines required for
    running on a server. Derive from this class if you plan
    using it in a server/client layout (e.g. heavy imports)
"""

from dataclasses import dataclass
import shutil
import sys
import os
from typing import Any
from abc import abstractmethod, ABC
from pathlib import Path

from oet.core.misc import (
    check_multi_progs,
    check_path,
    read_input,
    write_output,
    nat_from_xyzfile,
)
from argparse import ArgumentParser


class BasicSettings:
    """
    Class for the basic settings like file handling and input reading
    """

    def __init__(self, inputfile: str, program_names: set[str] | None) -> None:
        """
        Initialize the calculation settings. Also, create tmp dir

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
        # Basname of the calculation
        self.basename = inputfile.removesuffix(".extinp.tmp")
        # Outputfile read by ORCA
        self.orca_engrad = Path(self.basename + ".engrad")
        # Outputfile of the program
        self.prog_out = Path(self.basename + ".tmp.out")
        # Directory the calculation was called from
        self.start_dir = Path.cwd()
        # Tmp directory to run the calculation in
        self.tmp_dir = self.start_dir / Path(self.basename)
        # Prog Path that is used to call external binaries (and python)
        self.prog_path = None
        # If program_names is None or if they are not found, self.prog_path remains None
        self.set_program_path(program_names)
        # Read input file
        xyzfile, charge, mult, ncores, dograd, pointcharges = read_input(
            inputfile=self.orig_inputfile_path
        )
        # Origianl structure file location
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
        inp_tmp, xyz_tmp = self.make_tmp(
            files_to_copy=[self.orig_inputfile_path, self.orig_xyzfile],
            tmp_dir=self.tmp_dir,
        )
        # Input file in tmp directory
        self.inputfile_path = inp_tmp
        # Structure file in tmp directory (should be used for calculation)
        self.xyzfile = xyz_tmp

    def make_tmp(self, files_to_copy: list[Path], tmp_dir: Path) -> list[Path]:
        """
        Makes tmp directories and copies files

        Parameters
        ----------
        files_to_copy: list[Path]
            Paths of the files that should be copied
        tmp_dir: Path
            Path to the tmp directory

        Returns
        -------
        list[Path]: List of the Paths of the copied files
        """
        tmp_dir.mkdir(parents=True, exist_ok=True)
        final_file_paths = []
        for file_path in files_to_copy:
            new_path = tmp_dir / file_path.name
            shutil.copy2(file_path, new_path)
            final_file_paths.append(new_path)
        return final_file_paths

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
                self.prog_path = check_path(exe_path_or_name)
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
        Main routine for calculation. Recieves relevant settings
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
        settings: BasicSettings,
        directory: Path,
        args_not_parsed: list[str],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[float, list[float]]:
        """
        Main routing for calculating engrad.
        Gets information from input file and should return energy
        and optional gradient.

        Parameters
        ----------
        settings: BasicSettings
            Object with all information about the calculation
        directory: Path
            Directory where to work
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
        args_parsed: dict,
        args_not_parsed: list[str] = [],
        directory: Path | None = None,
    ) -> None:
        """
        Main routine that computes energy and gradient based on inputfile

        Parameters
        ----------
        inputfile: str
            Name of the inputfile generated by ORCA
        settings: dict
            Special settings for each calculation defined by respective arg parser extension
        args_not_parsed: list[str], default: []
            Arguments not parsed. Might be provided directly to executing program
        directory: Path
            Where to run the calculation

        Raises
        ------
        RuntimeError: If parsing or energy/gradient calculation failed
        """
        # Check if python version matches requirements if redefined by subclass
        self._check_python_version()
        # Set filenames and paths according to inputfile name. Also make tmpdir
        settings = BasicSettings(inputfile=inputfile, program_names=self.PROGRAM_NAMES)
        # Run the routine performing actual calculation
        try:
            # Go to tmp dir where the calculation should be performed
            os.chdir(settings.tmp_dir)
            # Perform calculation
            energy, gradient = self.calc(
                settings=settings,
                args_not_parsed=args_not_parsed,
                **args_parsed,
            )
            # Go back to starting dir
            os.chdir(settings.start_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to compute energy and/or gradient: {e}")
        # Get number of atoms
        nat = nat_from_xyzfile(settings.xyzfile)
        # Write ORCA input
        write_output(filename=settings.orca_engrad, nat=nat, etot=energy, grad=gradient)
        # Remove tmp dir
        settings.remove_tmp()

    def parse_args(self, input: list[str] | None = None) -> tuple[str, dict, list[str]]:
        """
        Main parser
        Can be extended by the subclasses with extend_parser routine

        Parameters
        ----------
        input: list[str] | None, default: None
            Optional list of arguments to parse. If not given, cmd arguments are used

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
            args, args_not_parsed = parser.parse_known_args(input)
        except Exception as e:
            raise RuntimeError(f"Failed to parse arguments: {e}")
        # Get inputfile
        inputfile = args.inputfile
        # Remove the inputfile from aragparser as it is already parsed
        delattr(args, "inputfile")

        return inputfile, vars(args), args_not_parsed

    def extend_parser(self, parser: ArgumentParser) -> None:
        """
        Subclasses override this to add custom arguments.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        pass

    def setup(self, args: dict) -> dict:
        """
        Only overwrite for using import heavy stuff that should run on server
        Sets class variables/objects that should remain identical for multiple calculations
        Usually the import heavy stuff
        Is called by the server and setup on start
        Afterwards, it is not changed
        Uses a dictionary to filter relevant settings, comes usually from command-line arguments
        Returns the respective dict where the processed arguments are removed

        Parameters
        ----------
        args: dict
            Arguments provided via cmd

        Returns
        -------
        dict: Arguments where all entries are removed that were processed
        """
        pass

    def _check_python_version(self) -> None:
        """
        Checks wether the Python version matches the minimum requirement

        Raises
        ------
        RuntimeError: If minimum requirement is not satisfied
        """
        if sys.version_info < self.minimal_python_version:
            raise RuntimeError(
                f"Python version must be higher than {self.minimal_python_version[0]}.{self.minimal_python_version[1]}"
            )


class CalcServer(BaseCalc):
    """
    Class for building a calculator that can also be used by a server.
    """

    def extend_parser(self, parser: ArgumentParser) -> None:
        """
        Extends the parser by options defined by the subclasses

        Parameters
        ----------
        parser: ArgumentParser
            Parser that is extended
        """
        self.extend_parser_setup(parser)
        self.extend_parser_settings(parser)

    def extend_parser_setup(self, parser: ArgumentParser) -> None:
        """
        Method that extends parser for setup related arguments

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        pass

    def extend_parser_settings(self, parser: ArgumentParser) -> None:
        """
        Method that extends parser for calculation settings related arguments

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        pass

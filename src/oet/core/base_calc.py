"""
Classes with basic functionalities provided by the oet.
They set the framework for easy implementation of new methods.

Provides
--------
class: CalcResult
    Results of a calculation as returned by BaseCalc.calc.
class: CalculationData
    Holds data and settings related to a specific calculation: file names, coordinates, etc.
class: BaseCalc(ABC)
    Abstract base class for all calculators, which must be derived from it.
    Takes care of communication with ORCA and can be used in both standalone and server/client mode.
"""

import os
import shutil
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from oet import __version__ as version
from oet.core.definitions import Property
from oet.core.misc import (
    check_multi_progs,
    check_path,
    copy_files_to_tmpdir,
    nat_from_xyzfile,
    print_filecontent,
    read_input,
    search_path,
    write_hess_output,
    write_output,
)

# Full list of all available calculator types.
# Add a new calculator here
# Used, e.g., by oet executable and server.py
CALCULATOR_CLASSES = {
    "aenet": ("oet.calculator.aenet", "AenetCalc"),
    "aimnet2": ("oet.calculator.aimnet2", "Aimnet2Calc"),
    "gxtb": ("oet.calculator.gxtb", "GxtbCalc"),
    "mace": ("oet.calculator.mace", "MaceCalc"),
    "mlatom": ("oet.calculator.mlatom", "MlatomCalc"),
    "mopac": ("oet.calculator.mopac", "MopacCalc"),
    "uma": ("oet.calculator.uma", "UmaCalc"),
    "xtb": ("oet.calculator.xtb", "XtbCalc"),
}


@dataclass
class CalculationResult:
    """
    Results of a calculation, returned by BaseCalc.calc.

    Attributes
    ----------
    energy: float
        Electronic energy in Hartree
    gradient: list[float]
        Gradient in Hartree/Bohr, leave empty if dograd=False
    hessian: list[float]
        Flattened row-major Hessian in Hartree/Bohr^2, leave empty if dohess=False
    """

    energy: float
    gradient: list[float] = field(default_factory=list)
    hessian: list[float] = field(default_factory=list)


class CalculationData:
    """
    Holds the data for a given calculation. Performs file handling and input reading
    """

    def __init__(self, inputfile: str, program_names: Sequence[str] | None) -> None:
        """
        Initialize the calculation data. Also, create tmp dir

        Parameters
        ----------
        inputfile: str
            Name of the input file
        program_names: Sequence[str] | None
            List of program names that are tried to detect for settings path to program
        """
        # Store the original input file location
        # This should not be needed after reading
        self._input_file: Path = check_path(Path(inputfile).resolve())

        # Read the input file
        # ------------------
        xyzfile, charge, mult, ncores, dograd, pointcharges, dohess = read_input(
            inputfile=self._input_file
        )
        # Original structure file location
        # It is relative to the input file directory (or already an absolute path).
        # This should not be touched after copying to tmp dir
        self._orig_xyzfile: Path = check_path(Path(self._input_file.parent / xyzfile).resolve())
        # Molecular charge
        self.charge: int = charge
        # Multiplicity
        self.mult: int = mult
        # Number of cores
        self.ncores: int = ncores
        # Do gradient?
        self.dograd: bool = dograd
        # Do Hessian?
        self.dohess: bool = dohess
        # Original point charge file location (if any)
        # This should not be touched after copying to tmp dir
        self._orig_pointcharges: Path | None = (
            check_path(Path(pointcharges).resolve()) if pointcharges else None
        )
        # Number of atoms (needed to write output for ORCA)
        self.natoms = nat_from_xyzfile(self._orig_xyzfile)

        # Paths and filenames
        # -------------------
        # Path to the orca input if it's in a different directory, e.g.,
        # orca /path/to/input
        self.orca_input_dir: Path = self._input_file.parent
        # Basename of the calculation (this is usually the ORCA BaseName + "_EXT")
        self.basename: str = self._orig_xyzfile.name.removesuffix(".xyz")
        # Directory the calculation was called from
        self.start_dir: Path = Path.cwd()
        # Tmp directory to run the calculation in
        self.tmp_dir: Path = self.orca_input_dir / self.basename
        # Output file read by ORCA
        self.orca_engrad = self.orca_input_dir / (self.basename + ".engrad")
        # Hessian output file read by ORCA (only written if self.dohess)
        self.orca_hess = self.orca_input_dir / (self.basename + ".hess")
        # Temporary output file for the external program
        # At the end of the run. this will be printed to STDOUT and deleted
        self.output_file = self.tmp_dir / (self.basename + ".out")
        # Prog Path that is used to call external binaries (and python)
        self.prog_path: Path | None = None
        # If program_names is None or if they are not found, self.prog_path remains None
        self.set_program_path(program_names)

        # Set up tmp dir and copy files
        # -----------------------------
        files_to_copy = [self._orig_xyzfile]
        if self._orig_pointcharges:
            files_to_copy.append(self._orig_pointcharges)
        # Make tmp directory so that every calculation is performed in its own directory
        copied_files = copy_files_to_tmpdir(
            files_to_copy=files_to_copy,
            tmp_dir=self.tmp_dir,
        )
        # Structure file in tmp directory (should be used for calculation)
        self.xyzfile: Path = copied_files[0]
        # Point charge file in tmp directory (should be used for calculation)
        self.pointcharges: Path | None = copied_files[1] if self._orig_pointcharges else None

    def remove_tmp(self) -> None:
        """
        Removes tmp dir. Goes back to starting dir, if necessary
        """
        # Go to starting directory (if not done before)
        if Path.cwd() == self.tmp_dir:
            os.chdir(self.start_dir)
        shutil.rmtree(self.tmp_dir)

    def set_program_path(self, exe_path_or_name: Sequence[str] | str | Path | None) -> bool:
        """
        Checks for executable of program and sets self.prog_path to it.
        If nothing was found, False is returned and self.prog_path is None

        Params
        ------
        exe_path_or_name: Sequence[str] | str | Path | None
            Full path to an executable, just its name or list of names.
            If a list is provided, the first match will set the program path.
            If just a name is provided, it is searched for and set it to the path.

        Returns
        -------
        bool: True if an executable was found, False otherwise
        """
        # If exe_path_or_name is None or empty, return False
        if not exe_path_or_name:
            return False
        # str | Path handling
        if isinstance(exe_path_or_name, (str, Path)):
            try:
                self.prog_path = search_path(exe_path_or_name)
                return True
            # The error could also be raised to the caller, but as
            # it should be standardized, it is printed here.
            except Exception as e:
                print(f"Warning: Provided executable not valid: {e}")
                return False
        # Sequence handling
        else:
            self.prog_path = check_multi_progs(exe_path_or_name)
            return self.prog_path is not None


class BaseCalc(ABC):
    """
    Abstract base class with basic functionality.
    Wrapper classes should inherit.

    Things that must be overwritten:
    calc:
        Main routine for calculation. Receives relevant settings
        and should return an energy in Hartree, a gradient in
        Hartree/Bohr, and a Hessian in Hartree/Bohr^2. Gradient can be
        empty if dograd = False, Hessian can be empty if dohess = False.

    Things that might be overwritten:
    minimal_python_version:
        Minimal python version required, default: 3.10
    PROGRAM_NAMES:
        Names of executable that is searched for in PATH
    extend_parser:
        Here, the parser can be extended with individual arguments
        They are provided to the calc routine unpacked
    DEFAULT_PROPERTIES:
        The properties every calculator has per default.
        Can be overwritten by the calculators.
    """

    # Minimal Python version required;
    # Overwrite in wrapper if you require newer versions
    minimal_python_version: tuple[int, int] = (3, 10)

    # A set of properties that will per default be assumed by every calculator to support.
    # Will be used to initialize `self.available_properties`.
    DEFAULT_PROPERTIES = frozenset(
        {
            Property.ENERGY,
            Property.GRADIENT,
        }
    )

    def __init__(self) -> None:
        """
        Initialize object variables.
        """
        # The properties that are supported by a specific calculator.
        # Per default, energies and gradients are assumed to be supported.
        # Can be adjusted by adding/removing specific properties of the calculator instance.
        self.available_properties: set[Property] = set(self.DEFAULT_PROPERTIES)

    @property
    def PROGRAM_NAMES(self) -> list[str] | None:
        """List of executables to search for in Path."""
        return None

    @abstractmethod
    def calc(
        self,
        calc_data: CalculationData,
        args_parsed: Namespace,
        args_not_parsed: list[str],
    ) -> CalculationResult:
        """
        Main routing for calculating energy, gradient and Hessian.
        Gets information from input file and should return energy
        and optional gradient and Hessian.

        Parameters
        ----------
        calc_data: CalculationData
            Object with all information about the calculation
        args_parsed: Namespace
            All arguments parsed with options defined by the subclasses
        args_not_parsed: list[str]
            Additional arguments not parsed so far

        Returns
        -------
        CalcResult: energy in Hartree, and, if requested, gradient in Hartree/Bohr
            and flattened row-major Hessian in Hartree/Bohr^2
        """

    def run(
        self,
        inputfile: str,
        args_parsed: Namespace,
        args_not_parsed: Sequence[str] = (),
        directory: Path | str | None = None,
    ) -> None:
        """
        Main routine that computes energy and gradient based on inputfile

        Parameters
        ----------
        inputfile: str
            Name of the inputfile generated by ORCA
        args_parsed: Namespace
            Arguments already parsed
        args_not_parsed: Sequence[str], default: ()
            Arguments not parsed. Might be provided directly to executing program
        directory: Path | str | None
            Where to run the calculation

        Raises
        ------
        RuntimeError: If parsing or energy/gradient/Hessian calculation failed,
            or if a Hessian was requested but this calculator doesn't support it
        """
        # Check if python version matches requirements if redefined by subclass
        self._check_python_version()
        # Check if calculation directory is located somewhere else (important for server)
        start_dir = Path.cwd()
        if directory:
            directory = check_path(directory)
            os.chdir(directory)
        try:
            # Set filenames and paths according to inputfile name. Also make tmpdir
            calc_data = CalculationData(inputfile=inputfile, program_names=self.PROGRAM_NAMES)
            # Abort early if a Hessian was requested but this calculator doesn't support it.
            # No calculation was attempted, so there is nothing to debug: remove the tmp dir
            # that CalculationData already created (unlike a genuine runtime failure below,
            # which leaves it on disk for debugging).
            if calc_data.dohess and not Property.HESSIAN in self.available_properties:
                calc_data.remove_tmp()
                raise RuntimeError(
                    f"{type(self).__name__} does not support Hessian calculations, "
                    "but one was requested in the ORCA input file."
                )
            # Abort early if a gradient calculation was requested and this calculator doesn't support it.
            if calc_data.dograd and not Property.GRADIENT in self.available_properties:
                calc_data.remove_tmp()
                raise RuntimeError(
                    f"{type(self).__name__} does not support gradient calculations, "
                    "but one was requested in the ORCA input file."
                )
            # Run the routine performing actual calculation
            try:
                # Go to tmp dir where the calculation should be performed
                os.chdir(calc_data.tmp_dir)
                # Perform calculation
                result = self.calc(
                    calc_data=calc_data,
                    args_parsed=args_parsed,
                    args_not_parsed=list(args_not_parsed),
                )
                # Go back to directory where input file was located
                os.chdir(calc_data.orca_input_dir)
            except Exception as e:
                raise RuntimeError("Failed to compute energy and/or gradient") from e
            # Write output for ORCA. A failure here (e.g. an unexpectedly-sized Hessian) is
            # left uncaught, so it surfaces as-is; the starting directory is still restored
            # below via the outer finally.
            write_output(
                filename=calc_data.orca_engrad,
                nat=calc_data.natoms,
                etot=result.energy,
                grad=result.gradient,
            )
            if calc_data.dohess:
                write_hess_output(
                    filename=calc_data.orca_hess, nat=calc_data.natoms, hessian=result.hessian
                )
            # Write program output to STDOUT
            if calc_data.output_file.exists():
                print_filecontent(calc_data.output_file)
            # Remove tmp dir
            calc_data.remove_tmp()
        finally:
            # Go back to start directory, whatever happened above
            if directory:
                os.chdir(start_dir)

    def parse_args(self, input_args: list[str] | None = None) -> tuple[str, Namespace, list[str]]:
        """
        Main parser
        Can be extended by the subclasses with extend_parser routine

        Parameters
        ----------
        args: list[str] | None, default: None
            Optional list of arguments to parse. If not given, command line arguments from `sys.argv` are used

        Returns
        -------
        str
            Name of inputfile
        Namespace
            The parsed arguments
        list[str]
            Remaining arguments parsed by the subclasses
        """
        # Setup new parser
        parser = ArgumentParser(
            prog="oet",
            description="ORCA external tools wrapper.",
        )
        parser.add_argument("inputfile")
        parser.add_argument("--version", action="version", version=version)
        # Extend parser with in subclass defined arguments
        self.extend_parser(parser)

        # Argument parsing
        try:
            args, args_not_parsed = parser.parse_known_args(input_args)
        except Exception as e:
            raise RuntimeError(f"Failed to parse arguments: {e}")
        # Get inputfile
        inputfile = args.inputfile
        # Remove the inputfile from aragparser as it is already parsed
        delattr(args, "inputfile")

        return inputfile, args, args_not_parsed

    @classmethod
    def extend_parser(self, parser: ArgumentParser) -> None:
        """
        Subclasses override this to add custom arguments.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """

    def release(self) -> None:
        """
        Release device-side resources held by this calculator.

        No-op by default. Calculators that hold device-resident state
        (GPU models, compiled artifacts, etc.) override to move models
        to CPU, drop references, and call torch.cuda.empty_cache().
        Server-mode worker cache eviction calls this before evicting
        a cached calculator. Must be idempotent.
        """

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

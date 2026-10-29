#!/usr/bin/env python3
"""
Wrapper for tblite (github.com/tblite/tblite), compatible with the 
ExtTool interface.

Provides
--------
class: TbliteCalc(BaseCalc)
    Class for performing a tblite calculation together with ORCA
main: function
    Main function
"""

import os
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path

from oet.core.base_calc import BaseCalc, CalculationData
from oet.core.misc import (
    check_path,
    mult_to_nue,
    nat_from_xyzfile,
    run_command,
)

# Header of a data block in the tblite tag file, e.g. "gradient           :real:2:3,42"
# Groups: name, data type, rank, dimensions
TAG_HEADER = re.compile(r"^(\S+)\s*:(\w+):(\d+):([\d,]*)\s*$")


class TbliteCalc(BaseCalc):
    @property
    def PROGRAM_NAMES(self) -> list[str]:
        """Program names to search for in PATH"""
        return ["tblite", "otool_tblite"]

    @classmethod
    def extend_parser(cls, parser: ArgumentParser) -> None:
        """Add tblite parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        parser.add_argument("-e", "--exe", dest="prog", help="Path to the tblite executable")

    def read_tag_file(self, tag_file: str | Path) -> dict[str, list[float]]:
        """
        Read a tblite tag file, which stores the results as blocks of the form

            gradient           :real:2:3,42
             <values, several per line>

        Only blocks of type 'real' are kept. Fortran writes the arrays in column-major
        order, so the gradient is already flattened as A1X, A1Y, A1Z, A2X, ...

        Parameters
        ----------
        tag_file: str | Path
            Tag file written by tblite (via '--grad')

        Returns
        -------
        dict[str, list[float]]
            The values of each block, keyed by the block name
        """
        blocks: dict[str, list[float]] = {}
        name: str | None = None
        tag_path = check_path(tag_file)
        with tag_path.open() as f:
            for line in f:
                if match := TAG_HEADER.match(line):
                    # Skip the data of all blocks that are not real-valued
                    name = match.group(1) if match.group(2) == "real" else None
                    if name:
                        blocks[name] = []
                elif name:
                    blocks[name] += [float(i) for i in line.split()]
        return blocks

    def read_energy(self, output_file: str | Path) -> float:
        """
        Read the total energy from the tblite output. Only needed if no gradient is
        requested, since otherwise the energy is taken from the tag file.

        Parameters
        ----------
        output_file: str | Path
            Output file of the tblite run

        Returns
        -------
        float
            The computed energy

        Raises
        ------
        ValueError
            If no total energy was found
        """
        output_path = check_path(output_file)
        with output_path.open() as f:
            for line in f:
                if "total energy" not in line.lower():
                    continue
                # Return the first field of the line that is a number
                for field in line.split():
                    try:
                        return float(field)
                    except ValueError:
                        continue
        raise ValueError(f"Total energy not found in file {output_file}")

    def read_tbliteout(
        self, calc_data: CalculationData, tag_file: str | Path, natoms: int
    ) -> tuple[float, list[float]]:
        """
        Read the results of a tblite run

        Parameters
        ----------
        calc_data: CalculationData
            Calculation data
        tag_file: str | Path
            Tag file written by tblite (only read if a gradient was requested)
        natoms: int
            Number of atoms in the system

        Returns
        -------
        float
            The computed energy
        list[float]
            The gradient (X,Y,Z) for each atom

        Raises
        ------
        ValueError
            If the energy or the gradient is missing or incomplete
        """
        if not calc_data.dograd:
            return self.read_energy(calc_data.output_file), []

        blocks = self.read_tag_file(tag_file)
        for key in ("energy", "gradient"):
            if not blocks.get(key):
                raise ValueError(f"No '{key}' block found in tag file {tag_file}")
        gradient = blocks["gradient"]
        if len(gradient) != 3 * natoms:
            raise ValueError(
                f"Number of gradient entries: {len(gradient)} does not match 3x number of atoms: {natoms}"
            )
        return blocks["energy"][0], gradient

    def run_tblite(
        self,
        calc_data: CalculationData,
        tag_file: str | Path,
        args: list[str],
    ) -> None:
        """
        Run the tblite program and redirect its STDOUT and STDERR to a logfile.

        Parameters
        ----------
        calc_data: CalculationData
            Object with calculation data for the run
        tag_file: str | Path
            File the results are written to by tblite
        args: list[str]
            Arguments not parsed so far
        """
        # tblite is parallelized with OpenMP only and has no command-line option for it
        os.environ["OMP_NUM_THREADS"] = str(calc_data.ncores)

        run_args = ["run", "--charge", str(calc_data.charge)]
        # tblite expects the number of unpaired electrons and defaults to a spin-restricted
        # calculation, so '--spin' is only set where it is actually needed.
        nue = mult_to_nue(calc_data.mult)
        if nue:
            run_args += ["--spin", str(nue)]
        if calc_data.dograd:
            run_args += ["--grad", str(tag_file)]
        # All options have to be passed before the structure file
        run_args += args + [str(calc_data.xyzfile)]

        if not calc_data.prog_path:
            raise RuntimeError("Path to program is None.")
        run_command(calc_data.prog_path, calc_data.output_file, run_args)

    def calc(
        self, calc_data: CalculationData, args_parsed: Namespace, args_not_parsed: list[str]
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient.
        Writes ORCA output

        Parameters
        ----------
        calc_data: CalculationData
            Object with calculation data for the run
        args_parsed: Namespace
            Arguments parsed as defined in extend_parser
        args_not_parsed: list[str]
            Arguments not parsed so far

        Returns
        -------
        float
            The computed energy (Eh)
        list[float]
            Flattened gradient vector (Eh/Bohr), if computed, otherwise empty
        """
        # Get parsed options
        prog = args_parsed.prog
        # Set and check the program path if its executable
        calc_data.set_program_path(prog)
        if calc_data.prog_path:
            print(f"Using executable {calc_data.prog_path}")
        else:
            raise FileNotFoundError(
                f"Could not find a valid executable from standard program names: {self.PROGRAM_NAMES}"
            )

        # File the results are written to
        tag_file = f"{calc_data.basename}.tag"

        # run tblite
        self.run_tblite(
            calc_data=calc_data,
            tag_file=tag_file,
            args=args_not_parsed,
        )

        # get the number of atoms from the xyz file
        natoms = nat_from_xyzfile(xyz_file=calc_data.xyzfile)

        # parse the tblite output
        energy, gradient = self.read_tbliteout(
            calc_data=calc_data,
            tag_file=tag_file,
            natoms=natoms,
        )

        return energy, gradient


def main() -> None:
    """
    Main routine for execution
    """
    calculator = TbliteCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()

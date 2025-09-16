#!/usr/bin/env python3
"""
This is a simple wrapper for the xtb binary (github.com/grimme-lab/xtb), compatible with ORCA's ExtTool interface.
It is mostly used for testing purposes, since ORCA has a native interface to xtb.

Provides
--------
class: XtbCalc(BaseCalc)
    Class for performing a xtb calculation together with ORCA
main: function
    Main function
"""

from argparse import ArgumentParser
from typing import Any

from oet.core.base_calc import BaseCalc, BasicSettings
from oet.core.misc import (
    check_path,
    mult_to_nue,
    nat_from_xyzfile,
    print_filecontent,
    run_command,
)


class XtbCalc(BaseCalc):
    @property
    def PROGRAM_NAMES(self) -> set[str]:
        """Program keys to search for in PATH"""
        return {"xtb", "otools_xtb"}

    def extend_parser(self, parser: ArgumentParser) -> None:
        """Add xtb parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        parser.add_argument("-e", "--exe", dest="prog", help="Path to the xtb executable")

    def read_xtbout(self, settings: BasicSettings, natoms: int) -> tuple[float, list[float]]:
        """
        Read the output from XTB

        Parameters
        ----------
        settings: BasicSettings
            Basic calculation settings
        natoms
            number of atoms in the system

        Returns
        -------
        energy: float
            The computed energy
        gradient: list[float]
            The gradient (X,Y,Z) for each atom
        """
        xtbgrad = f"{settings.basename}.gradient"
        energy = None
        gradient = []
        # read the energy from the output file
        xtbout = check_path(settings.prog_out)
        with xtbout.open() as f:
            for line in f:
                if "TOTAL ENERGY" in line:
                    energy = float(line.split()[3])
                    break
        # read the gradient from the .gradient file
        if settings.dograd:
            xtbgrad_path = check_path(xtbgrad)
            natoms_read = 0
            with xtbgrad_path.open() as f:
                for line in f:
                    if "$grad" in line:
                        break
                for line in f:
                    fields = line.split()
                    if len(fields) == 4:
                        natoms_read += 1
                    elif len(fields) == 3:
                        gradient += [float(i) for i in fields]
                    elif "$end" in line:
                        break
                if natoms_read != natoms:
                    print(
                        f"Number of atoms read: {natoms_read} does not match the expected: {natoms}"
                    )
                    exit(1)
                if len(gradient) != 3 * natoms:
                    print(
                        f"Number of gradient entries: {len(gradient)} does not match 3x number of atoms: {natoms}"
                    )
                    exit(1)
        if not energy:
            raise ValueError(f"Total energy not found in file {settings.prog_out}")
        return energy, gradient

    def run_xtb(
        self,
        settings: BasicSettings,
        args: list[str],
    ) -> None:
        """
        Run the xtb program with the given input file and redirect its STDOUT and STDERR to a logfile.

        Parameters
        ----------
        args: list[str]
            Arguments not parsed so far
        settings: BasicSettings
            Object with basic settings for the run
        """
        args += [
            str(i)
            for i in [
                settings.xyzfile,
                "-c",
                settings.charge,
                "-P",
                settings.ncores,
                "--namespace",
                settings.basename,
            ]
        ]
        nue = mult_to_nue(settings.mult)
        if nue:
            args += ["-u", str(nue)]
        if settings.dograd:
            args += ["--grad"]
        if not settings.prog_path:
            raise RuntimeError("Path to program is None.")
        run_command(settings.prog_path, settings.prog_out, args)

    def calc(
        self, settings: BasicSettings, args_parsed: dict[str, Any], args_not_parsed: list[str]
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient.
        Writes ORCA output

        Parameters
        ----------
        settings: BasicSettings
            Object with basic settings for the run
        args_parsed: dict[str, Any]
            Arguments parsed as defined in extend_parser
        args_not_parsed: list[str]
            Arguments not parsed so far

        Returns
        -------
        float: energy
        list[float]: gradients
        """
        # Get parsed options
        prog = args_parsed.get("prog")
        # Set and check the program path if its executable
        settings.set_program_path(prog)
        if settings.prog_path:
            print(f"Using executable {settings.prog_path}")
        else:
            raise FileNotFoundError(
                f"Could not find a valid executable from standard program names: {self.PROGRAM_NAMES}"
            )

        # run xtb
        self.run_xtb(
            settings=settings,
            args=args_not_parsed,
        )

        # get the number of atoms from the xyz file
        natoms = nat_from_xyzfile(xyz_file=settings.xyzfile)

        # parse the xtb output
        energy, gradient = self.read_xtbout(settings=settings, natoms=natoms)

        # Print filecontent
        print_filecontent(outfile=settings.prog_out)

        return energy, gradient


def main() -> None:
    """
    Main routine for execution
    """
    calculator = XtbCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()

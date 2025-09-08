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
from pathlib import Path
from oet.core.base_calc import BaseCalc
from oet.core.misc import run_command, mult_to_nue, nat_from_xyzfile, print_filecontent, check_path


class XtbCalc(BaseCalc):
    @property
    def PROGRAM_KEYS(self) -> set[str]:
        """Program keys to search for in PATH"""
        return {"xtb", "otools_xtb"}

    def extend_parser(self, parser: ArgumentParser):
        """Add xtb parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        parser.add_argument(
            "-e", "--exe", dest="prog", help="Path to the xtb executable"
        )

    def read_xtbout(self, natoms: int, dograd: bool) -> tuple[float, list[float]]:
        """
        Read the output from XTB

        Parameters
        ----------
        natoms
            number of atoms in the system
        dograd
            whether to read the gradient

        Returns
        -------
        energy: float
            The computed energy
        gradient: list[float]
            The gradient (X,Y,Z) for each atom
        """
        xtbgrad = f"{self.basename}.gradient"
        energy = None
        gradient = []
        # read the energy from the output file
        xtbout = check_path(self.prog_out)
        with xtbout.open() as f:
            for line in f:
                if "TOTAL ENERGY" in line:
                    energy = float(line.split()[3])
                    break
        # read the gradient from the .gradient file
        if dograd:
            xtbgrad = check_path(xtbgrad)
            natoms_read = 0
            with xtbgrad.open() as f:
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
        return energy, gradient

    def run_xtb(
        self,
        xyz_file: str,
        chrg: int,
        mult: int,
        ncores: int,
        dograd: bool,
        args: list[str],
    ) -> None:
        """
        Run the xtb program with the given input file and redirect its STDOUT and STDERR to a logfile.

        Parameters
        ----------
        xyz_file: str
            Filename of xyz structure
        chrg: int
            Molecular charge
        mult: int
            Multiplicity
        ncores: int
            Number of cores to use
        dograd: bool
            Whether to do a gradient calculation or not
        args: list[str]
            Arguments not parsed so far
        """
        args += [
            str(i)
            for i in [xyz_file, "-c", chrg, "-P", ncores, "--namespace", self.basename]
        ]
        nue = mult_to_nue(mult)
        if nue:
            args += ["-u", str(nue)]
        if dograd:
            args += ["--grad"]
        run_command(self.prog_path, self.prog_out, args)

    def calc(
        self, orca_input: dict, directory: Path, args_not_parsed: list[str], prog: str
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
        args_not_parsed: list[str]
            Arguments not parsed so far
        prog: str
            Which program executable to use

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
        # Set and check the program path if its executable
        self.set_program_path(prog)
        print("Using executable ", self.prog_path)

        # run xtb
        self.run_xtb(
            xyz_file=xyz_file,
            chrg=chrg,
            mult=mult,
            ncores=ncores,
            dograd=dograd,
            args=args_not_parsed,
        )

        # get the number of atoms from the xyz file
        natoms = nat_from_xyzfile(xyz_file=xyz_file)

        # parse the xtb output
        energy, gradient = self.read_xtbout(natoms=natoms, dograd=dograd)

        # Print filecontent
        print_filecontent(outfile=self.prog_out)

        # Delete files
        self.clean_files()

        return energy, gradient


def main():
    """
    Main routine for execution
    """
    calculator = XtbCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, settings=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
This is a simple wrapper for the g-xTB binary (github.com/grimme-lab/g-xTB), compatible with ORCA's ExtTool interface.
Note that this is currently a development version of g-xTB and that the final implementation will be available via tblite.
It currently runs only serial due to technical limitations of the development version.

Provides
--------
class: GxtbCalc(BaseCalc)
    Class for performing a g-xTB calculation together with ORCA
main: function
    Main function
"""
import shutil
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from oet.core.base_calc import BaseCalc


class GxtbCalc(BaseCalc):
    @property
    def PROGRAM_KEYS(self) -> set[str]:
        """Program keys to search for in PATH"""
        return {"gxtb", "g-xTB", "g-xtb"}

    def extend_parser(self, parser: ArgumentParser):
        """Add gxtb parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        parser.add_argument(
            "-x", "--exe", dest="prog", help="Path to the gxtb executable"
        )

    def run_gxtb(
        self,
        xyz_file: str,
        dograd: bool,
        ncores: int,
        args: list[str],
    ) -> None:
        """
        Run the gxtb program and redirect its STDOUT and STDERR to a file.

        Parameters
        ----------
        xyz_file : str
            name of the XYZ file
        dograd : bool
            whether to compute the gradient
        ncores: int
            number of cores to use
        args : list[str, ...]
            additional arguments to pass to gxtb
        """

        # Set number of cores by setting OMP_NUM_THREADS
        os.environ["OMP_NUM_THREADS"] = f"{ncores},1"

        args += [str(i) for i in ["-c", xyz_file]]

        if dograd:
            args += ["-grad"]

        self.run_command(self.prog_path, self.prog_out, args)

        return

    def read_gxtbout(
        self, energy_out: str | Path, grad_out: str | Path, natoms: int, dograd: bool
    ) -> tuple[float, list[float]]:
        """
        Read the output from gxtb

        Parameters
        ----------
        energy_out: str | Path
            file with energy
        grad_out: str | Path
            file with gradient
        natoms: int
            number of atoms in the system
        dograd: bool
            whether to read the gradient

        Returns
        -------
        energy: float
            The computed energy
        gradient: list[float]
            The gradient (X,Y,Z) for each atom
        """
        energy = None
        gradient = []
        # read the energy from the output file
        energy_path = self.check_path(energy_out)
        with energy_path.open() as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip() == "$energy":
                    # read the next line and split into values
                    parts = lines[i + 1].split()
                    # return the second value as float
                    energy = float(parts[1])
                    break

        if energy == None:
            raise ValueError("Energy couldn't be found on gxtb output file.")
        # read the gradient from the .gradient file
        if dograd:
            grad_path = self.check_path(grad_out)
            natoms_read = 0
            with grad_path.open() as f:
                for line in f:
                    if "$grad" in line:
                        break
                for line in f:
                    fields = line.split()
                    if len(fields) == 4:
                        natoms_read += 1
                    elif len(fields) == 3:
                        gradient += [float(i.replace("D", "E")) for i in fields]
                    elif "$end" in line:
                        break
                if natoms_read != natoms:
                    print(
                        f"Number of atoms read: {natoms_read} does not match the expected: {natoms}"
                    )
                    sys.exit(1)
                if len(gradient) != 3 * natoms:
                    print(
                        f"Number of gradient entries: {len(gradient)} does not match 3x number of atoms: {natoms}"
                    )
                    sys.exit(1)

        return energy, gradient

    def calc(
        self, orca_input: dict, directory: Path, clear_args: list[str], *, prog: str
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient.
        Writes ORCA output

        Parameters
        ----------
        orca_input: dict
            Input written by ORCA
        directory: Path
            Directory where to work in
        clear_args: list[str]
            Arguments not parser so far
        prog: str
            Executable to gxtb

        Returns
        -------
        float: energy
        list[float]: gradient; empty if not calculated
        """
        # Get the information needed
        xyz_file = orca_input["xyz_file"]
        chrg = orca_input["chrg"]
        mult = orca_input["mult"]
        dograd = orca_input["dograd"]
        ncores = orca_input["ncores"]
        xyz_file = directory / Path(xyz_file)
        # Set and check the program path if its executable
        self.set_program_path(prog)
        print("Using executable ", self.prog_path)

        # tmp directory named after basename
        tmp_dir = Path(self.basename)

        # make tmp file and copy xyz
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Copy input file(s) to work_dir
        shutil.copy(xyz_file, tmp_dir)

        # Change current directory to work_dir
        base_dir = Path.cwd()
        os.chdir(tmp_dir)

        # write .CHRG and .UHF file
        self.write_to_file(content=chrg, file=".CHRG")
        self.write_to_file(content=self.mult_to_nue(mult), file=".UHF")

        # run gxtb
        self.run_gxtb(xyz_file=xyz_file, dograd=dograd, ncores=ncores, args=clear_args)

        # get the number of atoms from the xyz file
        natoms = self.nat_from_xyzfile(xyz_file=xyz_file)

        # energy and gradient file
        energy_out = "energy"
        gradient_out = "gradient"

        # parse the gxtb output
        energy, gradient = self.read_gxtbout(
            energy_out=energy_out, grad_out=gradient_out, natoms=natoms, dograd=dograd
        )

        # print the output file to STDOUT
        self.print_filecontent(outfile=self.prog_out)

        # go back to parent dir
        os.chdir(base_dir)

        # remove tmp directory
        shutil.rmtree(tmp_dir)

        return energy, gradient


def main():
    """
    Main routine for execution
    """
    calculator = GxtbCalc()
    inputfile, args, clear_args = calculator.parse_args()
    calculator.run(inputfile=inputfile, settings=args, clear_args=clear_args)


# Python entry point
if __name__ == "__main__":
    main()

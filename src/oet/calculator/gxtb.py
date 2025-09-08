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
from oet.core.misc import run_command, write_to_file, mult_to_nue, nat_from_xyzfile, print_filecontent, check_path, check_file


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
        parser.add_argument(
            "-p",
            metavar="gxtb_parameterfile",
            dest="gxtb_parameterfile",
            help="path to the gxtb parameterfile"
        )
        parser.add_argument(
            "-e",
            metavar="eeq_parameterfile",
            dest="eeq_parameterfile",
            help="path to the eeq parameterfile"
        )
        parser.add_argument(
            "-b",
            metavar="basis_parameterfile",
            dest="basis_parameterfile",
            help="path to the basis parameterfile"
        )

    def check_parameter_files(self, file_path: str | None, filename: str) -> Path:
        """
        Check a parameter file. Looks first for CLAs, then GXTBHOME, 
        then HOME, and finally the current working directory.
        Terminates the program if the file cannot be found.

        Parameters
        ----------
        file_path: str | None
            CLA. None if no CLA was given
        filename: str
            filename of the parameterfile to look for

        Returns
        -------
        Path: Path to the parameterfile
        """
        # First the path given via cmd
        if file_path:
            param_file = Path(file_path).expanduser().resolve()
            if check_file(param_file):
                return param_file
            else:
                print(f"File {file_path} not found. Searching in other locations.")
        # Next the $GXTBHOME
        gxtb_home = os.getenv("GXTBHOME")
        if gxtb_home:
            gxtb_home = Path(gxtb_home).expanduser().resolve()
            param_file = (gxtb_home / filename).resolve()
            if check_file(param_file):
                print(f"Taking {filename} from GXTBHOME {gxtb_home}.")
                return param_file
        # Home directory
        param_file = (Path.home() / filename).resolve()
        if check_file(param_file):
            print(f"Taking {filename} from HOME.")
            return param_file
        # Current working dir
        cwd = Path.cwd()
        param_file = (cwd / filename).resolve()
        if check_file(param_file):
            print(f"Taking {filename} from cwd {cwd}.")
            return param_file
        # If nothing was found, terminate
        print(f"No {filename} found. Terminating")
        print("Please install gxtb correctly from GitHub.")
        sys.exit(1)

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

        args += ["-c", str(xyz_file), "-p", ".gxtb", "-e", ".eeq", "-b", ".basisq"]

        if dograd:
            args += ["-grad"]

        run_command(self.prog_path, self.prog_out, args)

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
        energy_path = check_path(energy_out)
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
            grad_path = check_path(grad_out)
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
        self, orca_input: dict, directory: Path, args_not_parsed: list[str], gxtb_parameterfile: str, eeq_parameterfile: str, basis_parameterfile: str, prog: str
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
        args_not_parsed: list[str]
            Arguments not parser so far
        gxtb_parameterfile: str
            Parameterfile .gxtb
        eeq_parameterfile: str
            Parameterfile .eeq
        basis_parameterfile: str
            Parameterfile .basisq
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

        # get parameter files
        gxtb_param = self.check_parameter_files(gxtb_parameterfile, ".gxtb")
        eeq_param = self.check_parameter_files(eeq_parameterfile, ".eeq")
        basis_param = self.check_parameter_files(basis_parameterfile, ".basisq")

        # tmp directory named after basename
        tmp_dir = Path(self.basename)

        # make tmp file and copy xyz
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Copy input file(s) to work_dir
        shutil.copy(xyz_file, tmp_dir)
        # Copy Parameterfiles to work_dir, so that they are provided to
        # the gxtb binary later on as relative paths.
        # This is necessary as the gxtb binary does not
        # allow for paths longer than 80 character.
        shutil.copy2(gxtb_param, tmp_dir)
        shutil.copy2(eeq_param, tmp_dir)
        shutil.copy2(basis_param, tmp_dir)

        # Change current directory to work_dir
        base_dir = Path.cwd()
        os.chdir(tmp_dir)

        # write .CHRG and .UHF file
        write_to_file(content=chrg, file=".CHRG")
        write_to_file(content=mult_to_nue(mult), file=".UHF")

        # run gxtb
        self.run_gxtb(xyz_file=xyz_file, dograd=dograd, ncores=ncores, args=args_not_parsed)

        # get the number of atoms from the xyz file
        natoms = nat_from_xyzfile(xyz_file=xyz_file)

        # energy and gradient file
        energy_out = "energy"
        gradient_out = "gradient"

        # parse the gxtb output
        energy, gradient = self.read_gxtbout(
            energy_out=energy_out, grad_out=gradient_out, natoms=natoms, dograd=dograd
        )

        # print the output file to STDOUT
        print_filecontent(outfile=self.prog_out)

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
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, settings=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()

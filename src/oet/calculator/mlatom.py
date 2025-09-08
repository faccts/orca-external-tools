#!/usr/bin/env python3
"""
This is a wrapper for MLatom (http://mlatom.com), compatible with ORCA's ExtTool interface.
Implementation by Pavlo O. Dral (http://dr-dral.com) on 2025.05.04,
based on the example https://github.com/ORCAQuantumChemistry/orca-external-tools/blob/71565ce53837d4d6bdedff71a1ff353f8a289b77/xtb.py

Usage instructions

./mlatom4orca.py <basename_EXT.extinp.tmp> [args for MLatom command line interface]

./mlatom4orca.py -x /path/to/mlatom/shell_cmd.py <basename_EXT.extinp.tmp> [args for MLatom command line interface]

Calculations can be performed with any method or ML model supported by MLatom, which can provide energies (and gradients).

Examples
1. Calculations with AIQM2:

./mlatom4orca.py ethanol_EXT.extinp.tmp AIQM2

2. Calculations with your ML model:

./mlatom4orca.py da_EXT.extinp.tmp useMLmodel MLmodelType=ANI MLmodelIn=da_energy_ani.npz

Note that the calculations via this wrapper calling MLatom for single-point calculations might be significantly slower than using MLatom directly in heavy simulations because of the disk I/O overhead.

Provides
--------
class: MlatomCalc(BaseCalc)
    Class for performing a MLAtom calculation together with ORCA
main: function
    Main function
"""
from argparse import ArgumentParser
from pathlib import Path
import shutil
import tempfile
import os
from oet.core.base_calc import BaseCalc
from oet.core.misc import run_command, print_filecontent, check_path, LENGTH_CONVERSION


class MlatomCalc(BaseCalc):
    @property
    def PROGRAM_KEYS(self) -> set[str]:
        """Program keys to search for in PATH"""
        return {"mlatom", "$mlatom"}

    def extend_parser(self, parser: ArgumentParser):
        """Add mlatom parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        parser.add_argument(
            "-e", "--exe", dest="prog", help="Path to the mlatom executable"
        )

    def run_mlatom(
        self,
        xyzname: str,
        charge: int,
        mult: int,
        ncores: int,
        dograd: bool,
        args: list[str],
    ) -> None:
        """
        Run the MLatom program and redirect its STDOUT and STDERR to a file.

        Parameters
        ----------
        xyzname : str
            name of the XYZ file
        charge : int
            total charge of the system
        mult : int
            spin multiplicity of the system
        ncores : int
            number of threads to use
        dograd : bool
            whether to compute the gradient
        args : list[str, ...]
            additional arguments to pass to MLatom
        """
        args = list(args)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdirname:
            mlatomenergy = os.path.join(cwd, f"{self.basename}.energy")
            mlatomgrad = os.path.join(cwd, f"{self.basename}.gradient")
            shutil.rmtree(tmpdirname)
            shutil.copytree(cwd, tmpdirname)
            args += [
                str(i)
                for i in [
                    f"XYZfile={os.path.join(cwd,xyzname)}",
                    f"charges={charge}",
                    f"multiplicities={mult}",
                    f"nthreads={ncores}",
                    f"YestFile={mlatomenergy}",
                ]
            ]
            if dograd:
                args += [f"YgradXYZestFile={mlatomgrad}"]
            run_command(self.prog_path, self.prog_out, args)

    def read_mlatomout(self, dograd: bool) -> tuple[float, list[float]]:
        """
        Read the output from MLatom

        Parameters
        ----------
        dograd: bool
            whether to read the gradient

        Returns
        -------
        energy: float
            The computed energy
        gradient: list[float]
            The gradient (X,Y,Z) for each atom
        """
        mlatomenergy = f"{self.basename}.energy"
        mlatomgrad = f"{self.basename}.gradient"
        energy = None
        gradient = []
        mlatomenergy = check_path(mlatomenergy)
        mlatomgrad = check_path(mlatomgrad)
        # read the energy from the .energy file
        with mlatomenergy.open() as f:
            for line in f:
                energy = float(line)
        # read the gradient from the .gradient file
        if dograd:
            icount = 0
            with mlatomgrad.open() as f:
                for line in f:
                    icount += 1
                    if icount > 2:
                        gradient += [float(i) * LENGTH_CONVERSION["Ang"] for i in line.split()]
        return energy, gradient

    def calc(
        self,
        orca_input: dict,
        directory: Path,
        args_not_parsed: list[str],
        prog: str,
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

        # run MLatom
        self.run_mlatom(
            xyzname=xyz_file,
            charge=chrg,
            mult=mult,
            ncores=ncores,
            dograd=dograd,
            args=args_not_parsed,
        )

        # parse the MLatom output
        energy, gradient = self.read_mlatomout(dograd=dograd)

        # Print filecontent
        print_filecontent(outfile=self.prog_out)

        # Delete files
        self.clean_files()

        return energy, gradient


def main():
    """
    Main routine for execution
    """
    calculator = MlatomCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, settings=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()

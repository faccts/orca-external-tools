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
import shutil
import tempfile
import os
from oet.core.base_calc import BaseCalc, BasicSettings
from oet.core.misc import run_command, print_filecontent, check_path, LENGTH_CONVERSION


class MlatomCalc(BaseCalc):
    @property
    def PROGRAM_NAMES(self) -> set[str]:
        """Program keys to search for in PATH"""
        return {"mlatom", "$mlatom"}

    def extend_parser(self, parser: ArgumentParser) -> None:
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
        settings: BasicSettings,
        args: list[str],
    ) -> None:
        """
        Run the MLatom program and redirect its STDOUT and STDERR to a file.

        Parameters
        ----------
        settings: BasicSettings
            Basic calculation settings
        args : list[str, ...]
            additional arguments to pass to MLatom
        """
        args = list(args)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdirname:
            mlatomenergy = os.path.join(cwd, f"{settings.basename}.energy")
            mlatomgrad = os.path.join(cwd, f"{settings.basename}.gradient")
            shutil.rmtree(tmpdirname)
            shutil.copytree(cwd, tmpdirname)
            args += [
                str(i)
                for i in [
                    f"XYZfile={settings.xyzfile}",
                    f"charges={settings.charge}",
                    f"multiplicities={settings.mult}",
                    f"nthreads={settings.ncores}",
                    f"YestFile={mlatomenergy}",
                ]
            ]
            if settings.dograd:
                args += [f"YgradXYZestFile={mlatomgrad}"]
            if not settings.prog_path:
                raise RuntimeError("Path to program is None.")
            run_command(settings.prog_path, settings.prog_out, args)

    def read_mlatomout(self, settings: BasicSettings) -> tuple[float, list[float]]:
        """
        Read the output from MLatom

        Parameters
        ----------
        settings: BasicSettings
            Basic calculation settings

        Returns
        -------
        energy: float
            The computed energy
        gradient: list[float] | None
            The gradient (X,Y,Z) for each atom
        """
        mlatomenergy = f"{settings.basename}.energy"
        mlatomgrad = f"{settings.basename}.gradient"
        energy = None
        gradient = []
        mlatomenergy_path = check_path(mlatomenergy)
        # read the energy from the .energy file
        with mlatomenergy_path.open() as f:
            for line in f:
                energy = float(line)
        # read the gradient from the .gradient file
        if settings.dograd:
            mlatomgrad_path = check_path(mlatomgrad)
            icount = 0
            with mlatomgrad_path.open() as f:
                for line in f:
                    icount += 1
                    if icount > 2:
                        gradient += [
                            float(i) * LENGTH_CONVERSION["Ang"] for i in line.split()
                        ]
        if not energy:
            raise ValueError(f"Total enery not found in file {settings.prog_out}")
        return energy, gradient

    def calc(
        self,
        settings: BasicSettings,
        args_not_parsed: list[str],
        prog: str,
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient.
        Writes ORCA output

        Parameters
        ----------
        settings: BasicSettings
            Basic calculation settings
        args_not_parsed: list[str]
            Arguments not parsed so far
        prog: str
            Which program executable to use

        Returns
        -------
        float: energy
        list[float]: gradients
        """
        settings.set_program_path(prog)
        if settings.prog_path:
            print(f"Using executable {settings.prog_path}")
        else:
            raise FileNotFoundError(
                f"Could not find a valid executable from standard program names: {self.PROGRAM_NAMES}"
            )

        # run MLatom
        self.run_mlatom(
            settings=settings,
            args=args_not_parsed,
        )

        # parse the MLatom output
        energy, gradient = self.read_mlatomout(settings=settings)

        # Print filecontent
        print_filecontent(outfile=settings.prog_out)

        return energy, gradient


def main() -> None:
    """
    Main routine for execution
    """
    calculator = MlatomCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(
        inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed
    )


# Python entry point
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Calculator for using the predict.x binary from
aenet (http://ann.atomistic.net), compatible with ORCA's ExtTool interface.

Provides
--------
class: AenetCalc(BaseCalc)
    Class for performing a Aenet calculation together with ORCA
main: function
    Main function
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping

from oet.core.base_calc import BaseCalc, BasicSettings
from oet.core.misc import (
    xyz2xsf,
    get_nns,
    run_command,
    print_filecontent,
    check_path,
    ENERGY_CONVERSION,
    LENGTH_CONVERSION,
)


class AenetCalc(BaseCalc):

    # directory, containing the NN files
    NNPATH: str | Path | None = None  # Path('/path/to/nns')
    # extension for the NN files (<Symbol>.<NNEXT>)
    NNEXT: str | None = None

    @property
    def PROGRAM_NAMES(self) -> set[str]:
        """Program keys to search for in PATH"""
        return {"predict.x"}

    def extend_parser(self, parser: ArgumentParser) -> None:
        """Add aenet parsing options."""
        parser.add_argument(
            "-x", "--exe", dest="prog", help="Path to the aenet executable"
        )
        parser.add_argument(
            "-n",
            "--nnpath",
            metavar="DIR",
            dest="nnpath",
            required=(not self.NNPATH),
            help="directory containing the NN files <Symbol>.<EXT>"
            + (f' (default: "{self.NNPATH}")' if self.NNPATH else ""),
            default=self.NNPATH,
        )
        parser.add_argument(
            "-e",
            "--nnext",
            metavar="EXT",
            dest="nnext",
            required=False,
            default=self.NNEXT,
            help="extension of the NN files. "
            + (
                f"(default: {self.NNEXT})"
                if self.NNEXT
                else "If not provided, there must be a single file that matches the glob <Symbol>.*"
            ),
        )

    @staticmethod
    def write_predict_input(
        xsfname: str | Path,
        inpname: str | Path,
        dograd: bool,
        nns: Mapping[str, Path],
    ) -> None:
        """Write the input file for predict.x

        Parameters
        ----------
        xsfname : str | Path
            The name of the XSF coordinates file
        inpname : str | Path
            The file to be written
        dograd : bool
            Whether to compute forces
        nns : Mapping[str, Path]
            Keys are element symbols and values are paths to the NN potential files
        """
        inpname = Path(inpname)
        xsfname = Path(xsfname)
        with inpname.open("w") as f:
            # write types
            f.write(f"TYPES\n{len(nns)}\n" + "\n".join(nns) + "\n\n")
            # write NN paths
            f.write(
                "NETWORKS\n" + "\n".join(f'{a}  "{p}"' for a, p in nns.items()) + "\n\n"
            )
            # forces tag if gradient is requested
            if dograd:
                f.write("FORCES\n\n")
            # write XSF file (only one supported)
            f.write(f"FILES\n1\n{xsfname}\n")

    def run_predict(self, predictexe: str | Path, inpname: str, ncores: int, prog_out: str) -> None:
        """
        Run the predict.x program and redirect its STDOUT and STDERR to a file. Exists on a non-zero return code.

        Parameters
        ----------
        predictexe : str | Path
            Path to the predict.x program
        inpname : str
            Path to the input file
        ncores : int
            Number of cores to use # TODO: currently only implemented in serial
        prog_out: str
            Output file of program
        """
        run_command(predictexe, prog_out, [inpname])

    def read_predict_output(
        self, natoms: int, dograd: bool, prog_out: str
    ) -> tuple[float, list[float]]:
        """Read the output from predict.x

        Parameters
        ----------
        natoms : int
            The number of atoms in the system
        dograd : bool
            Whether the gradient was computed

        Returns
        -------
        energy: float
            The computed energy
        gradient: list[float]
            The gradient (X,Y,Z) for each atom
        prog_out: str
            Outputfile to read from
        """
        energy = None
        gradient = []
        with open(prog_out) as f:
            for line in f:
                if "Total energy" in line:
                    fields = line.split()
                    unit = fields[-1]
                    if unit not in ENERGY_CONVERSION:
                        raise ValueError(f"Unknown energy unit: {unit}")
                    energy = float(fields[-2]) / ENERGY_CONVERSION[unit]
                elif dograd and "atomic forces" in line:
                    f.readline()  # empty
                    f.readline()  # x,y,z,Fx,Fy,Fz
                    eunit, lunit = f.readline().split()[-1].strip("()").split("/")
                    if eunit not in ENERGY_CONVERSION:
                        raise ValueError(f"Unknown energy unit: {eunit}")
                    if lunit not in LENGTH_CONVERSION:
                        raise ValueError(f"Unknown length unit: {lunit}")
                    # unit conversion & factor of -1 to convert from forces to gradient
                    fac = -LENGTH_CONVERSION[lunit] / ENERGY_CONVERSION[eunit]
                    f.readline()  # ---
                    for i, line2 in enumerate(f):
                        if i >= natoms:
                            break
                        fields = line2.split()
                        gradient += [float(i) * fac for i in fields[-3:]]
        if not energy:
            raise ValueError(f"Total enery not found in file {prog_out}")
        return energy, gradient

    def calc(
        self,
        settings: BasicSettings,
        args_not_parsed: list[str],
        prog: str,
        nnpath: str | Path,
        nnext: str,
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient.
        Writes ORCA output

        Parameters
        ----------
        settings: BasicSettings
            Object with basic settings for the run
        args_not_parsed: list[str]
            Arguments not parsed so far
        prog: str
            Path to program
        nnpath: str
            Path to nn files
        nnext: str
            extension of the nnfiles
        """
        settings.set_program_path(prog)
        if settings.prog_path:
            print(f"Using executable {settings.prog_path}")
        else:
            raise FileNotFoundError(
                f"Could not find a valid executable from standard program names: {self.PROGRAM_NAMES}"
            )
        # Check other files
        nnpath = check_path(nnpath)

        # set filenames
        namespace = settings.basename + ".predict"
        xsfname = namespace + ".xsf"
        inpname = namespace + ".in"
        prog_out = namespace + ".out"

        # process the XYZ file
        natoms, atomtypes = xyz2xsf(xyzname=settings.xyzfile, xsfname=xsfname)
        # find the NN files
        nns = get_nns(atomtypes=atomtypes, nnpath=nnpath, nnext=nnext)
        # write the input for predict.x
        self.write_predict_input(
            xsfname=xsfname, inpname=inpname, dograd=settings.dograd, nns=nns
        )
        # run predict.x
        self.run_predict(predictexe=settings.prog_path, inpname=inpname, ncores=settings.ncores, prog_out=prog_out)
        # parse the output
        energy, gradient = self.read_predict_output(natoms, settings.dograd, prog_out)

        # Print filecontent
        print_filecontent(outfile=prog_out)

        return energy, gradient


def main() -> None:
    """ Main entry point for wrapper"""
    calculator = AenetCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(
        inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed
    )


# Python entry point
if __name__ == "__main__":
    main()

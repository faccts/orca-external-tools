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
from typing import Iterable, Mapping

from oet.core.base_calc import BaseCalc


class AenetCalc(BaseCalc):

    # predict.x executable from aenet
    PREDICT_EXE: str | Path = "predict.x"
    # directory, containing the NN files
    NNPATH: str | Path | None = None  # Path('/path/to/nns')
    # extension for the NN files (<Symbol>.<NNEXT>)
    NNEXT: str | None = None

    @property
    def PROGRAM_KEYS(self) -> set[str]:
        """Program keys to search for in PATH"""
        return {"predict.x"}

    def extend_parser(self, parser: ArgumentParser):
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

    def xyz2xsf(self, xyzname: str | Path, xsfname: str | Path) -> tuple[int, set[str]]:
        """Convert a XYZ file to XSF format.

        Parameters
        ----------
        xyzname : str | Path
            The XYZ file to convert
        xsfname : str | Path
            The output XSF file name

        Returns
        -------
        tuple[int, set[str]]
            natoms: int
                The number of atoms in the XYZ file
            atomtypes: set[str]
                The elements present in the XYZ file
        """
        atomtypes = set()
        xyzname = self.check_path(xyzname)
        xsfname = Path(xsfname)
        with xyzname.open() as xyzf, xsfname.open("w") as xsff:
            natoms = int(xyzf.readline())
            xyzf.readline()  # comment line

            xsff.write("#\n\nATOMS\n")
            for i, line in enumerate(xyzf):
                if i >= natoms:
                    break
                # add the forces and print
                xsff.write(line.rstrip() + "  0.0  0.0  0.0\n")
                # collect the elements
                atomtypes.add(line.split()[0])
        return natoms, atomtypes

    def get_nns(
        self, atomtypes: Iterable[str], nnpath: str | Path, nnext: str | None = None
    ) -> dict[str, Path]:
        """Find the neural network potential files for each element in `atomtypes`.
        The files must all be in the same directory and be named "<ElementSymbol>.<Extension>" with the same extension.

        Parameters
        ----------
        atomtypes : Iterable[str]
            The elements needed
        nnpath : str | Path
            Path to the directory containing the neural network potential files
        nnext : str | None, default = None
            The extension for each NN file. If none is given '*' is used as a wildcard.
            However, then there must be a single file that matches, otherwise an exception is raised

        Returns
        -------
        dict[str, Path]
            The keys are element symbols and the values are paths to the NN files

        Raises
        ------
        RuntimeError
            If more than one or no NN files are found for a requested element
        """
        nnpath = self.check_path(nnpath)
        if not nnext:
            nnext = "*"
        nns = {}
        for a in atomtypes:
            matches = list(nnpath.glob(f"{a}.{nnext}"))
            if not matches:
                raise RuntimeError(f"No NN files found for {a} in {nnpath}")
            if len(matches) > 1:
                raise RuntimeError(
                    f"Multiple NN files found for {a}: {matches}. Set --nnext to specify the extension"
                )
            nns[a] = matches[0]
        return nns

    def write_predict_input(
        self,
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

    def run_predict(
        self, predictexe: str | Path, inpname: str, ncores: int
    ) -> None:
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
        """
        self.run_command(predictexe, self.prog_out, [inpname])

    def read_predict_output(
        self, natoms: int, dograd: bool
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
        """
        ENERGY_CONVERSION = {"eV": 27.21138625}
        LENGTH_CONVERSION = {"Ang": 0.529177210903}
        energy = None
        gradient = []
        with open(self.prog_out) as f:
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
        return energy, gradient

    def calc(
        self,
        orca_input: dict,
        directory: Path,
        clear_args: list[str],
        prog: str,
        nnpath: str,
        nnext: str,
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
        clear_args: list[str]
            Arguments not parsed so far
        prog: str
            Path to program
        nnpath: str
            Path to nn files
        nnext: str
            extension of the nnfiles
        """
        # Get the information needed
        xyz_file = orca_input["xyz_file"]
        xyz_file = directory / Path(xyz_file)
        # chrg = orca_input["chrg"]
        # mult = orca_input["mult"]
        ncores = orca_input["ncores"]
        dograd = orca_input["dograd"]
        # Set and check the program path if its executable
        self.set_program_path(prog)
        print("Using executable ", self.prog_path)
        # Check other files
        nnpath = self.check_path(nnpath)
        predictexe = self.check_path(prog)

        # set filenames
        namespace = self.basename + ".predict"
        xsfname = namespace + ".xsf"
        inpname = namespace + ".in"
        self.prog_out = namespace + ".out"

        # process the XYZ file
        natoms, atomtypes = self.xyz2xsf(xyzname=xyz_file, xsfname=xsfname)
        # find the NN files
        nns = self.get_nns(atomtypes=atomtypes, nnpath=nnpath, nnext=nnext)
        # write the input for predict.x
        self.write_predict_input(
            xsfname=xsfname, inpname=inpname, dograd=dograd, nns=nns
        )
        # run predict.x
        self.run_predict(predictexe=predictexe, inpname=inpname, ncores=ncores)
        # parse the output
        energy, gradient = self.read_predict_output(natoms, dograd)

        # Print filecontent
        self.print_filecontent(outfile=self.prog_out)

        # Delete files
        self.clean_files()

        return energy, gradient


def main():
    """ """
    calculator = AenetCalc()
    inputfile, args, clear_args = calculator.parse_args()
    calculator.run(inputfile=inputfile, settings=args, clear_args=clear_args)


# Python entry point
if __name__ == "__main__":
    main()

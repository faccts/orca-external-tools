#!/usr/bin/env python3
"""
This is a simple wrapper for the g-xTB binary (github.com/grimme-lab/g-xTB), compatible with ORCA's ExtTool interface.
Note that this is currently a development version of g-xTB and that the final implementation will be available via tblite.
It currently runs only serial due to technical limitations of the development version. 
"""

import shutil
import sys
import subprocess
import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Iterable


# path to the gxtb executable. If None, will look for all gxtb_names in the system PATH
gxtb_exe: str | Path | None = None
gxtb_names: list[str] = ["gxtb"]


# ----------------------------------------------------------------------------------------------------------------------
# Common functions: these are duplicated in all scripts to make them self-contained


def strip_comments(s: str) -> str:
    """Strip comment starting with '#' and continuing until the end of the string. Also strip whitespace."""
    return s.split("#")[0].strip()


def enforce_path_object(fname: str | Path) -> Path:
    """Enforce that the input is a Path object

    Parameters
    ----------
    fname : str | Path
        The filename which should be a string or a Path object

    Returns
    -------
    Path
        The filename as a Path object

    Raises
    ------
    TypeError
        If the input is not a string or a Path object (e.g. a list)
    """
    if isinstance(fname, str):
        return Path(fname)
    elif isinstance(fname, Path):
        return fname
    else:
        msg = "Input must be a string or a Path object."
        raise TypeError(msg)


def read_input(inpfile: str | Path) -> tuple[str, int, int, int, bool]:
    """Read the ORCA-generated input file

    Parameters
    ----------
    inpfile : str | Path
        The input file

    Returns
    -------
    tuple[str, int, int, int, bool]
        xyzname: str
            Name of the XYZ coordinates file
        charge: int
            Total charge
        mult: int
            Spin multiplicity
        ncores: int
            Number of parallel cores available
        dograd: bool
            Whether to compute the gradient
    """
    inpfile = enforce_path_object(inpfile)
    with inpfile.open() as f:
        xyzname = strip_comments(f.readline())
        charge = int(strip_comments(f.readline()))
        mult = int(strip_comments(f.readline()))
        ncores = int(strip_comments(f.readline()))
        dograd = bool(int(strip_comments(f.readline())))
        # TODO POINT CHARGES
    return xyzname, charge, mult, ncores, dograd


def write_engrad(
    outfile: str | Path,
    natoms: int,
    energy: float,
    dograd: bool,
    gradient: Iterable[float] | None = None,
) -> None:
    """Write the energy/gradient file to feed back to ORCA.

    Parameters
    ----------
    outfile : str | Path
        The engrad file
    natoms : int
        Number of atoms
    energy : float
        Total energy
    dograd : bool
        Whether the gradient is computed
    gradient
        The gradient (X,Y,Z) for each atom
    """
    outfile = enforce_path_object(outfile)
    with outfile.open("w") as f:
        output = "#\n"
        output += "# Number of atoms\n"
        output += "#\n"
        output += f"{natoms}\n"
        output += "#\n"
        output += "# Total energy [Eh]\n"
        output += "#\n"
        output += f"{energy:.12e}\n"
        if dograd and gradient is not None:
            output += "#\n"
            output += "# Gradient [Eh/Bohr] A1X, A1Y, A1Z, A2X, ...\n"
            output += "#\n"
            output += "\n".join(f"{g: .12e}" for g in gradient) + "\n"
        f.write(output)


def run_command(
    command: str | Path, outname: str | Path, *args: tuple[str, ...]
) -> None:
    """
    Run the given command and redirect its STDOUT and STDERR to a file. Exists on a non-zero return code.

    Parameters
    ----------
    command : str | Path
        The command to run or path to an executable
    outname : str | Path
        The output file to be written to (overwritten!)
    *args : tuple[str, ...]
        arguments to be passed to the command
    """
    command = enforce_path_object(command)
    outname = enforce_path_object(outname)
    with open(outname, "w") as of:
        try:
            subprocess.run(
                [str(command)] + list(args),
                stdout=of,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as err:
            print(err)
            sys.exit(err.returncode)


def print_filecontent(outfile: str | Path) -> None:
    """
    Print the file content, e.g. the output file, to STDOUT

    Parameters
    ----------
    outfile : str | Path
        The output file to print
    """
    # print the output to STDOUT
    outfile = enforce_path_object(outfile)
    with open(outfile) as f:
        for line in f:  # line by line to avoid memory overflow
            print(line, end="")


def run_gxtb(
    gxtbexe: str | Path,
    xyzname: str,
    dograd: bool,
    outfile: str | Path,
    *args: tuple[str, ...],
) -> None:
    """
    Run the gxtb program and redirect its STDOUT and STDERR to a file.

    Parameters
    ----------
    gxtbexe : str | Path
        path to the gxtb binary
    xyzname : str
        name of the XYZ file
    namespace : str
        filename prefix for the gxtb output files
    ncores : int
        number of threads to use
    dograd : bool
        whether to compute the gradient
    outfile : str | Path
        the output file
    *args : tuple[str, ...]
        additional arguments to pass to gxtb
    """
    args = list(map(str, args))
    args += [
        str(i) for i in ["-c", xyzname]
    ]

    if dograd:
        args += ["-grad"]
    
    run_command(gxtbexe, outfile, *args)

    return


def read_gxtbout(
    gxtb_out: str | Path, energy_out: str | Path, grad_out: str | Path, natoms: int, dograd: bool
) -> tuple[float, list[float]]:
    """
    Read the output from gxtb

    Parameters
    ----------
    namespace
        filename prefix of the gxtb output files
    gxtbout
        the main gxtb output file
    natoms
        number of atoms in the system
    dograd
        whether to read the gradient

    Returns
    -------
    tuple[float, list[float]]
        energy: float
            The computed energy
        gradient: list[float]
            The gradient (X,Y,Z) for each atom
    """
    energy = None
    gradient = []
    # read the energy from the output file
    energy_path = enforce_path_object(energy_out)
    grad_path = enforce_path_object(grad_out)
    with energy_path.open() as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == '$energy':
                # read the next line and split into values
                parts = lines[i + 1].split()
                # return the second value as float
                energy = float(parts[1])
                break
    # read the gradient from the .gradient file
    if dograd:
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
                    gradient += [float(i.replace('D', 'E')) for i in fields]
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

def remove_file(fname: str) -> None:
    """ Remove file if present."""
    file_path = Path(fname)
    if file_path.is_file():
        file_path.unlink()
    return
    
def write_chrg_uhf(
    charge: int, mult: int, charge_file: str, uhf_file: str,
) -> None:
    """
    Writes charge and number of unpaired e- to files

    Parameters
    ----------
    charge
        total charge of molecule
    mult
        multiplicity, will be converted to number of unpaired electrons
    charge_file
        filename to write charge into
    uhf_file
        whether to read the gradient

    Returns
    -------
    Nothing
    """

    # first check whether files are present and delete them if so
    remove_file(charge_file)
    remove_file(uhf_file)

    # get number of electrons
    nue = mult - 1

    # then, write the new files
    charge_path = Path(charge_file)
    uhf_path = Path(uhf_file)

    with open(charge_path, "w") as f:
        f.write(f"{charge}\n")

    with open(uhf_path, "w") as f:
        f.write(f"{nue}\n")

    return


def check_file(file_path: Path | str) -> bool:
    """ Check whether file is present or not. Returns boolean."""
    return Path(file_path).is_file()
    

def main(argv: list[str]) -> None:
    """Main function to run the script"""
    if not (gxtbexe := gxtb_exe):
        for gxtb in gxtb_names:
            if gxtbexe := shutil.which(gxtb):
                break

    # parse the CLI arguments
    parser = ArgumentParser(
        prog=argv[0],
        allow_abbrev=False,
        description="Wrapper for gxtb, compatible with ORCA's otool_external. "
        "Reads the ORCA-generated input <inputfile>, calls gxtb, "
        "parses its output and writes the BaseName.engrad file for ORCA.",
    )
    parser.add_argument("inputfile")
    parser.add_argument(
        "-x",
        "--exe",
        metavar="gxtbexe",
        dest="gxtbexe",
        required=(not gxtbexe),
        help="path to the gxtb executable"
        + (f' (default: "{gxtbexe}")' if gxtbexe else ""),
        default=gxtbexe,
    )
    args, gxtb_args = parser.parse_known_args(argv[1:])

    # sanitize the path to gxtb and check whether it is accessible
    gxtbexe = Path(args.gxtbexe).expanduser().resolve()
    if not check_file(gxtbexe):
        print(
            "No gxtb exe found."
            "Please install gxtb correctly from GitHub."
            )
        sys.exit(1)

    # check whether required parameter files are accessible
    parameter_file = Path("~/.gxtb").expanduser()
    eeq_file = Path("~/.eeq").expanduser()
    basis_file = Path("~/.basisq").expanduser()
    if not check_file(parameter_file):
        print(
            "No .gxtb parameter file found in home directory."
            "Please install gxtb correctly from GitHub."
            )
        sys.exit(1)
    if not check_file(eeq_file):
        print(
            "No .eeq parameter file found in home directory."
            "Please install gxtb correctly from GitHub."
            )
        sys.exit(1)
    if not check_file(basis_file):
        print(
            "No .basisq parameter file found in home directory."
            "Please install gxtb correctly from GitHub."
            )
        sys.exit(1)

    # delete gxtbrestart if present
    remove_file("gxtbrestart")

    # read the ORCA-generated input
    xyzname, charge, mult, ncores, dograd = read_input(args.inputfile)

    # set filenames
    basename = xyzname.rstrip(".xyz")
    orca_engrad = basename + ".engrad"
    gxtb_namespace = basename + ".gxtb"
    gxtbout = gxtb_namespace + ".out"

    # tmp directory
    tmp_dir = Path(gxtb_namespace)

    # make tmp file and copy xyz
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Copy input file(s) to work_dir
    shutil.copy(xyzname, tmp_dir)

    # Change current directory to work_dir
    base_dir = Path.cwd()
    os.chdir(tmp_dir)

    # write .CHRG and .UHF file
    write_chrg_uhf(charge, mult, ".CHRG", ".UHF")

    # run gxtb
    run_gxtb(
        gxtbexe, xyzname, dograd, gxtbout, *gxtb_args
    )

    # get the number of atoms from the xyz file
    with open(xyzname) as f:
        natoms = int(f.readline())

    # energy and gradient file
    energy_out = "energy"
    gradient_out = "gradient"

    # parse the gxtb output
    energy, gradient = read_gxtbout(gxtbout, energy_out, gradient_out, natoms, dograd)

    # print the output file to STDOUT
    print_filecontent(gxtbout)

    # go back to parent dir
    os.chdir(base_dir)

    # write the ORCA engrad file
    write_engrad(orca_engrad, natoms, energy, dograd, gradient)

    # remove tmp directory
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main(sys.argv)

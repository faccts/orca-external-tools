#!/usr/bin/env python3
'''
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
'''

from __future__ import annotations

import shutil
import sys, os
import subprocess
from pathlib import Path
from argparse import ArgumentParser
from typing import Iterable


# path to the mlatom executable. If None, will look for all MLATOM_NAMES in the system PATH.
# Note that here we use MLatom's command line interface, although MLatom can be directly imported in Python.
# Using command line interface provides direct access to all the command line options though.
MLATOM_EXE: str | Path | None = None
MLATOM_NAMES: list[str] = ["mlatom", "$mlatom"]


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
    gradient: Iterable[float] = None,
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
        if dograd:
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
            exit(err.returncode)


def clean_output(outfile: str | Path, namespace: str) -> None:
    """
    Print the output file to STDOUT and remove all files starting with `namespace`

    Parameters
    ----------
    outfile : str | Path
        The output file to print
    namespace : str
        The starting string of all files to remove.
    """
    # print the output to STDOUT
    outfile = enforce_path_object(outfile)
    with open(outfile) as f:
        for line in f:  # line by line to avoid memory overflow
            print(line, end="")
    # remove all file from the namespace
    for f in Path(".").glob(f"{namespace}*"):
        f.unlink()


# ----------------------------------------------------------------------------------------------------------------------


def run_mlatom(
    mlatomexe: str | Path,
    xyzname: str,
    namespace: str,
    charge: int,
    mult: int,
    ncores: int,
    dograd: bool,
    outfile: str | Path,
    *args: tuple[str, ...],
) -> None:
    """
    Run the MLatom program and redirect its STDOUT and STDERR to a file.

    Parameters
    ----------
    mlatomexe : str | Path
        path to the MLatom shell command script
    xyzname : str
        name of the XYZ file
    namespace : str
        filename prefix for the xtb output files
    charge : int
        total charge of the system
    mult : int
        spin multiplicity of the system
    ncores : int
        number of threads to use
    dograd : bool
        whether to compute the gradient
    outfile : str | Path
        the output file
    *args : tuple[str, ...]
        additional arguments to pass to MLatom
    """
    args = list(args)
    cwd = os.getcwd()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        mlatomenergy = os.path.join(cwd,f'{namespace}.energy')
        mlatomgrad = os.path.join(cwd,f'{namespace}.gradient')
        shutil.rmtree(tmpdirname)
        shutil.copytree(cwd, tmpdirname)
        args += [
            str(i) for i in [f'XYZfile={os.path.join(cwd,xyzname)}', f'charges={charge}', f'multiplicities={mult}', f'nthreads={ncores}', f'YestFile={mlatomenergy}']
        ]
        if dograd:
            args += [f'YgradXYZestFile={mlatomgrad}']
        with open(outfile, "w") as of:
            try:
                subprocess.run(
                    [str(mlatomexe)] + list(args),
                    stdout=of,
                    stderr=subprocess.STDOUT,
                    cwd=tmpdirname,
                    check=True,
                )
            except subprocess.CalledProcessError as err:
                print(err)
                exit(err.returncode)
        

def read_mlatomout(
    namespace: str, dograd: bool
) -> tuple[float, list[float]]:
    """
    Read the output from MLatom

    Parameters
    ----------
    namespace
        filename prefix of the MLatom output files
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
    mlatomenergy = f'{namespace}.energy'
    mlatomgrad = f'{namespace}.gradient'
    energy = None
    gradient = []
    mlatomenergy = enforce_path_object(mlatomenergy)
    mlatomgrad = enforce_path_object(mlatomgrad)
    # read the energy from the .energy file
    with mlatomenergy.open() as f:
        for line in f:
            energy = float(line)
    # read the gradient from the .gradient file
    if dograd:
        Bohr2Angstrom =  0.52917721092 # Peter J. Mohr, Barry N. Taylor, David B. Newell,
                                       # CODATA Recommended Values of the
                                       # Fundamental Physical Constants: 2010, NIST, 2012.
        icount = 0
        with mlatomgrad.open() as f:
            for line in f:
                icount += 1
                if icount > 2:
                    gradient += [float(i) * Bohr2Angstrom for i in line.split()]
    return energy, gradient

def main(argv: list[str]) -> None:
    """Main function to run the script"""
    if not (mlatomexe := MLATOM_EXE):
        for ml in MLATOM_NAMES:
            if mlatomexe := shutil.which(ml):
                break

    # parse the CLI arguments
    parser = ArgumentParser(
        prog=argv[0],
        allow_abbrev=False,
        description="Wrapper for MLatom, compatible with ORCA's otool_external. "
        "Reads the ORCA-generated input <inputfile>, calls MLatom, "
        "parses its output and writes the BaseName.engrad file for ORCA.",
    )
    parser.add_argument("inputfile")
    parser.add_argument(
        "-x",
        "--exe",
        metavar="mlatomexe",
        dest="mlatomexe",
        required=(not mlatomexe),
        help="path to the MLatom shell command script"
        + (f' (default: "{mlatomexe}")' if mlatomexe else ""),
        default=mlatomexe,
    )
    args, mlatom_args = parser.parse_known_args(argv[1:])

    # sanitize the path to xtb
    mlatomexe = Path(args.mlatomexe).expanduser().resolve()

    # read the ORCA-generated input
    xyzname, charge, mult, ncores, dograd = read_input(args.inputfile)

    # set filenames
    basename = xyzname.rstrip(".xyz")
    orca_engrad = basename + ".engrad"
    mlatom_namespace = basename + ".mlatom"
    mlatomout = mlatom_namespace + ".out"

    # run MLatom
    run_mlatom(
        mlatomexe, xyzname, mlatom_namespace, charge, mult, ncores, dograd, mlatomout, *mlatom_args
    )

    # get the number of atoms from the xyz file
    with open(xyzname) as f:
        natoms = int(f.readline())

    # parse the MLatom output
    energy, gradient = read_mlatomout(mlatom_namespace, dograd)

    # write the ORCA engrad file
    write_engrad(orca_engrad, natoms, energy, dograd, gradient)

    # print the MLatom output to STDOUT and remove leftover files
    clean_output(mlatomout, mlatom_namespace)


if __name__ == "__main__":
    main(sys.argv)
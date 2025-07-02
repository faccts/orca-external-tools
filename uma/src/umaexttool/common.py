#!/usr/bin/env python3

from __future__ import annotations

import subprocess
from enum import StrEnum, auto
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable
from ase import Atoms

import numpy as np


# Energy and length conversion to atomic units.
ENERGY_CONVERSION = {"eV": 27.21138625}
LENGTH_CONVERSION = {"Ang": 0.529177210903}
# Elements covered by UMA
ELEMENT_TO_ATOMIC_NUMBER = {
    "H": 1,  "He": 2,
    "Li": 3, "Be": 4, "B": 5,  "C": 6,  "N": 7,  "O": 8,  "F": 9,  "Ne": 10,
    "Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15, "S": 16, "Cl": 17,"Ar": 18,
    "K": 19, "Ca": 20,"Sc": 21,"Ti": 22,"V": 23, "Cr": 24,"Mn": 25,"Fe": 26,
    "Co": 27,"Ni": 28,"Cu": 29,"Zn": 30,"Ga": 31,"Ge": 32,"As": 33,"Se": 34,
    "Br": 35,"Kr": 36,"Rb": 37,"Sr": 38,"Y": 39, "Zr": 40,"Nb": 41,"Mo": 42,
    "Tc": 43,"Ru": 44,"Rh": 45,"Pd": 46,"Ag": 47,"Cd": 48,"In": 49,"Sn": 50,
    "Sb": 51,"Te": 52,"I": 53, "Xe": 54,"Cs": 55,"Ba": 56,"La": 57,"Ce": 58,
    "Pr": 59,"Nd": 60,"Pm": 61,"Sm": 62,"Eu": 63,"Gd": 64,"Tb": 65,"Dy": 66,
    "Ho": 67,"Er": 68,"Tm": 69,"Yb": 70,"Lu": 71,"Hf": 72,"Ta": 73,"W": 74,
    "Re": 75,"Os": 76,"Ir": 77,"Pt": 78,"Au": 79,"Hg": 80,"Tl": 81,"Pb": 82,
    "Bi": 83
}


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
            Whether to compute the gradient (currently ignored; gradient is always calculated)
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
    gradient : Iterable[float], optional
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


def run_command(command: str | Path, outname: str | Path, *args: tuple[str, ...]) -> None:
    """
    Run the given command and redirect its STDOUT and STDERR to a file. Exists on a non-zero return code.

    Parameters
    ----------
    command : str | Path
        The command to run or path to an executable
    outname : str | Path
        The output file to be written to (overwritten!)
    args : tuple[str, ...]
        arguments to be passed to the command
    """
    command = enforce_path_object(command)
    outname = enforce_path_object(outname)
    with outname.open("w") as of:
        try:
            subprocess.run(
                [command] + list(args), stdout=of, stderr=subprocess.STDOUT, check=True
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
    with outfile.open() as f:
        for line in f:  # line by line to avoid memory overflow
            print(line, end="")
    # remove all file from the namespace
    for f in Path(".").glob(namespace + "*"):
        f.unlink()


def read_xyzfile(xyzname: str | Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Read an XYZ file and return the atom types and coordinates.

    Parameters
    ----------
    xyzname : str | Path
        The XYZ file to read.

    Returns
    -------
    tuple[list[str], list[tuple[float, float, float]]]
        atom_types: list[str]
            A list of element symbols in order.
        coordinates: list[tuple[float, float, float]]
            A list of (x, y, z) coordinates.
    """
    atom_types = []
    coordinates = []
    xyzname = enforce_path_object(xyzname)
    with xyzname.open() as xyzf:
        natoms = int(xyzf.readline().strip())
        xyzf.readline()
        for _ in range(natoms):
            line = xyzf.readline()
            if not line:
                break
            parts = line.split()
            atom_types.append(parts[0])
            coords = tuple(float(c) for c in parts[1:4])
            coordinates.append(coords)
    return atom_types, coordinates


def atomic_symbol_to_number(symbol: str) -> int:
    """Convert an element symbol to an atomic number.

    Parameters
    ----------
    symbol
        Element symbol, e.g. "Cl"

    Returns
    -------
    int
        atomic number of the element

    Raises
    ------
    ValueError
        if the element is not in `ELEMENT_TO_ATOMIC_NUMBER`
    """
    try:
        return ELEMENT_TO_ATOMIC_NUMBER[symbol.title()]
    except KeyError:
        raise ValueError(f"Unknown element symbol: {symbol}")


def process_output(atoms: Atoms) -> tuple[float, list[float]]:
    """Process the output from umaCalculator and perform conversions to a.u.

    Parameters
    ----------
    atoms
        A molecule of the ASE atoms type

    Returns
    -------
    tuple[float, list[float]]
        energy : float
            The computed energy (Eh)
        gradient : list[float]
            Flattened gradient vector (Eh/Bohr), if computed, otherwise empty.
    """
    energy = atoms.get_potential_energy() / ENERGY_CONVERSION["eV"]
    gradient = []
    try:
        forces = atoms.get_forces()
        # Convert forces to gradient (-1) and unit conversion
        fac = -LENGTH_CONVERSION["Ang"] / ENERGY_CONVERSION["eV"]
        gradient = (fac*np.asarray(forces)).flatten().tolist()
    except Exception:
        # forces may not be available
        pass

    return energy, gradient


class RunMode(StrEnum):
    """Possible run modes for the wrapper"""
    Server = auto()
    Client = auto()
    Standalone = auto()


ProgName = {
    RunMode.Server: "umaserver",
    RunMode.Client: "umaclient",
    RunMode.Standalone: "umaexttool",
}


def cli_parse(args: list[str], mode: RunMode) -> Namespace:
    """Parse the command line arguments, depending on the run mode of the wrapper."""
    default_model_path = Path(__file__).resolve().parent / 'models'  #.dirname(os.path.abspath(__file__))

    parser = ArgumentParser(prog=ProgName[mode],
                            description=f'ORCA "external tool" interface for uma calculations ({mode} mode)')
    if mode in (RunMode.Standalone, RunMode.Client):
        parser.add_argument("inputfile", help="ORCA-generated input file.")
    if mode in (RunMode.Standalone, RunMode.Server):
        parser.add_argument(
            "-m", "--model",
            type=str,
            default="omol",
            help='The uma model file name (must be in MODEL_DIR) or absolute path. Default: "omol".')
        parser.add_argument(
            "-d", "--model-dir",
            metavar="MODEL_DIR",
            type=Path,
            default=default_model_path,
            help=f'The directory to look for uma model files. Default: "{default_model_path}".')
    if mode in (RunMode.Server, RunMode.Client):
        parser.add_argument(
            "-b", "--bind",
            metavar="hostname:port",
            default='127.0.0.1:8888',
            help=f'Server bind address and port. Default: 127.0.0.1:8888.')
    if mode is RunMode.Server:
        parser.add_argument(
            "-n", "--nthreads",
            metavar="N",
            type=int,
            default=1,
            help=f'Number of threads to use. Default: 1')

    return parser.parse_args(args)

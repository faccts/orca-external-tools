"""
General functions utilities used by oet
"""

from pathlib import Path
from shutil import which
import os
from typing import Iterable
import subprocess
import sys


# Energy conversition factors (Hartree -> unit)
ENERGY_CONVERSION = {"eV": 27.21138625, "kcal/mol": 627.509}

# Length converstion factors (Bohr -> unit)
LENGTH_CONVERSION = {"Ang": 0.529177210903}


def search_path(file: str | Path) -> Path:
    """
    Tries to find a file in current working directory
    and afterwards in Path. If something is found, the
    Path is returned.

    Parameters
    ----------
    file: str | Path
        Either string to file in PATH
        or Path to file

    Returns
    -------
    Path: Path to file

    Raises
    ------
    FileNotFoundError: File not found
    TypeError: Wrong input
    """
    # Step 1: Check if file exists in current directory
    local_path = Path(file)
    if local_path.exists():
        return local_path

    # Step 2: Check if file is found in system PATH
    path_str = which(file)
    if path_str:
        return Path(path_str)

    raise FileNotFoundError(
        f"File '{file}' not found in current directory or PATH."
    )

def check_path(file: str | Path) -> Path:
    """
    Checks if Path/file exists.

    Parameters
    ----------
    file: str | Path
        Either string to file in PATH
        or Path to file

    Returns
    -------
    Path: Path to file

    Raises
    ------
    FileNotFoundError: File not found
    TypeError: Wrong input
    """
    # Step 1: Check if file exists in current directory
    local_path = Path(file)
    if local_path.exists():
        return local_path

    raise FileNotFoundError(
        f"File '{file}' not found in current directory or PATH."
    )

def check_prog(prog: str | Path) -> Path:
    """
    Checks for executable

    Parameters
    ----------
    prog: str | Path
        Either string how executable is called in the PATH
        or Path to executable

    Returns
    -------
    Path: Path to program that is executable

    Raises
    ------
    PermissionError: Program not executable
    """
    # Sanitize Path
    path_to_prog = search_path(prog).resolve()
    # Check if executable
    if not os.access(path_to_prog, os.X_OK):
        raise PermissionError(f"Path '{path_to_prog}' is not executable.")
    return path_to_prog


def check_multi_progs(keys: set[str]) -> Path | None:
    """
    Checks multiple string for paths

    Parameters
    ----------
    keys: set[str]
        strings to be checked

    Returns
    -------
    Path | None: Path of executable or none
    """
    prog_path = None
    for key in keys:
        try:
            prog_path = check_prog(key)
            break
        except Exception:
            continue

    if prog_path:
        return prog_path
    else:
        return None


def print_filecontent(outfile: str | Path) -> None:
    """
    Print the file content, e.g. the output file, to STDOUT

    Parameters
    ----------
    outfile : str | Path
        The output file to print
    """
    # print the output to STDOUT
    outfile = Path(outfile)
    with open(outfile) as f:
        for line in f:  # line by line to avoid memory overflow
            print(line, end="")


def read_input(
    inputfile: str | Path,
) -> tuple[str, int, int, int, bool, str | None]:
    """
    Reads an input file written by ORCA and returns the parsed values as a tuple.

    Parameters
    ----------
    inputfile: str | Path
        Input file to read from

    Returns
    -------
    str: structure filename
    int: total molecular charge
    int: multiplicity
    int: number of cores
    bool: Do gradient?
    str | None: filename of pointcharges

    Raises
    ------
    FileNotFoundError: Input file not found
    ValueError: If input contained values in wrong format
    """
    # Get every first entry of each line of input file
    try:
        with open(inputfile, "r") as f:
            lines = [
                line.split(" ")[0].strip() for line in f.readlines() if line.strip()
            ]
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {inputfile}")
    # Save information
    try:
        xyz_filename = lines[0]
        charge = int(lines[1])
        multiplicity = int(lines[2])
        ncores = int(lines[3])
        # Check if gradient should be calculated or not
        if int(lines[4]) == 0:
            do_gradient = False
        elif int(lines[4]) == 1:
            do_gradient = True
        else:
            raise ValueError("do_gradient from ORCA input must be 0 or 1.")
    except ValueError as e:
        raise ValueError(f"Error reading ORCA input file: {e}")
    # Some sanity checks
    if multiplicity < 1:
        raise ValueError("Multiplicity must be a positive integer.")
    if ncores < 1:
        raise ValueError("NCores must be a positive integer.")
    # Optional pointcharges
    pointcharge_filename = lines[5] if len(lines) >= 6 else None
    return (
        xyz_filename,
        charge,
        multiplicity,
        ncores,
        do_gradient,
        pointcharge_filename,
    )


def get_ncores_from_input(
    inputfile: str | Path,
) -> int:
    """
    Reads an input file written by ORCA and returns the number of cores.

    Parameters
    ----------
    inputfile: str | Path
        Input file to read from

    Returns
    -------
    int: number of cores

    Raises
    ------
    FileNotFoundError: Input file not found
    ValueError: If input contained values in wrong format
    """
    # Get every first entry of each line of input file
    try:
        with open(inputfile, "r") as f:
            lines = [
                line.split(" ")[0].strip() for line in f.readlines() if line.strip()
            ]
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {inputfile}")
    # Save information
    try:
        ncores = int(lines[3])
    except ValueError as e:
        raise ValueError(f"Error reading ORCA input file: {e}")
    # Some sanity check
    if ncores < 1:
        raise ValueError("NCores must be a positive integer.")
    return ncores


def check_file(file_path: Path | str) -> bool:
    """Check whether file is present or not. Returns boolean."""
    return Path(file_path).is_file()


def write_output(
    filename: Path,
    nat: int,
    etot: float,
    grad: list[float] | None = None,
) -> None:
    """
    Writes an input for ORCA similar to external-tools format.
    Attributes
    ----------
    self._inputfile: Path
        Path to file to write to.
    nat: int
        number of atoms
    etot: int
        total energy in Hartree
    grad: list[float] | None, default: None
        gradients as plain list in Hartee/Bohr
        if not present or empty, it is not written

    Raises
    ------
    RuntimeError: If writing to file didn't work
    """
    try:
        with open(filename, "w") as f:
            output = "#\n"
            output += "# Number of atoms\n"
            output += "#\n"
            output += f"{nat}\n"
            output += "#\n"
            output += "# Total energy [Eh]\n"
            output += "#\n"
            output += f"{etot:.12e}\n"
            if grad:
                output += "#\n"
                output += "# Gradient [Eh/Bohr] A1X, A1Y, A1Z, A2X, ...\n"
                output += "#\n"
                output += "\n".join(f"{g: .12e}" for g in grad) + "\n"
            f.write(output)
    except OSError as e:
        raise RuntimeError(f"Failed to write ORCA output file {filename}: {e}")


def nat_from_xyzfile(xyz_file: str | Path) -> int:
    """
    Read number of atoms from xyz file

    Parameters
    ----------
    xyzname: str
        Name of xyz file

    Returns
    -------
    int: number of atoms
    """

    with open(xyz_file) as f:
        return int(f.readline())


def run_command(command: str | Path, outname: str | Path, args: list[str]) -> None:
    """
    Run the given command and redirect its STDOUT and STDERR to a file.
    Exits on a non-zero return code.

    Parameters
    ----------
    command : str | Path
        The command to run or path to an executable
    outname : str | Path
        The output file to be written to (overwritten!)
    args : list[str]
        arguments to be passed to the command
    """
    with open(outname, "w") as of:
        try:
            subprocess.run(
                [str(command)] + args,
                stdout=of,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as err:
            print(err)
            sys.exit(err.returncode)


def remove_file(fname: str | Path) -> None:
    """
    Remove file if present

    Parameters
    ----------
    fname: str
        filename to be removed
    """
    if isinstance(fname, str):
        fname = Path(fname)
    if fname.is_file():
        fname.unlink()
    return


def write_to_file(content: str | int | float, file: str) -> None:
    """
    Writes any str/int/float to file

    Parameters
    ----------
    content: str | int | float
        Content to be written to file
    file: str
        Name of file to be written to
    """
    # first check whether files are present and delete them if so
    remove_file(file)
    # Then, write to file
    file_path = Path(file)
    with open(file_path, "w") as f:
        f.write(f"{content}\n")


def mult_to_nue(mult: int) -> int:
    """
    Converts multiplicity to number of unpaired electrons.

    Parameters
    ----------
    mult: int
        Multiplicity

    Returns
    -------
    int: number of unpaired electrons
    """

    return mult - 1


def xyzfile_to_at_coord(
    xyzname: str | Path,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Read an XYZ file and return the atom types and coordinates.

    Parameters
    ----------
    xyzname : str | Path
        The XYZ file to read.

    Returns
    -------
    atom_types: list[str]
        A list of element symbols in order.
    coordinates: list[tuple[float, float, float]]
        A list of (x, y, z) coordinates.
    """
    atom_types = []
    coordinates = []
    xyzname = check_path(xyzname)
    with xyzname.open() as xyzf:
        natoms = int(xyzf.readline().strip())
        xyzf.readline()
        for _ in range(natoms):
            line = xyzf.readline()
            if not line:
                break
            parts = line.split()
            atom_types.append(parts[0])
            coords = (float(parts[1]), float(parts[2]), float(parts[3]))
            coordinates.append(coords)
    return atom_types, coordinates


def xyz2xsf(xyzname: str | Path, xsfname: str | Path) -> tuple[int, set[str]]:
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
    xyzname = check_path(xyzname)
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
    atomtypes: Iterable[str], nnpath: str | Path, nnext: str | None = None
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
    nnpath = search_path(nnpath).resolve()
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

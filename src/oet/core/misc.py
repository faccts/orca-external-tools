"""
General functions utilities used by oet
"""

from pathlib import Path
from shutil import which
import os


def check_path(file: str | Path) -> Path:
    """
    Checks for Path

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
    # Handle file names (strings)
    if isinstance(file, str):
        # Step 1: Check if file exists in current directory
        local_path = Path(file)
        if local_path.exists():
            return local_path.resolve()

        # Step 2: Check if file is found in system PATH
        path_str = which(file)
        if path_str:
            return Path(path_str).resolve()

        raise FileNotFoundError(
            f"File '{file}' not found in current directory or PATH."
        )

    elif isinstance(file, Path):
        if file.exists():
            return file.resolve()
        else:
            raise FileNotFoundError(f"Path '{file}' does not exist.")

    else:
        raise TypeError("Expected input to be of type str or Path.")


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
    path_to_prog = check_path(prog)
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


def print_filecontent(outfile: str | Path) -> None:
    """
    Print the file content, e.g. the output file, to STDOUT

    Parameters
    ----------
    outfile : str | Path
        The output file to print
    """
    # print the output to STDOUT
    outfile = check_path(outfile)
    with open(outfile) as f:
        for line in f:  # line by line to avoid memory overflow
            print(line, end="")


def read_orca_input(
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


def write_orca_input(
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


def nat_from_xyz(xyz_file: str | Path) -> int:
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

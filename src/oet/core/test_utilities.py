"""
Utilities used in the test suite
"""

import subprocess
from pathlib import Path

WATER = [
    ("O", 0.0000, 0.0000, 0.0000),
    ("H", 0.2774, 0.8929, 0.2544),
    ("H", 0.6068, -0.2383, -0.7169),
]

OH = [
    ("O", 0.0000, 0.0000, 0.0000),
    ("H", 0.2774, 0.8929, 0.2544),
]


def read_result_file(filename: str | Path) -> tuple[int, float, list[float]]:
    """
    Reads the engrad file written by the wrapper

    Parameters
    ----------
    filename: str
        Name of the output file

    Returns
    -------
    int: number of atoms
    float: total energy
    grad: gradient list

    Raises
    ------
    OSError: Failing to open file
    ValueError: Failed to convert the input to int/float
    """
    with open(filename) as f:
        lines = f.readlines()

    # Remove comments from '#' to the end of line:
    data_lines = [li for line in lines if (li := line.partition("#")[0].strip())]

    # Extract data
    num_atoms = int(data_lines[0])
    energy = float(data_lines[1])
    gradients = [float(val) for val in data_lines[2:]]

    return num_atoms, energy, gradients


def write_input_file(
    filename: str | Path,
    xyz_filename: str,
    charge: int,
    multiplicity: int,
    ncores: int,
    do_gradient: int | bool,
    pointcharges_filename: str | None = None,
) -> None:
    """
    Write an input file for the extopt wrapper script with the given parameters.

    Parameters
    ----------
    filename: str
        Output file name
    xyz_filename: str
        Filename of the structure file
    charge:int
        Molecular charge
    multiplicity: int
        Multiplicity
    ncores: int
        Number of cores to use
    do_gradient: int
        Compute gradient (1) or not (0)
    pointcharges_filename: str | None = None
        optional filename of the pointcharges
    """

    # Validate inputs (basic checks)
    if not xyz_filename.endswith(".xyz"):
        raise ValueError("xyz_filename did not end with '.xyz'")
    if multiplicity <= 0:
        raise ValueError("multiplicity must be a positive integer")
    if ncores <= 0:
        raise ValueError("ncores must be a positive integer")
    if type(do_gradient) is bool:
        do_gradient = int(do_gradient)
    if do_gradient not in (0, 1):
        raise ValueError("do_gradient must be 0 or 1")

    with open(filename, "w") as f:
        f.write(f"{xyz_filename} # xyz filename: string, ending in '.xyz'\n")
        f.write(f"{charge} # charge: integer\n")
        f.write(f"{multiplicity} # multiplicity: positive integer\n")
        f.write(f"{ncores} # NCores: positive integer\n")
        f.write(f"{do_gradient} # do gradient: 0 or 1\n")
        if pointcharges_filename:
            f.write(f"{pointcharges_filename} # point charge filename: string (optional)\n")
        else:
            f.write("\n")  # Write a blank line if no point charges file given


def write_xyz_file(filename: str | Path, atoms: list[tuple[str, float, float, float]]) -> None:
    """
    Write a file with the given format:

    Parameters
    ----------
    filename: str
        Output file name
    atoms: list[tuple[str, float, float, float]]
        atomic symbols and positions [(symbol, x, y, z), ...]
    """
    with open(filename, "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for atom in atoms:
            symbol, x, y, z = atom
            f.write(f"{symbol} {x:.4f} {y:.4f} {z:.4f}\n")


def run_wrapper(
        inputfile: str | Path,
        script_path: str | Path,
        outfile: str | Path,
        args: list[str] | None = None,
        timeout: float | None = 10.
) -> None:
    """
    Run the wrapper

    Parameters
    ----------
    inputfile: str | Path
        Inputfile
    script_path: str | Path
        Path to the oet script
    outfile: str | Path
        File to write the output to
    args: list[str] | None, default = None
        Additional arguments
    timeout: float | None, default: 10 s
        Default timeout time (seconds)
    """
    cmd = ["python3", str(script_path), str(inputfile)]
    if args:
        cmd += args

    with open(outfile, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)


def add_arguments(args: str | list[str], additions: list[str]) -> list[str]:
    """
    Add arguments

    Parameters
    ----------
    args: str | list[str]
        Arguments that should be extended
    additions: list[str]
        Arguments to add

    Returns
    -------
    list[str]: extended arguments
    """
    if isinstance(args, str):
        args = [args]
    args += additions
    return args


def get_filenames(basename: str) -> tuple[str, str, str, str]:
    """
    Set the filenames according to how ORCA would do and cleans any input existing
    """
    xyz_file = basename + ".xyz"
    input_file = basename + ".extinp.tmp"
    engrad_out = basename + ".engrad"
    output_file = basename + ".out"
    clear_files(basename=basename)
    return xyz_file, input_file, engrad_out, output_file


def clear_files(basename: str) -> None:
    """
    Remove every file starting with basename
    """
    dir_path = Path.cwd()
    for f in dir_path.glob(basename + "*"):
        if f.is_file():
            f.unlink()  # remove file

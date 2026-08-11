"""
General functions utilities used by oet
"""

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from shutil import which

# Energy conversion factors (Hartree -> unit)
ENERGY_CONVERSION = {"eV": 27.21138625, "kcal/mol": 627.509}

# Length conversion factors (Bohr -> unit)
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

    raise FileNotFoundError(f"File '{file}' not found in current directory or PATH.")


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

    raise FileNotFoundError(f"File '{file}' not found.")


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


def check_multi_progs(keys: Sequence[str]) -> Path | None:
    """
    Checks multiple string for paths

    Parameters
    ----------
    keys: Sequence[str]
        strings to be checked

    Returns
    -------
    Path | None: Path of executable or none
    """
    for key in keys:
        try:
            return check_prog(key)
        except Exception:
            continue
    return None


def resolve_model_file(
    model: str,
    model_dir: str | Path,
    *,
    fetch: Callable[[str], str | Path],
    alias_resolver: Callable[[str], str | None] | None = None,
    fetch_fallback: Callable[[str, Exception], str | None] | None = None,
) -> Path:
    """
    Resolve a model name / alias / absolute path to a local model file,
    downloading via `fetch` when necessary and caching the result under
    `model_dir`.

    Calculator-agnostic: callers supply the alias lookup and the fetch
    function so this helper does not depend on any specific MLIP package.

    Parameters
    ----------
    model : str
        Either an absolute path to an existing model file, or a model
        name / alias / relative filename that the calculator's registry
        understands.
    model_dir : str | Path
        Local directory used as the cache for downloaded model files.
        Created if missing; must be a directory if it exists.
    fetch : Callable[[str], str | Path]
        Function that takes a (possibly canonicalised) model name and
        returns a local filesystem path to the file (downloading on
        cache miss). Used for the actual model retrieval.
    alias_resolver : Callable[[str], str | None] | None
        Optional alias-to-canonical-name map function. Returns the
        canonical name for an alias, or None if `model` is not an alias.
        When None, `model` is passed unchanged to `fetch`.
    fetch_fallback : Callable[[str, Exception], str | None] | None
        Optional retry hook. Called with the original name and the
        exception raised by `fetch`; returns a different name to retry
        with, or None to re-raise the original exception.

    Returns
    -------
    Path
        Full path to the (possibly newly cached) model file.

    Raises
    ------
    FileNotFoundError
        If `model` is an absolute path that does not exist.
    FileExistsError
        If `model_dir` exists but is not a directory, or the cached path
        exists but is not a file.
    """
    # Absolute path: must already exist; return as-is.
    if (model_path := Path(model)).is_absolute():
        if not model_path.exists():
            raise FileNotFoundError(f'Model file "{model_path}" not found')
        return model_path

    # Resolve alias to a canonical name (if applicable).
    if alias_resolver is not None:
        canonical = alias_resolver(model)
        if canonical is not None:
            model_file = canonical
        else:
            model_file = model
    else:
        model_file = model

    # Strip any directories for the local-cache lookup.
    model_basename = Path(model_file).name

    # Make sure the cache directory exists.
    model_dir_path = Path(model_dir)
    if model_dir_path.exists() and not model_dir_path.is_dir():
        raise FileExistsError(f'Path "{model_dir}" exists but is not a directory')
    model_dir_path.mkdir(parents=True, exist_ok=True)

    # If a cached file with the same basename already exists, hand its path
    # to `fetch` so that fetchers which short-circuit on absolute paths
    # return immediately.
    cached_path = model_dir_path / model_basename
    if cached_path.exists():
        if cached_path.is_file():
            model = str(cached_path)
        else:
            raise FileExistsError(f'Path "{cached_path}" exists but is not a file')

    # Obtain the file from the fetcher; allow caller to retry with a
    # different name on failure (e.g. registry-subdirectory fallback).
    try:
        actual_path = Path(fetch(model))
    except Exception as e:
        if fetch_fallback is not None:
            retry_name = fetch_fallback(model, e)
            if retry_name is not None:
                actual_path = Path(fetch(retry_name))
            else:
                raise
        else:
            raise

    # Move the fetched file into the cache directory under its canonical
    # filename for subsequent runs.
    final_path = model_dir_path / actual_path.name
    if not (final_path.exists() and final_path.samefile(actual_path)):
        shutil.move(actual_path, final_path)
    return final_path


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
            lines = [line.split(" ")[0].strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {inputfile}")
    # Save information
    try:
        xyz_filename = Path(lines[0]).name
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
            lines = [line.split(" ")[0].strip() for line in f.readlines() if line.strip()]
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


def copy_files_to_tmpdir(files_to_copy: list[Path], tmp_dir: Path) -> list[Path]:
    """
    Makes a temporary directory and copies files

    Parameters
    ----------
    files_to_copy: list[Path]
        Paths of the files that should be copied
    tmp_dir: Path
        Path to the tmp directory to be created

    Returns
    -------
    list[Path]: List of the Paths of the copied files
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)  # FIXME both False?
    final_file_paths = []
    for file_path in files_to_copy:
        new_path = tmp_dir / file_path.name
        shutil.copy2(file_path, new_path)
        final_file_paths.append(new_path)
    return final_file_paths


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

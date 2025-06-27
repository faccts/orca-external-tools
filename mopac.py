#!/usr/bin/env python3
"""
This is a simple wrapper for the MOPAC binary (github.com/openmopac/mopac),
compatible with ORCA's ExtTool interface.
It converts ORCA-generated input files to a MOPAC input, runs MOPAC,
parses its output, converts energies and gradients from kcal to atomic units,
and writes the energy/gradient file for ORCA.
"""

from __future__ import annotations

import shutil
import sys
import subprocess
from pathlib import Path
from argparse import ArgumentParser
from typing import Iterable

# Path to the MOPAC executable. If None, the script will search for one of the names in MOPAC_NAMES.
MOPAC_EXE: str | Path | None = None
MOPAC_NAMES: list[str] = ["mopac", "otool_mopac"]

# ----------------------------------------------------------------------------------------------------------------------
# Common functions

def strip_comments(s: str) -> str:
    """Strip comment starting with '#' and continuing until the end of the string. Also strip whitespace."""
    return s.split("#")[0].strip()

def enforce_path_object(fname: str | Path) -> Path:
    """Ensure the given filename is a Path object."""
    if isinstance(fname, str):
        return Path(fname)
    elif isinstance(fname, Path):
        return fname
    else:
        raise TypeError("Input must be a string or a Path object.")

def read_input(inpfile: str | Path) -> tuple[str, int, int, int, bool]:
    """Read the ORCA-generated input file.

    Expected file format (one per line):
        1. Name of the XYZ coordinate file
        2. Total charge (integer)
        3. Spin multiplicity (integer)
        4. Number of cores (integer)
        5. Gradient flag (0 or 1)

    Returns
    -------
    tuple[str, int, int, int, bool]:
        xyzname, charge, multiplicity, ncores, dograd
    """
    inpfile = enforce_path_object(inpfile)
    with inpfile.open() as f:
        xyzname = strip_comments(f.readline())
        charge = int(strip_comments(f.readline()))
        mult = int(strip_comments(f.readline()))
        ncores = int(strip_comments(f.readline()))
        dograd = bool(int(strip_comments(f.readline())))
        # TODO: Handle point charges if needed.
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
        The engrad file.
    natoms : int
        Number of atoms.
    energy : float
        Total energy (in Eh).
    dograd : bool
        Whether the gradient is computed.
    gradient : Iterable[float], optional
        The gradient (Eh/Bohr for each atom), if available.
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
    Run the given command and redirect its STDOUT and STDERR to a file.
    Exits on non-zero return code.

    Parameters
    ----------
    command : str | Path
        The command or path to an executable.
    outname : str | Path
        The file to which output is written (overwritten).
    *args : tuple[str, ...]
        Additional arguments to pass to the command.
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

def clean_output(outfile: str | Path, results_file: str | Path, namespace: str) -> None:
    """
    Print the output file to STDOUT and remove all files starting with `namespace`.

    Parameters
    ----------
    outfile : str | Path
        The output file to print.
    namespace : str
        The starting string of all files to remove.
    """
    outfile = enforce_path_object(outfile)
    with open(outfile) as f:
        for line in f:
            print(line, end="")
    results_file = enforce_path_object(results_file)
    with open(results_file) as f:
        for line in f:
            print(line, end="")
    for f in Path(".").glob(f"*{namespace}*"):
        if f.name.endswith (".engrad"):
            continue  # Skip the .engrad file used by orca
        if f.name.endswith (".xyz"):
            continue  # Don't delete the xyz file
        f.unlink()

# ----------------------------------------------------------------------------------------------------------------------
# New functions for MOPAC support

def write_mopac_input(xyz_file: str | Path, method: str, multiplicity: int, charge: int,
                      ncores: int, dograd: bool, output_file: str | Path) -> None:
    """
    Create a MOPAC input file from the ORCA input variables and the provided XYZ file.

    The MOPAC input file format is as follows:

       {method} 1SCF XYZ MS={spin} CHARGE={charge} THREADS={ncores}[ UHF][ GRADIENTS]
       Generated by ORCA-to-MOPAC wrapper

       <coordinates from xyz file>

    Here the spin is calculated correctly from the multiplicity (2S+1) using:
         S = (multiplicity - 1) * 0.5

    Also:
      - "CHARGE={charge}" reports the total charge of the system.
      - "THREADS={ncores}" sets the number of processors.
      - "UHF" is appended if multiplicity is not 1.
      - "GRADIENTS" is appended if gradient computation is requested.
    """
    xyz_file = enforce_path_object(xyz_file)
    with xyz_file.open() as f:
        lines = f.readlines()
    # If file is in standard XYZ format (first line is atom count), skip first two lines.
    if lines and lines[0].strip().isdigit():
        coord_lines = lines[2:]
    else:
        coord_lines = lines

    spin = (multiplicity - 1) * 0.5
    # Format spin to one decimal place (e.g., "0.5" for a doublet)
    spin_formatted = f"{spin:.1f}"
    header = f"{method} 1SCF XYZ MS={spin_formatted} CHARGE={charge} THREADS={ncores}"
    if multiplicity != 1:
        header += " UHF"
    if dograd:
        header += " GRADIENTS"

    with enforce_path_object(output_file).open("w") as f_out:
        f_out.write(header + "\n")
        f_out.write("Generated by ORCA-to-MOPAC wrapper\n\n")
        for line in coord_lines:
            f_out.write(line)

def run_mopac(mopacexe: str | Path, inp_file: str | Path, logfile: str | Path) -> None:
    """
    Run the MOPAC program with the given input file and redirect its STDOUT and STDERR to a logfile.

    Parameters
    ----------
    mopacexe : str | Path
        Path to the MOPAC binary.
    inp_file : str | Path
        The generated MOPAC input file.
    logfile : str | Path
        The file where STDOUT and STDERR will be written.
    """
    run_command(mopacexe, logfile, str(inp_file))

def read_mopac_out(namespace: str, mopac_out: str | Path, mopac_results_out: str | Path, natoms: int, dograd: bool) -> tuple[float, list[float]]:
    """
    Read the output from MOPAC.

    It extracts the energy (by finding the "FINAL HEAT OF FORMATION" line)
    and, if requested, reads the gradient from the corresponding output block.
    
    The energy is originally in kcal/mol and the gradient components in kcal/Å.
    They are converted to atomic units as follows:
      - Energy: 1 kcal/mol = 1/627.5095 Eh
      - Gradient: 1 (kcal/Å) = (1/627.5095) / 1.889725988 Eh/Bohr

    Parameters
    ----------
    namespace : str
        Filename prefix for the MOPAC output files.
    mopac_out : str | Path
        The main MOPAC output file.
    natoms : int
        Number of atoms in the system.
    dograd : bool
        Whether to read the gradient.

    Returns
    -------
    tuple[float, list[float]]:
        energy : The computed energy in Eh.
        gradient : The gradient in Eh/Bohr (if computed), as a flat list of floats.
    """
    # Define conversion factors.
    KCAL_TO_EH = 1/627.5095            # 1 kcal/mol in Eh
    ANGSTROM_TO_BOHR = 1.889725988      # 1 Å in bohr
    GRAD_CONV_FACTOR = KCAL_TO_EH / ANGSTROM_TO_BOHR

    energy = None
    gradient = []
    mopac_out = enforce_path_object(mopac_out)
    with mopac_out.open() as f:
        for line in f:
            if "FINAL HEAT OF FORMATION" in line:
                # Expect a line like: "FINAL HEAT OF FORMATION =   -123.456 ..."
                tokens = line.split()
                if "=" in tokens:
                    idx = tokens.index("=")
                    try:
                        # Convert energy from kcal/mol to Eh.
                        energy = float(tokens[idx + 1]) * KCAL_TO_EH
                    except (IndexError, ValueError):
                        pass
                    break
    # Check also results file written by Mopac for energy
    if energy is None:
       mopac_results_out = enforce_path_object(mopac_results_out)
       with mopac_results_out.open() as f:
           for line in f:
               if "FINAL HEAT OF FORMATION" in line:
                   # Expect a line like: "FINAL HEAT OF FORMATION =   -123.456 ..."
                   tokens = line.split()
                   if "=" in tokens:
                       idx = tokens.index("=")
                       try:
                           # Convert energy from kcal/mol to Eh.
                           energy = float(tokens[idx + 1]) * KCAL_TO_EH
                       except (IndexError, ValueError):
                           pass
                       break

    if energy is None:
        print("Could not find energy in MOPAC output.")
        sys.exit(1)

    if dograd:
        # Read the entire output file into memory.
        with mopac_out.open() as f:
            lines = f.readlines()
        with mopac_results_out.open() as f:
            lines2 = f.readlines()
        lines = lines + lines2

        # Locate the header of the gradient table.
        header_idx = None
        for i, line in enumerate(lines):
            if "PARAMETER" in line and "GRADIENT" in line:
                header_idx = i + 1  # Data is expected to start on the next line.
                break
        if header_idx is None:
            print("Could not find gradient header in MOPAC output.")
            sys.exit(1)

        # Process the gradient table: each line should correspond to one gradient component.
        for line in lines[header_idx:]:
            line = line.strip()
            if not line:
                continue  # Skip empty lines.
            tokens = line.split()
            # Expect at least 7 tokens:
            # [0]: parameter index, [1]: atom number, [2]: element,
            # [3] & [4]: coordinate description (e.g., "CARTESIAN X"),
            # [5]: VALUE, [6]: GRADIENT, [7] (if present): unit.
            if len(tokens) < 7:
                continue
            try:
                # Extract the gradient value (seventh token) in kcal/Å and convert it to Eh/Bohr.
                grad_val = float(tokens[6]) * GRAD_CONV_FACTOR
            except ValueError:
                continue
            gradient.append(grad_val)
            # Stop once we have gathered the expected 3 * natoms entries.
            if len(gradient) >= 3 * natoms:
                break

        if len(gradient) != 3 * natoms:
            print(f"Gradient entries ({len(gradient)}) do not match expected 3x number of atoms ({3 * natoms}).")
            sys.exit(1)

    return energy, gradient

# ----------------------------------------------------------------------------------------------------------------------
# Main function

def main(argv: list[str]) -> None:
    """Main function to run the MOPAC wrapper script."""
    # Try to locate the MOPAC executable if not provided.
    mopacexe = MOPAC_EXE
    if not mopacexe:
        for mopac in MOPAC_NAMES:
            if mopacexe := shutil.which(mopac):
                break

    parser = ArgumentParser(
        prog=argv[0],
        allow_abbrev=False,
        description="Wrapper for MOPAC, compatible with ORCA's otool_external interface. "
                    "Reads the ORCA-generated input <inputfile>, converts it to a MOPAC input, "
                    "calls MOPAC, parses its output from an output file, converts energies and gradients to atomic units, "
                    "and writes the BaseName.engrad file for ORCA.",
    )
    parser.add_argument("inputfile", help="ORCA input file")
    parser.add_argument("--method", default="PM6-D3H4X",
                        help="MOPAC method (default: PM6-D3H4X)")
    parser.add_argument("-e", "--exe", metavar="MOPACEXE", dest="mopacexe",
                        required=(not mopacexe),
                        help="Path to the MOPAC executable" + (f' (default: "{mopacexe}")' if mopacexe else ""),
                        default=mopacexe)
    args, mopac_args = parser.parse_known_args(argv[1:])

    mopacexe = enforce_path_object(args.mopacexe).expanduser().resolve()

    # Read the ORCA-generated input.
    xyzname, charge, mult, ncores, dograd = read_input(args.inputfile)

    # Derive base names from the xyz file.
    basename = xyzname.rstrip(".xyz")
    orca_engrad = basename + ".engrad"
    mopac_inp = basename + ".mop"
    # Note: MOPAC writes its output to an output file (with extension .out); names with ".mop" are not tolerated.
    mopac_namespace = basename
    # Mopac writes the results automatically to the file mopac_results_out
    mopac_results_out = basename + ".out"
    # Standard output as mopac_out
    mopac_out = "mopac_" + basename + ".out"

    # Create the MOPAC input file, including CHARGE, THREADS, and the correctly calculated spin.
    write_mopac_input(xyzname, args.method, mult, charge, ncores, dograd, mopac_inp)

    # Run MOPAC (any additional command-line arguments are ignored).
    run_mopac(mopacexe, mopac_inp, mopac_out)

    # Get the number of atoms from the XYZ file.
    with open(xyzname) as f:
        first_line = f.readline().strip()
        if first_line.isdigit():
            natoms = int(first_line)
        else:
            # If not standard XYZ, assume one coordinate line per atom.
            natoms = len(f.readlines())

    # Parse the MOPAC output (from the output file) and, if requested, the gradient.
    energy, gradient = read_mopac_out(mopac_namespace, mopac_out, mopac_results_out, natoms, dograd)

    # Write the ORCA engrad file.
    write_engrad(orca_engrad, natoms, energy, dograd, gradient)

    # Print the MOPAC output to STDOUT and remove intermediate files.
    clean_output(mopac_out, mopac_results_out, mopac_namespace)

if __name__ == "__main__":
    main(sys.argv)

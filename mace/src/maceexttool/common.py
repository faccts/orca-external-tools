#!/usr/bin/env python3

from __future__ import annotations

import subprocess
from enum import StrEnum, auto
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable
import os
import socket
import numpy as np
from ase import Atoms


# Energy and length conversion to atomic units (same as UMA)
ENERGY_CONVERSION = {"eV": 27.21138625}
LENGTH_CONVERSION = {"Ang": 0.529177210903}


def strip_comments(s: str) -> str:
    return s.split("#")[0].strip()


def enforce_path_object(fname: str | Path) -> Path:
    if isinstance(fname, str):
        return Path(fname)
    elif isinstance(fname, Path):
        return fname
    else:
        raise TypeError("Input must be a string or a Path object.")


def read_input(inpfile: str | Path) -> tuple[str, int, int, int, bool]:
    inpfile = enforce_path_object(inpfile)
    with inpfile.open() as f:
        xyzname = strip_comments(f.readline())
        charge = int(strip_comments(f.readline()))
        mult = int(strip_comments(f.readline()))
        ncores = int(strip_comments(f.readline()))
        dograd = bool(int(strip_comments(f.readline())))
    return xyzname, charge, mult, ncores, dograd


def write_engrad(
    outfile: str | Path,
    natoms: int,
    energy: float,
    dograd: bool,
    gradient: Iterable[float] = None,
) -> None:
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
    outfile = enforce_path_object(outfile)
    with outfile.open() as f:
        for line in f:
            print(line, end="")
    for f in Path(".").glob(namespace + "*"):
        f.unlink()


def read_xyzfile(xyzname: str | Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    atom_types: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
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


def process_output(atoms: Atoms) -> tuple[float, list[float]]:
    """Convert ASE outputs (eV, Ang) to ORCA units (Eh, Eh/Bohr)."""
    energy = atoms.get_potential_energy() / ENERGY_CONVERSION["eV"]
    gradient: list[float] = []
    try:
        forces = atoms.get_forces()
        fac = -LENGTH_CONVERSION["Ang"] / ENERGY_CONVERSION["eV"]
        gradient = (fac * np.asarray(forces)).flatten().tolist()
    except Exception:
        pass
    return energy, gradient


class RunMode(StrEnum):
    Server = auto()
    Client = auto()
    Standalone = auto()


ProgName = {
    RunMode.Server: "maceserver",
    RunMode.Client: "maceclient",
    RunMode.Standalone: "maceexttool",
}


def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex((host, port))
        return result != 0


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def cli_parse(args: list[str], mode: RunMode) -> Namespace:
    """Parse CLI for MACE wrapper; options differ slightly by mode."""
    parser = ArgumentParser(
        prog=ProgName[mode],
        description=f'ORCA "external tool" interface for MACE calculations ({mode} mode)'
    )

    if mode in (RunMode.Standalone, RunMode.Client):
        parser.add_argument("inputfile", help="ORCA-generated input file.")

    # Common MACE options (for Standalone and Server)
    if mode in (RunMode.Standalone, RunMode.Server):
        parser.add_argument(
            "-s", "--suite",
            choices=["mp", "omol", "mace-mp", "mace-omol"],
            default="omol",
            help="Select MACE suite: mp/mace-mp or omol/mace-omol. Default: omol"
        )
        parser.add_argument(
            "-m", "--model",
            type=str,
            default=None,
            help="Model spec or local path. MP: small/medium/large/medium-mpa-0/... OMOL: extra_large or path."
        )
        parser.add_argument(
            "--default-dtype",
            choices=["float32", "float64"],
            default=None,
            help="Default float precision (recommended: float64 for opt, float32 for MD)."
        )
        parser.add_argument(
            "--device",
            type=str,
            default="",
            help="Device string for torch/ASE calculator (e.g., cuda, cpu)."
        )
        # MP-specific extras
        parser.add_argument(
            "--dispersion",
            action="store_true",
            help="Enable D3 dispersion (MP suite only)."
        )
        parser.add_argument("--damping", type=str, default="bj",
                            help="D3 damping (zero,bj,zerom,bjm). MP only.")
        parser.add_argument("--dispersion-xc", type=str, default="pbe",
                            help="XC functional for D3. MP only.")
        parser.add_argument("--dispersion-cutoff", type=float, default=None,
                            help="Cutoff radius for D3 in Bohr (default: 40 Bohr). MP only.")
        parser.add_argument("--head", type=str, default=None,
                            help="Advanced: select MACE head (MP only).")

    if mode in RunMode.Server:
        parser.add_argument(
            "-b", "--bind", metavar="hostname:port", default="127.0.0.1:8888",
            help="Server bind address and port. Default: 127.0.0.1:8888."
        )
    if mode is RunMode.Client:
        default_bind = os.getenv("MACE_BIND", "127.0.0.1:8888")
        parser.add_argument(
            "-b", "--bind", metavar="hostname:port", default=default_bind,
            help="Server bind address and port."
        )
    if mode is RunMode.Server:
        parser.add_argument("-n", "--nthreads", metavar="N", type=int, default=1,
                            help="Number of threads to use. Default: 1")

    parsed = parser.parse_args(args)

    if mode is RunMode.Server:
        try:
            host, port = parsed.bind.split(":")
            port = int(port)
        except ValueError:
            parser.error("Invalid --bind format. Use host:port")
        if not is_port_available(host, port):
            print(f"Port {port} on {host} is already in use. Selecting a free one...")
            port = get_free_port()
            parsed.bind = f"{host}:{port}"
            os.system(f"export MACE_BIND={host}:{port}")
            print(f"Using new port: {parsed.bind}")

    return parsed

#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse
from collections.abc import Sequence
from pathlib import Path
import shutil
from xmlrpc.client import Boolean


# Available extras
EXTRAS = ["uma", "aimnet2", "mlatom"]


def create_venv(venv_dir: Path) -> None:
    """
    Create virtual environment, if not present.

    Parameters
    ----------
    venv_dir: Path
        Path to the virtual environment
    """
    print(f"Creating virtual environment in '{venv_dir}'...")
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    print("Virtual environment created.")


def pip_install_target(venv_dir: Path, script_dir: Path) -> None:
    """
    Install oet to virtual environment

    Parameters
    ----------
    venv_dir: Path
        Path to the virtual environment
    script_dir: Path
        Path to the final scripts
    """
    pip_path = (
        venv_dir
        / ("Scripts" if os.name == "nt" else "bin")
        / ("pip.exe" if os.name == "nt" else "pip")
    )
    if not pip_path.exists():
        raise RuntimeError(f"pip not found in venv: {pip_path}")

    script_dir.mkdir(parents=True, exist_ok=True)

    print(f"Installing package to {script_dir} using pip in venv...")
    subprocess.check_call(
        [
            str(pip_path),
            "install",
            "-e" ".",  # install from current directory
            #f"--target={script_dir}",
        ]
    )
    print("Installation complete.")


def install_extra_requirements(venv_dir: Path, extras: list[str]) -> None:
    """
    Installs extra requirements located in the requirements directory.
    Use for, e.g., uma or AIMNet2

    Parameters
    ----------
    venv_dir: Path
        Path to the virtual environment that should be installed to.
    extras: list[str]
        Requirements that should be additionally installed
    """
    pip_path = (
        venv_dir
        / ("bin")
        / ("pip.exe" if os.name == "nt" else "pip")
    )

    for extra in extras:
        req_path = Path("requirements") / f"{extra}.txt"
        if not req_path.exists():
            print(f"Requirements file not found: {req_path}, skipping.")
            continue

        print(f"Installing extra requirements from {req_path}...")
        subprocess.check_call([str(pip_path), "install", "-r", str(req_path)])


def install_dev_tools(venv_dir: Path) -> None:
    """
    Installs the developer tools like nox.

    Parameters
    ----------
    venv_dir: Path
        Path to the virtual environment that should be installed to.
    """
    pip_path = (
        venv_dir
        / ("bin")
        / ("pip.exe" if os.name == "nt" else "pip")
    )

    print("Installing developer tools.")
    subprocess.check_call([str(pip_path), "install", ".[dev]"])


def copy_oet_scripts(venv_dir: Path, dest_dir: Path, extras: Sequence[str]) -> None:
    """
    Copy all scripts starting with 'oet' from venv/bin to the destination directory.
    Scripts which are "extras" are only copied if actually installed.

    Parameters
    ----------
    venv_dir : Path
        Path to the virtual environment root directory.
    dest_dir : Path
        Directory where the scripts should be copied.
    extras : Sequence[str]
        Installed extras
    """
    bin_dir = venv_dir / ("bin")
    if not bin_dir.exists():
        raise FileNotFoundError(f"bin directory not found in venv: {bin_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for script in bin_dir.glob("oet*"):
        if script.is_file():
            # skip not installed extras
            if (module := script.name.removeprefix("oet_")) in EXTRAS and module not in extras:
                continue
            target = dest_dir / script.name
            shutil.copy2(script, target)  # copy with metadata (executable bit)
            print(f"Copied {script.name} → {target}")
            count += 1

    if count == 0:
        print("No oet_ scripts found.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Installation for orca-external-tools package."
    )
    parser.add_argument(
        "--venv-dir",
        "-vd",
        type=Path,
        default=Path(".venv"),
        help="Path to the virtual environment directory",
    )
    parser.add_argument(
        "--script-dir",
        "-sd",
        type=Path,
        default=Path("./bin"),
        help="Custom directory where bin/packages should be installed",
    )
    parser.add_argument(
        "--extra",
        "-e",
        nargs="+",
        choices=EXTRAS,
        default=[],
        help="Install optional extra package sets from requirements/<name>.txt",
    )
    parser.add_argument(
        "--dev",
        "-d",
        action="store_true",
        help="Install optional developer tools.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create venv
    if not args.venv_dir.exists():
        create_venv(args.venv_dir)
    else:
        print(
            f"Virtual environment already exists in '{args.venv_dir}'.\n"
            "Installing oet to this venv."
        )
    # Install dev tools (nox)
    if args.dev:
        install_dev_tools(args.venv_dir)

    # Install extra dependencies in requirements.txt
    if args.extra:
        install_extra_requirements(args.venv_dir, args.extra)

    # Install oet
    pip_install_target(args.venv_dir, args.script_dir)

    # Copy scripts for easier usability
    copy_oet_scripts(venv_dir=args.venv_dir, dest_dir=args.script_dir, extras=args.extra)

if __name__ == "__main__":
    main()

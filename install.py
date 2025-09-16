#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse
from pathlib import Path
import shutil


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
        / ("Scripts" if os.name == "nt" else "bin")
        / ("pip.exe" if os.name == "nt" else "pip")
    )

    for extra in extras:
        req_path = Path("requirements") / f"{extra}.txt"
        if not req_path.exists():
            print(f"Requirements file not found: {req_path}, skipping.")
            continue

        print(f"Installing extra requirements from {req_path}...")
        subprocess.check_call([str(pip_path), "install", "-r", str(req_path)])


def copy_otool_scripts(venv_dir: Path, dest_dir: Path) -> None:
    """
    Copy all scripts starting with 'otool_' from venv/bin to the destination directory.

    Parameters
    ----------
    venv_dir : Path
        Path to the virtual environment root directory.
    dest_dir : Path
        Directory where the scripts should be copied.
    """
    bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    if not bin_dir.exists():
        raise FileNotFoundError(f"bin directory not found in venv: {bin_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for script in bin_dir.glob("otool_*"):
        if script.is_file():
            target = dest_dir / script.name
            shutil.copy2(script, target)  # copy with metadata (executable bit)
            print(f"Copied {script.name} → {target}")
            count += 1

    if count == 0:
        print("No otool_ scripts found.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Installation for orca-exteranl-tools package."
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
        default=Path("./scripts"),
        help="Custom directory where scripts/packages should be installed",
    )
    parser.add_argument(
        "--extra",
        "-e",
        nargs="*",
        choices=["uma", "aimnet2", "mlatom"],
        default=[],
        help="Optional extra package sets to install from requirements/<name>.txt",
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

    # Install extra dependencies in requirements.txt
    if args.extra:
        install_extra_requirements(args.venv_dir, args.extra)

    # Install oet
    pip_install_target(args.venv_dir, args.script_dir)

    # Copy scripts for easier usability
    copy_otool_scripts(venv_dir=args.venv_dir ,dest_dir=args.script_dir)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import sys
import time

import torch
from ase import Atoms

from maceexttool import common, calculator


def run_mace(
    atom_types: list[str],
    coordinates: list[tuple[float, float, float]],
    charge: int,
    mult: int,
    suite: str,
    model: str | None,
    dograd: bool,
    nthreads: int,
    default_dtype: str | None,
    device: str,
    dispersion: bool,
    damping: str,
    dispersion_xc: str,
    dispersion_cutoff: float | None,
    head: str | None,
) -> tuple[float, list[float]]:
    """Run a MACE calculation and return energy and gradient in ORCA units."""
    calc = calculator.init(
        suite=suite,
        model=model,
        device=device,
        default_dtype=default_dtype,
        dispersion=dispersion,
        damping=damping,
        dispersion_xc=dispersion_xc,
        dispersion_cutoff=dispersion_cutoff,
        head=head,
    )

    torch.set_num_threads(nthreads)

    atoms = Atoms(symbols=atom_types, positions=coordinates)
    atoms.info = {"charge": charge, "spin": mult}
    atoms.calc = calc

    return common.process_output(atoms)


def run(arglist: list[str]):
    args = common.cli_parse(arglist, mode=common.RunMode.Standalone)

    # Canonicalize suite and set defaults
    suite = args.suite
    if suite.startswith("mace-"):
        suite = suite.split("-", 1)[1]
    default_dtype = args.default_dtype
    if default_dtype is None:
        default_dtype = "float64" if suite == "omol" else "float32"

    # read ORCA input
    xyzname, charge, mult, ncores, dograd = common.read_input(args.inputfile)
    basename = xyzname.removesuffix(".xyz")
    orca_engrad = basename + ".engrad"

    atom_types, coordinates = common.read_xyzfile(xyzname)
    natoms = len(atom_types)

    start_time = time.perf_counter()
    energy, gradient = run_mace(
        atom_types=atom_types,
        coordinates=coordinates,
        charge=charge,
        mult=mult,
        suite=suite,
        model=args.model,
        dograd=dograd,
        nthreads=ncores,
        default_dtype=default_dtype,
        device=args.device,
        dispersion=args.dispersion,
        damping=args.damping,
        dispersion_xc=args.dispersion_xc,
        dispersion_cutoff=args.dispersion_cutoff,
        head=args.head,
    )

    common.write_engrad(orca_engrad, natoms, energy, dograd, gradient)
    print("Total time:  {:6.3f} seconds".format(time.perf_counter() - start_time))


def main():
    run(sys.argv[1:])


if __name__ == "__main__":
    main()

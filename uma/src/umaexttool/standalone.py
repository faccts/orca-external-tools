#!/usr/bin/env python3

from __future__ import annotations

import time
start_time = time.perf_counter()

import sys

import torch

from umaexttool import common, calculator

from ase import Atoms

import_time = time.perf_counter() - start_time


def run_uma(
        atom_types: list[str],
        coordinates: list[tuple[float, float, float]],
        charge: int,
        mult: int,
        model: str,
        dograd: bool,
        nthreads: int,
) -> tuple[float, list[float]]:
    """
    Runs an UMA calculation.

    Parameters
    ----------
    atom_types : list[str]
        List of element symbols (e.g., ["O", "H", "H"])
    coordinates : list[tuple[float, float, float]]
        List of (x, y, z) coordinates
    charge : int
        Molecular charge
    mult : int
        Spin multiplicity
    model : str
        The UMA model to use (name)
    dograd : bool
        Whether to compute the gradient (currently always computed)
    nthreads : int
        Number of threads to use for the calculation

    Returns
    -------
    tuple[float, list[float]]
        energy : float
            The computed energy (Eh)
        gradient : list[float]
            Flattened gradient vector (Eh/Bohr), if computed, otherwise empty.
    """

    # set up calculator
    calc = calculator.init(model)
    # set the number of threads
    torch.set_num_threads(nthreads)

    # make ase atoms object for calculation
    atoms=Atoms(symbols=atom_types, positions=coordinates)
    atoms.info = {"charge": charge, "spin": mult}

    atoms.calc = calc

    return common.process_output(atoms)


def run(arglist: list[str]):
    """Run the UMA calculation on a given structure using a specified model file."""
    args = common.cli_parse(arglist, mode=common.RunMode.Standalone)

    # get the model as a plain string
    model = str(args.model)

    # read the ORCA-generated input
    xyzname, charge, mult, ncores, dograd = common.read_input(args.inputfile)

    # set filenames
    basename = xyzname.rstrip(".xyz")
    orca_engrad = basename + ".engrad"

    # process the XYZ file
    atom_types, coordinates = common.read_xyzfile(xyzname)
    natoms = len(atom_types)
    # run UMA calculator
    energy, gradient = run_uma(atom_types=atom_types, coordinates=coordinates, charge=charge, mult=mult,
                                   model=model, dograd=dograd, nthreads=ncores)
    # convert to ORCA engrad
    common.write_engrad(orca_engrad, natoms, energy, dograd, gradient)

    # Print total timing
    print("Total time:  {:6.3f} seconds".format(time.perf_counter() - start_time))


def main():
    """Entry point for CLI execution"""
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
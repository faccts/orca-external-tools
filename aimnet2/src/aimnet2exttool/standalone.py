#!/usr/bin/env python3

from __future__ import annotations

# FIXME debug
import time
start_time = time.perf_counter()

import sys

import torch

from aimnet2exttool import common, calculator


def run_aimnet2(
        atom_types: list[str],
        coordinates: list[tuple[float, float, float]],
        charge: int,
        mult: int,
        model: str,
        dograd: bool,
        nthreads: int,
) -> tuple[float, list[float]]:
    """
    Runs an AIMNet2 calculation.

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
        The AIMNet2 model to use (file path or name)
    dograd : bool
        Whether to compute the gradient
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

    kwargs = common.serialize_input(atom_types=atom_types,
                                    coordinates=coordinates,
                                    mult=mult,
                                    charge=charge,
                                    dograd=dograd)

    calc = calculator.init(model=model)

    # set the number of threads
    torch.set_num_threads(nthreads)

    start_run = time.perf_counter()  # FIXME debug
    result = calc(**kwargs)
    end_run = time.perf_counter()  # FIXME debug

    return common.process_output(result)


def run(arglist: list[str]):
    """Run the AIMNet2 calculation on a given structure using a specified model file."""
    args = common.cli_parse(arglist, mode=common.RunMode.Standalone)

    # get the absolute path of the model file as a plain string
    model = str(args.model_dir / args.model)

    # read the ORCA-generated input
    xyzname, charge, mult, ncores, dograd = common.read_input(args.inputfile)

    # set filenames
    basename = xyzname.rstrip(".xyz")
    orca_engrad = basename + ".engrad"

    # process the XYZ file
    atom_types, coordinates = common.read_xyzfile(xyzname)
    natoms = len(atom_types)
    # run aimnet2 calculator
    energy, gradient = run_aimnet2(atom_types=atom_types, coordinates=coordinates, charge=charge, mult=mult,
                                   model=model, dograd=dograd, nthreads=ncores)
    # convert to ORCA engrad
    common.write_engrad(orca_engrad, natoms, energy, dograd, gradient)

    print("Total time:  {:6.3f} seconds".format(time.perf_counter() - start_time))


def main():
    """Entry point for CLI execution"""
    run(sys.argv[1:])


if __name__ == "__main__":
    main()

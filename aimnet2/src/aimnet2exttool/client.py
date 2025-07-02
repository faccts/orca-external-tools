#!/usr/bin/env python3

from __future__ import annotations

# FIXME debug
import time
start_time = time.perf_counter()

import requests
import sys

from aimnet2exttool import common


def submit_aimnet2(server_url: str,
        atom_types: list[str],
        coordinates: list[tuple[float, float, float]],
        charge: int,
        mult: int,
        dograd: bool,
        nthreads: int,
) -> tuple[float, list[float]]:
    """
    Sends an AIMNet2 calculation to the server and returns the result.

    Parameters
    ----------
    server_url : str
        Host:port address of the server
    atom_types : list[str]
        List of element symbols (e.g., ["O", "H", "H"])
    coordinates : list[tuple[float, float, float]]
        List of (x, y, z) coordinates
    charge : int
        Molecular charge
    mult : int
        Spin multiplicity
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

    payload = common.serialize_input(atom_types=atom_types,
                                     coordinates=coordinates,
                                     mult=mult,
                                     charge=charge,
                                     dograd=dograd)
    # add the number of threads
    payload['nthreads'] = nthreads

    start_run = time.perf_counter()  # FIXME debug

    response = requests.post('http://' + server_url + "/calculate", json=payload)
    response.raise_for_status()

    data = response.json()
    energy = data["energy"]
    gradient = data["gradient"]

    end_run = time.perf_counter()  # FIXME debug

    # FIXME debug
    run_time = end_run - start_run
    print("Calc time:   {:6.3f} seconds".format(run_time))

    return energy, gradient


def run(arglist: list[str]):
    """Run a calculation on a given structure on the AIMNet2 server."""
    args = common.cli_parse(arglist, mode=common.RunMode.Client)

    # read the ORCA-generated input
    xyzname, charge, mult, ncores, dograd = common.read_input(args.inputfile)

    # set filenames
    basename = xyzname.rstrip(".xyz")
    orca_engrad = basename + ".engrad"

    # process the XYZ file
    atom_types, coordinates = common.read_xyzfile(xyzname)
    natoms = len(atom_types)

    # submit to the aimnet2 calculator server
    energy, gradient = submit_aimnet2(server_url=args.bind, atom_types=atom_types, coordinates=coordinates,
                                      charge=charge, mult=mult, dograd=dograd, nthreads=ncores)
    # convert to ORCA engrad
    common.write_engrad(orca_engrad, natoms, energy, dograd, gradient)

    # FIXME debug timings
    print("Total time:  {:6.3f} seconds".format(time.perf_counter() - start_time))


def main():
    """Entry point for CLI execution"""
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
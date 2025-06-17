#!/usr/bin/env python3

from __future__ import annotations

#Timings including imports
import time
start_time = time.perf_counter()

import requests
import sys

from umaexttool import common

end_import = time.perf_counter()

def submit_uma(server_url: str,
        atom_types: list[str],
        coordinates: list[tuple[float, float, float]],
        charge: int,
        mult: int,
        dograd: bool,
        nthreads: int
) -> tuple[float, list[float]]:
    """
    Sends an uma calculation to the server and returns the result.

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
        Whether to compute the gradient (currently ignored; gradients are always computed)
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

    payload = {
        "atom_types": atom_types,
        "coordinates": coordinates,
        "mult": mult,
        "charge": charge,
        "dograd": dograd,
        "nthreads": nthreads
        }

    try:
        response = requests.post('http://' + server_url + "/calculate", json=payload)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as http_err:
        print("HTTP error occurred:", http_err)
        print("The server is probably not running.")
        print("Please start the server with the umaserver.sh script.")
        sys.exit(1)
    except requests.exceptions.ConnectionError as conn_err:
        print("Connection error: could not reach the server.")
        print("Details:", conn_err)
        sys.exit(1)
    except requests.exceptions.Timeout as timeout_err:
        print("Request to UMA server timed out:", timeout_err)
        sys.exit(1)
    except requests.exceptions.RequestException as req_err:
        print("General error:", req_err)
        sys.exit(1)

    energy = data["energy"]
    gradient = data["gradient"]

    return energy, gradient


def run(arglist: list[str]):
    """Run a calculation on a given structure on the uma server."""
    args = common.cli_parse(arglist, mode=common.RunMode.Client)

    # read the ORCA-generated input
    xyzname, charge, mult, ncores, dograd = common.read_input(args.inputfile)

    # set filenames
    basename = xyzname.rstrip(".xyz")
    orca_engrad = basename + ".engrad"

    # process the XYZ file
    atom_types, coordinates = common.read_xyzfile(xyzname)
    natoms = len(atom_types)

    # submit to the uma calculator server
    energy, gradient = submit_uma(server_url=args.bind, atom_types=atom_types, coordinates=coordinates,
                                      charge=charge, mult=mult, dograd=dograd, nthreads=ncores)
    
    # convert to ORCA engrad
    common.write_engrad(orca_engrad, natoms, energy, dograd, gradient)

    # Timings
    print("Total time:  {:6.3f} seconds".format(time.perf_counter() - start_time))

def main():
    """Entry point for CLI execution"""
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
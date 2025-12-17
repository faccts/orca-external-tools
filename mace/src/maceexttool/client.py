from __future__ import annotations

import json
import sys
from argparse import Namespace
from typing import Tuple

import requests

from maceexttool import common


def submit_mace(server_url: str,
                atom_types: list[str],
                coordinates: list[tuple[float, float, float]],
                charge: int,
                mult: int,
                nthreads: int) -> Tuple[float, list[float]]:
    payload = {
        "atom_types": atom_types,
        "coordinates": coordinates,
        "charge": charge,
        "mult": mult,
        "nthreads": nthreads,
    }

    try:
        response = requests.post('http://' + server_url + "/calculate", json=payload)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("The server is probably not running.")
        print("Please start the server with the maceserver.sh script.")
        raise
    except requests.exceptions.Timeout as timeout_err:
        print("Request to MACE server timed out:", timeout_err)
        raise

    data = response.json()
    return data['energy'], data['gradient']


def run(arglist: list[str]):
    args: Namespace = common.cli_parse(arglist, mode=common.RunMode.Client)

    # read ORCA-generated input
    xyzname, charge, mult, ncores, dograd = common.read_input(args.inputfile)
    basename = xyzname.removesuffix(".xyz")
    orca_engrad = basename + ".engrad"

    atom_types, coordinates = common.read_xyzfile(xyzname)
    natoms = len(atom_types)

    energy, gradient = submit_mace(server_url=args.bind,
                                  atom_types=atom_types,
                                  coordinates=coordinates,
                                  charge=charge,
                                  mult=mult,
                                  nthreads=ncores)

    common.write_engrad(orca_engrad, natoms, energy, dograd, gradient)


def main():
    run(sys.argv[1:])


if __name__ == '__main__':
    main()


from __future__ import annotations

import logging
import threading
import sys
from typing import Callable

import torch
from flask import Flask, request, jsonify
from ase import Atoms

from maceexttool import common, calculator

app = Flask('maceserver')

# Global configuration chosen at server startup
_suite: str = 'omol'
_model: str | None = None
_default_dtype: str | None = None
_device: str = ''
_dispersion: bool = False
_damping: str = 'bj'
_dispersion_xc: str = 'pbe'
_dispersion_cutoff: float | None = None
_head: str | None = None

# One calculator per server thread
calculators: dict[int | Callable] = {}


@app.route('/calculate', methods=['POST'])
def run_mace():
    input = request.get_json()

    atoms = Atoms(symbols=input["atom_types"], positions=input["coordinates"])
    atoms.info = {"charge": input["charge"], "spin": input["mult"]}

    nthreads = input.get('nthreads', 1)
    torch.set_num_threads(nthreads)

    thread_id = threading.get_ident()
    global calculators
    if thread_id not in calculators:
        calculators[thread_id] = calculator.init(
            suite=_suite,
            model=_model,
            device=_device,
            default_dtype=_default_dtype,
            dispersion=_dispersion,
            damping=_damping,
            dispersion_xc=_dispersion_xc,
            dispersion_cutoff=_dispersion_cutoff,
            head=_head,
        )
    calc = calculators[thread_id]
    atoms.calc = calc

    energy, gradient = common.process_output(atoms)
    return jsonify({'energy': energy, 'gradient': gradient})


def run(arglist: list[str]):
    args = common.cli_parse(arglist, mode=common.RunMode.Server)

    global _suite, _model, _default_dtype, _device, _dispersion, _damping, _dispersion_xc, _dispersion_cutoff, _head
    _suite = args.suite
    if _suite.startswith("mace-"):
        _suite = _suite.split("-", 1)[1]
    _model = args.model
    _default_dtype = args.default_dtype or ("float64" if _suite == "omol" else "float32")
    _device = args.device
    _dispersion = bool(args.dispersion)
    _damping = args.damping
    _dispersion_xc = args.dispersion_xc
    _dispersion_cutoff = args.dispersion_cutoff
    _head = args.head

    # Try waitress; fallback to Flask dev server for testing if waitress missing
    try:
        import waitress  # type: ignore
        logger = logging.getLogger('waitress')
        logger.setLevel(logging.INFO)
        waitress.serve(app, listen=args.bind, threads=args.nthreads)
    except Exception:
        host, port = args.bind.split(":")
        port = int(port)
        app.run(host=host, port=port, threaded=True)


def main():
    run(sys.argv[1:])


if __name__ == '__main__':
    main()

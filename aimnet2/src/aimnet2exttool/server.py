from __future__ import annotations

import logging
import sys
import threading
from typing import Callable

import torch
import waitress
from flask import Flask, request, jsonify

from aimnet2exttool import common, calculator

app = Flask('aimnet2server')

model: str = ''  # will hold the selected model

calculators: dict[int | Callable] = {}  # will hold one AIMNet2Calculator per server thread


@app.route('/calculate', methods=['POST'])
def run_aimnet2():
    """
    Runs an AIMNet2 calculation.
    Expects a JSON payload, which can be deserialized directly as kwargs to AIMNet2Calculator, i.e.:
    {
        "data": {
            "coord": list[list[tuple[float, float, float]]],
            "numbers": list[list[int]],
            "charge": list[list[float]],
            "mult": list[list[int]],
        },
        "forces": bool (optional),
        "stress": bool (optional),
        "hessian": bool (optional)
        "nthreads": int  # passed to torch.set_num_threads()
    }
    Returns JSON with energy and flattened gradient in a.u.
    """
    input = request.get_json()

    # Set the number of torch threads
    nthreads = input.pop('nthreads', 1)
    torch.set_num_threads(nthreads)

    # Get the initialized AIMNet2Calculator
    # Since the object is not thread-safe, we initialize one per server thread
    thread_id = threading.get_ident()
    global calculators
    if thread_id not in calculators:
        calculators[thread_id] = calculator.init(model=model)
    calc = calculators[thread_id]

    # run the calculation
    result = calc(**input)

    # get the output
    energy, gradient = common.process_output(result)

    return jsonify({'energy': energy, 'gradient': gradient})


def run(arglist: list[str]):
    """Start the AIMNet2 calculation server using a specified model file."""
    args = common.cli_parse(arglist, mode=common.RunMode.Server)

    # get the absolute path of the model file as a plain string
    global model
    model = str(args.model_dir / args.model)

    # set up logging
    logger = logging.getLogger('waitress')
    logger.setLevel(logging.DEBUG)

    # start the server
    waitress.serve(app, listen=args.bind, threads=args.nthreads)


def main():
    """Entry point for CLI execution"""
    run(sys.argv[1:])


if __name__ == '__main__':
    main()
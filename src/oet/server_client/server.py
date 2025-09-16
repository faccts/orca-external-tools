#!/usr/bin/env python3
"""
Provides the class for running a server loaded with a calculator

Provides
--------
class: OtoolServer
    Main class for running a server
function: main
    Main function for executing
"""
from __future__ import annotations

import importlib
from flask import Flask, Response, request, jsonify
from waitress import serve
from typing import Any, Dict, Tuple, List
from argparse import ArgumentParser
from pathlib import Path
import logging
import threading
from concurrent.futures import ProcessPoolExecutor
import traceback

from oet.core.misc import get_ncores_from_input
from oet.core.base_calc import CALCULATOR_CLASSES

# Per-process cache of initialized calculators
# key: (module, class, frozenset(setup_items)) -> calc instance
_WORKER_CALC_CACHE: dict[tuple[str, str, frozenset[tuple[str, Any]]], Any] = {}


def _run_calc_in_process(
    calc_module: str, calc_class: str, setup_kwargs: dict[str, Any], run_kwargs: dict[str, Any]
) -> None:
    """
    Worker entrypoint. Runs in a separate process.
    We lazily create & cache a calculator instance per unique setup.

    Parameters
    ----------
    calc_module: str
        Module to load for calculator
    calc_class: str
        Calculator type
    setup_kwargs: dict
        Keywords of setup
    run_kwargs: dict
        Infos from client about the run
    """
    import importlib

    key = (calc_module, calc_class, frozenset(setup_kwargs.items()))
    calc = _WORKER_CALC_CACHE.get(key)
    if calc is None:
        mod = importlib.import_module(calc_module)
        Cls = getattr(mod, calc_class)
        calc = Cls()
        # calculators expected setup(dict)
        # calc.setup(setup_kwargs.copy())
        _WORKER_CALC_CACHE[key] = calc

    # Run calc.run
    calc.run(**run_kwargs)


class CoreLimiter:
    """
    Enforces a global core budget across concurrent jobs.
    Blocks until enough cores are free.
    """

    def __init__(self, total_cores: int) -> None:
        # Total cores of server
        self.total = int(total_cores)
        # Available cores of server
        self.available = int(total_cores)
        # Waiting
        self._cv = threading.Condition()

    def acquire(self, n: int) -> None:
        """
        Wait until the required cores for the job are available again

        Parameters
        ----------
        n: int
            Number of cores that are requested for the job
        """
        n = int(n)
        with self._cv:
            if n > self.total:
                # Fail fast: this job can never run on this server
                raise ValueError(
                    f"Requested {n} cores but only {self.total} total available."
                )
            while n > self.available:
                self._cv.wait()
            self.available -= n

    def release(self, n: int) -> None:
        """
        Releases number of cores after the job is done

        Parameters
        ----------
        n: int
            Number of cores to release
        """
        n = int(n)
        with self._cv:
            self.available += n
            if self.available > self.total:
                self.available = self.total
            self._cv.notify_all()


class CalculatorPool:
    """
    Stores and manages free calculators
    """

    def __init__(self) -> None:
        # Class of the calculators
        self._cls: Any = None

    def set_calculator_type(self, calc_type: str) -> None:
        """
        Determine the calculator type, import it and set the class variable

        Parameters
        ----------
        calc_type: str
            Type of calculator
        """
        import_module, calculator_name = CALCULATOR_CLASSES[calc_type]
        mod = importlib.import_module(import_module)
        self._cls = getattr(mod, calculator_name)

    def build_full_parser(self, hook_name: str, parser: ArgumentParser) -> None:
        """
        Extends parser based on subroutine hook_name

        hook_name: str
            Name of the subroutine of the calculator
        parser: ArgumentParser
            Parser to be extended
        """
        if self._cls is None:
            raise RuntimeError("Call set_calculator_type() first.")
        tmp = self._cls()  # assumes default constructor
        meth = getattr(tmp, hook_name)
        meth(parser)


class OtoolServer:
    def __init__(
        self,
        pool: CalculatorPool,
        total_cores: int,
        executor: ProcessPoolExecutor,
        calc_spec: tuple[str, str],
        setup_kwargs: dict[str, Any],
    ):
        self.pool = pool
        self.core_limiter = CoreLimiter(total_cores)
        self.executor = executor
        self.calc_module, self.calc_class = calc_spec
        self.setup_kwargs = setup_kwargs

    def handle_client(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes the input from client and runs the calculation

        Parameters
        ----------
        content: Dict[str, Any]
            Message received from client

        Returns
        -------
        Dict[str, Any]: Message to be sent to client
        """
        arguments: List[str] = content["arguments"]
        working_dir = Path(content["directory"]).resolve()

        # Parse client args
        inputfile, args, clear_args = self.parse_client_input(arguments)

        # Make inputfile absolute (per-request, thread-safe)
        inputfile_path = (working_dir / inputfile).resolve()

        # Get per-job core demand
        ncores_job = get_ncores_from_input(inputfile_path)

        # Gate by cores
        self.core_limiter.acquire(ncores_job)
        try:
            # Submit the job to a separate process
            run_kwargs = {
                "inputfile": str(inputfile_path),
                "args_parsed": args,
                "args_not_parsed": clear_args,
                "directory": str(working_dir),
            }
            fut = self.executor.submit(
                _run_calc_in_process,
                self.calc_module,
                self.calc_class,
                self.setup_kwargs,
                run_kwargs,
            )
            # Will raise if the worker raised
            _ = fut.result()
        finally:
            self.core_limiter.release(ncores_job)

        return {"status": "Success"}

    def parse_client_input(self, arguments: List[str]) -> Tuple[str, dict[str, Any], List[str]]:
        """
        Handles the input sent by client

        Parameters
        ----------
        arguments: list[str]
            Arguments from client

        Returns
        -------
        str: inputfile name
        dict: parsed settings
        list[str]: not parsed settings
        """
        # First server-related settings
        parser = ArgumentParser(
            prog="otool_server",
            description="Client arguments parser.",
        )
        parser.add_argument("inputfile")

        # Let the calculator define its per-request flags
        self.pool.build_full_parser("extend_parser", parser)

        args, remaining_args = parser.parse_known_args(arguments)

        # Transform to dict
        args_dict = vars(args)
        inputfile = args_dict.pop("inputfile")

        return inputfile, args_dict, remaining_args


def create_app(server: OtoolServer) -> Flask:
    """
    Takes the OtoolServer and returns a Flask application
    """
    app = Flask(__name__)

    @app.get("/healthz")
    def healthz() -> Response:
        return jsonify({"status": "OK"})

    @app.post("/calculate")
    def calculate() -> Response:
        try:
            data = request.get_json(force=True, silent=False)
            if not isinstance(data, dict):
                return jsonify(
                    {
                        "status": "Error",
                        "error_message": "Invalid JSON payload",
                        "error_type": "ValueError",
                    }
                )

            arguments = data.get("arguments")
            directory = data.get("directory")

            if not isinstance(arguments, list) or not isinstance(directory, str):
                return jsonify(
                    {
                        "status": "Error",
                        "error_message": "Payload must have list 'arguments' and str 'directory'",
                        "error_type": "ValueError",
                    }
                )

            # Validate directory exists and is a dir
            p = Path(directory)
            if not p.exists() or not p.is_dir():
                return jsonify(
                    {
                        "status": "Error",
                        "error_message": f"Invalid directory: {p}",
                        "error_type": "ValueError",
                    }
                )

            # Delegate to server
            result = server.handle_client(
                {"arguments": arguments, "directory": directory}
            )
            return jsonify(result)

        except Exception as e:
            return jsonify(
                {
                    "status": "Error",
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                }
            )

    return app


def main() -> None:
    """
    Main routine of otools
    """
    # First parse arguments
    parser = ArgumentParser(
        prog="otool_server",
        description="ORCA external tools wrapper.",
        epilog="Specific keywords of the calculators should also set here.",
    )
    parser.add_argument(
        "method",
        choices=CALCULATOR_CLASSES.keys(),
        help="Type of calculation to execute.",
    )
    # Server related things already here to have it in help
    parser.add_argument(
        "-b",
        "--bind",
        metavar="hostname:port",
        default="127.0.0.1:8888",
        dest="host_port",
        help="Server bind address and port. Default: 127.0.0.1:8888.",
    )
    parser.add_argument(
        "-n",
        "--nthreads",
        metavar="nthreads",
        type=int,
        default=1,
        dest="nthreads",
        help="Number of threads to use. Default: 1",
    )

    # Logging for printout of infos
    logging.basicConfig(level=logging.INFO)
    # Suppress the warnings (e.g. jobs are queued)
    logging.getLogger("waitress.queue").setLevel(logging.ERROR)

    # First parse only the known args to get method/nthreads/etc.
    args, _ = parser.parse_known_args()

    # Make a pool for getting the hooks on calculators argument parsing
    pool = CalculatorPool()
    pool.set_calculator_type(args.method)

    # Let the calculator add any *setup* options to the CLI (if they exist)
    pool.build_full_parser("extend_parser", parser)

    # Re-parse now that setup flags may be registered
    args, _ = parser.parse_known_args()

    # Prepare the calculator spec & setup kwargs for the worker
    calc_module, calc_class = CALCULATOR_CLASSES[args.method]
    setup_kwargs = vars(args).copy()

    # Create workers
    workers = args.nthreads
    # Initialize the ProcessPool
    executor = ProcessPoolExecutor(max_workers=workers)

    # Then initialize a server instance that uses the pool
    server = OtoolServer(
        pool=pool,
        total_cores=args.nthreads,  # or another CLI flag like --total-cores
        executor=executor,
        calc_spec=(calc_module, calc_class),
        setup_kwargs=setup_kwargs,
    )

    # Start the server
    host, port = args.host_port.split(":")
    app = create_app(server)
    # Waitress will create a thread per incoming request (bounded by 'threads').
    # Each request acquires/releases a calculator safely.
    serve(app, host=host, port=int(port), threads=args.nthreads)


if __name__ == "__main__":
    main()

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
import io
import logging
import threading
import traceback
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

from flask import Flask, Response, jsonify, request
from waitress import serve

from oet.core.base_calc import CALCULATOR_CLASSES, BaseCalc
from oet.core.misc import get_ncores_from_input

# Cache of initialized calculators: each worker process populates its own copy of the cache
# This guarantees that for calculators which are inadvertently not thread-safe
# (because they temporarily hold calculation data in member variables)
# we never use the same instance by multiple processes in parallel.
# key: (module, class, frozenset(setup_items)) -> calc instance
_WORKER_CALC_CACHE: dict[tuple[str, str, frozenset[tuple[str, Any]]], Any] = {}


def _run_calc_in_process(
    calc_module: str, calc_class: str, setup_kwargs: dict[str, Any], run_kwargs: dict[str, Any]
) -> str:
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

    Returns
    -------
    str
        The STDOUT of the calculator's `run` function
    """

    key = (calc_module, calc_class, frozenset(setup_kwargs.items()))
    calc = _WORKER_CALC_CACHE.get(key)
    if calc is None:
        mod = importlib.import_module(calc_module)
        Cls = getattr(mod, calc_class)
        calc = Cls()
        # calculators expected setup(dict)
        # calc.setup(setup_kwargs.copy())
        _WORKER_CALC_CACHE[key] = calc

    # Run calc.run and return its STDOUT
    buf = io.StringIO()
    with redirect_stdout(buf):
        calc.run(**run_kwargs)
    return buf.getvalue()


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
                raise ValueError(f"Requested {n} cores but only {self.total} total available.")
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


class CalculatorClass:
    """
    Stores the calculator class and is used to build the full parser multiple times
    """

    def __init__(self, calc_type: str) -> None:
        """
        Set the calculator types and import it

        Parameters
        ----------
        calc_type: str
            Type of calculator
        """
        # Determine module and calculator name to be imported from dict
        import_module, calculator_name = CALCULATOR_CLASSES[calc_type]
        # Module to import from
        self.import_module = import_module
        # Name of calculator class
        self.calculator_name = calculator_name
        # Import module
        mod = importlib.import_module(import_module)
        # Set calculator type
        self._cls: Type[BaseCalc] = getattr(mod, calculator_name)

    def build_full_parser(self, parser: ArgumentParser) -> None:
        """
        Extends the given argument parser by calling the class method `extend_parser` of the calculator class

        parser: ArgumentParser
            Parser to be extended
        """
        if self._cls is None:
            raise RuntimeError("Call set_calculator_type() first.")
        self._cls.extend_parser(parser)


class OtoolServer:
    def __init__(
        self,
        calc_class: CalculatorClass,
        total_cores: int,
        executor: ProcessPoolExecutor,
        setup_kwargs: dict[str, Any],
    ):
        # Calculator class used to build parser
        self.calc_class = calc_class
        # Core limiter to prevent using more cores than the server has
        self.core_limiter = CoreLimiter(total_cores)
        # Executor to run the actual calculation
        self.executor = executor
        # Keywords for setup
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
                self.calc_class.import_module,
                self.calc_class.calculator_name,
                self.setup_kwargs,
                run_kwargs,
            )
            # Will raise if the worker raised
            output = fut.result()
        except:
            raise
        else:
            return {"status": "Success", "stdout": output}
        finally:
            self.core_limiter.release(ncores_job)

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
        self.calc_class.build_full_parser(parser)

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
            result = server.handle_client({"arguments": arguments, "directory": directory})
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
    Main routine of otool_server
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

    # Make a CalculatorClass getting the hooks on calculators argument parsing
    # Info on calculator type is store in the object for client requests
    calcClass = CalculatorClass(args.method)

    # Let the calculator add any *setup* options to the CLI (if they exist)
    calcClass.build_full_parser(parser)

    # Re-parse now that setup flags may be registered
    args, _ = parser.parse_known_args()

    # Prepare the calculator spec & setup kwargs for the worker
    setup_kwargs = vars(args).copy()

    # Create workers
    workers = args.nthreads
    # Initialize the ProcessPool
    executor = ProcessPoolExecutor(max_workers=workers)

    # Then initialize a server instance that uses the calc_class
    server = OtoolServer(
        calc_class=calcClass,
        total_cores=args.nthreads,  # or another CLI flag like --total-cores
        executor=executor,
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

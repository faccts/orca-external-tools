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

from ast import arg
import importlib
from blinker import Namespace
from flask import Flask, request, jsonify
from waitress import serve
from typing import Any, Dict, Tuple, List
from argparse import ArgumentParser
from pathlib import Path
import queue
import threading

# The following defines first the key that is provided
# to the otool script
# Second, the path to import and third the calculator class
CALCULATOR_CLASSES = {
    "aimnet2": ("oet.calculator.aimnet2", "Aimnet2Calc"),
    "uma": ("oet.calculator.uma", "UmaCalc"),
}

class CalculatorPool:
    """
    Stores and manages free calculators
    """
    def __init__(self):
        # Class of the calculators
        self._cls: Any = None
        # Pool of calculators
        self._pool: dict[int, Any] = {}
        # Queue for the calculators
        self._free: queue.LifoQueue[int] | None = None

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
        self._cls = getattr(mod, calculator_name)  # store the class, not an instance

    def init_pool(self, size: int, args: Namespace) -> None:
        """
        Initialize the pool by creating the calculators

        Parameters
        ----------
        size: int
            Size of pool
        args: Namespace
            Arguments (options) from parsing
        """
        # Create independent instances (no shallow clones)
        self._pool = {i: self._cls() for i in range(size)}
        for calculator in self._pool.values():
            calculator.setup(vars(args))
        self._free = queue.LifoQueue(maxsize=size)
        # Make them all free
        for i in range(size):
            self._free.put(i)

    def acquire(self, timeout: float | None = None) -> Tuple[int, Any]:
        """
        Waits until the next calculator is available and returns it

        Parameters
        ----------
        timeout: float, default = None
            Optional timeout time
        
        Returns
        -------
        int: idx of the calculator
        Any: calculator
        """
        if self._free is None:
            raise RuntimeError("Pool not initialized.")
        idx = self._free.get(timeout=timeout)   # thread-safe
        return idx, self._pool[idx]

    def release(self, idx: int):
        """Return the calculator to the pool."""
        if self._free is None:
            raise RuntimeError("Pool not initialized.")
        self._free.put(idx)

    def _call_parser_hook(self, hook_name: str, parser: ArgumentParser) -> None:
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

    def build_full_parser(self, parser: ArgumentParser) -> None:
        """Add calculator *setup* options to the server CLI (once, at startup)."""
        self._call_parser_hook("extend_parser_setup", parser)

    def extend_request_parser(self, parser: ArgumentParser) -> None:
        """Add calculator *request* options to per-request parser."""
        self._call_parser_hook("extend_parser_settings", parser)


class OtoolServer:
    def __init__(self, pool: CalculatorPool):
        self.pool = pool

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
        working_dir: str = content["directory"]

        inputfile, args, clear_args = self.parse_client_input(arguments)

        # Acquire a free calculator instance for this request
        idx, calc = self.pool.acquire()
        try:
            # Run calculation (in requested working directory)
            calc.run(
                inputfile=inputfile,
                args_parsed=args,
                args_not_parsed=clear_args,
                directory=working_dir,
            )
        finally:
            # Always release back to the pool
            self.pool.release(idx)

        return {"status": "Success"}

    def parse_client_input(self, arguments: List[str]) -> Tuple[str, dict, List[str]]:
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
        self.pool.extend_request_parser(parser)

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
    def healthz():
        return jsonify({"status": "OK"})

    @app.post("/calculate")
    def calculate():
        try:
            data = request.get_json(force=True, silent=False)
            if not isinstance(data, dict):
                return jsonify({"status": "Error", "error_message": "Invalid JSON payload", "error_type": "ValueError"})

            arguments = data.get("arguments")
            directory = data.get("directory")

            if not isinstance(arguments, list) or not isinstance(directory, str):
                return jsonify({"status": "Error", "error_message": "Payload must have list 'arguments' and str 'directory'", "error_type": "ValueError"})

            # Optional: validate directory exists and is a dir
            p = Path(directory)
            if not p.exists() or not p.is_dir():
                return jsonify({"status": "Error", "error_message": f"Invalid directory: {p}", "error_type": "ValueError"})

            # Delegate to your server logic
            result = server.handle_client({"arguments": arguments, "directory": directory})
            return jsonify(result)

        except Exception as e:
            return jsonify({"status": "Error", "error_message": str(e), "error_type": type(e).__name__})

    return app


def main():
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

    # ---- Pool and calculator setup (one-time, startup) -----------
    # We first parse only the known args to get method/nthreads/etc.
    args, _ = parser.parse_known_args()

    pool = CalculatorPool()
    pool.set_calculator_type(args.method)
    nthreads = args.nthreads

    # Let the calculator add any *setup* options to the CLI (if they exist)
    pool.build_full_parser(parser)

    # Re-parse now that setup flags may be registered
    args, _ = parser.parse_known_args()

    # Initialize pool of independent instances
    pool.init_pool(size=nthreads, args=args)

    # Then initialize a server instance that uses the pool
    server = OtoolServer(pool)

    # Start the server
    host, port = args.host_port.split(":")
    app = create_app(server)
    # Waitress will create a thread per incoming request (bounded by 'threads').
    # Each request acquires/releases a calculator safely.
    serve(app, host=host, port=int(port), threads=args.nthreads)

if __name__ == "__main__":
    main()

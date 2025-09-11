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
import importlib
from flask import Flask, request, jsonify
from waitress import serve
from typing import Any, Dict
from argparse import ArgumentParser
from pathlib import Path

# The following defines first the key that is provided
# to the otool script
# Second, the path to import and third the calculator class
CALCULATOR_CLASSES = {
    "aimnet2": ("oet.calculator.aimnet2", "Aimnet2Calc"),
    "uma": ("oet.calculator.uma", "UmaCalc"),
}

app = Flask(__name__)

class OtoolServer:

    calculator: Any

    def handle_client(self, content: Dict[str, Any]) -> Dict[str, Any]:
        settings = content["arguments"]
        working_dir = content["directory"]

        inputfile, args, clear_args = self.parse_client_input(settings)

        # Run calculation (in requested working directory)
        self.calculator.run(
            inputfile=inputfile,
            args_parsed=args,
            args_not_parsed=clear_args,
            directory=working_dir
        )
        return {"status": "Success"}

    def set_calculator_type(self, calc_type: str):
        """
        Sets and imports everything required for settings up the calculator
        """
        # Get module to be imported and name of class
        import_module, calculator_name = CALCULATOR_CLASSES[calc_type]
        # Import everything
        mod = importlib.import_module(import_module)
        # Get class
        calculator = getattr(mod, calculator_name)
        # Initialize object
        self.calculator = calculator()

    def parse_client_input(self, arguments: list[str]) -> tuple[str, dict, list[str]]:
        """
        Function for parsing client input

        Parameters
        ----------
        arrguments: list[str]
            Arguments from client

        Returns
        -------
        str: inputfile name
        dict: parsed settings
        list[str]: not parsed settings
        """
        # First server related settings
        parser = ArgumentParser(
            prog="otool_server",
            description="Client arguments parser.",
        )
        parser.add_argument("inputfile")
        self.calculator.extend_parser_settings(parser)
        args, remaining_args = parser.parse_known_args(arguments)

        # Transform to dict
        args = vars(args)
        inputfile = args.pop("inputfile")

        return inputfile, args, remaining_args

    def build_full_parser(self, parser: ArgumentParser) -> None:
        """
        Function for building the full parser

        parser: ArgumentParser
            Will be extended to build the full parser
        """
        # Also add calculator setup related arguments
        self.calculator.extend_parser_setup(parser)


def create_app(server: OtoolServer) -> Flask:
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
    args, remaining_args = parser.parse_known_args()

    # Then initialize a server instance
    server = OtoolServer()
    # Set the calcuator type of OtoolServer
    server.set_calculator_type(args.method)
    # Extend the parser
    server.build_full_parser(parser)
    args, unkown_args = parser.parse_known_args()
    # Set the calculator defaults
    server.calculator.setup(vars(args))
    # Start the server
    host, port = args.host_port.split(":")
    app = create_app(server)
    # For production, run under gunicorn/uwsgi. app.run is fine for dev.
    serve(app, host=host, port=int(port), threads=args.nthreads)

if __name__ == "__main__":
    main()

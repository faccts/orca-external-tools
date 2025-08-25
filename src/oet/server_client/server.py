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
import socket
from typing import Any
from argparse import ArgumentParser
import json

# The following defines first the key that is provided
# to the otool script
# Second, the path to import and third the calculator class
CALCULATOR_CLASSES = {
    "aimnet2": ("oet.calculator.aimnet2", "Aimnet2Calc"),
    "uma": ("oet.calculator.uma", "UmaCalc"),
}


class OtoolServer:

    calculator: Any

    def handle_client(self, conn) -> bool:
        # First 10 bytes = integer
        data = conn.recv(4096)
        content = json.loads(data.decode())

        settings = content["arguments"]
        working_dir = content["directory"]
        inputfile, args, clear_args = self.parse_client_input(settings)
        self.calculator.run(
            inputfile=inputfile,
            settings=args,
            clear_args=clear_args,
            directory=working_dir,
        )

    def start_server(self, id_port: str) -> None:
        """ "
        Starts the server.

        Parameters
        ----------
        id_port: str
            ID:Port of the server
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            id, port = id_port.split(":")
            s.bind((id, int(port)))
            s.listen()
            print(f"Server listening on port {port}.")

            while True:
                conn, addr = s.accept()
                try:
                    self.handle_client(conn)
                    server_message = {"status": "Success"}
                    conn.sendall(json.dumps(server_message).encode())
                except Exception as e:
                    server_message = {"status": "Error", "error_message": str(e), "error_type": type(e).__name__}
                    conn.sendall(json.dumps(server_message).encode())

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
        dest="id_port",
        help=f"Server bind address and port. Default: 127.0.0.1:8888.",
    )
    parser.add_argument(
        "-n",
        "--nthreads",
        metavar="nthreads",
        type=int,
        default=1,
        dest="nthreads",
        help=f"Number of threads to use. Default: 1",
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
    server.start_server(args.id_port)


if __name__ == "__main__":
    main()

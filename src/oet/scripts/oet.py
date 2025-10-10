#!/usr/bin/env python3
"""
Single script for redirecting to wrapper
"""

from argparse import ArgumentParser
import importlib
import sys

from oet import __version__ as version
from oet.core.base_calc import CALCULATOR_CLASSES


def parse_oet() -> tuple[str, str, list[str]]:
    """
    Function for parsing oet keywords

    Returns:
    str: Argument for calculator type
    str: Input file name
    list[str]: Remaining arguments parsed by the subclasses
    """
    parser = ArgumentParser(
        prog="oet",
        description="ORCA external tools wrapper.",
        epilog="Provide the arguments for the individual wrapper scripts in addition when calling this oet script.",
        add_help=len(sys.argv) < 3,  # show help if there are insufficient arguments
    )
    parser.add_argument("inputfile")
    parser.add_argument(
        "method",
        choices=CALCULATOR_CLASSES.keys(),
        help="Type of calculation to execute. Call 'oet inputfile method --help' to see method-specific arguments.",
    )
    parser.add_argument("--version", action="version", version=version)
    args, remaining_args = parser.parse_known_args()

    # show help only if requested without a chosen method,
    # otherwise the method-specific parser will do it.
    if not args.method and ('--help' in remaining_args or '-h' in remaining_args):
        parser.print_help()
        exit(0)

    return args.method, args.inputfile, remaining_args


def main():
    """
    Main routine of oet
    """
    # First parse calculator type and cmd arguments
    calc_type, inputfile, args = parse_oet()
    # Get module to be imported and name of class
    import_module, calculator_name = CALCULATOR_CLASSES[calc_type]
    # Import everything
    mod = importlib.import_module(import_module)
    # Get class
    calculator = getattr(mod, calculator_name)
    # Initialize object
    calculator = calculator()
    # Parsing and run the calculation
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()

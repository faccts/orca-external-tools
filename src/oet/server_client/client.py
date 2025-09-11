#!/usr/bin/env python3
"""
Module for sending input to server
"""
import requests
import sys
import os
import traceback
from argparse import ArgumentParser

def send_to_server(
    host_port: str,
    arguments: list[str],
) -> None:
    """
    Sends a calculation to the server. Server should handle file writing.

    Parameters
    ----------
    host_port: str
        Host:Port of the server
    inputfile: str
        Name of the inputfile written by ORCA
    nthreads : int
        Number of threads to use for the calculation
    """

    host, port = host_port.split(":")
    url = f"http://{host}:{port}/calculate"
    payload = {"arguments": arguments, "directory": os.getcwd()}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "Error":
            print(f"Server error {data.get("error_type")}: {data.get("error_message")}.")
            sys.exit(1)
    except requests.exceptions.Timeout:
        print("Connection timed out.")
        sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        traceback.print_exc() 
        print(f"Recieved the following from server: {data}")
        sys.exit(1)


def client():
    """Entry point for CLI execution"""
    parser = ArgumentParser(
        prog="ORCA otool client",
        description="ORCA external tools wrapper for clients. Calls a sever.",
    )
    # parser.add_argument("inputfile", help="ORCA-generated input file.")
    parser.add_argument(
        "-b",
        "--bind",
        metavar="hostname:port",
        default="127.0.0.1:8888",
        dest="host_port",
        help="Server bind address and port. Default: 127.0.0.1:8888.",
    )

    args, remaining_args = parser.parse_known_args(sys.argv[1:])

    send_to_server(host_port=args.host_port, arguments=remaining_args)


if __name__ == "__main__":
    client()

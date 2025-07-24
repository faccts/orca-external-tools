#!/usr/bin/env python3
"""
Module for sending input to server
"""
import socket
import sys
import os
from argparse import ArgumentParser
import json


def send_to_server(
    id_port: str,
    arguments: list[str],
) -> None:
    """
    Sends a calculation to the server. Server should handle file writing.

    Parameters
    ----------
    id_port: str
        ID:Port of the server
    inputfile: str
        Name of the inputfile written by ORCA
    nthreads : int
        Number of threads to use for the calculation
    """

    try:
        id, port = id_port.split(":")
        s = socket.socket()
        s.connect((id, int(port)))
        message = {"arguments": arguments, "directory": os.getcwd()}
        s.sendall(json.dumps(message).encode())
        data = s.recv(1024)
        response = json.loads(data.decode())
        if response["status"] == "Error":
            print(f"Server error {response['error']}.")
            sys.exit(1)
    except socket.timeout:
        print("Connection timed out.")
        sys.exit(1)
    except socket.error as e:
        print(f"Socket error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        s.close()


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
        dest="id_port",
        help=f"Server bind address and port. Default: 127.0.0.1:8888.",
    )

    args, remaining_args = parser.parse_known_args(sys.argv[1:])

    send_to_server(id_port=args.id_port, arguments=remaining_args)


if __name__ == "__main__":
    client()

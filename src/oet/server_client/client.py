#!/usr/bin/env python3
"""
Module for sending input to server.

ORCA starts the client once per gradient, so its start-up is part of every call.
HTTP is spoken over a raw stdlib socket: importing `requests` (or even http.client)
costs 25-240 ms, which for xTB-class methods is more than the calculation itself.
"""

import json
import os
import socket
import sys
from argparse import ArgumentParser

from oet import __version__ as version


def _dechunk(body: bytes) -> bytes:
    """Decode a body sent with 'Transfer-Encoding: chunked'."""
    out: list[bytes] = []
    while True:
        line, _, rest = body.partition(b"\r\n")
        size = int(line.split(b";")[0], 16) if line else 0
        if size == 0:
            return b"".join(out)
        out.append(rest[:size])
        body = rest[size + 2 :]


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
    arguments: list[str]
        Arguments to forward to the server's calculator
    """
    host, _, port = host_port.rpartition(":")
    payload = json.dumps({"arguments": arguments, "directory": os.getcwd()}).encode()
    request = (
        b"POST /calculate HTTP/1.1\r\n"
        b"Host: " + host_port.encode() + b"\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(payload)).encode() + b"\r\n"
        b"Connection: close\r\n\r\n" + payload
    )
    try:
        with socket.create_connection((host, int(port))) as sock:
            sock.sendall(request)
            chunks = []
            while data := sock.recv(65536):
                chunks.append(data)
    except (OSError, ValueError) as e:
        print(f"Connection error: {e}")
        sys.exit(1)

    raw = b"".join(chunks)
    head, sep, body = raw.partition(b"\r\n\r\n")
    if b"transfer-encoding: chunked" in head.lower():
        body = _dechunk(body)
    try:
        data = json.loads(body)
    except ValueError:
        status = head.split(b"\r\n", 1)[0].decode(errors="replace")
        print(f"HTTP error: {status or 'no response'}. Received: {body!r}")
        sys.exit(1)

    # print the stdout of the calculation, if any
    print(data.get("stdout", ""), end="")
    if data.get("status") == "Error":
        print(f"Server error {data.get('error_type')}: {data.get('error_message')}.")
        print(f"Exact traceback: {data.get('traceback')}")
        sys.exit(1)


def client() -> None:
    """Entry point for CLI execution"""
    parser = ArgumentParser(
        prog="oet_client",
        description="Sends the calculation info to a server that must previously be started with the oet_server script.",
    )
    parser.add_argument("--version", action="version", version=version)
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

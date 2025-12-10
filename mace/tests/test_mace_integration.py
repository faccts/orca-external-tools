import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _write_extinp(tmp: Path, name: str, xyz_content: str, charge=0, mult=1, ncores=1, dograd=1):
    xyz_path = tmp / f"{name}_EXT.xyz"
    xyz_path.write_text(xyz_content)
    ext = tmp / f"{name}_EXT.extinp.tmp"
    ext.write_text("\n".join([
        str(xyz_path),
        str(charge),
        str(mult),
        str(ncores),
        str(dograd),
    ]) + "\n")
    return ext


def test_integration_standalone_cli(tmp_path, monkeypatch):
    # Prepare simple H2 in XYZ
    xyz = """2
H2
H 0 0 0
H 0 0 0.75
"""
    ext = _write_extinp(tmp_path, "h2_cli", xyz)

    env = os.environ.copy()
    src_path = Path(__file__).resolve().parents[1] / "src"
    mace_repo = Path(__file__).resolve().parents[3] / "mace"
    env["PYTHONPATH"] = os.pathsep.join([str(src_path), str(mace_repo), env.get("PYTHONPATH", "")])
    # call entry module instead of venv shell script
    cmd = [sys.executable, "-m", "maceexttool.standalone", "-s", "omol", "-m", "extra_large", "--device", "cpu", "--default-dtype", "float64", str(ext)]
    cp = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert cp.returncode == 0, cp.stderr

    engrad = Path(str(ext).replace(".extinp.tmp", ".engrad")).with_suffix(".engrad")
    assert engrad.exists()
    # sanity: energy is written, gradient lines exist
    out = engrad.read_text()
    assert "Total energy" in out


def test_integration_server_client_cli(tmp_path):
    # simple water molecule
    xyz = """3
H2O
O 0.0 0.0 0.0
H 0.0 0.0 0.96
H 0.0 0.75 -0.24
"""
    ext = _write_extinp(tmp_path, "h2o_cli", xyz)

    env = os.environ.copy()
    src_path = Path(__file__).resolve().parents[1] / "src"
    mace_repo = Path(__file__).resolve().parents[3] / "mace"
    env["PYTHONPATH"] = os.pathsep.join([str(src_path), str(mace_repo), env.get("PYTHONPATH", "")])

    port = _free_port()
    bind = f"127.0.0.1:{port}"

    # Start server
    server_cmd = [sys.executable, "-m", "maceexttool.server", "-s", "omol", "-m", "extra_large", "--device", "cpu", "-b", bind]
    server = subprocess.Popen(server_cmd, cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Wait for port to accept connections
    ready = False
    t0 = time.time()
    while time.time() - t0 < 60:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect(("127.0.0.1", port))
                ready = True
                break
            except OSError:
                time.sleep(0.1)
                continue
    assert ready, "Server did not start in time"

    try:
        # Run client
        client_cmd = [sys.executable, "-m", "maceexttool.client", "-b", bind, str(ext)]
        cp = subprocess.run(client_cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
        assert cp.returncode == 0, cp.stderr

        engrad = Path(str(ext).replace(".extinp.tmp", ".engrad")).with_suffix(".engrad")
        assert engrad.exists()
        out = engrad.read_text()
        assert "Total energy" in out
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

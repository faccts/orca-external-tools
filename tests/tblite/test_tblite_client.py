import os
import shutil
import signal
import subprocess
import unittest
from pathlib import Path

from oet.core.test_utilities import (
    WATER,
    get_filenames,
    read_result_file,
    run_wrapper,
    wait_for_server,
    write_input_file,
    write_xyz_file,
)

# Get the paths to the scripts that should be tested
resolved_client = shutil.which("oet_client")
resolved_server = shutil.which("oet_server")
if resolved_client is None or resolved_server is None:
    raise RuntimeError(
        "The 'oet_client' or 'oet_server' script was not found in PATH. "
        "Run the tests with the project's virtual environment activated."
    )
client_path = Path(resolved_client)
server_path = Path(resolved_server)

# Default ID and port of server. Change if needed
ip_port = "127.0.0.1:18890"


class TbliteClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Start the server"""
        server_out = Path("server.out").resolve()
        print(f"Starting the server. A detailed server log can be found on file {server_out}")
        with open(server_out, "a") as f:
            cls.server = subprocess.Popen(
                [server_path, "tblite", "--bind", ip_port, "--nthreads", "2"],
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        wait_for_server(process=cls.server, ip_port=ip_port, timeout=30.0)

    @classmethod
    def tearDownClass(cls):
        """Shut the server at the end"""
        os.killpg(os.getpgid(cls.server.pid), signal.SIGTERM)
        cls.server.wait(timeout=10)

    def _write_water(self, basename: str) -> tuple[Path, Path, Path]:
        xyz_file, input_file, engrad_out, output_file = get_filenames(basename)
        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=1,
            do_gradient=1,
        )
        return input_file, engrad_out, output_file

    def test_H2O_engrad(self):
        input_file, engrad_out, output_file = self._write_water("H2O_client")
        run_wrapper(
            inputfile=input_file,
            script_path=client_path,
            outfile=output_file,
            args=["--bind", ip_port],
            timeout=30,
        )
        natoms, energy, gradient = read_result_file(engrad_out)
        self.assertEqual(natoms, 3)
        self.assertEqual(len(gradient), 9)
        self.assertLess(energy, 0.0)

    def test_failing_calculation_returns(self):
        """A calculator that calls sys.exit() must give an error, not a hanging client."""
        input_file, _, _ = self._write_water("H2O_client_fail")
        result = subprocess.run(
            [client_path, "--bind", ip_port, input_file, "--bogus"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Server error", result.stdout)


if __name__ == "__main__":
    unittest.main()

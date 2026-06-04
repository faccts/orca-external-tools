import os
import signal
import subprocess
import time
import unittest

from oet import ROOT_DIR
from oet.core.test_utilities import (
    OH,
    WATER,
    get_filenames,
    read_result_file,
    run_wrapper,
    write_input_file,
    write_xyz_file,
)

# Path to the script, adjust if needed.
mace_server_path = ROOT_DIR / "../../bin_mace/oet_server"
mace_client_path = ROOT_DIR / "../../bin_mace/oet_client"
# Default maximum time (in sec) to download the model files if not present
timeout = 600
# Default ID and port of server. Change if needed
id_port = "127.0.0.1:9000"


def run_mace(inputfile: str, output_file: str, args: list[str]) -> None:
    # Run the wrapper with an increased timeout as loading the MACE model files might take a while
    args.extend(["--bind", id_port])
    run_wrapper(
        inputfile=inputfile,
        script_path=mace_client_path,
        outfile=output_file,
        timeout=30,
        args=args,
    )


class MACETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Test starting the server
        """
        print("Starting the server. A detailed server log can be found on file server.out")
        with open("server.out", "a") as f:
            cls.server = subprocess.Popen(
                [mace_server_path, "mace", "--bind", id_port, "--nthreads", "2"],
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
        # Wait a little to make sure it is setup
        time.sleep(5)

    @classmethod
    def tearDownClass(cls):
        """
        Shut the server at the end
        """
        print("Killing the server.")
        os.killpg(os.getpgid(cls.server.pid), signal.SIGTERM)
        cls.server.wait(timeout=10)

    def test_H2O_engrad(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("H2O")

        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        args = ["-s", "mace-mp", "-m", "medium", "head", "mh0"]
        run_mace(input_file, output_file, args)
        expected_num_atoms = 3
        expected_energy = -5.203530407103e-01
        expected_gradients = [
            2.897306550196e-03,
            2.144325522181e-03,
            -1.515869870431e-03,
            -7.767454655073e-04,
            -3.378720118971e-03,
            -1.222818329722e-03,
            -2.120561084688e-03,
            1.234394596790e-03,
            2.738688200153e-03,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=7)

    def test_OH_anion_eng_grad(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("OH_anion")
        write_xyz_file(xyz_file, OH)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=-1,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        args = ["-s", "omol"]
        run_mace(input_file, output_file, args)
        expected_num_atoms = 2
        expected_energy = -7.580657311911e01
        expected_gradients = [
            -1.082263844645e-03,
            -3.483610415645e-03,
            -9.925303607707e-04,
            1.082263844645e-03,
            3.483610415645e-03,
            9.925303607707e-04,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=7)

    def test_OH_rad_eng_grad(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("OH_rad")
        write_xyz_file(xyz_file, OH)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=2,
            ncores=2,
            do_gradient=1,
        )
        # Test the defaults (no arguments)
        run_mace(input_file, output_file, args=[])
        expected_num_atoms = 2
        expected_energy = -7.574239878105e01
        expected_gradients = [
            1.056554797887e-03,
            3.400856384066e-03,
            9.689529220712e-04,
            -1.056554797887e-03,
            -3.400856384066e-03,
            -9.689529220712e-04,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=7)


if __name__ == "__main__":
    unittest.main()

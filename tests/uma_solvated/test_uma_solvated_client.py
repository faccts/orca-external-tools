import os
import signal
import subprocess
import time
import unittest

from oet import ROOT_DIR
from oet.calculator.uma import DEFAULT_CACHE_DIR
from oet.calculator.uma_solvated import UmaSolvatedCalc
from oet.core.test_utilities import (
    WATER,
    TimeoutCall,
    TimeoutCallError,
    get_filenames,
    read_result_file,
    run_wrapper,
    write_input_file,
    write_xyz_file,
)

# Path to the scripts, adjust if needed.
uma_solvated_script_path = ROOT_DIR / "../../bin/oet_client"
uma_solvated_server_path = ROOT_DIR / "../../bin/oet_server"
# Default maximum time (in sec) to download the model files if not present
timeout = 600
# Default ID and port of server. Change if needed
id_port = "127.0.0.1:9001"
# UMA model to use
uma_model = "uma-s-1p1"


def cache_model_files(
    basemodel: str, param: str = "omol", device: str = "cpu", cache_dir: str = DEFAULT_CACHE_DIR
) -> None:
    """
    Wrapper to set up an UMA calculator that downloads the model files into the same cache-directory used for actual oet calculations.

    basemodel: str
        Basemodel used to calculate the test cases
    param: str, default: omol
        Parameter set.
    device str, default: cpu
        Device used for the calculations.
    cache_dir: str, default: DEFAULT_CACHE_DIR
        The cache directory used to store the model data.
    """
    calculator = UmaSolvatedCalc()
    calculator.set_calculator(param=param, basemodel=basemodel, device=device, cache_dir=cache_dir)


def run_uma_solvated(inputfile: str, output_file: str, extra_args: list[str] | None = None) -> None:
    args = ["--bind", id_port, "--model", uma_model]
    if extra_args:
        args.extend(extra_args)
    run_wrapper(
        inputfile=inputfile,
        script_path=uma_solvated_script_path,
        outfile=output_file,
        args=args,
    )


class UmaSolvatedClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Test starting the server
        """
        # Pre-download UMA model files
        print("Checking the model files and downloading them if necessary.")
        # Make a timeout call to avoid hanging forever
        get_pretrained_mlip_timeout = TimeoutCall(fn=cache_model_files)
        ok, payload = get_pretrained_mlip_timeout(uma_model, timeout=timeout)
        # Check if the model files could not be loaded
        if not ok:
            # Timeout
            if payload == TimeoutCallError.TIMEOUT:
                print(
                    "Loading the model files timed out. "
                    "Please check your internet connection and consider increasing the time before timing out."
                )
                raise unittest.SkipTest("Timed out.")
            # General errors and crashes
            elif payload == TimeoutCallError.CRASH or payload == TimeoutCallError.ERROR:
                print(
                    "Loading the model files failed. Make sure that "
                    "the virtual environment with UMA installed is active."
                )
                raise unittest.SkipTest("Loading failed.")
            # Unresolved error
            else:
                print("Could not load the model files.")
                raise unittest.SkipTest("Loading failed.")
        print("Starting the server. A detailed server log can be found on file server_solvated.out")
        with open("server_solvated.out", "a") as f:
            cls.server = subprocess.Popen(
                [uma_solvated_server_path, "uma_solvated", "--bind", id_port, "--nthreads", "2"],
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
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

    def test_H2O_engrad_no_solvent(self):
        """Test with solvent=none (should match regular UMA results)"""
        xyz_file, input_file, engrad_out, output_file = get_filenames("H2O_client_nosolv")

        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        run_uma_solvated(input_file, output_file, extra_args=["--solvent", "none"])
        expected_num_atoms = 3
        expected_energy = -76.43352090249
        expected_gradients = [
            -0.007321094162762,
            -0.005420647095889,
            0.003829096443951,
            0.002653303323314,
            0.006170153152198,
            0.001055987784639,
            0.004667790140957,
            -0.0007495055906475,
            -0.004885083995759,
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

    @unittest.skipIf(True, "xTB not available - skipping solvation tests")
    def test_H2O_engrad_with_water_solvent(self):
        """Test with water solvent"""
        xyz_file, input_file, engrad_out, output_file = get_filenames("H2O_client_water")

        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        run_uma_solvated(input_file, output_file, extra_args=["--solvent", "water"])
        expected_num_atoms = 3

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        # Just verify we get valid results
        self.assertIsNotNone(energy)
        self.assertEqual(len(gradients), 9)


if __name__ == "__main__":
    unittest.main()

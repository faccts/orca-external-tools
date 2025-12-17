import unittest

from oet import ROOT_DIR
from oet.calculator.uma import DEFAULT_CACHE_DIR, UmaCalc
from oet.core.test_utilities import (
    OH,
    WATER,
    TimeoutCall,
    TimeoutCallError,
    get_filenames,
    read_result_file,
    run_wrapper,
    write_input_file,
    write_xyz_file,
)

# Path to the script, adjust if needed.
uma_script_path = ROOT_DIR / "../../bin/oet_uma"
# Default maximum time (in sec) to download the model files if not present
timeout = 600
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
    calculator = UmaCalc()
    calculator.set_calculator(param=param, basemodel=basemodel, device=device, cache_dir=cache_dir)


def run_uma(inputfile: str, output_file: str) -> None:
    # Run the wrapper with an increased timeout as loading the UMA model files might take a while
    run_wrapper(inputfile=inputfile, script_path=uma_script_path, outfile=output_file, timeout=30)


class UmaTests(unittest.TestCase):
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
        run_uma(input_file, output_file)
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
        run_uma(input_file, output_file)
        expected_num_atoms = 2
        expected_energy = -75.80575637958
        expected_gradients = [
            -0.001200547791086,
            -0.003864351427183,
            -0.001101008034311,
            0.001200547791086,
            0.003864351427183,
            0.001101008034311,
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
        run_uma(input_file, output_file)
        expected_num_atoms = 2
        expected_energy = -75.74201333130
        expected_gradients = [
            0.001247821375728,
            0.004016515333205,
            0.001144362613559,
            -0.001247821375728,
            -0.004016515333205,
            -0.001144362613559,
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

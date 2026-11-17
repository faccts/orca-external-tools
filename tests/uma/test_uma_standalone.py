import shutil
import subprocess
import unittest
from pathlib import Path

from oet.core.test_utilities import (
    OH,
    WATER,
    get_filenames,
    read_hess_file,
    read_result_file,
    run_wrapper,
    write_input_file,
    write_xyz_file,
)

# Get the path to the script that should be tested
resolved_uma_script = shutil.which("oet_uma")
if resolved_uma_script is None:
    raise RuntimeError(
        "The 'goet_xtb' script was not found in PATH. "
        "Run the tests with the project's virtual environment activated."
    )
uma_script_path = Path(resolved_uma_script)

# Default maximum time (in sec) to download the model files if not present
timeout = 600
# UMA model to use
uma_model = "uma-s-1p1"


def cache_model_files(basemodel: str, param: str = "omol") -> None:
    """
    Wrapper to set up an UMA calculator that downloads the model files into the same cache-directory used for actual oet calculations.

    basemodel: str
        Basemodel used to calculate the test cases
    param: str, default: omol
        Parameter set.
    """
    subprocess.run(
        [
            uma_script_path,
            "--download-only",
            "--model",
            basemodel,
            "--task",
            param,
        ],
        check=True,
        timeout=timeout,
    )


def run_uma(inputfile: str, output_file: str) -> None:
    # Run the wrapper with an increased timeout as loading the UMA model files might take a while
    run_wrapper(inputfile=inputfile, script_path=uma_script_path, outfile=output_file, timeout=60)


class UmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Test starting the server
        """
        # Pre-download UMA model files
        print("Checking the model files and downloading them if necessary.")
        try:
            cache_model_files(uma_model)
        except subprocess.TimeoutExpired:
            raise unittest.SkipTest(
                "Loading the model files timed out. Please check your internet connection."
            )
        except subprocess.CalledProcessError:
            raise unittest.SkipTest("Loading the model files failed. ")

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

    def test_H2O_hessian(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("H2O_hess")
        hess_out = "H2O_hess.hess"

        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=0,
            do_hessian=1,
        )
        run_uma(input_file, output_file)

        expected_num_atoms = 3
        expected_energy = -76.43352090249
        # Reference Hessian taken directly from the UMA calculator (uma-s-1p1, omol) on
        # this geometry, cross-checked against the semi-numerical Hessian of a manual ORCA
        # `! extopt numfreq` run (see tmp-hess-test/orca-uma-numhess.hess): they agree to
        # ~4 significant figures (max abs deviation 1.3e-5 Eh/Bohr^2).
        expected_hessian = [
            0.27651517536245174,
            0.07451102807662664,
            -0.21193898341599532,
            -0.06798138496475879,
            -0.11255250404396952,
            -0.00477389942324221,
            -0.2085337560482475,
            0.03804147842087469,
            0.21671286934481254,
            0.07451104279781756,
            0.48876815196638074,
            0.20788568996798207,
            -0.15126690036889365,
            -0.42305161731570584,
            -0.09907852187236428,
            0.07675586247813972,
            -0.06571649048710217,
            -0.1088071680956178,
            -0.21193902267250442,
            0.20788568996798207,
            0.3315890544887967,
            -0.027269404000211637,
            -0.13598302628820655,
            -0.05740208453903188,
            0.23920841931212058,
            -0.07190266367977553,
            -0.27418695032151025,
            -0.06798139477888605,
            -0.15126690036889365,
            -0.027269418721402542,
            0.0627461123775765,
            0.12226907889067874,
            0.014864759322013447,
            0.005235282401309549,
            0.028997821478214895,
            0.012404659399389095,
            -0.11255249422984225,
            -0.42305157805919674,
            -0.13598302628820655,
            0.12226904944829693,
            0.4365941707945591,
            0.13492240373981684,
            -0.00971656073890127,
            -0.01354261113685097,
            0.0010606162612144228,
            -0.0047738847020513024,
            -0.09907852187236428,
            -0.05740207963196824,
            0.014864751961417995,
            0.1349224135539441,
            0.06803817931928063,
            -0.0100908684861326,
            -0.035843901495707105,
            -0.010636096007014668,
            -0.20853377567650205,
            0.07675586247813972,
            0.23920839968386604,
            0.005235282401309549,
            -0.009716559512135362,
            -0.010090862965686011,
            0.2032984883681289,
            -0.0670393023526214,
            -0.2291175532795198,
            0.03804146369968378,
            -0.06571649048710217,
            -0.07190264895858463,
            0.028997826385278532,
            -0.01354260868331915,
            -0.03584389413511165,
            -0.0670392876314305,
            0.07925908812952814,
            0.1077465455472281,
            0.216712849716558,
            -0.1088071680956178,
            -0.27418695032151025,
            0.012404659399389095,
            0.0010606201715307577,
            -0.01063609232671694,
            -0.22911751402301073,
            0.1077465455472281,
            0.2848230500088227,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        # No gradient was requested, so none must be written
        self.assertEqual(gradients, [])

        try:
            hessian = read_hess_file(hess_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error Hessian outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(len(hessian), (3 * expected_num_atoms) ** 2)
        for h1, h2 in zip(hessian, expected_hessian):
            self.assertAlmostEqual(h1, h2, places=5)

        # Check symmetry
        ndim = 3 * expected_num_atoms
        for i in range(ndim):
            for j in range(ndim):
                self.assertAlmostEqual(hessian[i * ndim + j], hessian[j * ndim + i], places=5)

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

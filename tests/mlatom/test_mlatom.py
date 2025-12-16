import unittest

import torchani

from oet import ROOT_DIR
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
mlatom_script_path = ROOT_DIR / "../../bin/oet_mlatom"
# Default maximum time (in sec) to download the model files if not present
timeout = 300
# Leave mlatom_executable_path empty, if mlatom from system path should be called
mlatom_executable_path = ""


def run_mlatom(inputfile: str, output_file: str) -> None:
    arguments = []
    if mlatom_executable_path:
        arguments = ["--exe", mlatom_executable_path]
    arguments.append("ANI-1ccx")
    # print(inputfile, arguments)
    run_wrapper(
        inputfile=inputfile, script_path=mlatom_script_path, outfile=output_file, args=arguments
    )


class MLatomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Force download / initialization of ANI-1ccx once
        print("Checking the model files and downloading them if necessary.")
        # Make a timeout call to avoid hanging forever
        get_ani1ccx_timeout = TimeoutCall(fn=torchani.models.ANI1ccx)
        ok, payload = get_ani1ccx_timeout(timeout=timeout, periodic_table_index=True)
        if not ok:
            if payload == TimeoutCallError.TIMEOUT:
                print(
                    "Loading the model files timed out. "
                    "Please check your internet connection and consider increasing the time before timing out."
                )
                raise unittest.SkipTest("Timed out.")
            if payload == TimeoutCallError.CRASH or payload == TimeoutCallError.ERROR:
                print(
                    "Loading the model files failed. Make sure that "
                    "the virtual environment with MLAtoms installed is active."
                )
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
        run_mlatom(input_file, output_file)

        expected_num_atoms = 3
        expected_energy = -76.38342071002
        expected_gradients = [
            -9.34811007e-03,
            -6.92128305e-03,
            4.88938529e-03,
            2.98246744e-03,
            9.27055785e-03,
            2.54374613e-03,
            6.36564281e-03,
            -2.34927474e-03,
            -7.43313148e-03,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=9)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=9)

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
        run_mlatom(input_file, output_file)
        expected_num_atoms = 2
        expected_energy = -75.76385998084
        expected_gradients = [
            -1.07623156e-02,
            -3.46419311e-02,
            -9.86998337e-03,
            1.07623156e-02,
            3.46419311e-02,
            9.86998337e-03,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=9)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=9)

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
        run_mlatom(input_file, output_file)
        expected_num_atoms = 2
        expected_energy = -75.76385998084
        expected_gradients = [
            -1.07623156e-02,
            -3.46419311e-02,
            -9.86998337e-03,
            1.07623156e-02,
            3.46419311e-02,
            9.86998337e-03,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=9)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=9)


if __name__ == "__main__":
    unittest.main()

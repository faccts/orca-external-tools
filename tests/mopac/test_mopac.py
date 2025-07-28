import unittest
import subprocess
from oet.core.test_utilities import (
    read_result_file,
    write_input_file,
    write_xyz_file,
    get_filenames,
    WATER,
    OH,
)

mopac_script_path = "../../src/oet/calculator/mopac.py"
# Leave moppac_executable_path empty, if mopac from system path should be called
mopac_executable_path = ""
output_file = "wrapper.out"


def run_wrapper(arguments: str) -> None:
    args = arguments
    if mopac_executable_path:
        args += "--exe " + mopac_executable_path

    with open(output_file, "w") as f:
        subprocess.run(
            ["python", mopac_script_path, args], stdout=f, stderr=subprocess.STDOUT
        )


class MopacTests(unittest.TestCase):
    def test_H2O_engrad(self):
        xyz_file, input_file, engrad_out = get_filenames("OH")
        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        run_wrapper(input_file)
        
        expected_num_atoms = 3
        expected_energy = -7.849280369e-02
        expected_gradients = [
            -7.93659660e-03,
            -5.85954874e-03,
            4.14782076e-03,
            6.58260052e-03,
            -6.09236043e-03,
            -9.82276238e-03,
            1.35399608e-03,
            1.19519092e-02,
            5.67494162e-03,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except FileNotFoundError:
            print("Error wrapper outputfile not found. Check wrapper.out for details")

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=9)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=9)

    def test_OH_eng_grad(self):
        xyz_file, input_file, engrad_out = get_filenames("OH")
        write_xyz_file(xyz_file, OH)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=-1,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        run_wrapper(input_file)
        expected_num_atoms = 2
        expected_energy = -4.909933634471e-02
        expected_gradients = [
            -1.08238106e-02,
            -3.48519048e-02,
            -9.92625017e-03,
            1.08238106e-02,
            3.48519048e-02,
            9.92625017e-03,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except FileNotFoundError:
            print("Error wrapper outputfile not found. Check wrapper.out for details")

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=9)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=9)


if __name__ == "__main__":
    unittest.main()

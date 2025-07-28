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

gxtb_script_path = "../../src/oet/calculator/gxtb.py"
# Leave uma_executable_path empty, if gxtb from system path should be called
gxtb_executable_path = ""
output_file = "wrapper.out"


def run_wrapper(arguments: str) -> None:
    args = arguments
    if gxtb_executable_path:
        args += "--exe " + gxtb_executable_path

    with open(output_file, "w") as f:
        subprocess.run(
            ["python", gxtb_script_path, args], stdout=f, stderr=subprocess.STDOUT
        )


class GxtbTests(unittest.TestCase):
    def test_H2O_engrad(self):
        xyz_file, input_file, engrad_out = get_filenames("H2O")

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
        expected_energy = -76.437068989
        expected_gradients = [
            -0.008729628293243,
            -0.006455060140335,
            0.00456420451466,
            0.003703035104271,
            0.005500728080203,
            -0.0003292290623828,
            0.005026593188973,
            0.0009543320601324,
            -0.004234975452277
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
        expected_energy = -75.80251013875
        expected_gradients = [
            0.001412983928617,
            0.004540856987963,
            0.001295824899709,
            -0.001412983928617,
            -0.004540856987963,
            -0.001295824899709
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

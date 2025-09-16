import unittest
import subprocess
from oet.core.test_utilities import (
    read_result_file,
    write_input_file,
    write_xyz_file,
    get_filenames,
    run_wrapper,
    WATER,
    OH,
)

mlatom_script_path = "../../scripts/otool_mlatom"
# Leave mlatom_executable_path empty, if mlatom from system path should be called
mlatom_executable_path = ""
output_file = "wrapper.out"


def run_mlatom(inputfile: str) -> None:
    arguments = []
    if mlatom_executable_path:
        arguments = ["--exe", mlatom_executable_path]
    arguments.append("ANI-1ccx")
    #print(inputfile, arguments)
    run_wrapper(inputfile=inputfile, script_path=mlatom_script_path, outfile=output_file, args=arguments)


class MLatomTests(unittest.TestCase):
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
        run_mlatom(input_file)
        
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
        except FileNotFoundError:
            print("Error wrapper outputfile not found. Check wrapper.out for details")

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=9)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=9)

    def test_OH_anion_eng_grad(self):
        xyz_file, input_file, engrad_out = get_filenames("OH_anion")
        write_xyz_file(xyz_file, OH)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=-1,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        run_mlatom(input_file)
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
        except FileNotFoundError:
            print("Error wrapper outputfile not found. Check wrapper.out for details")

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=9)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=9)

    def test_OH_rad_eng_grad(self):
        xyz_file, input_file, engrad_out = get_filenames("OH_rad")
        write_xyz_file(xyz_file, OH)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=2,
            ncores=2,
            do_gradient=1,
        )
        run_mlatom(input_file)
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
        except FileNotFoundError:
            print("Error wrapper outputfile not found. Check wrapper.out for details")

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=9)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=9)


if __name__ == "__main__":
    unittest.main()

import unittest
import subprocess
import time
from oet.core.test_utilities import (
    read_result_file,
    write_input_file,
    write_xyz_file,
    get_filenames,
    WATER,
    OH,
)

aimnet2_script_path = "../../scripts/otool_client"
aimnet2_server_path = "../../scripts/otool_server"
output_file = "wrapper.out"
# Default ID and port of server. Change if needed
id_port = "127.0.0.1:9000"


def run_wrapper(arguments: str) -> None:
    args = arguments# + " --bind " + id_port

    with open(output_file, "a") as f:
        subprocess.run(
            ["python3", aimnet2_script_path, args, "--bind", id_port], stdout=f, stderr=subprocess.STDOUT
        )


class Aimnet2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Test starting the server
        """
        with open(output_file, "a") as f:
            cls.server = subprocess.Popen(
                ["python3", aimnet2_server_path, "aimnet2", "--bind", id_port], stdout=f, stderr=subprocess.STDOUT
            )
        # Wait a little to make sure it is setup
        time.sleep(5)

    @classmethod
    def tearDownClass(cls):
        """
        Shut the server at the end
        """
        cls.server.terminate()
        cls.server.wait()

    def test_H2O_engrad(self):
        xyz_file, input_file, engrad_out = get_filenames("H2O_client")

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
        expected_energy = -76.47682538331
        expected_gradients = [
            -0.01020942255855,
            -0.007558935321867,
            0.005339920055121,
            0.003577792551368,
            0.009023879654706,
            0.001832913258113,
            0.0066316309385,
            -0.001464945613407,
            -0.007172833196819
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
        expected_energy = -75.82629634884
        expected_gradients = [
            -0.000485832511913,
            -0.001563806785271,
            -0.0004455488233361,
            0.000485832511913,
            0.001563805621117,
            0.0004455488233361
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
        xyz_file, input_file, engrad_out = get_filenames("OH_client")
        write_xyz_file(xyz_file, OH)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=2,
            ncores=2,
            do_gradient=1,
        )
        run_wrapper(input_file)
        expected_num_atoms = 2
        expected_energy = -75.68258695326
        expected_gradients = [
            -3.78393149e-03,
            -1.21797854e-02,
            -3.47019313e-03,
            3.78393149e-03,
            1.21797854e-02,
            3.47019313e-03,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except FileNotFoundError:
            print("Error wrapper outputfile not found. Check wrapper.out for details")

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=7)

if __name__ == "__main__":
    unittest.main()


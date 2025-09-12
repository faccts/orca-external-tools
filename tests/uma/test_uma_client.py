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

uma_script_path = "../../scripts/otool_client"
uma_server_path = "../../scripts/otool_server"
output_file = "wrapper.out"
# Default ID and port of server. Change if needed
id_port = "127.0.0.1:9000"


def run_wrapper(arguments: str) -> None:
    args = arguments# + " --bind " + id_port

    with open(output_file, "a") as f:
        subprocess.run(
            ["python3", uma_script_path, args, "--bind", id_port], stdout=f, stderr=subprocess.STDOUT
        )


class UmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Test starting the server
        """
        with open(output_file, "a") as f:
            cls.server = subprocess.Popen(
                ["python3", uma_server_path, "uma", "--bind", id_port], stdout=f, stderr=subprocess.STDOUT
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
        expected_energy = -76.43349724311
        expected_gradients = [
            -0.007337533868849,
            -0.005432820878923,
            0.003837696509436,
            0.002658477984369,
            0.006186702288687,
            0.001060670707375,
            0.004679056815803,
            -0.0007538812351413,
            -0.004898367449641,
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
        xyz_file, input_file, engrad_out = get_filenames("OH_client")
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
        expected_energy = -75.80600885514
        expected_gradients = [
            -0.001075187698007,
            -0.003460831707343,
            -0.0009860402205959,
            0.001075187698007,
            0.003460831707343,
            0.0009860402205959,
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

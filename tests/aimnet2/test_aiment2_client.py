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

aimnet2_script_path = ROOT_DIR / "../../bin/oet_client"
aimnet2_server_path = ROOT_DIR / "../../bin/oet_server"
# Default ID and port of server. Change if needed
id_port = "127.0.0.1:9000"


def run_aimnet2(inputfile: str, output_file: str) -> None:
    run_wrapper(
        inputfile=inputfile,
        script_path=aimnet2_script_path,
        outfile=output_file,
        args=["--bind", id_port],
    )


class Aimnet2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Test starting the server
        """
        with open("server.out", "a") as f:
            cls.server = subprocess.Popen(
                [aimnet2_server_path, "aimnet2", "--bind", id_port, "--nthreads", "2"],
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
        os.killpg(os.getpgid(cls.server.pid), signal.SIGTERM)
        cls.server.wait(timeout=10)

    def test_H2O_engrad(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("H2O_client")

        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        run_aimnet2(input_file, output_file)
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
            -0.007172833196819,
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
        xyz_file, input_file, engrad_out, output_file = get_filenames("OH_anion_client")
        write_xyz_file(xyz_file, OH)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=-1,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        run_aimnet2(input_file, output_file)
        expected_num_atoms = 2
        expected_energy = -75.82629634884
        expected_gradients = [
            -0.000485832511913,
            -0.001563806785271,
            -0.0004455488233361,
            0.000485832511913,
            0.001563805621117,
            0.0004455488233361,
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
        xyz_file, input_file, engrad_out, output_file = get_filenames("OH_rad_client")
        write_xyz_file(xyz_file, OH)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=2,
            ncores=2,
            do_gradient=1,
        )
        run_aimnet2(input_file, output_file)
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

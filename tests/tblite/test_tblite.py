import shutil
import unittest
from pathlib import Path

from oet.core.test_utilities import (
    OH,
    WATER,
    get_filenames,
    read_result_file,
    run_wrapper,
    write_input_file,
    write_xyz_file,
)

# Get the path to the script that should be tested
resolved_tblite_script = shutil.which("oet_tblite")
if resolved_tblite_script is None:
    raise RuntimeError(
        "The 'oet_tblite' script was not found in PATH. "
        "Run the tests with the project's virtual environment activated."
    )
tblite_script_path = Path(resolved_tblite_script)

# Leave tblite_executable_path empty, if tblite from system path should be called
tblite_executable_path = ""


def run_tblite(inputfile: str, output_file: str) -> None:
    if tblite_executable_path:
        arguments = ["--exe", tblite_executable_path]
    else:
        arguments = None
    run_wrapper(
        inputfile=inputfile,
        script_path=tblite_script_path,
        outfile=output_file,
        args=arguments,
    )


class TbliteTests(unittest.TestCase):
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
        run_tblite(input_file, output_file)

        expected_num_atoms = 3
        expected_energy = -5.070208558737
        expected_gradients = [
            -0.011127011280,
            -0.008238194875,
            0.005819915662,
            0.004860162825,
            0.006535176840,
            -0.000835599844,
            0.006266848455,
            0.001703018035,
            -0.004984315818,
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
            self.assertAlmostEqual(g1, g2, places=6)

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
        run_tblite(input_file, output_file)

        expected_num_atoms = 2
        expected_energy = -4.681597355444
        expected_gradients = [
            0.002282296472,
            0.007346296035,
            0.002093064970,
            -0.002282296472,
            -0.007346296035,
            -0.002093064970,
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
            self.assertAlmostEqual(g1, g2, places=6)

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
        run_tblite(input_file, output_file)

        expected_num_atoms = 2
        expected_energy = -4.428349083738
        expected_gradients = [
            -0.001375464787,
            -0.004427370254,
            -0.001261421203,
            0.001375464787,
            0.004427370254,
            0.001261421203,
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
            self.assertAlmostEqual(g1, g2, places=6)

    def test_H2O_energy_only(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("H2O_energy")
        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=0,
        )
        run_tblite(input_file, output_file)

        expected_num_atoms = 3
        expected_energy = -5.070208558737

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        self.assertEqual(gradients, [])


if __name__ == "__main__":
    unittest.main()

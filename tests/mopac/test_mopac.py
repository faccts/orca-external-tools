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

mopac_script_path = Path(__file__).parent / "../../scripts/oet_mopac"
# Leave moppac_executable_path empty, if mopac from system path should be called
mopac_executable_path = ""


def run_mopac(inputfile: str, output_file: str) -> None:
    if mopac_executable_path:
        arguments = ["--exe", mopac_executable_path]
    else:
        arguments = None
    run_wrapper(
        inputfile=inputfile, script_path=mopac_script_path, outfile=output_file, args=arguments
    )


class MopacTests(unittest.TestCase):
    def test_H2O_engrad(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("OH")
        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        run_mopac(input_file, output_file)

        expected_num_atoms = 3
        expected_energy = -7.849286623778e-02
        expected_gradients = [
            -7.93660235e-03,
            -5.85955298e-03,
            4.14782376e-03,
            6.58260529e-03,
            -6.09236485e-03,
            -9.82276949e-03,
            1.35399706e-03,
            1.19519178e-02,
            5.67494573e-03,
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
        run_mopac(input_file, output_file)
        expected_num_atoms = 2
        expected_energy = -4.909937546712478e-02
        expected_gradients = [
            -1.08238184e-02,
            -3.48519301e-02,
            -9.92625736e-03,
            1.08238184e-02,
            3.48519301e-02,
            9.92625736e-03,
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
        run_mopac(input_file, output_file)
        expected_num_atoms = 2
        expected_energy = 0.0212775115576
        expected_gradients = [
            1.41467827e-03,
            4.53843190e-03,
            1.29739489e-03,
            -1.41467827e-03,
            -4.53843190e-03,
            -1.29739489e-03,
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

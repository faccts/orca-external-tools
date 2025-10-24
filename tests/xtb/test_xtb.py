import unittest
from pathlib import Path

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

xtb_script_path = ROOT_DIR / "../../bin/oet_xtb"
# Leave xtb_executable_path empty, if xtb from system path should be called
xtb_executable_path = ""


def run_xtb(inputfile: str, output_file: str) -> None:
    if xtb_executable_path:
        arguments = ["--exe", xtb_executable_path]
    else:
        arguments = None
    run_wrapper(
        inputfile=inputfile,
        script_path=xtb_script_path,
        outfile=output_file,
        args=arguments,
    )


class XtbTests(unittest.TestCase):
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
        run_xtb(input_file, output_file)

        expected_num_atoms = 3
        expected_energy = -5.07020855616
        expected_gradients = [
            -0.01112651673624,
            -0.008237828744874,
            0.005819656982873,
            0.004859971621349,
            0.006534801222881,
            -0.0008356358368765,
            0.006266545114891,
            0.001703027521994,
            -0.004984021145997,
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
        run_xtb(input_file, output_file)
        expected_num_atoms = 2
        expected_energy = -4.68159735481
        expected_gradients = [
            0.002282292426847,
            0.007346283013453,
            0.002093061259517,
            -0.002282292426847,
            -0.007346283013453,
            -0.002093061259517,
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
        run_xtb(input_file, output_file)
        expected_num_atoms = 2
        expected_energy = -4.42834908239
        expected_gradients = [
            -1.37551849e-03,
            -4.42754310e-03,
            -1.26147045e-03,
            1.37551849e-03,
            4.42754310e-03,
            1.26147045e-03,
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

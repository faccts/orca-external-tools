import shutil
import unittest
from pathlib import Path

from oet.core.test_utilities import (
    OH,
    WATER,
    get_filenames,
    read_hess_file,
    read_result_file,
    run_wrapper,
    write_input_file,
    write_xyz_file,
)

# Get the path to the script that should be tested
resolved_xtb_script = shutil.which("oet_xtb")
if resolved_xtb_script is None:
    raise RuntimeError(
        "The 'oet_xtb' script was not found in PATH. "
        "Run the tests with the project's virtual environment activated."
    )
xtb_script_path = Path(resolved_xtb_script)

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

    def test_H2O_hessian(self):
        basename = "H2O_hess"
        xyz_file, input_file, engrad_out, output_file = get_filenames(basename)
        hess_out = basename + ".hess"
        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=0,
            do_hessian=1,
        )
        run_xtb(input_file, output_file)

        expected_num_atoms = 3
        expected_energy = -5.07020855616
        # Reference Hessian (Eh/Bohr^2), flattened row-major, computed with GFN2-xTB
        # for the same WATER geometry/charge/multiplicity as test_H2O_engrad.
        expected_hessian = [
            0.2430965910,
            0.0745472590,
            -0.1884155965,
            -0.0608292195,
            -0.1043009764,
            -0.0026203284,
            -0.1822673715,
            0.0297537174,
            0.1910359249,
            0.0745472590,
            0.4172496600,
            0.1713881066,
            -0.1338212069,
            -0.3630080293,
            -0.0833669269,
            0.0592739479,
            -0.0542416307,
            -0.0880211797,
            -0.1884155965,
            0.1713881066,
            0.2791959884,
            -0.0197735972,
            -0.1115071858,
            -0.0459440378,
            0.2081891938,
            -0.0598809208,
            -0.2332519506,
            -0.0608292195,
            -0.1338212069,
            -0.0197735972,
            0.0535907569,
            0.1064876235,
            0.0107910085,
            0.0072384626,
            0.0273335834,
            0.0089825888,
            -0.1043009764,
            -0.3630080293,
            -0.1115071858,
            0.1064876235,
            0.3784810458,
            0.1184136358,
            -0.0021866471,
            -0.0154730165,
            -0.0069064499,
            -0.0026203284,
            -0.0833669269,
            -0.0459440378,
            0.0107910085,
            0.1184136358,
            0.0585198396,
            -0.0081706801,
            -0.0350467089,
            -0.0125758017,
            -0.1822673715,
            0.0592739479,
            0.2081891938,
            0.0072384626,
            -0.0021866471,
            -0.0081706801,
            0.1750289089,
            -0.0570873008,
            -0.2000185137,
            0.0297537174,
            -0.0542416307,
            -0.0598809208,
            0.0273335834,
            -0.0154730165,
            -0.0350467089,
            -0.0570873008,
            0.0697146472,
            0.0949276296,
            0.1910359249,
            -0.0880211797,
            -0.2332519506,
            0.0089825888,
            -0.0069064499,
            -0.0125758017,
            -0.2000185137,
            0.0949276296,
            0.2458277523,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=9)
        self.assertEqual(gradients, [])

        try:
            hessian = read_hess_file(hess_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error Hessian outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(len(hessian), (3 * expected_num_atoms) ** 2)
        for h1, h2 in zip(hessian, expected_hessian):
            self.assertAlmostEqual(h1, h2, places=8)
        # Check symmetry
        ndim = 3 * expected_num_atoms
        for i in range(ndim):
            for j in range(ndim):
                self.assertAlmostEqual(hessian[i * ndim + j], hessian[j * ndim + i], places=8)

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

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
resolved_mace_script = shutil.which("oet_mace")
if resolved_mace_script is None:
    raise RuntimeError(
        "The 'goet_xtb' script was not found in PATH. "
        "Run the tests with the project's virtual environment activated."
    )
mace_script_path = Path(resolved_mace_script)


def run_mace(inputfile: str, output_file: str, args: list[str] | None = None) -> None:
    # Run the wrapper with an increased timeout as loading the MACE model files might take a while
    run_wrapper(
        inputfile=inputfile,
        script_path=mace_script_path,
        outfile=output_file,
        timeout=30,
        args=args,
    )


class MACETests(unittest.TestCase):
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
        args = ["-s", "mace-mp", "-m", "medium", "--head", "mh0"]
        run_mace(input_file, output_file, args)
        expected_num_atoms = 3
        expected_energy = -5.203530407103e-01
        expected_gradients = [
            2.897306550196e-03,
            2.144325522181e-03,
            -1.515869870431e-03,
            -7.767454655073e-04,
            -3.378720118971e-03,
            -1.222818329722e-03,
            -2.120561084688e-03,
            1.234394596790e-03,
            2.738688200153e-03,
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

    def test_H2O_hessian(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("H2O_hess")
        hess_out = "H2O_hess.hess"

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
        run_mace(input_file, output_file)

        expected_num_atoms = 3
        expected_energy = -76.43356260247
        # Reference Hessian from a manual ORCA `! extopt freq` run (oet_client, default
        # aimnet2 model) on this exact geometry (see tmp-hess-test/orca-aimnet2-exthess.hess).
        # Energy/gradient from that same run agree with test_H2O_engrad's reference values
        # at places=6, confirming geometry/model consistency.
        expected_hessian = [
            2.7614297069e-01,
            7.4401343694e-02,
            -2.1164841923e-01,
            -6.7916408250e-02,
            -1.1230389988e-01,
            -4.6911567735e-03,
            -2.0822656244e-01,
            3.7902556183e-02,
            2.1633957601e-01,
            7.4401343694e-02,
            4.8811794319e-01,
            2.0761272193e-01,
            -1.5107954175e-01,
            -4.2241856845e-01,
            -9.8889386950e-02,
            7.6678198055e-02,
            -6.5699374745e-02,
            -1.0872333498e-01,
            -2.1164841923e-01,
            2.0761272193e-01,
            3.3115301489e-01,
            -2.7222273308e-02,
            -1.3585229517e-01,
            -5.7370825654e-02,
            2.3887069254e-01,
            -7.1760426763e-02,
            -2.7378218924e-01,
            -6.7916408250e-02,
            -1.5107954175e-01,
            -2.7222273308e-02,
            6.2692412922e-02,
            1.2207548020e-01,
            1.4807335852e-02,
            5.2239953286e-03,
            2.9004061548e-02,
            1.2414937456e-02,
            -1.1230389988e-01,
            -4.2241856845e-01,
            -1.3585229517e-01,
            1.2207548020e-01,
            4.3591753101e-01,
            1.3471136798e-01,
            -9.7715803234e-03,
            -1.3498962561e-02,
            1.1409271925e-03,
            -4.6911567735e-03,
            -9.8889386950e-02,
            -5.7370825654e-02,
            1.4807335852e-02,
            1.3471136798e-01,
            6.7974356994e-02,
            -1.0116179078e-02,
            -3.5821981026e-02,
            -1.0603531340e-02,
            -2.0822656244e-01,
            7.6678198055e-02,
            2.3887069254e-01,
            5.2239953286e-03,
            -9.7715803234e-03,
            -1.0116179078e-02,
            2.0300256711e-01,
            -6.6906617732e-02,
            -2.2875451346e-01,
            3.7902556183e-02,
            -6.5699374745e-02,
            -7.1760426763e-02,
            2.9004061548e-02,
            -1.3498962561e-02,
            -3.5821981026e-02,
            -6.6906617732e-02,
            7.9198337307e-02,
            1.0758240779e-01,
            2.1633957601e-01,
            -1.0872333498e-01,
            -2.7378218924e-01,
            1.2414937456e-02,
            1.1409271925e-03,
            -1.0603531340e-02,
            -2.2875451346e-01,
            1.0758240779e-01,
            2.8438572058e-01,
        ]

        try:
            num_atoms, energy, _ = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=6)

        try:
            hessian = read_hess_file(hess_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error Hessian outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(len(hessian), (3 * expected_num_atoms) ** 2)
        for h1, h2 in zip(hessian, expected_hessian):
            self.assertAlmostEqual(h1, h2, places=5)

        # Check symmetry
        ndim = 3 * expected_num_atoms
        for i in range(ndim):
            for j in range(ndim):
                self.assertAlmostEqual(hessian[i * ndim + j], hessian[j * ndim + i], places=5)

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
        args = ["-s", "omol"]
        run_mace(input_file, output_file, args)
        expected_num_atoms = 2
        expected_energy = -7.580657311911e01
        expected_gradients = [
            -1.082263844645e-03,
            -3.483610415645e-03,
            -9.925303607707e-04,
            1.082263844645e-03,
            3.483610415645e-03,
            9.925303607707e-04,
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
        run_mace(input_file, output_file)
        expected_num_atoms = 2
        expected_energy = -7.574239878105e01
        expected_gradients = [
            1.056554797887e-03,
            3.400856384066e-03,
            9.689529220712e-04,
            -1.056554797887e-03,
            -3.400856384066e-03,
            -9.689529220712e-04,
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

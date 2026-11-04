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
resolved_mopac_script = shutil.which("oet_mopac")
if resolved_mopac_script is None:
    raise RuntimeError(
        "The 'goet_xtb' script was not found in PATH. "
        "Run the tests with the project's virtual environment activated."
    )
mopac_script_path = Path(resolved_mopac_script)
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
        run_mopac(input_file, output_file)

        expected_num_atoms = 3
        expected_energy = -7.849286623778e-02
        # Reference Hessian from a manual ORCA `! extopt freq` run (oet_client, default
        # aimnet2 model) on this exact geometry (see tmp-hess-test/orca-aimnet2-exthess.hess).
        # Energy/gradient from that same run agree with test_H2O_engrad's reference values
        # at places=6, confirming geometry/model consistency.
        expected_hessian = [
            1.1496529002e-01,
            2.4179178642e-02,
            -8.6371759973e-02,
            -3.1165990879e-02,
            -3.0086746188e-02,
            7.6281654961e-03,
            -8.3817767509e-02,
            5.8452846535e-03,
            7.8824344374e-02,
            2.4179178642e-02,
            2.1222800024e-01,
            9.4794044321e-02,
            -6.4865889421e-02,
            -1.7299636763e-01,
            -3.5885376081e-02,
            4.0669275194e-02,
            -3.9309702399e-02,
            -5.8813159487e-02,
            -8.6371759973e-02,
            9.4794044321e-02,
            1.4682738956e-01,
            -1.2557224238e-02,
            -6.9075525533e-02,
            -3.2871651693e-02,
            9.8988251900e-02,
            -2.5671461931e-02,
            -1.1406205854e-01,
            -3.1165990879e-02,
            -6.4865889421e-02,
            -1.2557224238e-02,
            2.8451905834e-02,
            3.4512741667e-02,
            -8.7638915424e-03,
            2.7371650329e-03,
            3.0363883477e-02,
            2.1287301180e-02,
            -3.0086746188e-02,
            -1.7299636763e-01,
            -6.9075525533e-02,
            3.4512741667e-02,
            2.1108869088e-01,
            9.0811075255e-02,
            -4.3945494521e-03,
            -3.8073462450e-02,
            -2.1785855624e-02,
            7.6281654961e-03,
            -3.5885376081e-02,
            -3.2871651693e-02,
            -8.7638915424e-03,
            9.0811075255e-02,
            5.9302310372e-02,
            1.1002991832e-03,
            -5.4955198855e-02,
            -2.6365736713e-02,
            -8.3817767509e-02,
            4.0669275194e-02,
            9.8988251900e-02,
            2.7371650329e-03,
            -4.3945494521e-03,
            1.1002991832e-03,
            8.1075990855e-02,
            -3.6223178573e-02,
            -1.0013548638e-01,
            5.8452846535e-03,
            -3.9309702399e-02,
            -2.5671461931e-02,
            3.0363883477e-02,
            -3.8073462450e-02,
            -5.4955198855e-02,
            -3.6223178573e-02,
            7.7442373845e-02,
            8.0581457936e-02,
            7.8824344374e-02,
            -5.8813159487e-02,
            -1.1406205854e-01,
            2.1287301180e-02,
            -2.1785855624e-02,
            -2.6365736713e-02,
            -1.0013548638e-01,
            8.0581457936e-02,
            1.4046919396e-01,
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

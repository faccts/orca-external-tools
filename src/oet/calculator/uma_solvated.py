"""
This module provides UMA calculations with xTB ALPB implicit solvation corrections for ORCA's ExtTool interface.

Provides
--------
class: UmaSolvatedCalc(UmaCalc)
    Class for performing a solvated UMA calculation together with ORCA
main: function
    Main function
"""

import os
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path

from oet.calculator.uma import UmaCalc
from oet.core.base_calc import CalculationData
from oet.core.misc import mult_to_nue, run_command


class UmaSolvatedCalc(UmaCalc):
    @classmethod
    def extend_parser(cls, parser: ArgumentParser) -> None:
        """Add UmaSolvated parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        # First, add the UMA parser options
        super().extend_parser(parser)

        # Then add the solvation-specific options
        parser.add_argument(
            "--solvent",
            type=str,
            default="none",
            metavar="SOLVENT",
            dest="solvent",
            help="Solvent name for xTB ALPB solvation correction. "
            "Use 'none' to disable solvation correction. "
            "Examples: water, thf, toluene, acetonitrile, dmso, methanol. "
            "Default: none.",
        )
        parser.add_argument(
            "--xtb-exe",
            type=str,
            default="xtb",
            metavar="PATH",
            dest="xtb_exe",
            help="Path to xTB executable. Default: xtb.",
        )

    def run_xtb(
        self,
        xyzfile: Path,
        charge: int,
        mult: int,
        ncores: int,
        dograd: bool,
        solvent: str | None = None,
        xtb_exe: str = "xtb",
    ) -> tuple[float, list[float]]:
        """
        Runs an xTB calculation with optional ALPB solvation.

        Parameters
        ----------
        xyzfile : Path
            Path to XYZ file
        charge : int
            Molecular charge
        mult : int
            Multiplicity
        ncores : int
            Number of cores to use
        dograd : bool
            Whether to compute gradient
        solvent : str | None, default = None
            Solvent name for ALPB solvation (e.g., "water", "thf")
        xtb_exe : str, default = "xtb"
            Path to xTB executable

        Returns
        -------
        float
            The computed energy (Eh)
        list[float]
            Flattened gradient vector (Eh/Bohr), if computed, otherwise empty
        """
        # Create a temporary directory for xTB calculation
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Build xTB command
            args = [
                str(xyzfile),
                "-c",
                str(charge),
                "-P",
                str(ncores),
            ]

            # Add multiplicity (unpaired electrons)
            nue = mult_to_nue(mult)
            if nue:
                args += ["-u", str(nue)]

            # Add gradient calculation
            if dograd:
                args += ["--grad"]

            # Add solvation
            if solvent:
                args += ["--alpb", solvent]

            # Set output file
            output_file = tmpdir_path / "xtb.out"

            # Change to temp directory for xTB execution
            orig_dir = Path.cwd()
            os.chdir(tmpdir_path)
            try:
                # Run xTB
                run_command(xtb_exe, output_file, args)
            finally:
                # Always return to original directory
                os.chdir(orig_dir)

            # Parse energy from output
            energy = None
            with output_file.open() as f:
                for line in f:
                    if "TOTAL ENERGY" in line:
                        energy = float(line.split()[3])
                        break

            if energy is None:
                raise ValueError(f"Total energy not found in xTB output: {output_file}")

            # Parse gradient if requested
            gradient = []
            if dograd:
                gradient_file = tmpdir_path / "gradient"
                if gradient_file.exists():
                    natoms_read = 0
                    with gradient_file.open() as f:
                        for line in f:
                            if "$grad" in line:
                                break
                        for line in f:
                            fields = line.split()
                            if len(fields) == 4:
                                natoms_read += 1
                            elif len(fields) == 3:
                                gradient += [float(i) for i in fields]
                            elif "$end" in line:
                                break
                else:
                    raise FileNotFoundError(f"Gradient file not found: {gradient_file}")

            return energy, gradient

    def calc(
        self,
        calc_data: CalculationData,
        args_parsed: Namespace,
        args_not_parsed: list[str],
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient with solvation correction.

        Parameters
        ----------
        calc_data: CalculationData
            Object with calculation data for the run
        args_parsed: dict[str, Any]
            Arguments parsed as defined in extend_parser
        args_not_parsed: list[str]
            Arguments not parsed so far

        Returns
        -------
        float
            The computed energy (Eh)
        list[float]
            Flattened gradient vector (Eh/Bohr), if computed, otherwise empty
        """
        # Get solvation-specific arguments
        if args_parsed.solvent:
            solvent = args_parsed.solvent
        else:
            solvent = "none"
        xtb_exe = args_parsed.xtb_exe

        if not isinstance(solvent, str) or not isinstance(xtb_exe, str):
            raise TypeError("Problems handling solvation input parameters.")

        # Initialize solvation corrections
        e_solv_correction = 0.0
        grad_solv_correction = []

        # Calculate solvation correction if solvent is specified
        if solvent and solvent.lower() != "none":
            print(f"Calculating xTB ALPB solvation correction with solvent: {solvent}")

            # Run xTB in vacuum
            e_vacuum, grad_vacuum = self.run_xtb(
                xyzfile=calc_data.xyzfile,
                charge=calc_data.charge,
                mult=calc_data.mult,
                ncores=calc_data.ncores,
                dograd=calc_data.dograd,
                solvent=None,
                xtb_exe=xtb_exe,
            )

            # Run xTB with solvation
            e_solvated, grad_solvated = self.run_xtb(
                xyzfile=calc_data.xyzfile,
                charge=calc_data.charge,
                mult=calc_data.mult,
                ncores=calc_data.ncores,
                dograd=calc_data.dograd,
                solvent=solvent,
                xtb_exe=xtb_exe,
            )

            # Calculate solvation correction
            e_solv_correction = e_solvated - e_vacuum
            print(f"  E(xtb,vacuum)   = {e_vacuum:16.10f} Eh")
            print(f"  E(xtb,solvated) = {e_solvated:16.10f} Eh")
            print(f"  E(solvation)    = {e_solv_correction:16.10f} Eh")

            # Calculate gradient correction if needed
            if calc_data.dograd and grad_vacuum and grad_solvated:
                grad_solv_correction = [
                    g_solv - g_vac for g_solv, g_vac in zip(grad_solvated, grad_vacuum)
                ]

        # Run UMA calculation
        energy_uma, gradient_uma = super().calc(calc_data, args_parsed, args_not_parsed)

        # Add solvation correction
        energy_total = energy_uma + e_solv_correction
        gradient_total = gradient_uma

        if grad_solv_correction:
            gradient_total = [
                g_uma + g_solv for g_uma, g_solv in zip(gradient_uma, grad_solv_correction)
            ]

        print(f"  E(UMA)          = {energy_uma:16.10f} Eh")
        if e_solv_correction != 0.0:
            print(f"  E(UMA+solv)     = {energy_total:16.10f} Eh")

        return energy_total, gradient_total


def main() -> None:
    """
    Main routine for execution
    """
    calculator = UmaSolvatedCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()

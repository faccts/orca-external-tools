"""Contract tests for the AIMNet2 v0.2 salvage.

Lives alongside test_aiment2_standalone.py / test_aiment2_client.py
(unittest-style; preserved untouched). New behaviors get pytest-style
tests here.
"""

import argparse
import os
import warnings
from unittest.mock import MagicMock, patch

import pytest

from oet.calculator.aimnet2 import Aimnet2Calc


@pytest.fixture
def extinp_water_fixture(tmp_path):
    """Build a minimal ORCA extinp + xyz for water (charge=0, mult=1, ncores=1).

    O-H stretched to 1.06 A so the gradient is non-trivial — useful for
    smoke tests that want to exercise the gradient pathway, not just
    assert finiteness near equilibrium.
    """
    xyz = tmp_path / "h2o_EXT.xyz"
    xyz.write_text(
        "3\n\n"
        "O 0.000000 0.000000 0.000000\n"
        "H 0.000000 0.000000 1.060000\n"
        "H 0.930000 0.000000 -0.240000\n"
    )
    extinp = tmp_path / "h2o_EXT.extinp.tmp"
    # extinp format: xyz_filename, charge, mult, ncores, dograd, [pointcharges]
    extinp.write_text(f"{xyz}\n0\n1\n1\n1\n")
    return extinp


@pytest.fixture
def extinp_h2_no_grad(tmp_path):
    """Minimal H2 extinp with dograd=0 — used by argparse-rejection tests
    that need a valid extinp but don't actually run the calculator.
    """
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")
    extinp = tmp_path / "h2_EXT.extinp.tmp"
    extinp.write_text(f"{xyz}\n0\n1\n1\n0\n")
    return extinp


class TestPeriodicTableCoverage:
    """Test C from the design spec — periodic-table translator coverage."""

    def setup_method(self):
        self.calc = Aimnet2Calc()

    def test_first_row(self):
        assert self.calc.atomic_symbol_to_number("H") == 1
        assert self.calc.atomic_symbol_to_number("He") == 2

    def test_main_group(self):
        assert self.calc.atomic_symbol_to_number("C") == 6
        assert self.calc.atomic_symbol_to_number("O") == 8
        assert self.calc.atomic_symbol_to_number("Cl") == 17

    def test_transition_metal(self):
        assert self.calc.atomic_symbol_to_number("Pd") == 46

    def test_actinide(self):
        assert self.calc.atomic_symbol_to_number("U") == 92

    def test_transactinide(self):
        assert self.calc.atomic_symbol_to_number("Og") == 118

    def test_case_insensitive(self):
        assert self.calc.atomic_symbol_to_number("h") == 1
        assert self.calc.atomic_symbol_to_number("cl") == 17
        assert self.calc.atomic_symbol_to_number("CL") == 17

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown element symbol"):
            self.calc.atomic_symbol_to_number("Xx")
        with pytest.raises(ValueError, match="Unknown element symbol"):
            self.calc.atomic_symbol_to_number("Bq")  # ghost atoms not supported


class TestNseMultGating:
    """Test B from the design spec - mult key only included for NSE models."""

    def setup_method(self):
        self.calc = Aimnet2Calc()

    def _make_input_kwargs(self, is_nse: bool):
        self.calc._calc = MagicMock()
        self.calc._calc.is_nse = is_nse
        return self.calc.serialize_input(
            atom_types=["H", "H"],
            coordinates=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)],
            charge=0,
            mult=1,
            dograd=True,
        )

    def test_mult_omitted_for_non_nse_model(self):
        out = self._make_input_kwargs(is_nse=False)
        assert "mult" not in out["data"]
        assert "coord" in out["data"]
        assert "numbers" in out["data"]
        assert "charge" in out["data"]

    def test_mult_included_for_nse_model(self):
        out = self._make_input_kwargs(is_nse=True)
        assert out["data"]["mult"] == [1]

    def test_forces_passes_through_dograd(self):
        out = self._make_input_kwargs(is_nse=False)
        assert out["forces"] is True
        assert out["stress"] is False
        assert out["hessian"] is False

    def test_open_shell_with_non_nse_model_warns(self):
        """mult != 1 + non-NSE model emits a UserWarning that points to
        aimnet2-nse (open-shell guidance)."""
        self.calc._calc = MagicMock()
        self.calc._calc.is_nse = False
        with pytest.warns(UserWarning, match="aimnet2-nse"):
            self.calc.serialize_input(
                atom_types=["O", "H"],
                coordinates=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.97)],
                charge=0,
                mult=2,
                dograd=True,
            )

    def test_open_shell_with_nse_model_no_warn(self):
        """mult != 1 + NSE model passes mult through and does not warn."""
        self.calc._calc = MagicMock()
        self.calc._calc.is_nse = True
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # turn warnings into errors
            out = self.calc.serialize_input(
                atom_types=["O", "H"],
                coordinates=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.97)],
                charge=0,
                mult=2,
                dograd=True,
            )
        assert out["data"]["mult"] == [2]


class TestSetupSignature:
    """Sanity checks on the new setup() method (no model load)."""

    def test_cuda_raises_when_unavailable(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        calc = Aimnet2Calc()
        with pytest.raises(RuntimeError, match="CUDA requested but not available"):
            calc.setup(model="aimnet2", model_dir="/tmp", device="cuda", ncores=1)

    def test_args_match_short_circuit(self):
        """Second setup() call with same args is a no-op."""
        calc = Aimnet2Calc()
        # Pre-populate as if first setup ran. MagicMock() is sufficient
        # here because the args-match branch returns before touching
        # any method on _calc.
        calc._calc = MagicMock()
        calc._setup_args = frozenset(
            {
                "model": "aimnet2",
                "model_dir": "/tmp",
                "device": "cpu",
                "ncores": 1,
                "compile_model": False,
                "nb_threshold": 120,
                "ensemble_member": 0,
                "coulomb": "auto",
                "dispersion": "auto",
                "coulomb_method": None,
                "coulomb_cutoff": 15.0,
                "dftd3_cutoff": None,
                "dftd3_smoothing_fraction": None,
            }.items()
        )
        # Same args -> no-op (does not raise, does not touch _calc).
        calc.setup(
            model="aimnet2",
            model_dir="/tmp",
            device="cpu",
            ncores=1,
        )
        assert calc._calc is not None  # still the original mock

    def test_args_mismatch_raises(self):
        """Second setup() call with different args raises."""
        calc = Aimnet2Calc()
        # MagicMock() is sufficient here because the args-mismatch branch
        # raises before touching any method on _calc.
        calc._calc = MagicMock()
        calc._setup_args = frozenset(
            {
                "model": "aimnet2",
                "model_dir": "/tmp",
                "device": "cpu",
                "ncores": 1,
                "compile_model": False,
                "nb_threshold": 120,
                "ensemble_member": 0,
                "coulomb": "auto",
                "dispersion": "auto",
                "coulomb_method": None,
                "coulomb_cutoff": 15.0,
                "dftd3_cutoff": None,
                "dftd3_smoothing_fraction": None,
            }.items()
        )
        with pytest.raises(RuntimeError, match="different args"):
            calc.setup(
                model="aimnet2-2025",
                model_dir="/tmp",
                device="cpu",
                ncores=1,
            )

    def test_auto_and_none_device_compare_equal(self):
        """device='auto' and device=None should both normalize to None
        in _setup_args, so back-to-back setup() calls swapping between
        them are idempotent (not a different-args raise)."""
        calc = Aimnet2Calc()
        calc._calc = MagicMock()
        # Stash as if first call was device="auto" (normalized to None)
        calc._setup_args = frozenset(
            {
                "model": "aimnet2",
                "model_dir": "/tmp",
                "device": None,
                "ncores": 1,
                "compile_model": False,
                "nb_threshold": 120,
                "ensemble_member": 0,
                "coulomb": "auto",
                "dispersion": "auto",
                "coulomb_method": None,
                "coulomb_cutoff": 15.0,
                "dftd3_cutoff": None,
                "dftd3_smoothing_fraction": None,
            }.items()
        )
        # Second call with device=None — same after normalization → no-op
        calc.setup(
            model="aimnet2",
            model_dir="/tmp",
            device=None,
            ncores=1,
        )
        assert calc._calc is not None

    def test_invalid_coulomb_choice_raises(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        calc = Aimnet2Calc()
        with pytest.raises(ValueError, match="coulomb='yes'"):
            calc.setup(
                model="aimnet2",
                model_dir="/tmp",
                device="cpu",
                ncores=1,
                coulomb="yes",
            )

    def test_invalid_coulomb_method_raises(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        calc = Aimnet2Calc()
        with pytest.raises(ValueError, match="coulomb_method='wolf'"):
            calc.setup(
                model="aimnet2",
                model_dir="/tmp",
                device="cpu",
                ncores=1,
                coulomb_method="wolf",
            )

    def test_rxn_with_non_trained_cutoff_warns(self, monkeypatch, tmp_path):
        """aimnet2-rxn family + --coulomb-method set + cutoff != 4.6 must
        emit a UserWarning pointing the user at the trained cutoff."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        # Stub get_model_file so we don't hit the network; produce a path
        # whose stem encodes the rxn family.
        rxn_path = tmp_path / "aimnet2-rxn_0.pt"
        rxn_path.write_bytes(b"")
        calc = Aimnet2Calc()
        with patch.object(Aimnet2Calc, "get_model_file", return_value=rxn_path):
            with patch("oet.calculator.aimnet2.AIMNet2Calculator") as MockCalc:
                MockCalc.return_value = MagicMock()
                with pytest.warns(UserWarning, match="aimnet2-rxn training cutoff"):
                    calc.setup(
                        model="aimnet2-rxn",
                        model_dir=str(tmp_path),
                        device="cpu",
                        ncores=1,
                        coulomb_method="dsf",
                        coulomb_cutoff=12.0,
                    )

    def test_rxn_with_trained_cutoff_no_warn(self, monkeypatch, tmp_path):
        """aimnet2-rxn + cutoff = 4.6 must NOT emit the rxn-cutoff warning."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        rxn_path = tmp_path / "aimnet2-rxn_0.pt"
        rxn_path.write_bytes(b"")
        calc = Aimnet2Calc()
        with patch.object(Aimnet2Calc, "get_model_file", return_value=rxn_path):
            with patch("oet.calculator.aimnet2.AIMNet2Calculator") as MockCalc:
                MockCalc.return_value = MagicMock()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    calc.setup(
                        model="aimnet2-rxn",
                        model_dir=str(tmp_path),
                        device="cpu",
                        ncores=1,
                        coulomb_method="dsf",
                        coulomb_cutoff=4.6,
                    )
                rxn_warnings = [
                    w for w in caught if "aimnet2-rxn training cutoff" in str(w.message)
                ]
                assert rxn_warnings == []

    def test_rxn_without_coulomb_method_no_warn(self, monkeypatch, tmp_path):
        """aimnet2-rxn without --coulomb-method ignores --coulomb-cutoff
        entirely; warning should NOT fire on the default 15.0."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        rxn_path = tmp_path / "aimnet2-rxn_0.pt"
        rxn_path.write_bytes(b"")
        calc = Aimnet2Calc()
        with patch.object(Aimnet2Calc, "get_model_file", return_value=rxn_path):
            with patch("oet.calculator.aimnet2.AIMNet2Calculator") as MockCalc:
                MockCalc.return_value = MagicMock()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    calc.setup(
                        model="aimnet2-rxn",
                        model_dir=str(tmp_path),
                        device="cpu",
                        ncores=1,
                    )
                rxn_warnings = [
                    w for w in caught if "aimnet2-rxn training cutoff" in str(w.message)
                ]
                assert rxn_warnings == []


class TestReleaseHook:
    """Server-side eviction calls release(); verify it clears state."""

    def test_release_drops_calc_and_args(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        calc = Aimnet2Calc()
        calc._calc = MagicMock()
        calc._setup_args = frozenset({"sentinel": "dummy"}.items())
        calc.release()
        assert calc._calc is None
        assert calc._setup_args is None

    def test_release_no_op_on_uninitialized(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        calc = Aimnet2Calc()
        # Must not raise even though _calc was never set.
        calc.release()
        assert calc._calc is None

    def test_release_then_setup_with_different_args(self, monkeypatch):
        """After release(), a fresh setup() with different args should work
        (release() clears _setup_args so the args-mismatch check passes).
        Verifies the lifecycle: setup() -> release() -> setup() -> ...
        """
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        calc = Aimnet2Calc()
        # Pre-populate with first-setup state
        calc._calc = MagicMock()
        calc._setup_args = frozenset({"sentinel": "first"}.items())

        calc.release()
        assert calc._calc is None
        assert calc._setup_args is None

        # Second setup with completely different args should construct fresh
        # (not raise the "different args" RuntimeError).
        with patch("oet.calculator.aimnet2.AIMNet2Calculator") as MockCalc:
            MockCalc.return_value = MagicMock()
            calc.setup(
                model="aimnet2-2025",
                model_dir="/tmp",
                device="cpu",
                ncores=2,
            )
        assert calc._calc is not None
        assert calc._setup_args is not None

    def test_setup_post_ctor_failure_clears_calc(self, monkeypatch):
        """If set_lrcoulomb_method raises after the ctor moved the model to
        device, setup() must release device-resident state before re-raising
        (otherwise a repeat-bad-config server loop accumulates VRAM).
        """
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        calc = Aimnet2Calc()

        bad_calc = MagicMock()
        bad_calc.set_lrcoulomb_method.side_effect = RuntimeError("bad config")
        bad_calc.model = MagicMock()
        bad_calc.external_coulomb = MagicMock()
        bad_calc.external_dftd3 = None

        with patch("oet.calculator.aimnet2.AIMNet2Calculator", return_value=bad_calc):
            with pytest.raises(RuntimeError, match="bad config"):
                calc.setup(
                    model="aimnet2",
                    model_dir="/tmp",
                    device="cpu",
                    ncores=2,
                    coulomb_method="dsf",
                    coulomb_cutoff=12.0,
                )

        # Rollback contract: components walked back to cpu BEFORE _calc dropped.
        bad_calc.model.to.assert_called_with("cpu")
        bad_calc.external_coulomb.to.assert_called_with("cpu")
        assert calc._calc is None
        assert calc._setup_args is None


class TestBaseCalcReleaseDefault:
    """BaseCalc.release() exists, is callable, and is a no-op by default
    (M4 — makes the sibling server-VRAM PR genuinely order-independent
    by ensuring polymorphic `calc.release()` calls are safe on any
    BaseCalc subclass that hasn't overridden it).
    """

    def test_release_default_is_callable_and_no_op(self):

        from oet.core.base_calc import BaseCalc

        class _StubCalc(BaseCalc):
            def calc(self, calc_data, args_parsed, args_not_parsed):  # pragma: no cover
                return 0.0, []

        stub = _StubCalc()
        # Must not raise; must return None.
        assert stub.release() is None
        # Idempotent.
        assert stub.release() is None


class TestArgparseContract:
    """Test A from the design spec - flag presence, defaults, choices, mappings.

    Catches argparse-side regressions (rename, accidental drop) at near-zero
    cost. No network, no model load.
    """

    def _make_parser(self):
        parser = argparse.ArgumentParser()
        Aimnet2Calc.extend_parser(parser)
        # extend_parser doesn't add the inputfile positional (BaseCalc does);
        # add a stub so parse_args() works for tests.
        if not any(a.dest == "inputfile" for a in parser._actions):
            parser.add_argument("inputfile", nargs="?", default="dummy.tmp")
        return parser

    def test_all_v02_flags_present(self):
        parser = self._make_parser()
        all_opts = {opt for a in parser._actions for opt in a.option_strings}
        for flag in [
            "-m",
            "--model",
            "-p",
            "--model-path",
            "-d",
            "--device",
            "--compile",
            "--nb-threshold",
            "--ensemble-member",
            "--coulomb",
            "--coulomb-method",
            "--coulomb-cutoff",
            "--dispersion",
            "--dftd3-cutoff",
            "--dftd3-smoothing-fraction",
        ]:
            assert flag in all_opts, f"{flag!r} missing from parser options"

    def test_dropped_flags_absent(self):
        parser = self._make_parser()
        all_opts = {opt for a in parser._actions for opt in a.option_strings}
        for flag in [
            "--revision",
            "--token",
            "--dsf-alpha",
            "--ewald-accuracy",
            "--needs-coulomb",
            "--no-needs-coulomb",
            "--needs-dispersion",
            "--no-needs-dispersion",
        ]:
            assert flag not in all_opts, f"{flag!r} should have been dropped"

    def test_defaults(self):
        parser = self._make_parser()
        ns = parser.parse_args(["dummy.tmp"])
        assert ns.model == "aimnet2"
        assert ns.device == "cpu"
        assert ns.compile is False
        assert ns.nb_threshold == 120
        assert ns.ensemble_member == 0
        assert ns.coulomb == "auto"
        assert ns.dispersion == "auto"
        assert ns.coulomb_method is None
        assert ns.coulomb_cutoff == 15.0
        assert ns.dftd3_cutoff is None
        assert ns.dftd3_smoothing_fraction is None

    def test_device_choices(self):
        parser = self._make_parser()
        ns = parser.parse_args(["--device", "auto", "dummy.tmp"])
        assert ns.device == "auto"
        with pytest.raises(SystemExit):
            parser.parse_args(["--device", "tpu", "dummy.tmp"])

    def test_coulomb_choices(self):
        parser = self._make_parser()
        ns = parser.parse_args(["--coulomb", "on", "dummy.tmp"])
        assert ns.coulomb == "on"
        with pytest.raises(SystemExit):
            parser.parse_args(["--coulomb", "yes", "dummy.tmp"])

    def test_coulomb_method_choices(self):
        parser = self._make_parser()
        for m in ("simple", "dsf", "ewald"):
            ns = parser.parse_args(["--coulomb-method", m, "dummy.tmp"])
            assert ns.coulomb_method == m
        with pytest.raises(SystemExit):
            parser.parse_args(["--coulomb-method", "wolf", "dummy.tmp"])

    def test_ensemble_member_choices(self):
        parser = self._make_parser()
        # Valid choices 0..3
        for n in (0, 1, 2, 3):
            ns = parser.parse_args(["--ensemble-member", str(n), "dummy.tmp"])
            assert ns.ensemble_member == n
        # Out of range rejects
        with pytest.raises(SystemExit):
            parser.parse_args(["--ensemble-member", "4", "dummy.tmp"])

    def test_dispersion_choices(self):
        parser = self._make_parser()
        ns = parser.parse_args(["--dispersion", "off", "dummy.tmp"])
        assert ns.dispersion == "off"
        with pytest.raises(SystemExit):
            parser.parse_args(["--dispersion", "yes", "dummy.tmp"])

    def test_coulomb_cutoff_without_method_rejects(self, extinp_h2_no_grad, monkeypatch):
        """--coulomb-cutoff without --coulomb-method must raise SystemExit.

        The validation is post-parse (in calc()), so we must invoke the
        full main() entry point — parser.parse_args() alone wouldn't
        trigger it.
        """
        # Force CPU so the test doesn't try to use CUDA in CI
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)

        from oet.calculator.aimnet2 import main

        with pytest.raises(SystemExit):
            main([str(extinp_h2_no_grad), "--coulomb-cutoff", "12.0"])


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("OET_RUN_NETWORK_TESTS") != "1",
    reason="set OET_RUN_NETWORK_TESTS=1 to run network tests",
)
class TestSmokeDefaultModel:
    """Test D from the design spec — end-to-end on the default model.

    Downloads aimnet2 (= aimnet2-wb97m-d3_0) on first run; subsequent
    runs use the local cache.
    """

    def test_default_model_round_trip(self, extinp_water_fixture, tmp_path):
        """Default model loads, runs SP+grad, writes engrad parseable by ORCA.

        Uses the perturbed-water fixture (one O-H stretched to 1.06 A) so the
        gradient is non-trivial; the bound below catches sign-flip / unit
        bugs that an equilibrium geometry would silently mask.
        """
        from oet.calculator.aimnet2 import main

        main([str(extinp_water_fixture)])

        # Assert the engrad file exists and parses.
        engrad = tmp_path / "h2o_EXT.engrad"
        assert engrad.exists(), (
            f"engrad file not written; tmp_path contains: {list(tmp_path.iterdir())}"
        )
        text = engrad.read_text()

        # Parse the engrad: skip comments and the natoms line, then
        # the next float is the energy and the following 9 are gradient.
        floats: list[float] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                floats.append(float(line))
            except ValueError:
                continue
        # First float is the natoms-line value (3); skip it.
        # Then: energy + 9 gradient components.
        assert floats[0] == 3.0, (
            f"first non-comment line should be natoms=3, got {floats[0]}\nengrad text:\n{text}"
        )
        energy = floats[1]
        grad = floats[2:11]
        assert -100.0 < energy < 0.0, f"water energy {energy} Eh implausible\nengrad text:\n{text}"
        assert len(grad) == 9, f"expected 9 grad components, got {len(grad)}\nengrad text:\n{text}"
        # Stretched O-H at 1.06 A gives a non-trivial gradient on the
        # stretched H (~0.03-0.10 Eh/Bohr). Assert at least one component
        # is in the order-of-magnitude band for a real gradient (catches
        # sign-flip / unit bugs that would either zero the gradient or
        # blow it up).
        max_grad = max(abs(g) for g in grad)
        assert 0.005 < max_grad < 0.5, (
            f"max grad component {max_grad} outside expected (0.005, 0.5) Eh/Bohr "
            f"— possible unit/sign bug\nengrad text:\n{text}"
        )

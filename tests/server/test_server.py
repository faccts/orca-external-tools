import tempfile
import unittest
from argparse import Namespace

from oet.server_client.server import (
    CalculatorRuntimeException,
    _run_calc_in_process,
    create_app,
)


class ExitingCalc:
    """Stand-in for a calculator that calls sys.exit() on failure, as run_command does."""

    def run(self, **kwargs):
        print("calculation failed")
        raise SystemExit(1)


class ExitingServer:
    """Stand-in for OtoolServer whose calculation ends with sys.exit()."""

    def handle_client(self, content):
        raise SystemExit(1)


class ServerTests(unittest.TestCase):
    def test_worker_turns_sys_exit_into_error(self):
        # This module is the calculator module, so that no backend is needed
        run_kwargs = {"args_parsed": Namespace(), "args_not_parsed": []}
        with self.assertRaises(CalculatorRuntimeException) as cm:
            _run_calc_in_process(__name__, "ExitingCalc", run_kwargs, max_memory_per_thread=10**6)
        self.assertIn("calculation failed", cm.exception.stdout)

    def test_calculate_answers_on_sys_exit(self):
        # Without an answer, the client would wait forever
        app = create_app(ExitingServer())
        with tempfile.TemporaryDirectory() as directory:
            response = app.test_client().post(
                "/calculate", json={"arguments": [], "directory": directory}
            )
        self.assertEqual(response.get_json()["status"], "Error")
        self.assertEqual(response.get_json()["error_type"], "SystemExit")


if __name__ == "__main__":
    unittest.main()

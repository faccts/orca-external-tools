import os

# > External packages
import nox

# > Making sure Nox session only see their packages and not any globally installed packages.
os.environ.pop("PYTHONPATH", None)
# > Hiding any virtual environments from outside.
os.environ.pop("VIRTUAL_ENV", None)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                               GLOBAL NOX OPTIONS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# > Stop after first session that fails.
# > Can be turned off with '--no-stop-on-first-error'
nox.options.stop_on_first_error = True
# > NOX can detect if a binary is called from outside the currently running session.
# > Make sure we use that the binary from the session specific virtual environment.
nox.options.error_on_external_run = True
# > Mark sessions as failed if NOX cannot find the desired Python interpreter.
# > By default, NOX just skips these sessions.
nox.options.error_on_missing_interpreters = True
# > Always start from a clean environment.
nox.options.reuse_existing_virtualenvs = True
# > Using "venv" as default backend
nox.options.default_venv_backend = "venv"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                   NOX SESSIONS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# //////////////////////////////////////////
# ///     STATIC TYPE CHECKING: mypy     ///
# //////////////////////////////////////////
@nox.session(tags=["static_check"])
def type_check(session):
    session.install(".[type-check]")
    session.run("mypy")


# //////////////////////////////////////////////////
# ///        REMOVING UNUSED IMPORTS: Ruff      ///
# ////////////////////////////////////////////////
@nox.session(tags=["style", "fix", "static_check"])
def remove_unused_imports(session):
    session.install(".[lint]")
    # > Sorting imports with ruff instead of isort
    session.run("ruff", "check", "--select", "F401", "--fix", "--exit-non-zero-on-fix")


# //////////////////////////////////////////
# ///        SORTING IMPORTS: Ruff      ///
# //////////////////////////////////////////
@nox.session(tags=["style", "fix", "static_check"])
def sort_imports(session):
    session.install(".[lint]")
    # > Sorting imports with ruff instead of isort
    session.run("ruff", "check", "--select", "I", "--fix", "--exit-non-zero-on-fix")


# ////////////////////////////////////////
# ///         LINTING: Ruff            ///
# ////////////////////////////////////////
@nox.session(tags=["style", "static_check"])
def lint(session):
    session.install(".[lint]")
    session.run("ruff", "check")


# //////////////////////////////////////////
# ///         CODE FORMATTING: Ruff     ///
# //////////////////////////////////////////
@nox.session(tags=["style", "fix", "static_check"])
def format_code(session):
    # Installs the project + the "lint" extra into this nox venv using pip
    session.install(".[lint]")
    session.run("ruff", "format", "--exit-non-zero-on-format")


# ////////////////////////////////////////////////////
# ///         SPELL CHECKING: codespell            ///
# ////////////////////////////////////////////////////
@nox.session(tags=["static_check"])
def spell_check(session):
    session.install(".[spell-check]")
    session.run("codespell", "src/oet")


# //////////////////////////////////////////////
# ///         DEAD CODE: vulture            ///
# /////////////////////////////////////////////
@nox.session(tags=["static_check"], default=True)
def dead_code(session):
    session.install(".[dead-code]")
    session.run("vulture")

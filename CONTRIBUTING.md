# Contributing to orca-external-tools

Thank you for your interest in contributing to this project! We're excited to collaborate with you.

Before you start, please familiarize yourself with the following guidelines.
This ensures that your contribution can be merged as soon as possible so that everyone can benefit from it.

## Reporting a Bug

A bug is a problem or unexpected behavior caused by the code in this repository.  
Please report any bugs you encounter—this is extremely valuable to us for improving the code.

Before opening a bug report:

1. Check if the issue has already been reported.
2. Check whether it still occurs or has already been fixed.  
   You might try reproducing it using the latest version from the `master` branch.
3. (Optional) Isolate the problem and create a minimal test case to demonstrate the issue.

Please try to be as detailed as possible in your report, and  
answer at least the following questions or choose the appropriate issue template:

1. Which version of `ORCA` are you using?
2. What is your environment (OS, Python version, etc.)?
3. What steps will reproduce the issue?  
   (We need all relevant input files to reproduce it.)
4. What is the expected outcome?
5. What did you observe instead?

All these details will help us fix potential bugs more efficiently.

## Suggesting a New Feature

Feature requests are welcome. It's helpful if you explain your requested feature in detail, describe what you want to achieve with it, and (if possible) suggest a potential solution.

## Implementing a New Feature

Contributions are very welcome via GitHub pull requests.

- Each pull request should implement *one* feature or fix *one* bug.  
  If you want to add or fix more than one thing, please submit separate pull requests.  
  This makes the history easier to track.
- Do not commit changes to files that are unrelated to your feature or bug fix.
- Ensure that the code compiles and works correctly.
- Follow the code guidelines and use the code quality tool `Nox` before starting a PR (see below)

After submitting the pull request, you will be asked in the PR thread to sign  
a [CLA](CLA.md) with your GitHub account.  
We will review the code as soon as possible, provide feedback, and  
you will then be able to merge the code.

### Installing and using Nox

[Nox](https://nox.thea.codes/en/stable/) is a framework for automated testing.
It is configured via a central configuration file `noxfile.py`.
**Only touch this file if you know what you are doing!**

We use Nox not just for testing but also to drive all our tools that we use to ensure code quality and safety.

Before using `Nox`, please install the `oet` with some additional dependencies.
Therefore, use the `install.py` script and add the `--dev` argument.

```
python install.py --dev
```

This will create a new virtual environment (venv) with a matching Python version.
The path to the virtual environment and to the script directory can be adjusted as described in the `README.md` file.

To execute Nox navigate to the project's root directory and use the following command:

```
nox <additional nox arguments>
```

Without any additional arguments, Nox will execute all (default) sessions that are defined in `noxfile.py`.
These will also automatically be checked when starting a PR and must be passed before merging.
A session is a predefined Python function that executes some code and returns a status.
To list all available sessions use:

```
nox -l
```
Each session is executed in its own and isolated Python environment, which is set up and managed by Nox according to the
configuration.
This ensures that the outcome of each session is consistent across different environments.

The following sessions are included in the `oet` defaults:
- Static Type Checking: mypy
- Linting and formatting: Ruff
- Spell Checking: Codespell

### Code Guidelines

We strictly follow the rules employed in [Black](https://black.readthedocs.io/en/stable/).
For performance reasons, we use Ruff, which is designed as a drop-in replacement for Black.
Ruff is not fully compliant to Black ([https://docs.astral.sh/ruff/formatter/#intentional-deviations](https://docs.astral.sh/ruff/formatter/#intentional-deviations)).
Further, we follow PEP 8 guidelines for naming variables, functions, and other identifiers to ensure consistency and readability across our codebase.
Indentation is standardized to 4 spaces per level—tabs must not be used.
Additionally, we've extended the maximum line length to 100 characters to enhance code clarity and make better use of modern screen widths.
Please name files using snake_case, with words separated by underscores (e.g. `this_is_my_file.py`).
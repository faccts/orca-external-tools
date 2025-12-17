# MACE ExtTool for ORCA

Wrapper around MACE calculators to use ORCA's `otool_external` interface. Mirrors the UMA tool interface with standalone and server–client modes.

- Suites: `mace-mp` (Materials Project) and `mace-omol` (OMOL foundation model)
- Extras: `dispersion` (MP only), `default_dtype` (`float32` or `float64`), optional `device`, and `head` for advanced MP heads.

## Quick start

- Standalone (single call per ORCA task):
  - `./mace.sh -s mp -m medium-mpa-0 your_job_EXT.extinp.tmp`
  - `./mace.sh -s omol your_job_EXT.extinp.tmp`

- Server–client (faster for many calls in one ORCA job):
  - Start: `./maceserver.sh -s mp -m medium-mpa-0 -n 2 -b 127.0.0.1:8888`
  - Use in ORCA input: `progext ".../maceclient.sh"` (ORCA passes the temp input filepath)

## Arguments

Common:
- `-s, --suite`: `mp` or `omol` (default: `omol`)
- `-m, --model`: model spec or local path (MP: e.g. `medium-mpa-0`, `medium`, `small`; OMOL: `extra_large` or path)
- `--default-dtype`: `float32` (MD speed) or `float64` (opt accuracy)
- `--device`: `cpu`, `cuda`, etc.

MP only:
- `--dispersion`: enable D3 dispersion (off by default)
- `--damping`, `--dispersion-xc`, `--dispersion-cutoff` advanced dispersion controls
- `--head`: MACE head selection for multi-head variants

Server:
- `-b, --bind`: `host:port` (default: `127.0.0.1:8888`)
- `-n, --nthreads`: number of threads per server

## Install

- `cd mace && ./install.sh`
- This installs a venv, the wrapper, and tries to `pip install -e ../../mace` if the local MACE repo exists. Otherwise it relies on `mace-torch`.


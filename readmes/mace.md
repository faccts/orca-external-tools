# MACE ExtTool for ORCA

Wrapper around MACE calculators to use ORCA's `otool_external` interface.

- Suites: `mace-mp` (Materials Project) and `mace-omol` (OMOL foundation model)
- Extras: `dispersion` (MP only), `default_dtype` (`float32` or `float64`), optional `device`, and `head` for advanced MP heads.

## Arguments

| Category | Argument | Values / Description | Default |
|---|---|---|---|
| Common | `-s, --suite` | `mp` or `omol` | `omol` |
| Common | `-m, --model` | Model spec or local path. MP examples: `medium-mpa-0`, `medium`, `small`; OMOL: `extra_large` or local path | — |
| Common | `--default-dtype` | `float32` (MD speed) or `float64` (optimization accuracy) | — |
| Common | `--device` | Compute device, e.g. `cpu`, `cuda` | Auto / framework default |
| MP only | `--dispersion` | Enable D3 dispersion correction | Off |
| MP only | `--damping` | Advanced dispersion damping controls | — |
| MP only | `--dispersion-xc` | Exchange-correlation functional for dispersion | — |
| MP only | `--dispersion-cutoff` | Dispersion cutoff distance | — |
| MP only | `--head` | MACE head selection for multi-head variants | — |
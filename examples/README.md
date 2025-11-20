# MACE GOAT Examples (Custom Model)

This folder shows how to run ORCA's global optimization tool (GOAT) using the MACE wrapper with a custom model file.

Two variants are provided:
- Server–client (recommended for many external calls): `goat_water_mace_omol_server.inp`
- Standalone (simpler, slower for many calls): `goat_water_mace_omol_standalone.inp`

Replace `/abs/path/to/your/MACE-omol-custom.model` with your actual model file path before running.

## Server–Client (recommended)

1) Start the MACE server in another terminal with your custom model:

```
../maceserver.sh \
  -s omol \
  -m /abs/path/to/your/MACE-omol-custom.model \
  --default-dtype float64 \
  --device cpu \
  -n 4 \
  -b 127.0.0.1:8888
```

2) Run ORCA on the provided input:

```
orca goat_water_mace_omol_server.inp > goat_water_mace_omol_server.out
```

- The input references `../maceclient.sh` and points it to the server via `Ext_Params "-b 127.0.0.1:8888"`.
- This example uses `pal1` to avoid requiring MPI. If you have MPI, you can change to `pal4`.

## Standalone (simple, no server)

Run ORCA on the standalone input:

```
orca goat_water_mace_omol_standalone.inp > goat_water_mace_omol_standalone.out
```

- The input references `../mace.sh` and passes the suite and custom model via `Ext_Params`.
- Recommended flags for optimization: `--default-dtype float64` and `--device cpu`.

Notes
- Ensure ORCA is available on your PATH (or call it via absolute path).
- Use absolute paths for the custom model so ORCA can find it regardless of working directory.
- For Materials Project models (mp), change `-s omol` to `-s mp` and add flags like `--dispersion` or `--head mh0` as needed.


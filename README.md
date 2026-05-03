# orca-external-tools

This repository contains wrapper scripts compatible with the `otool_external` interface in ORCA.
The scripts call an external program, which computes the energy and gradient of a system, 
then pass this information back to ORCA for use in optimization, NEB, GOAT, MD, etc.

## Installation

Use the `install.py` to install the scripts. 
It will create a virtual environment, whose path should later on not be changed due to scripts linking the absolute path. 
The installation name and path of the virtual environment can be set upon installation:

```
python install.py --venv-dir <path/to/venv/dir/>
```

If you want to use either AIMNet2 or UMA, you can add `-e aimnet2` or `-e uma` to additionally install the required dependencies. 
As AIMNet2 and UMA require dependencies that are not compatible with each other, 
we recommend creating separate installations for each by specifying different virtual environment and script directories.

After installation, you should have a directory called `bin` (by default)
which contains all wrapper scripts that are usable out of the box. 
They may be moved and renamed freely as long as the original virtual environment stays in place. 
You can also modify the path to these scripts upon installation with the `--script-dir path/to/scripts/` keyword. 
If you want to have multiple installations, e.g., to use UMA as well as AIMNet2, 
be careful to provide different script paths to avoid overwriting.

The minimum Python version is currently 3.11.

### Testing
To test your installation, you can use the tests provided in the `tests` directory.
There are different subdirectories depending on which interface you want to test.
For testing, please activate the respective virtual `oet` environment that was installed with the `install.py` script, e. g., `source .venv/bin/activate`.
Afterward, execute the `test_<interface>.py` script in the respective `tests` subdirectories.
If you installed the scripts to a different directory, set the path to the script you want to test at the beginning of the `test_<interface>.py` file.

## Usage

### ORCA 5
A link named `otool_external` must be created in the ORCA executables directory, 
which points to the chosen script.
Optional arguments are not supported, so additional wrappers or hard-coded modifications may be necessary.

### ORCA 6
In addition to the `otool_external` route which is backwards-compatible,
it is also possible to set the full path to the chosen script via the environment variable `EXTOPTEXE`,
or via the ORCA input:
``` 
%method
  ProgExt "/full/path/to/script"
  Ext_Params "optional command line arguments"
end
```

### Server
For MLIPs like AIMNet2 and UMA, we recommend to use a server/client combination, 
as the calculations will otherwise take significantly longer,
due to heavy imports of dependencies like `torch`. 
Therefore, start a calculation server with the `oet_server` script, e.g., `oet_server aimnet2`. 
It will handle the single-point and gradient calculations and can remain active for multiple ORCA runs. 
The number of cores it is allowed to use can be specified with `-n <integer>`. 
In your ORCA input, you then have to specify the `oet_client` as `ProgExt`. 
It will forward all the calculation requests to the server. 
If you want to keep multiple servers running for different types of calculations, 
you have to specify different ports for the server and clients with the `-b <hostname>:<port>` keyword. 
Provide the keyword to the client via the ORCA input line `Ext_Params "-b <hostname>:<port>"`.

### AIMNet2 options

Default model is `aimnet2` (= `aimnet2-wb97m-d3_0`, ωB97M-D3 trained).
Other models are selectable via `-m`. For non-covalent or screening
work, `aimnet2-2025` (B97-3c trained, faster) is a good alternative;
for general thermochemistry and reaction barriers, `aimnet2` remains
the recommended default per upstream guidance.

> **Open-shell users — read this.** The default `aimnet2` (and
> `aimnet2-2025`, `aimnet2-b973c-d3*`, `aimnet2-rxn*`) are
> closed-shell-trained. Passing `mult ≠ 1` (e.g. an ORCA input with
> `* xyzfile 0 2 OH.xyz`) is **silently accepted** and produces a
> spin-restricted energy, not a true UKS-equivalent open-shell value.
> For genuine open-shell or charged-species energetics, use
> `-m aimnet2-nse`.
>
> No AIMNet2 family does broken-symmetry, so a closed-shell-singlet
> biradical (`mult = 1` with two unpaired electrons coupled
> antiferromagnetically — carbenes, nitrenes, stretched singlet σ
> bonds) is not modelable correctly by **any** model in the suite,
> NSE included. Treat such systems with a multireference method.

#### Choosing a model

**WARNING — energies from different model families are NOT comparable.**
Different families were trained on different reference data, with
different functionals, with or without solvation. Mixing models within
a single workflow (e.g. optimize with `aimnet2-rxn`, compute SP energies
with `aimnet2-2025`) produces wrong reaction energies of order tens of
kcal/mol — comparable to or larger than typical reaction barriers, with
no failure indication. The aimnet runtime emits a one-time
`UserWarning` when two families are constructed in the same Python
process; this is informational, not a bug.

Recommended pattern: TS search in `aimnet2-nse` → SP energies in
`aimnet2-nse`. Never optimize in one family and compute SP in another.
The exception is `aimnet2-pd`, which can be used end-to-end for
Pd-catalyzed THF-solvated workflows but must not be mixed with any
other family at all.

| Workflow | Recommended model | Notes |
|----------|-------------------|-------|
| Closed-shell, neutral, equilibrium geometry, gas-phase | `aimnet2` | ωB97M-D3 trained |
| Closed-shell, neutral, non-covalent / screening / large systems | `aimnet2-2025` | B97-3c trained; faster but barriers typically 3-5 kcal/mol off vs the ωB97M-D3 default. Fine for relative ranking and screening, not absolute kinetics. |
| Open-shell or charged | `aimnet2-nse` | NSE: handles arbitrary charge / multiplicity. 14-element coverage (H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I). Covers single-reference DFT regimes. NOT reliable for biradicals (two unpaired electrons on different atoms with weak coupling), near-degenerate spin states, or stretched singlet bonds (e.g. homolytic dissociation on a singlet surface). |
| Reactive (TS, NEB, IRC), broad element set | `aimnet2-nse` | Same caveats as above; covers reactive open-shell trajectories. |
| Reactive, neutral H/C/N/O only, faster | `aimnet2-rxn` | NEUTRAL only (charge ≠ 0 raises `ValueError`). H/C/N/O only (other elements raise `ValueError`). Coulomb cutoff locked at 4.6 Å (other values fire an upstream `UserWarning`). |
| Pd-containing (catalysis, organometallic) | `aimnet2-pd` | B97-3c with **implicit CPCM/THF solvation BAKED IN** — energies are NOT gas-phase. Replaces As with Pd vs default; AsR3 ligands (arsine ligands on Pd, e.g. Pd(AsPh3)4-style complexes) are not supported. |

#### Reproducibility

Aliases (`aimnet2`, `aimnet2-2025`, …) may be repointed in future
aimnet releases. For bit-stable reproducibility across versions, pin
to canonical keys: `-m aimnet2-wb97m-d3_0` rather than `-m aimnet2`.

#### Performance flags

| Flag | Default | Effect |
|------|---------|--------|
| `--compile` | False | torch.compile JIT. **Server mode only** — standalone re-pays JIT every ORCA call. First-call latency 10–60s. Recompiles on shape change (catastrophic in NEB). Incompatible with Hessian. Do not use with NEB / OptTS / IRC. For server mode, set `TORCHINDUCTOR_CACHE_DIR=/persistent/path` to keep the warm cache across worker restarts. |
| `--nb-threshold N` | 120 | Adaptive neighbor-list batch size. |
| `--ensemble-member {0,1,2,3}` | 0 | Use a single ensemble member (default 0). OET runs ONE model per call. To get an ensemble mean for production accuracy, run with each `--ensemble-member` value (0,1,2,3) and average outside ORCA. Upstream recommends ensemble averaging for production calculations. |

#### Numerical precision

AIMNet2 internally evaluates in float32. For large systems (energy magnitudes
above ~1000 eV), absolute energies are precise to ~1e-4 eV ≈ 4e-6 Eh. ORCA's
default `TolE=5e-6 Eh` is at this noise floor; if optimization stalls on
`Energy convergence not reached`, loosen with `! TightOpt` → `%scf TolE 1e-5 end`
or accept slightly looser convergence.

#### GPU server deployment

Each `oet_server aimnet2 -d cuda` worker holds the model resident on GPU
until evicted. `torch.cuda.empty_cache()` returns memory to PyTorch's allocator,
not to the OS — `nvidia-smi` shows steady-state usage equal to the sum of
resident workers' models. Plan worker count by VRAM/model-size, not per-call
peak. For multi-worker GPU deployments, set `CUDA_VISIBLE_DEVICES` per worker
or use `--workers 1`. (Server-side worker-cache eviction with `release()` is
in a sibling PR.)

> **`aimnet2-rxn` family CAUTION**: the Coulomb cutoff is locked at 4.6 Å
> in training. Always pass `--coulomb-cutoff 4.6` when using `-m aimnet2-rxn*`,
> otherwise the upstream `UserWarning` fires and your electrostatics are
> physically suspect.

#### Long-range Coulomb flags

| Flag | Default | Effect |
|------|---------|--------|
| `--coulomb {auto,on,off}` | `auto` | Force on/off the long-range Coulomb module. `auto` defers to model. |
| `--coulomb-method {simple,dsf,ewald}` | unset | Override the model's method. |
| `--coulomb-cutoff R` | 15.0 | Cutoff in Å (used by dsf/ewald). Rejected without `--coulomb-method`. **For `aimnet2-rxn` family, pass `--coulomb-cutoff 4.6`** (training-frozen value). |

#### Dispersion (DFT-D3) flags

| Flag | Default | Effect |
|------|---------|--------|
| `--dispersion {auto,on,off}` | `auto` | Force on/off the D3 module. |
| `--dftd3-cutoff R` | unset | Override D3 cutoff in Å. |
| `--dftd3-smoothing-fraction f` | unset | Override D3 smoothing fraction. |

#### Examples

CLI smoke (default model, water gradient):

```bash
oet_aimnet2 input.extinp.tmp
```

Condensed phase, DSF Coulomb at 12 Å:

```bash
oet_aimnet2 input.extinp.tmp --coulomb-method dsf --coulomb-cutoff 12.0
```

ORCA OptTS with the NSE model on a charged radical (no `--compile`):

```text
! OptTS
%method
  ProgExt "/path/to/oet_aimnet2"
  Ext_Params "-m aimnet2-nse -d cuda"
end
* xyzfile -1 2 ts_radical.xyz
```

ORCA NEB-CI with the RXN model (neutral H/C/N/O only; no `--compile`,
`--coulomb-cutoff 4.6`):

```text
! NEB-CI
%neb Product "product.xyz" NImages 8 end
%method
  ProgExt "/path/to/oet_aimnet2"
  Ext_Params "-m aimnet2-rxn --coulomb-method dsf --coulomb-cutoff 4.6"
end
* xyzfile 0 1 reactant.xyz
```

## Interface

All scripts must be executable as:
```
scriptname <basename_EXT.extinp.tmp> [args]
```
where `basename_EXT.extinp.tmp` is the name of an input file generated 
by ORCA (see below) and `args` are optional command line arguments.
The latter can be provided in the ORCA input file (starting with ORCA 6) 
and are directly passed to the external script.

### Input syntax
The `extinp` file has the following format:
```
basename_EXT.xyz # xyz filename: string, ending in '.xyz'
0 # charge: integer
1 # multiplicity: positive integer
1 # NCores: positive integer
0 # do gradient: 0 or 1
pointcharges.pc # point charge filename: string (optional)
```
Comments from `#` until the end of the line should be ignored.

The file `basename_EXT.xyz` will also be present in the working directory with standard XYZ format:
```
<NAtoms>
comment line
<Element> <X> <Y> <Z>
...
```

### Output syntax
The script must generate a file called `basename_EXT.engrad` using the same `basename` as the XYZ file. 
This file must have the following format:
```
#
# Number of atoms: must match the XYZ
#
3
#
# The current total energy in Eh
#
-5.504066223730
#
# The current gradient in Eh/bohr: Atom1X, Atom1Y, Atom1Z, Atom2X, etc.
#
-0.000123241583
0.000000000160
-0.000000000160
0.000215247283
-0.000000001861
0.000000001861
-0.000092005700
0.000000001701
-0.000000001701
```
In ORCA 5, exactly 3 comment lines must be present between entries (as above).
In ORCA 6, comments from `#` until the end of the line are ignored, 
as are the (now optional) comment-only lines.

The script may also print relevant output to STDOUT and/or STDERR. 
STDOUT will either be printed in the ORCA standard output, 
or redirected to a temporary file and removed afterwards,
depending on the type of job and ORCA output settings.

# License
## Open Source License
This open source project is released publicly under the following open source license: `GPL-3.0`. 
This license governs all public releases of the code and allows anyone to use, modify, 
and distribute the project freely, in accordance with its terms.
## Proprietary License
The program, including all contributions, may also be included in our proprietary software products under a commercial license. 
This enables us to:
- Combine open source and closed source components into a single product,
- Offer the project under alternative licensing terms to customers with specific commercial needs,
- Ensure open source compliance for all public parts, while simplifying license obligations in private or embedded distributions.

## Contributor License Agreement (CLA)
To maintain this licensing model, all contributors must sign our Contributor License Agreement (CLA). 
This CLA is an adapted industry-standard CLA (Apache CLA) with minor modifications. 
By signing the CLA, you
- Retain ownership of your contributions,
- Grant us a non-exclusive license to use, sublicense, relicense and distribute your contributions 
  under both open source and proprietary terms.

## We use a two-part CLA system:
- [Individual CLA (ICLA) for personal contributions](CLA.md),
- Corporate CLA (CCLA) for contributions made on behalf of an employer (available upon request to info@faccts.de).

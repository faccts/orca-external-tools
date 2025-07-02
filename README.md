# orca-external-tools

This repository contains wrapper scripts compatible with the `otool_external` interface in ORCA.
The scripts call an external program, which computes the energy and gradient of a system, 
then pass this information back to ORCA for use in optimization, NEB, GOAT, MD, etc.

## Installation

The scripts should be put in some directory and made executable.
The interpreter required for each script (e.g. `python`) should be available the user environment.

## Usage

### ORCA 5
A link named `otool_external` must be created in the ORCA executables directory, 
which points to the chosen script.
Optional arguments are not supported, so additional wrappers or hard-coded modifications may be necessary.

### ORCA 6
In addition to the `otool_external` route which is backwards-compatible,
it is also possible to set the full path to the chosen scipt via the environment variable `EXTOPTEXE`,
or via the ORCA input:
``` 
%method
  ProgExt "/full/path/to/script"
  Ext_Params "optional command line arguments"
end
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
This open source project is released publicly under the following open source license: `GPL-3.0`. This license governs all public releases of the code and allows anyone to use, modify, and distribute the project freely, in accordance with its terms.
## Proprietary License
The program, including all contributions, may also be included in our proprietary software products under a commercial license. This enables us to:
- Combine open source and closed source components into a single product,
- Offer the project under alternative licensing terms to customers with specific commercial needs,
- Ensure open source compliance for all public parts, while simplifying license obligations in private or embedded distributions.

## Contributor License Agreement (CLA)
To maintain this licensing model, all contributors must sign our Contributor License Agreement (CLA). This CLA is an adapted industry-standard CLA (Apache CLA) with minor modifications. By signing the CLA, you
- Retain ownership of your contributions,
- Grant us a non-exclusive license to use, sublicense, relicense and distribute your contributions under both open source and proprietary terms.

## We use a two-part CLA system:
- [Individual CLA (ICLA) for personal contributions](CLA.md),
- Corporate CLA (CCLA) for contributions made on behalf of an employer (available upon request to info@faccts.de).

# UMA ExtTool

This package provides [fairchem](https://github.com/facebookresearch/fairchem) wrappers for ORCA's ExtTool interface.
Before starting to use this module, please make sure your have access to the [fairchem repository](https://huggingface.co/facebook/UMA) and logged in with your
huggingface account. For details, please see the GitHub repository or the [respective tutorials](https://fair-chem.github.io/).
There are two alternative workflows.

## Standalone mode
The script `uma.sh` can be used directly as a standalone "external tool".
```
usage: uma.sh [-h] [-m MODEL] [-d MODEL_DIR] inputfile

positional arguments:
  inputfile             ORCA-generated input file.

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The UMA model. Default: "omol".
```

However, importing the `torch` package takes a significant amount of time, 
possibly longer than the actual calculation for small systems.
In such situations the server-client mode is faster.

## Server-client mode
This mode is based on a server that is started once and runs locally in the background.
Therefore, please use the script `umaserver.sh` which is also used to define the respective UMA model.
```
usage: umaserver [-h] [-m MODEL] [-b hostname:port]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The UMA model. Default: "omol".
  -b hostname:port, --bind hostname:port
                        Server bind address and port. Default: 127.0.0.1:8888.
```
Remember to close it once all the required calculations are done.
It listens for inputs from the client script and sends the results in response.
This client script `umaclient.sh` can be used as the "external tool" and its path can be defined via the ORCA input file.

```
usage: umaclient [-h] [-b hostname:port] inputfile

ORCA "external tool" interface for UMA calculations (client mode)

positional arguments:
  inputfile             ORCA-generated input file.

options:
  -h, --help            show this help message and exit
  -b hostname:port, --bind hostname:port
                        Server bind address and port. Default: 127.0.0.1:8888.
```

## Installation
If not detected, fairchem will be installed to a `.venv` directory by the wrapper scripts the first time you are using them.
Depending on your download speed, this might take a while.
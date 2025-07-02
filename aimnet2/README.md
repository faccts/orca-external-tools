# AIMNet2 ExtTool

This package provides [AIMNet2](https://github.com/isayevlab/AIMNet2) wrappers for ORCA's ExtTool interface.
There are two alternative workflows.

## Standalone mode
The script `aimnet2exttool` can be used directly as a standalone "external tool".
```
usage: aimnet2exttool [-h] [-m MODEL] [-d MODEL_DIR] inputfile

ORCA "external tool" interface for AIMNet2 calculations (standalone mode)

positional arguments:
  inputfile             ORCA-generated input file.

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The AIMNet2 model file name (must be in MODEL_DIR) or absolute path. Default: "aimnet2_wb97m_0.jpt".
  -d MODEL_DIR, --model-dir MODEL_DIR
                        The directory to look for AIMNet2 model files. Default: "orca-external-tools/aimnet2/src/aimnet2exttool/models".
```


However, importing the `torch` package takes a significant amount of time, 
possibly longer than the actual calculation time for small systems.
In such situations the server-client mode is faster.

## Server-client mode
The script `aimnet2server` starts a calculation server with the requested AIMNet2 model.
```
usage: aimnet2server [-h] [-m MODEL] [-d MODEL_DIR] [-b hostname:port]

ORCA "external tool" interface for AIMNet2 calculations (server mode)

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The AIMNet2 model file name (must be in MODEL_DIR) or absolute path. Default: "aimnet2_wb97m_0.jpt".
  -d MODEL_DIR, --model-dir MODEL_DIR
                        The directory to look for AIMNet2 model files. Default: "orca-external-tools/aimnet2/src/aimnet2exttool/models".
  -b hostname:port, --bind hostname:port
                        Server bind address and port. Default: 127.0.0.1:8888.
```
It then listens for inputs from the client script and sends the results in response.
The server must be started before any ORCA calculations and manually terminated after the calculations are done!

The `aimnet2client` script acts as the ORCA "external tool".
```
usage: aimnet2client [-h] [-b hostname:port] inputfile

ORCA "external tool" interface for AIMNet2 calculations (client mode)

positional arguments:
  inputfile             ORCA-generated input file.

options:
  -h, --help            show this help message and exit
  -b hostname:port, --bind hostname:port
                        Server bind address and port. Default: 127.0.0.1:8888.
```

## Installation
Run the `install.sh` script to initialize a virtual environment, download all dependencies, and install the scripts.
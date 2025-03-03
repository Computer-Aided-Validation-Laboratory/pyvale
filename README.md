# pyvale

A python validation engine (`pyvale`). Used to simulate experimental data from an input multi-physics simulation by explicitly modelling sensors with realistic uncertainties. Useful for experimental design, sensor placement optimisation, testing simulation validation metrics and testing digital shadows/twins.

## Installation: Ubuntu
### Managing Python Versions

To be compatible with `bpy` (the Blender python interface), `pyvale` uses python 3.11. To install python 3.11 without corrupting your operating systems python installation first add the deadsnakes repository to apt:
```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update && sudo apt upgrade -y
```

Install python 3.11:
```shell
sudo apt install python3.11
```

Add `venv` to your python 3.11 install:
```shell
sudo apt install python3.11-venv
```

Check your python 3.11 install is working using the following command which should open an interactive python interpreter:
```shell
python3.11
```

### Virtual Environment

We recommend installing `pyvale` in a virtual environment using `venv` or `pyvale` can be installed into an existing environment of your choice. To create a specific virtual environment for `pyvale` navigate to the directory you want to install the environment and use:

```shell
python3.11 -m venv .pyvale-env
source .pyvale-env/bin/activate
```

### Standard & Developer Installation

Clone `pyvale` to your local system and `cd` to the root directory of `pyvale`. Ensure you virtual environment is activated and run the following commmand from the `pyvale` directory:

```
pip install .
```

To create an editable/developer installation of `pyvale` follow the instructions for a standard installation but run:

```
pip install -e .
```

### Mooseherder
`pyvale` requires `mooseherder` to be able to load exodus output files from `moose` finite element simulations. Go to the `mooseherder` github page (https://github.com/Digital-Validation-Laboratory/mooseherder) and install `mooseherder` into the same virtual environment as `pyvale`.

### MOOSE
`pyvale` come pre-packaged with example `moose` physics simulation outputs to demonstrate its functionality. If you need to run additional simulation cases we recommend `proteus` (https://github.com/aurora-multiphysics/proteus) which has build scripts of common linux distributions.

## Getting Started
The examples folder in "src/pyvale/examples" includes a sequence of examples of increasing complexity that demonstrate the functionality of `pyvale`.

## Contributors
The Computer Aided Validation Team at UKAEA:
- Lloyd Fletcher, UK Atomic Energy Authority
- Adel Tayeb, UK Atomic Energy Authority
- Alex Marsh, UK Atomic Energy Authority
- Rory Spencer, UK Atomic Energy Authority
- Michael Atkinson, UK Atomic Energy Authority
- Lorna Sibson, UK Atomic Energy Authority
- John Charlton, UK Atomic Energy Authority
- Joel Hirst, UK Atomic Energy Authority



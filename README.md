# pyvale
The python validation engine (`pyvale`): An all-in-one package for sensor simulation, sensor uncertainty quantification, sensor placement optimisation and simulation calibration/validation.​ Used to simulate experimental data from an input multi-physics simulation by explicitly modelling sensors with realistic uncertainties. Useful for experimental design, sensor placement optimisation, testing simulation validation metrics and virtually testing digital shadows/twins.

## Quick Demo: Simulating Point Sensors
Here we demonstrate how `pyvale` can be used to simulate thermocouples and strain gauges applied to a [MOOSE](https://mooseframework.inl.gov/index.html) thermo-mechanical simulation of a fusion divertor armour heatsink. The figures below show visualisations of the virtual thermocouple and strain gauge locations on the simualtion mesh as well as time traces for each sensor over a series of simulated experiments.

The code to run the simulated experiments and produce the output shown here comes from [this example](https://computer-aided-validation-laboratory.github.io/pyvale/examples/point/ex4_2.html). You can find more examples and details of `pyvale` python API in the `pyvale` [documentation](https://computer-aided-validation-laboratory.github.io/pyvale/index.html).

|![fig_thermomech3d_tc_vis](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/thermomech3d_tc_vis.svg)|![fig_thermomech3d_sg_vis](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/thermomech3d_sg_vis.svg)|
|--|--|
|*Visualisation of the thermcouple locations.*|*Visualisation of the strain gauge locations.*|

|![fig_thermomech3d_tc_traces](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/thermomech3d_tc_traces.png)|![fig_thermomech3d_sg_traces](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/thermomech3d_sg_traces.png)|
|--|--|
|*Thermocouples time traces over a series of simulated experiments.*|*Strain gauge time traces over a series of simulated experiments.*|


## Quick Install
`pyvale` can be installed from pypi:
```shell
pip install pyvale
```

## Detailed Install: Ubuntu
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

### Standard Installation
`pyvale` can be installed from pypi. Ensure you virtual environment is activated and run:
```shell
pip install pyvale
```

### Developer Installation

Clone `pyvale` to your local system along with submodules using
```shell
git clone --recurse-submodules git@github.com:Computer-Aided-Validation-Laboratory/pyvale.git
```

`cd` to the root directory of `pyvale`. Ensure you virtual environment is activated and run the following commmand from the `pyvale` directory:
```shell
pip install -e .
pip install -e ./dependencies/mooseherder
```

### Running Physics Simulations with MOOSE
`pyvale` come pre-packaged with example `moose` physics simulation outputs (as *.e exodus files) to demonstrate its functionality. If you need to run additional simulation cases we recommend `proteus` (https://github.com/aurora-multiphysics/proteus) which has build scripts for common linux distributions.

## Contributors
The Computer Aided Validation Team at UKAEA:
- Lloyd Fletcher ([ScepticalRabbit](https://github.com/ScepticalRabbit)), UK Atomic Energy Authority
- Joel Hirst ([JoelPhys](https://github.com/JoelPhys)), UK Atomic Energy Authority
- John Charlton ([coolmule0](https://github.com/coolmule0)), UK Atomic Energy Authority
- Lorna Sibson ([lornasibson](https://github.com/lornasibson)), UK Atomic Energy Authority
- Megan Sampson ([meganasampson](https://github.com/meganasampson)), UK Atomic Energy Authority
- Michael Atkinson ([mikesmic](https://github.com/mikesmic)), UK Atomic Energy Authority
- Adel Tayeb ([3adelTayeb](https://github.com/3adelTayeb)), UK Atomic Energy Authority
- Alex Marsh ([alexmarsh2](https://github.com/alexmarsh2)), UK Atomic Energy Authority
- Rory Spencer ([fusmatrs](https://github.com/orgs/Computer-Aided-Validation-Laboratory/people/fusmatrs)), UK Atomic Energy Authority






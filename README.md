# pyvale
![fig_pyvale_logo](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/pyvale_logo.png)

The python validation engine (`pyvale`) is your virtual engineering laboratory: An all-in-one package for sensor uncertainty quantification simulations, experimental design/sensor placement optimisation and simulation calibration/validation. Used to simulate experimental data from an input multi-physics simulation by explicitly modelling sensors with realistic uncertainties. Useful for experimental design, sensor placement optimisation, testing simulation validation metrics and virtually testing digital shadows/twins.

We are actively developing dedicated tools for simulation and uncertainty quantification of imaging sensors including digital image correlation (DIC) and infra-red thermography (IRT). Check out the [documentation](https://computer-aided-validation-laboratory.github.io/pyvale/index.html) to get started with some of our examples.

## Quick Demo: Simulating Point Sensors
Here we demonstrate how `pyvale` can be used to simulate thermocouples and strain gauges applied to a [MOOSE](https://mooseframework.inl.gov/index.html) thermo-mechanical simulation of a fusion divertor armour heatsink. The figures below show visualisations of the virtual thermocouple and strain gauge locations on the simualtion mesh as well as time traces for each sensor over a series of simulated experiments.

The code to run the simulated experiments and produce the output shown here comes from [this example](https://computer-aided-validation-laboratory.github.io/pyvale/examples/basicsensorsim/ex0_quickstart.html). You can find more examples and details of `pyvale` python API in the `pyvale` [documentation](https://computer-aided-validation-laboratory.github.io/pyvale/index.html).

|![fig_thermomech3d_tc_vis](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/thermomech3d_tc_vis.png)|![fig_thermomech3d_sg_vis](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/thermomech3d_sg_vis.png)|
|--|--|
|*Visualisation of the thermocouple locations.*|*Visualisation of the strain gauge locations.*|

|![fig_thermomech3d_tc_traces](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/thermomech3d_tc_traces.png)|![fig_thermomech3d_sg_traces](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/thermomech3d_sg_traces.png)|
|--|--|
|*Thermocouple time traces over a series of simulated experiments.*|*Strain gauge time traces over a series of simulated experiments.*|


## Quick Install
`pyvale` can be installed from pypi:
```shell
pip install pyvale
```

We recommend installing `pyvale` into a virtual environment of your choice as `pyvale` requires python 3.11. If you need help setting up your virtual environment and installing `pyvale` head over to the [installation guide](https://computer-aided-validation-laboratory.github.io/pyvale/install/install.html) in our docs.

## Contributors
The Computer Aided Validation Team at UKAEA:
- Lloyd Fletcher ([ScepticalRabbit](https://github.com/ScepticalRabbit)), UK Atomic Energy Authority
- Joel Hirst ([JoelPhys](https://github.com/JoelPhys)), UK Atomic Energy Authority
- Lorna Sibson ([lornasibson](https://github.com/lornasibson)), UK Atomic Energy Authority
- Megan Sampson ([meganasampson](https://github.com/meganasampson)), UK Atomic Energy Authority
- Wiera Bielajewa ([WieraB](https://github.com/WieraB)), UK Atomic Energy Authority
- Chris Dawson ([ctdaws](https://github.com/ctdaws)), UK Atomic Energy Authority
- Michael Darcy ([AnalogArnold](https://github.com/AnalogArnold)), Swansea University
- Rob Hamill ([rob-hamill](https://github.com/rob-hamill)), UK Atomic Energy Authority
- Michael Atkinson ([mikesmic](https://github.com/mikesmic)), UK Atomic Energy Authority
- Adel Tayeb ([3adelTayeb](https://github.com/3adelTayeb)), UK Atomic Energy Authority
- Alex Marsh ([alexmarsh2](https://github.com/alexmarsh2)), UK Atomic Energy Authority
- Rory Spencer ([fusmatrs](https://github.com/orgs/Computer-Aided-Validation-Laboratory/people/fusmatrs)), UK Atomic Energy Authority
- John Charlton ([coolmule0](https://github.com/coolmule0)), UK Atomic Energy Authority






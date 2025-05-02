# `pyvale` Visualisation Module: Design Specification

## NOTES:

- What is done?


- What is not done?


Principles:
- Formatting for print quality: 300dpi, vector graphics where possible, suitable for single or two column journal papers


## Motivation

The cost of performing large-scale validation tests on a complex components such as a breeder blankets will be on the order of £M's. Therefore, significant cost and risk reduction can be achieved by maximising the information obtained from an optimised set of targeted experiments. A key parameter of validation experiments is the deployment of sensor arrays to measure the components response. There are currently no commercial tools available that can simulate and optimise the placement of diverse arrays of sensors for multi-physics conditions with realistic constraints (e.g., cost, reliability, and accuracy).

To address this we are developing the `pyvale` python package which is intended to be an all-in-one package for sensor simulation, sensor uncertainty quantication, sensor placement optimisation and simulation calibration/validation. For all functionality of `pyvale` visualisation tools are key to allow users to setup their sensor simulations and interpret the results of their analysis. A key application of `pyvale` is the simulation of imaging sensors such as infra-red thermography and digital image correlation. Imaging sensors produce visual output and `pyvale` requires a set of tools that allow users to visualise the output of these sensors...

## Aims & Objectives
The aim of this project is to develop the visualisation toolbox for `pyvale` that will allow users to visualise the setup of sensor simulations; the output from sensor simulations; and the output of further analysis such as sensor placement optimisation and the calculation of validation metrics. The objectives of this project are to develop a visualisation module for `pyvale` that supports:

- Plotting of time traces for point sensors and extracted data from camera sensors
- Visualisation of point and camera sensor locations including orientation of sensors for vector and tensor fields
- TODO
- Visualisation of digital image correlation data including raw images, extracted displacement fields and post-processed data such as strain fields
- Producing print quality graphics suitable for journal publications


## Summary of Visualisation Tools
This functionality exists in `pyvale` for visualisation:
- Visualisation of point sensors on single mesh, including perturbed sensor locations
- Visualisation of point sensor traces for a single physics and experiment

TODO
These are new visualisation features to be developed:
- Visualisation of multiple sensor types on a single mesh
- Visualisation of sensor area + integration points
- Visualisation of sensor angles for vector ad tensor field
- Subplots for traces of multi-physics point sensors
- Animation/video for trace/mesh vis
- Animation/video of camera image stacks
- Extract point trace for pixels in an image
- Extract line plot for pixels in an image
- Extract area average for image data
- Scene visualisation for camera rendering

## Sub-Module: VisTimeTraces
This sub-module will print time traces of physical variables for point sensors or extracted groups of pixel data from camera sensors.

This sub-module will use `matplotlib` for plotting sensor traces.

### Inputs
- A list of `SensorArray` objects that have been used
### Workflow
- TODO
### Outputs
- Interactive display of a plot or subplots of sensor traces




## Sub-Module: VisExpTraces
This sub-module will print time traces of physical variables for point sensors or extracted groups of pixel data from camera sensors.

This sub-module will use `matplotlib` for plotting sensor traces.

### Inputs
- A list of `SensorArray` objects that have been used
### Workflow
- TODO
### Outputs
- Interactive display of a plot or subplots of sensor traces




## Sub-Module: VisAnimateTraces
This sub-module will print time traces of physical variables for point sensors or extracted groups of pixel data from camera sensors.

This sub-module will use `matplotlib` for plotting sensor traces.

### Inputs
- A list of `SensorArray` objects that have been used
### Workflow
- TODO
### Outputs
- Interactive display of a plot or subplots of sensor traces




## Sub-Module: VisSimSensors
This sub-module will

This sub-module will utilise `pyvista` for visualising the simulation fields as well as the sensor parameters.

### Inputs
- A list of `SensorArray` objects that have been used
### Workflow
- TODO
### Outputs
- TODO




## Sub-Module: VisAnimateSim
This sub-module will

This sub-module will utilise `pyvista` for visualising the simulation fields as well as the sensor parameters.

### Inputs
- A list of `SensorArray` objects that have been used
### Workflow
- TODO
### Outputs
- TODO




## Sub-Module: VisRenderScene
This sub-module will

This sub-module will utilise `pyvista` for visualising the simulation fields as well as the sensor parameters.

### Inputs
- A `RenderScene` object containing a list of cameras, meshes, lights and any other objects to be displayed
### Workflow
- TODO
### Outputs
- An



## Sub-Module: VisDIC
This sub-module will

This sub-module will utilise a combination of `matplotlib` and `pyvista` for visualising DIC data.

### Inputs
- TODO
### Workflow
- TODO
### Outputs
- TODO



## Deliverables
- A visualisation module fully integrated and merged into the main branch of `pyvale` with the following sub-modules (note these are just suggested names and are not binding, use whatever structure makes most sense during developement to achieve the desired functionality):
    - VisTimeTraces
    - VisExpTraces
    - VisAnimateTraces
    - VisSimSensors
    - VisRenderScene
    - VisDIC
- Full doc-strings and auto generated documentation for all modules and sub-modules
- A pragmatic suite of software tests including unit and regression tests for all modules and sub-modules
- Example/tutorial scripts demonstrating the functionality of the visualisation module



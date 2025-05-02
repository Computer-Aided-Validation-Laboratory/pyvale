# `pyvale` Visualisation Module: Design Specification

## NOTES:

- What is done?
    - Visualisation of point sensors on single mesh, including perturbed sensor locations
    - Visualisation of point sensor traces for a single physics and experiment

- What is not done?
    - Visualisation of sensor area + integration points
    - Visualisation of sensor angles
    - Subplots for traces of multi-physics point sensors
    - Animation/video for trace/mesh vis
    - Animation/video of camera image stacks
    - Extract point trace for pixels in an image
    - Extract line plot for pixels in an image
    - Extract area average for image data


TODO:
- Subplots for multiple sensors and multi-physics cases (e.g. temperature and strain)
- Showing multiple sensors (point and camera) on a single mesh visualisation

- POINT SENSORS:
    - Visualisation of sensor location and orientation for vector/tensor fields on the mesh using pyvista
    - Visualisation of point sensor traces using matplotlib

- Graphing capabilities for point sensor traces
- Animation module for output videos (various formats) of:
    - Simulated point sensor traces
    - Imaging sensor output as videos
- 2D visualisation tools for images using matplotlib
- 3D visualisation for stereo DIC data
- Interactive:
    - 3D scene visualisation using pyvista showing: point sensors, multiple cameras, multiple meshes and camera view frustrums as specified by user options.
- Need to support vector graphics (*.svg) where possible
- Default formatting options for print quality (300dpi, figures sized to fit one or two columns of a journal article)


## Motivation

The cost of performing large-scale validation tests on a complex components such as a breeder blankets will be on the order of £M's. Therefore, significant cost and risk reduction can be achieved by maximising the information obtained from an optimised set of targeted experiments. A key parameter of validation experiments is the deployment of sensor arrays to measure the components response. There are currently no commercial tools available that can simulate and optimise the placement of diverse arrays of sensors for multi-physics conditions with realistic constraints (e.g., cost, reliability, and accuracy).

To address this we are developing the `pyvale` python package which is intended to be an all-in-one package for sensor simulation, sensor uncertainty quantication, sensor placement optimisation and simulation calibration/validation. For all functionality of `pyvale` visualisation tools are key to allow users to setup their sensor simulations and interpret the results of their analysis. A key application of `pyvale` is the simulation of imaging sensors such as infra-red thermography and digital image correlation. Imaging sensors produce visual output and `pyvale` requires a set of tools that allow users to visualise the output of these sensors...

## Aims & Objectives
The aim of this project is to develop the visualisation toolbox for `pyvale` that will allow users to visualise the setup of sensor simulations; the output from sensor simulations; and the output of further analysis such as sensor placement optimisation and the calculation of validation metrics. The objectives of this project are to develop a visualisation module for `pyvale` that supports:

- TODO

## Sub-Module: TODO
This sub-module will TODO

### Inputs
- TODO
### Workflow
- TODO
### Outputs
- TODO

## Deliverables
- A visualisation module fully integrated and merged into the main branch of `pyvale` with the following sub-modules (note these are just suggested names and are not binding, use whatever structure makes most sense during developement to achieve the desired functionality):
    - TODO
- Full doc-strings and auto generated documentation for all modules and sub-modules
- A pragmatic suite of software tests including unit and regression tests for all modules and sub-modules
- Example/tutorial scripts demonstrating the functionality of the visualisation module



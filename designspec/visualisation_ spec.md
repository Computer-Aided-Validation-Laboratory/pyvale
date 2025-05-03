# `pyvale` Visualisation Module: Design Specification

## NOTES:

- What is done?


- What is not done?


Principles:


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


## Overview Visualisation Tool Requirements
This functionality already exists in `pyvale` for visualisation:
- Visualisation of point sensors on single mesh, including perturbed sensor locations
- Visualisation of point sensor traces for a single physics and experiment

These are new visualisation features to be developed:
- Visualisation of multiple sensor types on a single mesh
- Visualisation of sensor area and integration points
- Visualisation of sensor angles for vector ad tensor fields
- Subplots for traces of multi-physics point sensors
- Animation/video for simultaneous time traces and mesh sensor visualisation
- Animation/video of camera image stacks
- Extract and plot point trace for pixels in an image
- Extract and produce line plots for pixels in an image
- Extract and plot time trace of an area average for image data
- Scene visualisation for camera rendering including: visualisation of any meshes in the scene; any cameras in the scene as well as their orientation and view frustrum.
- Display digital image correlation data (e.g. displacement fiels, strain fields, correlation criterion, ) including: tranparent overlays on the raw images

All visualisation modules should support:
- It is the users responsibility to display the figure in interactive mode (i.e. calling `.show()` for `matplotlib` and `pyvista`) or to save the figure using a method provided by the `pyvale` visualisation toolbox. Therefore all plot functions should return a handle to the created figure to allow the user to show and/or save the figure.
- Formatting for print quality: 300dp for raster formats, vector graphics where possible (*.svg), suitable for single or two column journal papers.


## Sub-Module: VisTraces
This sub-module plots time or other user defined traces (e.g. force/displacement) of physical variables for point sensors or extracted groups of pixel data from camera sensors. This sub-module uses `matplotlib` for plotting sensor traces.

### Inputs
- An options dataclass (with set defaults) that includes general visualisation parameters (e.g. fonts, figure sizes/resolution etc.)
- An options dataclass (with set defaults) that controls specific parameters for plotting sensor time traces (e.g. line styles and colours, axis labels, legend parameters etc.)
- A list of `SensorArray` objects to be plotted
- Data to configure subplots for multiple sensor arrays applied to multi-physics cases (e.g side by side plots of thermocouples and strain gauges)

### Workflow
- Define and configure the figure canvas and subplots using the general or user specified parameters, defaults should size figures for single column journal articles
- Default the horizontal axis to time unless user specified (for example the user might want to plot a load/displacement curve from a tensile test)
- Label axes using the `SensorDescriptor` information from the `SensorArray`
- Loop over the sensors in the array and plot all traces OR use the user specified sensor numbers
- For each trace plot the truth from the simulation as a solid line and the simulated sensor values as dashed lines (line styles should also be user configurable using the appropriate dataclass)
- For large sensor arrays (more than ~10 or whatever looks best) automatically split the sensors into subplots so the traces are clear
- For extremely large sensor arrays (~1000's) plot every n'th sensor and warn the user
- Create a legend with the sensor tag for each trace
- Configure multiple subplots assigning different sensor arrays and sensor numbers to each subplot.

### Outputs
- Return figure and axis handles to the user to allow for additional user defined configuration with `matplotlib`
- Interactive display of a plot or subplots of sensor traces if the user calls `.show()` on the returned handle
- Function to save the plot as a vector graphic (.svg) or raster graphic (.png)




## Sub-Module: VisTracesExp
This sub-module plots time traces of physical variables for point sensors or extracted groups of pixel data from camera sensors. This sub-module uses `matplotlib` for plotting sensor traces.

### Inputs
- An options dataclass (with set defaults) that includes general visualisation parameters (e.g. fonts, figure sizes/resolution etc.)
- An options dataclass (with set defaults) that controls specific parameters for plotting sensor time traces (e.g. line styles and colours, axis labels, legend parameters etc.)
- A list of `SensorArray` objects to be plotted
- Data to configure subplots for multiple sensor arrays applied to multi-physics cases (e.g side by side plots of thermocouples and strain gauges)

### Workflow
The workflow for this sub-module is the same as for `VisTraces` above but applied over N Monte-Carlo simulations for each sensor requiring mean sensor traces and shaded uncertainty bounds to be plotted.

### Outputs
NOTE: the outputs are the same as for `VisTraces` above.
- Return figure and axis handles to the user to allow for additional user defined configuration with `matplotlib`
- Interactive display of a plot or subplots of sensor traces if the user calls `.show()` on the returned handle
- Function to save the plot as a vector graphic (.svg) or raster graphic (.png)




## Sub-Module: VisTracesAnimate
This sub-module plots animated time traces of physical variables for point sensors or extracted groups of pixel data from camera sensors. This sub-module uses `matplotlib` for plotting sensor traces.

### Inputs
The same as for the `VisTraces` witht the addition of:
- A dataclass specifying the animation options (e.g. frames per second etc.)

### Workflow
The workflow for this sub-module is the same as for `VisTraces` above but provides an animation highlighting the data point at each time step.

### Outputs
- An image sequence of raster graphics (.jpg or .png)
AND/OR
- A animation/video with configurable quality in at least mp4 and/or gif format




## Sub-Module: VisSimSensors
This sub-module shows the simulation mesh and displays labelled virtual sensor locations to the user. This sub-module utilises `pyvista` for visualising the simulation fields as well as the sensor parameters (location, orientation and sensor area).

### Inputs
- A configuration dataclass that includes the parameters to control the plotting behaviour (e.g. colour bar parameters, subplots and fonts)
- A list of `SensorArray` objects to visualise including: `Field` objects for
### Workflow
- TODO
### Outputs
- TODO




## Sub-Module: VisSimAnimate
This sub-module will

This sub-module will utilise `pyvista` for visualising the simulation fields as well as the sensor parameters.

### Inputs
- A list of `SensorArray` objects that have been used
### Workflow
- TODO
### Outputs
- TODO




## Sub-Module: VisRenderScene
This sub-module utilisse `pyvista` for visualising...

### Inputs
- A `RenderScene` object containing a list of cameras, meshes, lights and any other objects to be displayed
### Workflow
- TODO
### Outputs
- An



## Sub-Module: VisRenderData
This sub-module utilises a combination of `matplotlib` and `pyvista` for visualising renderer camera data (e.g. infra-red camera or digital image correlation data).

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
    - VisRenderData
- Full doc-strings and auto generated documentation for all modules and sub-modules
- A pragmatic suite of software tests including unit and regression tests for all modules and sub-modules
- Example/tutorial scripts demonstrating the functionality of the visualisation module



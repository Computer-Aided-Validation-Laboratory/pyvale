# `pyvale` Visualisation Module: Design Specification

## Motivation

The cost of performing large-scale validation tests on a complex components such as a breeder blankets will be on the order of £M's. Therefore, significant cost and risk reduction can be achieved by maximising the information obtained from an optimised set of targeted experiments. A key parameter of validation experiments is the deployment of sensor arrays to measure the components response. There are currently no commercial tools available that can simulate and optimise the placement of diverse arrays of sensors for multi-physics conditions with realistic constraints (e.g., cost, reliability, and accuracy).

To address this we are developing the `pyvale` python package which is intended to be an all-in-one package for sensor simulation, sensor uncertainty quantication, sensor placement optimisation and simulation calibration/validation. For all functionality of `pyvale` visualisation tools are key to allow users to setup their sensor simulations and interpret the results of their analysis. A key application of `pyvale` is the simulation of imaging sensors such as infra-red thermography and digital image correlation. Imaging sensors produce visual output and `pyvale` requires a set of tools that allow users to visualise the output of these sensors.

## Aims & Objectives
The aim of this project is to develop the visualisation toolbox for `pyvale` that will allow users to visualise the setup of sensor simulations; the output from sensor simulations; and the output of further analysis such as sensor placement optimisation and the calculation of validation metrics. The objectives of this project are to develop a visualisation module for `pyvale` that supports:

- Plotting of time traces for point sensors and extracted data from camera sensors.
- Visualisation of virtual point and camera sensor locations including orientation of sensors for vector and tensor fields to aid users in setting up their sensor simulations.
- Visualisation of virtual camera images such as infra-red or  digital image correlation data including raw images, thermal fields, extracted displacement fields and post-processed data such as strain fields.
- Producing print quality graphics suitable for journal publications.
- Producing animations for presentations and demonstration purposes.

In the future additional visualisation tools will need to be developed to support the sensor placement optimisation and calibration/validation modules in `pyvale`.


## Overview of Visualisation Tools for `pyvale`
A detailed list of suggested sub-modules is given below to implement the functionality listed here. This functionality already exists in `pyvale` for visualisation:

- Visualisation of point sensors on single mesh, including perturbed sensor locations
- Visualisation of point sensor traces for a single physics and experiment
- Basic single physics plotting of uncertainty bounds for Monte-Carlo sensor simulation experiments

These are new visualisation features to be developed on top of the existing visualisation modules:

- Subplots for visualising multiple sensor traces for field components or multi-physics cases
- Subplots for visualising multiple field components or multi-physics cases showing sensor locations on the simulation mesh
- Visualisation of multiple sensor types on a single mesh
- Visualisation of sensor area and integration points
- Visualisation of sensor angles for vector ad tensor fields
- Animations/videos for simultaneous time traces and mesh sensor visualisation
- Animations/videos of camera image stacks
- Extract and plot time traces for pixels in an image
- Extract and produce line plots for pixels in an image
- Extract and plot time trace of an area average for image data
- Scene visualisation for camera rendering including: visualisation of any meshes in the scene; any cameras in the scene as well as their orientation and view frustrum.
- Display synthetic infra-red camera and digital image correlation data (e.g. displacement fiels, strain fields, correlation criterion, ) including: tranparent overlays on the raw images

All visualisation modules should support:
- Giving the user responsibility to display the figure in interactive mode (i.e. calling `.show()` for `matplotlib` and `pyvista`) or to save the figure using a method provided by the `pyvale` visualisation toolbox. Therefore all plot functions should return a handle to the created figure to allow the user to show and/or save the figure or perform additional modifications as necessary.
- Formatting for print quality: 300dp for raster formats, vector graphics where possible (*.svg), suitable for single or two column journal papers.


## Sub-Module: VisTraces
This sub-module plots time or other user defined traces (e.g. force/displacement) of physical variables for point sensors or extracted groups of pixel data from camera sensors. This sub-module uses `matplotlib` for plotting sensor traces.

### Inputs
- A list of `SensorArray` objects to be plotted
- An options dataclass (with set defaults) that controls specific parameters for plotting sensor time traces (e.g. line styles and colours, axis labels, legend parameters etc.)
- Data to configure subplots for multiple sensor arrays applied to multi-physics cases (e.g side by side plots of thermocouples and strain gauges)
- A options dataclass (with set defaults) that includes general visualisation parameters (e.g. fonts, figure sizes/resolution etc.)

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
This sub-module plots time traces of physical variables for point sensors or extracted groups of pixel data from camera sensors over a series of Monte-Carlo experiments. This sub-module uses `matplotlib` for plotting the sensor traces.

### Inputs
- An `ExperimentSimulator` object containing the a list of `SensorArray` objects and an array of Monte-Carlo experiments to plot.
- Data to configure subplots for multiple sensor arrays applied to multi-physics cases (e.g side by side plots of thermocouples and strain gauges). This might need to be combined into the options dataclass or be a separate dataclass.
- An options dataclass (with set defaults) that controls specific parameters for plotting sensor time traces and the fill between lines to show uncertainty bounds (e.g. line styles and colours, axis labels, legend parameters etc.)
- An options dataclass (with set defaults) that includes general visualisation parameters (e.g. fonts, figure sizes/resolution etc.)


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
The same as for the `VisTraces` but the specific options class controls the animation parameters:
- A dataclass specifying the animation options (e.g. frames per second, save file options for image sequences or video files etc.)

### Workflow
The workflow for this sub-module is the same as for `VisTraces` above but provides an animation highlighting the data point at each time step.

This module should also be able to be synchronised with `VisSimAnimate` to show the animation of simulation fields alongside the virtual sensor traces.

### Outputs
- An image sequence of raster graphics (.jpg or .png)
AND/OR
- A animation/video with configurable quality in at least mp4 and/or gif format




## Sub-Module: VisSimSensors
This sub-module shows the simulation mesh with field data as well as displaying labelled virtual sensor locations to the user. This sub-module utilises `pyvista` for visualising the simulation fields as well as the sensor parameters (location, orientation and sensor area).

### Inputs
- A list of `SensorArray` objects to visualise including: references to `SimData` objects containing the simulation fields to display.
- Configuration dataclass for displaying multiple sensor arrays and fields in subplots for example: 1) displaying different components of a vector or tensor field such as strain on displacement, 2) thermal and strain fields with different sensor arrays.
- A configuration dataclass that includes the generic parameters to control the plotting behaviour (e.g. colour bar parameters, subplots and fonts)

### Workflow
NOTE: If the virtual sensors are sampling at time steps that do not match the simulation it will be necessary to interpolate the field data using the `Field` object associated with the `SensorArray`.

- Create the `pyvista` scene/window with the default setup parameters splitting into subplots as necessary
- Add simulation meshes to the scene displaying the relevant physical field
- Add labelled symbols to the scene showing the sensor locations, the sensor spatial dimensions and sensor orientation (for vector/tensor fields)
- Add any required annotations showing: the simulation/sensor time step
- If sensor location/orientation perturbation errors are added then display these to the user in a different colour to the nominal locations

### Outputs
- Return a handle to the `pyvista` plot object so the user can perform additional manipulation on the scene
- If required from the options dataclass then a saved image to disk




## Sub-Module: VisSimAnimate
This sub-module is similar in function to `VisSimSensors` but animates simulation fields to produce image sequences and/or video files.

### Inputs
Inputs for this module are the same as `VisSimSensors` with the addition of:
- A dataclass to control the animation options (e.g. frames per second and output format which could be image sequence or video format)

### Workflow
The workflow for this submodule is the same as for `VisSimSensors` but it will step through a sequence of time steps showing how the fields and sensor parameters change over time.

NOTE: If the virtual sensors are sampling at time steps that do not match the simulation it will be necessary to interpolate the field data using the `Field` object associated with the `SensorArray`.

### Outputs
- An animation as an image sequence of raster graphics (.jpg or .png) or as gif or in a video format (.mp4)




## Sub-Module: VisRenderScene
NOTE: The camera rendering capability of `pyvale` is still under developement so this module will need to be developed in the later half of 2025.

This sub-module utilises `pyvista` for visualising the setup of simulation scenes which will be used to generate virtual camera sensor data.

### Inputs
- A `RenderScene` object containing a list of cameras, meshes, lights and any other objects to be displayed such as point sensor arrays.
- An options dataclass (with set defaults) that includes general visualisation parameters (e.g. fonts, figure sizes/resolution etc.)
- An options dataclass (with set defaults) that controls specific parameters for plotting the scene.
### Workflow
- Create the `pyvista` scene/window with the default setup parameters splitting into subplots as necessary
- Add simulation meshes to the scene displaying the relevant physical field
- Add labelled symbols to the scene showing the camera locations, the camera orientations, the camera label, and the camera view frustrum as a transparency as a user defined option
- Add any required annotations showing: the simulation/sensor time step
- Set the user viewing location such that all meshes, cameras and other objects are visible in the scene.
### Outputs
- A handle to the `pyvista` plot object so the user can call `.show()` or save an image of the scene



## Sub-Module: VisRenderData
NOTE: The camera rendering capability of `pyvale` is still under developement so this module will need to be developed in the later half of 2025.

This sub-module utilises `matplotlib` for visualising rendered camera data (e.g. infra-red camera or digital image correlation data) in 2D and 3D. It might also be necessary to user `pyvista` for visualisation of stereo digital image correlation data in 3D.

### Inputs
- A camera or stereo-camera object that renderer the data to display
- An options dataclass (with set defaults) that controls specific parameters for plotting the image data (e.g. colormap and colorbar settings, axis setting to pixel or length units etc.)
- An options dataclass (with set defaults) that includes general visualisation parameters (e.g. fonts, figure sizes/resolution etc.)


For comparison of camera to input simulation data:
- The `SimData` and/or `Field` object that was used to generate the camera data

### Workflow
- Define and configure the figure canvas and subplots using the general or user specified parameters, defaults should size figures for single column journal articles
- Default the horizontal and vertical axes to pixel coordinates unless specified by the user options. Also, allow the user to suppress the pixel axes labels with the options.
- Add and configure the colorbar for displaying the image data. For greyscale images default to grey. Otherwise use a Red-White-Blue colormap as default for diverging.
- Label the title and colorbar units using the `CameraDescriptor` information for the camera

Additional workflows:
1) Allow the user to produce comparison and difference subplots for example: subplot 1 = camera image data, subplot 2 = simulation data interpolated onto image pixel grid, subplot 3 = different between the camera image data and the simulation data
2) Allow the user to overlay digital image correlation field data (displacement fields, strain fields etc.) as a transparency on top of the greyscale image.
3) Display stereo digital image correlation data in 3D for static non-interactive plots use `matplotlib` for interactive plots use `pyvista`


### Outputs
- Return figure and axis handles to the user to allow for additional user defined configuration with `matplotlib`
- Interactive display of a plot or subplots of camera image heatmaps if the user calls `.show()` on the returned handle
- Function to save the plot as a vector graphic (.svg) or raster graphic (.png)



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
- A set of example/tutorial scripts demonstrating the functionality of the visualisation module



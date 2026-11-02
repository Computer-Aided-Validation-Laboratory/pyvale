# `pyvale` Design Specification

## Motivation
Qualification of fusion technology is reliant on simulations to predict the performance of components in extreme (e.g., thermal and electromagnetic) and untestable (e.g., fusion neutron fluxes) environments. Enabling the use of simulations for risk-informed decision making requires that they are validated over testable domains to reduce uncertainty when extrapolating into irradiated conditions. The cost of performing large-scale validation tests on a complex components such as a breeder blankets will be on the order of £M's. Therefore, significant cost and risk reduction can be achieved by maximising the information obtained from an optimised set of targeted experiments.

A key parameter of validation experiments is the deployment of sensor arrays to measure the components response. There are currently no commercial tools available that can simulate and optimise the placement of diverse arrays of sensors for multi-physics conditions with realistic constraints (e.g., cost, reliability, and accuracy). Such a tool would have immediate benefits for reducing costs of the experimental programme required to qualify fusion components such as the breeder blankets and divertors. Enabling digital twins for future fusion power plants will also require sensors and data transfer to the digital model. It is envisaged that a sensor simulation and optimisation tool would decrease the cost and increase the reliability of the hardware and cyber/physical interfaces for fusion plant digital twins.

## Aims & Objectives
The aim of the `pyvale` project is to develop a software engine that can use an input multi-physics simulation to produce a set of simulated sensor dataset with realistic uncertainties to be used for experiment optimisation. This software engine will be developed as a python package to allow for ease of use by the scientific and engineering community.  `pyvale` is made open-source with an MIT license allowing for commercial use. The main objectives of `pyvale` are to:

1. Provide a toolbox of sensor simulation models with realistic uncertainties which can be applied to an input multi-physics simulation with a particular focus on camera-based sensors.
2. Create an experimental design and sensor placement optimisation workflow that utilises the sensor model library.
3. Develop a simulation calibration and validation framework that can be used with the sensor model library or with real experimental data.

A key focus of `pyvale` is developement of open-source scalable and accesible tools that can be used to simulate uncertainties for camera-based sensors. Current commercial solutions for simulating cameras have restrictive licenses that do not allow deployment on clusters for large optimisation analyses.

## `pyvale` Design Philosophy
- Scientists and engineers want a python interface: Write computationally demanding algorithms in lower level languages like Cython, C/C++ but provide a Python interface for users
- Users want to get started quickly but have the option to customise everything: provide good defaults to simplify the interface
- Good docs and examples make it easy for users:

## Sensor Simulation Toolbox Specification
A simulated sensor will produce a simulated measurement $M$ which is defined to be: $M = T + E_{S} + E_{R}$ where $T$ is the ground truth value taken from the given input multi-physics simulation, $E_{S}$ includes any sources of systematic measurement error (e.g. averaging, digitisation, calibration), and $E_{R}$ includes any sources of random errors (e.g. noise).

The models in the sensor simulation library should provide:

- Simulated measurements of scalar, vector and tensorial fields accounting for sensor orientation in the vector and tensor cases.
- Fast interpolation routines to extract groung truth sensor values from the input multi-physics simulation at user specified positions and sampling times.
- Models of sources of systematic errors including:
    - Digitisation, analog to digital conversion and saturation.
    - Uncertainty in the sensor position, angle or sampling time.
    - Spatial and temporal averaging.
    - Calibration error and drift.
    - Interval based systematic errors based on a given probability distribution.
- Models of sources of random errors including:
    - Noise with a given probability distribution.
    - Noise as a percentage of ground truth or accumulated sensor reading.
- A system for chaining different error sources for a given sensor in a plug and play manner.
- A set of tools to run Monte-Carlo type simulated experiments for a given set of sensor arrays applied to a set of simulations.
- Statisitical analysis methods for extracting the main contributors to the total sensor error over a set of experiments as well as summary statisitics.
- Quickstart pre-built sensors for common cases such as thermocouples, strain gauges, load cells etc.
- A camera simulation engine for uncertainty quantification of infra-red thermography (IRT) and digital image correlation (DIC) systems supporting:
    - Rasterisation and ray tracing using a custom engine and/or Blender's EEVEE or Cycles.
    - Direct projection of physical fields onto the camera (e.g. temperature or displacement) or texture warping of speckle patterns for DIC.
- A 2D DIC engine including:
    - Speckle size analysis tools, speckle pattern quality assessment and speckle pattern generation.
    - Length calibration and region of interest selection tools.
    - 2D DIC: supporting affine shape functions, spline interpolation and the ZNSSD correlation criteria.
- A stereo DIC engine including:
    - A stereo calibration module supporting various distortion models
    - Calculation of correlation quality statistics such as the epi-polar distance
    - Stereo correlation for shape reconstruction, displacement and strain calculation

## Experimental Design & Sensor Placement Optimisation Specification
TODO

## Simulation Calibration & Validation Framework Specification
TODO

## `pyvale` Developement Milestones for 2025
The key milestones for `pyvale` developement for 2025 include:

- A camera simulation engine for simulating infra-red thermography (IRT) and digital image correlation (DIC) supporting:
    - Rasterisation: using a custom engine and Blender's EEVEE
    - Ray Tracing: using a custom engine and Blender's Cycles
    - Direct projection of physical fields onto the camera or texture warping for DIC
    - Interoperability with existing `pyvale` sensor and error models
- A DIC engine supporting:
    - Speckle pattern texture generation
    - Speckle quality analysis tools
    - 2D DIC with affine shape functions and ZNSSD correlation criteria
    - Extension: Stereo DIC and stereo calibration
    - Interoperability with existing `pyvale` sensor and error models
- Software engineering tasks to support sharing of the above tools:
    - Internal review of the code, tests and documentation
    - Documentation of the code
    - Appropriate tests to ensure functionality of the code
    - Example/tutorial scripts demonstrating the usage of the code
- Publication of at least one paper on/or using `pyvale`, possibly:
    - Point sensor simulation and comparison to experiments - SoftwareX
    - A comparison of (image rendering techniques for) rasterisation and ray tracing for DIC UQ in 2D - Exp Mech
    - Assessment of image-based validation metrics - VVUQ journal

To ensure performance for the computationally intensive rendering and DIC processes above the underlying processing will be done in Cython, C and HIP. A python interface will be provided through `pyvale` for ease of use for the engineering and scientific community.


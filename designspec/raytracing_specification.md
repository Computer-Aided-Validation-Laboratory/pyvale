# `pyvale` Ray-Tracing Engine: Design Specification

## Motivation
The `pyvale` python package is intended to be an all-in-one package for sensor simulation, sensor uncertainty quantication, sensor placement optimisation and simulation calibration/validation. A particular focus of `pyvale` is to develop sensor simulation methods specifically focused on cameras including infra-red thermography and digital image correlation (DIC).


## Aims & Objectives

The objectives of this project are to develop a ray tracing engine for `pyvale` that supports:

- A Python interface with underlying performant code in Cython, C and/or vendor agnostic GPU code (HIP).
- TODO

A non-exhaustive list of sub-modules is provided below including the required inputs, processing and outputs for each. Note that these requirements may change as the DIC engine module is developed.

## TODO
- Apply a speckle pattern / texture to a 3D object
- Bring additional objects into the scene to accurately model the field of view
- Allow modelling of windows
- Render a series of static images
- Render a series of deformed images
- Allow addition of grey level noise to the images
- Allow for multiple cameras
- Allow for the specification of light sources

## Sub-Module: TODO
### Inputs
- TODO
### Example Workflow
- TODO
### Outputs
- TODO


## Deliverables
- A ray tracing engine module integrated into `pyvale` with the following sub-modules (note these are just suggested names and are not binding, use whatever structure makes most sense during developement):
    - TODO
- Full doc-strings and auto generated documentation
- A pragmatic suite of software tests including unit and regression tests
- Example/tutorial scripts demonstrating the functionality of the ray tracing module with increasing complexity of use
- A short markdown report analysing ray tracing benchmarks for DIC (comparing speed and accuracy) compared to the same ray tracing benchmarks performed in Blender using the Cycles renderer.
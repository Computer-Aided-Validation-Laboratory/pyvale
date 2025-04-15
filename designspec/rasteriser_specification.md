# `pyvale` Rasterisation Module: Design Specification

## Motivation
The `pyvale` python package is intended to be an all-in-one package for sensor simulation, sensor uncertainty quantication, sensor placement optimisation and simulation calibration/validation. A particular focus of `pyvale` is to develop sensor simulation methods specifically focused on cameras including infra-red thermography (IRT) and digital image correlation (DIC) allowing for uncertainty quantification of these systems. The computer graphics field has undertaken extensive research in rendering and camera simulation methods.

There are two main methods in computer graphics for rendering scenes onto simulated cameras, this includes rasterisation and ray tracing. Rasterisation is faster and can be performed in real time for video game applications. Ray tracing is more computationally intensive but has the benefit that it can more accurately simulate lighting effects such as shadows and reflections. Given the speed of rasterisation it is desirable for simulating uncertainties for imaging sensors when they are coupled to optimisation loops.

There are open-source graphics libraries that support rasterisation, a notable example being Blender. However, computer graphics libraries are targeted at cases where the geometry is defined by high element count surface meshes and typically only support linear triangular elements. For solid mechanics finite element simulations in engineering higher order elements (quadratic with midside nodes) are preferred and surface element counts are low given that the full 3D geometry is meshed for displacment, strain and stress calculations. Computer graphics engines will typically triangulate higher order meshes which can lead to interpolation errors.


## Aims & Objectives
The objectives of this project are to develop a rasteriation engine for `pyvale` that supports:

- A Python interface with underlying performant code in Cython, C and/or vendor agnostic GPU code (HIP).
- TODO

A non-exhaustive list of sub-modules is provided below including the required inputs, processing and outputs for each. Note that these requirements may change as the DIC engine module is developed.

## Sub-Module: TODO
### Inputs
- TODO
### Example Workflow
- TODO
### Outputs
- TODO


## Deliverables
- A rasterisation module integrated into `pyvale` with the following sub-modules (note these are just suggested names and are not binding, use whatever structure makes most sense during developement):
    - TODO
- Full doc-strings and auto generated documentation
- A pragmatic suite of software tests including unit and regression tests
- Example/tutorial scripts demonstrating the functionality of the rasteriation module with increasing complexity of use
- A short markdown report analysing rasteriation benchmarks for DIC (comparing speed and accuracy) compared to the same rasteriation benchmarks performed in Blender using the EEVEE renderer.
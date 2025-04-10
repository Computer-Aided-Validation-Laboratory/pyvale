# `pyvale` Ray-Tracing Module: Design Specification

## Motivation
The `pyvale` python package is intended to be an all-in-one package for sensor simulation, sensor uncertainty quantication, sensor placement optimisation and simulation calibration/validation. A particular focus of `pyvale` is to develop sensor simulation methods specifically focused on cameras including infra-red thermography (IRT) and digital image correlation (DIC) allowing for uncertainty quantification of these systems. The computer graphics field has undertaken extensive research in rendering and camera simulation methods.

There are two main methods in computer graphics for rendering scenes onto simulated cameras, this includes rasterisation and ray tracing. Rasterisation is faster and can be performed in real time for video game applications. Ray tracing is more computationally intensive but has the benefit that it can more accurately simulate lighting effects such as shadows and reflections. Given that lighting can have a significant impact on measurements made with DIC systems it is desirable for `pyvale` to be able to replicate these effects.

There are open-source graphics libraries that support ray tracing, a notable example being Blender. However, computer graphics libraries are targeted at cases where the geometry is defined by high element count surface meshes and typically only support linear triangular elements. For solid mechanics finite element simulations in engineering higher order elements (quadratic with midside nodes) are preferred and surface element counts are low given that the full 3D geometry is meshed for displacment, strain and stress calculations. Computer graphics engines will typically triangulate higher order meshes which can lead to interpolation errors.


## Aims & Objectives
The aim of this project is to develop a custom ray tracing module for `pyvale` that supports rendering of higher order finite element simulations without interpolation bias. This ray tracing module will be used to simulate images that would be taken by IRT and DIC systems to simulate the uncertainties associted with these sensors. The objectives of this project are to develop a ray tracing module for `pyvale` that supports:

- A Python interface with underlying performant code in Cython, C and/or vendor agnostic GPU code (HIP)
- Scene building allowing the user to specify parts that will implement physics (e.g. having temperature or displacement fields) and parts that are only meant to block line of sight or reflect light
- Speckle pattern texture application to 3D surface meshes and specification of surface optical properties in the visible and IR spectrum
- User configurable cameras and light sources with multiple cameras and light sources allowed in a scene for simulating combined IRT and DIC systems
- Rendering of triangular and quadrilateral, linear and quadratic elements with a mix of different element types in the scene
- Rendering infra-red images allowing for black/grey bodies radiating in the scene and accounting for the spectral response of a given IR camera in a user specified wave length range
- Rendering static images and deformed speckle pattern images from a solid mechanics finite element simulation with the option to add various types of grey level camera sensor noise for Monte-Carlo analysis
- Simulation of focus, depth of field, thin/thick lense effects and optical distortion
- CPU and GPU rendering
- Parallelisation options for CPU and GPU to increase performance and scalability

A non-exhaustive list of sub-modules is provided below including the required inputs, processing and outputs for each. Note that the specific organisation of sub-modules will change as the project develops, it most important that the key functionality and workflows are supported for the objectives above.


## Sub-Module: Scene Builder
This sub-module will add all relevant objects, cameras, lights and physics to the scene before using a rendering submodule to create the desired images from the scene.
### Inputs
- Meshes for the objects to be present in the scene with or without physical fields as `SimData` objects
- The position and orientation of each object in the scene with respect to a world coordinate system
- The position, orientation and type of each camera and light source in the scene, it may be helpful here to group cameras as IR, 2D DIC or Stereo DIC pairs. Note that it should also be possible to pair an IR camera with a stereo DIC camera to map temperatures onto a deformed object.
### Workflow
- Load meshes of all objects in the scene along with any relevant physics into `SimData` objects
- Specify the optical properties for all meshes in the scene
- Specify all light sources in the scene as well as the background light intensity
- Specify all cameras in the scene and their parameters
- Create the scene and add the meshes, camera and light sources
### Outputs
- A scene dataclass containing all information required to render the scene
- An option to save the scene dataclass to file for later use


## Sub-Module: Speckle Mapper
### Inputs
- One or more geometric meshes (of various element types)
- A high resolution speckle pattern texture image
- A scaling factor to map the speckle pattern texture to the mesh with the desired speckle size
### Workflow
- Load the meshes and the speckle pattern texture image
- Specify the scaling factor, surfaces on the meshes to map the texture to and the texture mapping algorithm
- Map the speckle texture to the image and view the result
### Outputs
- An interactive visualisation of the speckle texture applied to the mesh using `pyvista`
- A mapping for the speckle texture onto the mesh


## Sub-Module: Infra-Red Renderer
This sub-module is part of the core functionality of the ray tracing module allowing the user to simulate infra-red images. The effects of reflections and parasitic radiation from surrounding objects to the measured temperature are of most interest here.
### Inputs
- A scene dataclass describing: the objects/meshes in the scene; their surface properties in the IR spectrum; their positions in world coordinates; the constant temperature or temperature field for objects in the scene radiating infra-red light; and the parameters for any infra-red cameras in the scene.
- A dataclass specifying the rendering options such as parallelisation, CPU/GPU rendering, number of samples, number of ray bounces and wavelengths of the IR spectrum to analyse etc.
### Workflow
- Build a scene with the desired objects/meshes, physics (nominally temperature and displacements), infra-red cameras and light sources
- Specify the infra-red wavelength range to render, the emissivity and temperature of all objects radiating in the scene
- Render the image sequence for the specified temperature and displacement fields to memory or optionally save image by image to disk (in a lossless bitmap format) to save memory
- Optionally add sensor noise to the images in terms of temperature or counts
### Outputs
- A "noise free" static reference image per camera
- A set of "noise free" deformed images for each camera in the scene using the input displacement time history for each deformable object in the scene
- A set of "noisy" static reference images for each camera in the scene with different copies of user defined grey level noise to allow for noise floor analysis
- Sets of "noisy" deformed images with different copies of user defined grey level noise to allow for Monte-Carlo analysis


## Sub-Module: Calibration Target Renderer
This will be similar to the deformed image renderer below but will allow the user to build a specific calibration scene without a speckled target. The module will then let the user specify parameters for a calibration target (a grid of black dots with 3 key white on black dots specifying the target plane) before rendering a stack of images translating and rotating the target through all degrees of freedom in the field of view of specific camera pairs.

### Inputs
- A scene dataclass describing: the objects/meshes in the scene; their optical properties; and their positions in world coordinates.
- A dataclass specifying the rendering options such as parallelisation, CPU/GPU rendering, number of samples and number of ray bounces etc.
- A dataclass describing the calibration target (number of dots, dot spacing and location of key dots)
- A dataclass describing the translations and rotations to perform for the
### Workflow
- Build a scene with the desired objects/meshes, stereo DIC camera systems (optionally paired with IR cameras for temperature mapping) and light sources
- Create the calibration target and the sequence of translation/rotation combinations to be rendered
- Render the image sequence of the calibration target to memory (to pass directly to the stereo calibration module) for all cameras in the scene or optionally save an image at a time to disk (in a lossles bitmap format) to save memory
### Outputs
- A sequence of calibration target images translating the calibration target through all degrees of freedom throughout the field of view


## Sub-Module: Deformed Image Renderer
This sub-module is part of the core functionality of the ray tracing module allowing the user to create deformed speckle pattern images from a solid mechanics simulation for processing with 2D or Stereo DIC. A key functionality of this module will be to render speckle pattern images with realistic lighting effects for uncertainty quantification.
### Inputs
- A scene dataclass describing: the objects/meshes in the scene; their optical properties; their positions in world coordinates; and the displacement fields for deforming objects in the scene as a function of time or an option to render a static reference image only.
- A dataclass specifying the rendering options such as parallelisation, CPU/GPU rendering, number of samples, number of ray bounces and option to save images to render all images in memory or save directly to disk etc.
### Workflow
- Build a scene with the desired objects/meshes, 2D or stereo DIC camera systems and light sources
- Apply speckle patterns to the deforming objects in the scene scaling them to the appropiate pixel/speckle resolution
- Render the image sequence of the deforming objects either to memory (to be passed directly to the DIC module) or one frame at a time to disk to conserve memory
- Optionally add grey level sensor noise to the images
### Outputs
- A "noise free" static reference image per camera
- A set of "noise free" deformed images for each camera in the scene using the input displacement time history for each deformable object in the scene
- A set of "noisy" static reference images for each camera in the scene with different copies of user defined grey level noise to allow for noise floor analysis
- Sets of "noisy" deformed images with different copies of user defined grey level noise to allow for Monte-Carlo analysis

## Deliverables
- A ray tracing module fully integrated and merged into the main branch of `pyvale` with the following sub-modules (note these are just suggested names and are not binding, use whatever structure makes most sense during developement to achieve the desired functionality):
    - RayTrace.SceneBuilder
    - RayTrace.SpeckleMapper
    - RayTrace.CalTargetRenderer
    - RayTrace.IRRender
    - RayTrace.DICRender
- Full doc-strings and auto generated documentation for all modules and sub-modules
- A pragmatic suite of software tests including unit and regression tests for all modules and sub-modules
- Example/tutorial scripts demonstrating the functionality of the ray tracing module with increasing complexity of use
- A short markdown report analysing ray tracing benchmarks for DIC (comparing speed and accuracy) compared to the same ray tracing benchmarks performed in Blender using the Cycles renderer.
- A short markdown report comparing the ray tracing module to the 2D image deformation code already in `pyvale` based on processing the deformed speckle images with DIC for the test cases here: https://github.com/Computer-Aided-Validation-Laboratory/dicbenchmarks
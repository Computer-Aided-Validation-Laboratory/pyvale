# `pyvale` Digital Image Correlation Engine: Design Specification

## Motivation
Most digital image correlation (DIC) software packages are commercial with restrictive license that prevent deployment on supercomputing clusters. Open source alternative are available however, many of these only support 2D digital image correlation and not stereo correlation. A notable exception to this is the DICe package developed by Sandia National Laboratory. However, DICe requires installation of the Sandia software stack and is mainly targeted at Redhat Enterprise Linux distributions preventing portability across operating systems. Given the prevalence of Python in the scientific and engineering communities it would be desirable to have a DIC software package with a Python interface and underlying performant code in Cython, C and GPU languages for use across operating systems and on supercomputing clusters.

The `pyvale` python package is intended to be an all-in-one package for sensor simulation, sensor uncertainty quantication, sensor placement optimisation and simulation calibration/validation. A particular focus of `pyvale` is to develop sensor simulation methods specifically focused on cameras including infra-red thermography and DIC. Testing camera simulation methods such as rasterisation and ray-tracing for DIC requires a DIC engine for verification. Therefore, we intend to build and integrate a DIC engine into `pyvale` supporting both 2D and stereo DIC.


## Aims & Objectives
The aim of this project is to develop a performant DIC engine with a Python interface that is fully integrated with the `pyvale` sensor simulation package. The objectives of this project are to develop a DIC engine that supports:

- A Python interface with underlying performant code in Cython, C and/or vendor agnostic GPU code (HIP).
- A set of tools for speckle pattern generation and speckle quality analysis
- Subset based 2D DIC
- Stereo calibration
- Subset based stereo DIC
- Functions for performing convergence studies for the DIC parameters

A non-exhaustive list of sub-modules is provided below including the required inputs, processing and outputs for each. Note that these requirements may change as the DIC engine module is developed.

## Sub-Module: Speckle Pattern Generator
### Inputs
- Number of pixels in the image in the horizontal (X) and vertical directions (Y)
- Number of pixels sampling each speckle, default to 5 pixels/speckle
- Bit depth of the image, default to 12 bits stored in a 16 bit wrapper
- Target black/white balance
- Specified contrast and mean grey level for the image, both specified as a fraction of the grey level
- Option to apply a gaussian blur to the generate speckle image
- Any other options required to specify the 'randomness' of the speckle pattern
- Option to return the image as a numpy array of `np.float64` or to apply digitisation error and return as a `np.uint16` or similar supporting the bit depth of the image
- Save options to save the image to hard disk (note the capability to pass the image via RAM to other algorithms is also required, so save functionality should be separated from the speckle image generation)
### Example Workflow
- Create a dataclass with the required options to generate the speckle pattern setting desired parameters and leaving others as defaults.
- Call a function to generate and return the speckle pattern in memory
- Call a function to view the speckle pattern
- Call a function to save the speckle pattern in memory to disk
- Optional follow up workflows:
    - Pass the speckle pattern image to the pattern quality sub-module
    - Pass the pattern to one of the image deformation modules in `pyvale` to be directly deformed in 2D or used as texture in 3D
### Outputs
- A speckle pattern image as a numpy array (where the user specifies `np.float64` or `np.uint16`) allowing it to be passed to the DIC processing submodule directly.
and/or
- A speckle pattern image saved to an uncompressed format such as .tiff or .bmp


## Sub-Module: Speckle Pattern Quality
### Inputs
- One or more grey level images of the speckle pattern to analyse
### Example Workflow
### Outputs
- Average speckle size calculated from the image(s)
- Black white balance calculated from the image(s)
- Mean intensity gradient of the image(s)
- Shannon entropy of the image(s)
- If at least two images are provided then: the noise as a function of grey level


## Sub-Module: Region of Interest
### Inputs
- TODO
### Example Workflow
- TODO
### Outputs
- TODO


## Sub-Module: 2D DIC
### Inputs
- A static reference grey level image (or pair of images for stereo)
- One or more deformed images
- A region of interest geometric mask defining where the correlation is to be performed in the reference image
- A set of options specifying (allowing for sensible defaults):
    - Subset size in pixels
    - Step size in pixels
    - Subset shape function: at minimum rigid and affine
    - Correlation criterion: at minimum supporting Zero Normalised Sum of Square Differences (ZNSSD)
    - Interpolation method: at minimum supporting b-splines
    - Image pre-filtering: at minimum Gaussian blurring over a specified window in pixels
    - A correlation residual threshold for discarding poorly correlated subsets.
    - Parallelisation: CPU or GPU based
- A pixel resolution in units of length per pixel
### Example Workflow
- TODO
### Outputs
- Coordinates of the subsets in pixel [x,y] and world coordinates [x,y]
- Displacement vector components for each subset, [x,y] in pixel and world length units
- Correlation residual for each subset


## Sub-Module: Stereo Calibration
For this module there are probably a large number of function in OpenCV that can help, especially for dot detection on the calibration target. Blender can be used to generate known calibration target images for testing this submodule.

### Inputs
- A set of images of a calibration target moved through all degrees of freedom in the image space
- Parameters for the calibration target including dot spacing, dot size and number of dots
### Example Workflow
### Outputs
- A set of intrinsic and extrinsic calibration constants as a dataclass which can be passed directly to the stereo DIC sub module
- The calibration residual
and/or
- A human readable file containing the calibration constants


## Sub-Module: Stereo DIC
### Inputs
- A static reference grey level image (or pair of images for stereo)
- One or more deformed images
- A region of interest geometric mask defining where the correlation is to be performed in the reference image
- A set of options specifying (allowing for sensible defaults):
    - Subset size in pixels
    - Step size in pixels
    - Subset shape function: at minimum rigid and affine
    - Correlation criterion: at minimum supporting Zero Normalised Sum of Square Differences (ZNSSD)
    - Interpolation method: at minimum supporting b-splines
    - Image pre-filtering: at minimum Gaussian blurring over a specified window in pixels
    - A correlation residual threshold for discarding poorly correlated subsets.
    - Parallelisation: CPU or GPU based
- Stereo calibration parameters (intrinsic and extrinsic)
### Example Workflow
### Outputs
- Coordinates of the subsets in pixel [x,y] and world coordinates [x,y,z]
- Displacement vector components for each subset, [x,y,z] in pixel and world length units
- Correlation residual for each subset
- Epi-polar distance



## Sub-Module: DIC Post-Processing
### Inputs
- Subset coordinates and displacements from stereo or 2D DIC
- Options for spatial differentiation and spatial smoothing of DIC data to calculate the deformation gradient
- Options for calculating strain tensors from the deformation gradient
- Options for temporal differentiation and temporal smoothing of DIC data
### Example Workflow
-
### Outputs

## DIC Benchmarks
Analyse accuracy and correlation speed for:
- Correlation on rigid body motion of a targt in 0.01 pixel intervals up to 0.1 pixels of total motion
- Correlation for a uniform hydrostatic strain in 0.01 pixel intervals up to 0.1 pixels of total deformation
- Correlation for a shear deformation in 0.01 pixel intervals up to 0.1 pixels of total deformation
- Correlation for a tensile test on a plate with a hole with deformation in 0.01 pixel intervals up to 0.1 pixels of total deformation
- DIC challenge benchmarks

This [repository](https://github.com/Computer-Aided-Validation-Laboratory/dicbenchmarks) can be used to generate benchmarks for cases other than the DIC challenge which is openly available elsewhere.

## Deliverables
- A DIC engine module integrated into `pyvale` with the following sub-modules (note these are just suggested names and are not binding, use whatever structure makes most sense during developement):
    - DICSpeckleQuality
    - DICSpeckleGen
    - DICRegionOfInterest
    - DIC2D
    - DICCalibration
    - DICStereo
    - DICPost including DICStrain, DICTempDiff
- Full doc-strings and auto generated documentation
- A pragmatic suite of software tests including unit and regression tests
- Example/tutorial scripts demonstrating the functionality of the DIC engine module with increasing complexity of use
- A short markdown report analysing the benchmarks in the DIC challenge
- A short markdown report benchmarking the DIC engine against anonymised data from commercial DIC software as well as the open source DICe

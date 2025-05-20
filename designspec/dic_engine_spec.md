# `pyvale` Digital Image Correlation Module: Design Specification

## Motivation
Most digital image correlation (DIC) software packages are commercial with restrictive license that prevent deployment on supercomputing clusters. Open source alternative are available however, many of these only support 2D digital image correlation and not stereo correlation. A notable exception to this is the DICe package developed by Sandia National Laboratory. However, DICe requires installation of the Sandia software stack and is mainly targeted at Redhat Enterprise Linux distributions. Furthermore, DICe must be built from the source to be used on Linux systems and clusters requiring advanced user knowledge. Given the prevalence of Python in the scientific and engineering communities it would be desirable to have a DIC software package with a Python interface and underlying performant code in Cython, C and GPU languages for use across operating systems and on supercomputing clusters.

The `pyvale` python package is intended to be an all-in-one package for sensor simulation, sensor uncertainty quantication, sensor placement optimisation and simulation calibration/validation. A particular focus of `pyvale` is to develop sensor simulation methods specifically focused on cameras including infra-red thermography and DIC. Testing camera simulation methods such as rasterisation and ray-tracing for DIC requires a DIC module for verification. Therefore, we intend to build and integrate a DIC module into `pyvale` supporting both 2D and stereo DIC.


## Aims & Objectives
The aim of this project is to develop a performant DIC module with a Python interface that is fully integrated with the `pyvale` sensor simulation package. The objectives of this project are to develop a DIC module that supports:

- A Python interface with underlying performant code in Cython, C and/or vendor agnostic GPU code (HIP).
- A set of tools for speckle pattern generation and speckle quality analysis
- Subset based 2D DIC
- Stereo calibration
- Subset based stereo DIC
- Functions for performing convergence studies for the DIC parameters
- Parallelisation options for CPU and GPU to increase performance and scalability

A non-exhaustive list of sub-modules is provided below including the required inputs, processing and outputs for each. Note that the specific organisation of sub-modules will change as the project develops, it most important that the key functionality and workflows are supported.

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
### Workflow
- Create a dataclass with the required options to generate the speckle pattern setting desired parameters and leaving others as defaults.
- Generate and return the speckle pattern in memory
- View the speckle pattern
- Save the speckle pattern in memory to disk in *.tiff or *.bmp formate
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
### Workflow
- Load one or more grey level images into a numpy array
- Calculate the speckle pattern quality statistics and return them as a dataclass
- Calculate grey level noise as a function of grey level for a stack of reference images
- Output the speckle pattern quality statistics to a human readable file
### Outputs
- Average speckle size calculated from the image(s)
- Black white balance calculated from the image(s)
- Mean intensity gradient of the image(s)
- Shannon entropy of the image(s)
- If at least two images are provided then: the noise as a function of grey level


## Sub-Module: Region of Interest
Gmsh uses OpenCASCADE to perform boolean operations on shapes in 3D. We might be able to leverage this to help but the 3D engine is probably overkill for something in 2D. The easiest method is probably going to be leveraging any interactive drawing tools available in matplotlib. There may also be tools in OpenCV for auto thresholding a mask.

### Inputs
- A static reference image from which the region of interest will be selected
### Workflow
- Load the reference image
- Interactively draw shapes on the image in 2D performing boolean operations between shapes to specify the region of interest, or
- Automatically threshold the image to extract the mask
- Determine the mask specifying the region of interest and return it as a 2D numpy array
### Outputs
- A numpy array mask the same size as the image specifying the valid pixels for subsets to run the correlation
- Save this mask to file as an image


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
    - Interpolation method: at minimum supporting cubic splines
    - Image pre-filtering: at minimum Gaussian blurring over a specified window in pixels
    - A correlation residual threshold for discarding poorly correlated subsets.
    - Parallelisation: CPU or GPU based
    - Option to load and perform the correlation image by image to save RAM or to load all images in one shot and perform the correlation over everything
- A pixel resolution in units of length per pixel
### Workflow
- Build a dataclass specifying the correlation options and the length to pixel conversion
- Load a region of interest mask file or specify this in code
- Load a reference image and optionally a set of deformed images
- Run the correlation and return the subset coordinates in world coordinates (i.e. the imaged surface shape) and the subset displacement vector components for the deformed images
    - Allow various parallelisation options here by subset and/or image or run single threaded to allow paralleisation in an outer loop.
- Save the subset coordinates, the displacement vector components, the correlation residual to file in either *.csv per image or an open binary format like *.hdf5
### Outputs
- Coordinates of the subsets in pixel [x,y] and world coordinates [x,y]
- Displacement vector components for each subset, [x,y] in pixel and world length units
- Correlation residual for each subset


## Sub-Module: Stereo Calibration
For this module there are probably a large number of function in OpenCV that can help, especially for dot detection on the calibration target. Blender can be used to generate known calibration target images for testing this submodule.

### Inputs
- A set of images of a calibration target moved through all degrees of freedom in the image space
- Parameters for the calibration target including dot spacing, dot size and number of dots
### Workflow
- Input the parameters of the calibration target including dot spacing and locations of the three dots used to find the plane of the target
- Load a stack of images of the calibration target moved through all degrees of freedom with the field of view
- Iterate through all of the images and perform dot detection to extract the grid of dots, locating the plane detection dots
- Knowing the parameters of the calibration target and the orientation of the plane it is in solve for the unknown stereo calibration parameters by minimising the cost function over all calibration images
- Return the residual for each calibration image and remove any images below a given threshold
- Repeat the cost function minimisation with poor images removed
- Save the calibration parameters to file or pass directly to the stereo DIC sub-module
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
    - Option to load an perform the correlation image by image to save RAM or to load all images in one shot and perform the correlation over everything
- Stereo calibration parameters (intrinsic and extrinsic)
### Workflow
- Build a dataclass specifying the stereo correlation options
- Load a stereo calibration parameters file or specify this in code
- Load a region of interest mask file or specify this in code
- Load a reference image and optionally a set of deformed images
- Run the correlation and return the subset coordinates in world coordinates (i.e. the imaged surface shape) and the subset displacement vector components for the deformed images
    - Allow various parallelisation options here by subset and/or image or run single threaded to allow paralleisation in an outer loop.
- Save the subset coordinates, the displacement vector components, the correlation residual and the epi-polar distance to file in either *.csv per image or an open binary format like *.hdf5
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
### Workflow
- Build a dataclass specifying the post-processing options
- Pass in RAM or load from file the DIC displacement data
- Post-process the displacement data with smoothing and spatial and/or temporal differentiation
- Return the post-processed quantities of interest (e.g. strain) and optionally save to disk
### Outputs
- The post processed quantities of interest (e.g. strain) as numpy arrays
- Coordinates at which the post processed quantities of interest were evaluated


## DIC Benchmarks
Analyse accuracy and correlation speed for:
- Correlation on rigid body motion of a targt in 0.01 pixel intervals up to 0.1 pixels of total motion
- Correlation for a uniform hydrostatic strain in 0.01 pixel intervals up to 0.1 pixels of total deformation
- Correlation for a shear deformation in 0.01 pixel intervals up to 0.1 pixels of total deformation
- Correlation for a tensile test on a plate with a hole with deformation in 0.01 pixel intervals up to 0.1 pixels of total deformation
- DIC challenge benchmarks

This [repository](https://github.com/Computer-Aided-Validation-Laboratory/dicbenchmarks) can be used to generate benchmarks for cases other than the DIC challenge which is openly available elsewhere.


## Deliverables
- A DIC module module fully integrated and merged into the main branch of `pyvale` with the following sub-modules (note these are just suggested names and are not binding, use whatever structure makes most sense during developement):
    - DICSpeckleQuality
    - DICSpeckleGen
    - DICRegionOfInterest
    - DIC2D
    - DICCalibration
    - DICStereo
    - DICPost including DICStrain, DICTempDiff
- Full doc-strings and auto generated documentation
- A pragmatic suite of software tests including unit and regression tests
- Example/tutorial scripts demonstrating the functionality of the DIC module module with increasing complexity of use
- A short markdown report analysing the benchmarks in the DIC challenge
- A short markdown report benchmarking the DIC module against anonymised data from commercial DIC software as well as the open source DICe

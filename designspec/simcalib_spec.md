# `pyvale` Simulation Calibration Module: Design Specification

## Motivation
`pyvale` is a virtual engineering laboratory and a key use-case of `pyvale` is enabling quantitative comparison between high-fidelity experimental data and simulations (with uncertainty quantification). There are two main applications of quantitative comparison between experimental data and simulations: 1) validation (the process of assessing the degree to which a simulation agrees with the real world system); and 2) calibration (the process of tuning unknown simulation inputs to match real world systems). The focus of this software module in `pyvale` is to provide a toobox for calibration of unknown simulation parameters for sub-component and full-component multi-physics models. Given the complexity of component models (mult-material with complex 3D geometry) it is generally not possible to make the simplifications that can be made in Materials Testing 2.0 that enable much faster inverse techniques such as the Virtual Fields Method (VFM). Therefore, this toolbox will focus on Finite Element Model Updating (FEMU) and other methods that can be used for component scale multi-physics models.   

## Aims & Objectives
The aim of this project is to develop a module for performing simulation calibration using input experimental data in `pyvale`. The input experimental data may be simulated experimental data generated with the sensor simulation or imaging modules in which case the purpose is to optimise the calibration error with respect to the sensor parameters (for DIC this will be selection of processing parameters such as the subset size and virtual strain gauge size). Otherwise the experimental data is taken from a real experiment and the calibration yields the simulation parameters that best agree with the experimental data within the bounds of the experimental uncertainty. The simulation calibration module in `pyvale` will provide:

- Specific tools for extracting boundary conditions or initial condition data directly from imaging sensor data (e.g. extracting boundary displacements from DIC data).
- A finite element updating (FEMU) module supporting:
    - User configuration to calibrate: material properties, boundary conditions, geometry or any other model inputs. 
    - Multi-objective optimisation for the above to identify pareto front of different combinations of parameters.
    - Use of full-fidelity FE models or surrogates
    - Steady state and transient simulation calibration
    - Multi-physics calibration (e.g. thermo-mechanical) either as a combined system or as a multi-step calibration procedure. 
    - Multi-fidelity sensors (e.g. point and cameras)
- Workflow management tools to connect the calibration module with other parts of `pyvale` supported by integration into `Matflow`.

As an extension this toolbox should support Bayesian parameter calibration and/or inverse identification allowing for the extraction of probability distributions for unknown simulation parameters. 

## Sub-Module: Image Data Boundary Condition Extractor (`ImageBCExtractor`)
A common problem when using DIC data for FEMU is the need to extract boundary conditions or initial geometry information from the DIC data. For example: take a tensile test on a complex geometry sample. We would like to extract the displacements along the top and bottom edges of the sample as some sort of spline or lookup table and impose this on our finite element model. Another example might be extracting the true shape of a component or the relative orientation and positiosn of multiple components from stereo DIC shape information. For IR camera data we might also want to impose a measured temperature field on simulation. 

### Inputs
- Image-based experimental data (normally IR camera or DIC data) either as a single frame of data or a time series of data. 
### Workflow
- Show the user the image of the field and allow them to draw geometry to extract data from (see the DIC module ROI tool for inspiration), including lines and areas.
- Either interpolate the DIC data to the user defined extraction geometry 
- Plot the extracted data as a function of space or time based on yse
- Allow the user to fit polynomials or splines and extract coefficients.
OR
- Produce a lookup table for linear interpolation
- Save the extracted data to file in csv format for parsing by simulation codes.
### Outputs 
- Function coefficients or interpolation/look-up table in simulation coords for the output distribution saved to a csv

### Test Cases 
1. Creates a MOOSE function for a complex displacement boundary condition on a 2D tension test of a complex geometry sample (see Creep MT2 open D shape sample).
    - As above but for thermal data extracting a line of temperature values.
2. Extract the coordinate transform that best maps DIC data (2D and stereo) onto the shape of a complex MT2 sample.
3. Use stereo DIC coordinate data to extract the relative positions of two 3D components in a scene.

## Sub-Module: Finite Element Model Updating (`FEMU`)
- This module will need to be able to support parallelisation either within the model/simulation itself as well as calculation of the cost function (especially for genetic algorithms and particle swarms). Model parallelisation should be handled the simulation engine and be user configurable whereas cost function parallelisation can be implemented with python multi-processing and `apply_async`. 
- This module will also need to implement logging to track iterations of the optimiser in case the optimisation fails or optimiser hits and iteration limit. 

### Inputs
- Experimental data to calibrate against: can be multi-fidelity (point, images etc) and multi-physics (thermal, mechanical etc) 
- A model to calibrate (finite element model or surrogate)
    - Define interface (abstract base class) to allow this to be: surrogate from `pyapprox` or `scikit-learn` or full model from `MOOSE`, `ngsolve` of `fenics`
- The input variables to the finite element model that will calibrated/identified 
- Initial guess(es) / sampling for the input variables and their feasible ranges and/or PDFs
- A cost function (loss function, objective function etc) to optimise including:
    - Define an interface for this (abstract base class) that the user must implement or that has concrete implementations the user can select.
    - General sum of square differences
    - Presets for different sensor types and how they can be combined 
    - Option to include regularisation
    - Sensivity-based weighting
    - Sensor noise normalisation
- A sampler (such as grid search, latin-hypercube or Monte-Carlo) or optimiser (nominally gradient free) such as Nelder-Mead, genetic algorithms, and particle swarms.g
    - Define interface (abstract base class) for this to wrap `Scipy` and `pymoo` to start with.
    - Optimiser parameters, convergence criteria and iteration limits
    - Provide ability to use Bayesian optimisation with `Botorch` for expensive models
 
### Workflow
- Define inputs above to setup the problem
- Generate initial guesses for the input parameters
- Calculate the cost function for the initial input parameters (possible in parallel)
    - Run finite element model or surrogate
    - For a full finite element model:
        - Read simulation output from disk to a `SimData` object
        - Interpolate the model output to the sensor locations and times (using existing `pyvale` sensor simulation module). 
    - For a field-based surrogate:
        - Evaluate the field at the sensor locations
    - For a direct surrogate:
        - 
    - Calculate individual sensor cost using squared difference (or other suitable user configurable norm) between sensor values. 
    - Apply any weighting, noise normalisation and regularisation combining the sensor values into a single cost.  
- Update input variables based on optimiser rules and repeat cost function calculation until convergence is achieved or an iteration limit is reached. 

### Outputs
- Option to save all model evaluations throughout the optimisation
- The cost function history 
- The identified input variables (and/or distributions) that minimise the cost function (material properties, boundary conditions )

### Test Cases
A list of suggested test cases is provided below for thermo-mechanical problems. This list is not exhaustive and some test cases can be removed and replaced with other to demonstarte core functionality as long as the test cases cover the following:
- Steady state and transient problems
- Single and multi-physics problems
- Point, imaging aand multi-fidelity sensor arrays

**Thermal Test Cases:**
- Identify input heat flux and/or heat transfer coefficient on a 2D plate: can be steady state or transient; 
- Identify thermal material properties based on  

**Mechanical Test Cases:**
- Identify coefficients of a polynomial distributed tensile load applied to a 2D linear elastic plate with known material properties: can be steady state or transient; can include load cell and strain gauges or load cell and DIC.
- Identify the yield stress and hardening modulus (i.e. plastic material properties assuming elastic properties are known) from a plate with hole loaded in tension: must be transient to obtain the hardening modulus: will need to have load cell data and some combination of kinematic
- MT2/WPDIV test sample simulation and experimental data: identify elastic modulue, Poisson's ratio and Voce hardening law. 

**Multi-Physics Test Cases:**
- Identify the temperature dependent thermal expansion coefficient for a 2D plate with thermal and mechanical loads.
- Simultaneous identification of thermal and mechanical material properties
- Simultaneous identification of thermal and mechanical boundary conditions

## Sub-Module: Matflow Integration
- This module should integrate the simulation calibration module as a 'task schema' in Matflow allowing integration with  
- See `pyvale` workflows design specification. 

## Deliverables
- A simulation calibration module integrated into `pyvale` demonstrated on the test cases listed above.
- A user guide page in the `pyvale` documentation describing any theory required for the user to understand how the module works
- Full doc-strings and auto generated documentation for all modules and sub-modules
- A pragmatic suite of software tests including unit and regression tests where appropriate
- Example/tutorial scripts demonstrating the functionality of the module with increasing complexity of use integrated with the `pyvale` documentation

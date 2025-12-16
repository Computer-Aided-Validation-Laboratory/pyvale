# `pyvale` Simulation Calibration Module: Design Specification

## Motivation


`pyvale` is a virtual engineering laboratory. 

## Aims & Objectives
The aim of this project is to develop a module for performing inverse identification of simulation parameters (i.e. simulation calibration) using input experimental data in `pyvale`. The input experimental data may be simulated experimental data generate with the sensor simulation or imaging modules in which case the purpose is to optimise the identificatione error with respect to the sensor parameters (for DIC this will be selection of processing parameters such as the subset size and virtual strain gauge size.)

- Supports multi-physics calibration with data from multi-fidelity sensors (e.g. point and cameras)
- User configurable to calibrate: material properties, boundary conditions, geometry or other model inputs 
- Allows for multi-objective optimisation for the above to identify pareto front of different combinations of parameters

**Scratch**
- IO: experiment and simulation data as well as ExpData and SimData data structures
- Python interfaces linking and orchestrating things?
- Integrates with Matflow to orchestrate the FEMU loop?  
- Allows different optimisers? Bayesian optmisation and other gradient free (genetic and particle swarm)
- Plug and play surrogate modelling capability? 
    - Will need to define some interfaces for Sim, IO, Cost
- Bayesian calibration capability to identify parameter distributions with MCMC?
- Tested with sensor simulation, blender and DIC modules?
- Tested with thermal, mechanical and thermo-mechanical problems.
- Need plug and play field cost function and point cost function as well as method to combine and weight different sensors in the cost function
- **NOTE** need to be able to interpolate to sensor locations for FEMU when building cost function. Use shape functions or mesh-free interpolation - should just integrate with pyvale sensor sim?
- Will need IO for simulation step into SimData.

- User configurable sensor positioning and orientation in the toolbox, will need to integrate with `pyvale` sensor simulation toolbox

Inverse identification of:
- Unknown loading distribution on a plate given DIC data and the load cell information
- Unknown heat flux distribution or htc given some temperature measurements / IR camera data
- Unknown material constitutive law given DIC data and load cell information, 
- Multi-physics calibration of a thermo-mechanical simulation, identify thermal expansion coeff?
- Identify plasticity law on a plate with a hole (sigma_y and H)?

## Sub-Module: Experimental Data Boundary Condition Extractor
- 

## Sub-Module: Finite Element Model Updating (FEMU)
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
- An optimiser (nominally gradient free) such as Nelder-Mead, genetic algorithms, and particle swarms.
    - Define interface (abstract base class) for this to wrap Scipy and Pymoo to start with.
    - Optimiser parameters, convergence criteria and iteration limits
### Workflow
- Define inputs above to setup the problem
- Generate initial guesses for the input parameters
- Calculate the cost function for the initial input parameters (possible in parallel)
    - Run finite element model or surrogate
    - For a full finite element model:
        - Read simulation output from disk to a `SimData` object
    - For a field-based surrogate:
        - 
    - For a direct surrogate:
        - 
- Update input variables based on optimiser rules and repeat previous step 
### Outputs
- The cost function history 
- The identified input variables (and/or distributions) that minimise the cost function (material properties, boundary conditions )

## Test Cases: Finite Element Model Updating (FEMU)
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

## Deliverables
- An inverse identification module integrated into `pyvale` demonstrated on the test cases listed above.
- A user guide page in the `pyvale` documentation describing any theory required for the user to understand how the module works
- Full doc-strings and auto generated documentation
- A pragmatic suite of software tests including unit and regression tests where appropriate
- Example/tutorial scripts demonstrating the functionality of the rasteriation module with increasing complexity of use integrated with the `pyvale` documentation

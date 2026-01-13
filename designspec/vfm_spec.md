

# `pyvale` VFM Module: Design Specification

## Motivation
The `pyvale` VFM (Virtual Fields Method) module seeks to provide a toolkit for inverse identification of mechanical properties from full-field strain measurements. 

Knowledge of the mechanical properties of materials is required for the design, analysis and qualification of  engineering products. Capitalising on information-rich, full-field optical measurements, inverse identification techniques can provide spatially varying maps of mechanical properties. In turn, identified properties can be used to inform computer models, manufacturing processes, design codes  etc. - accelerating engineering design and qualification.
###### Spatially-varying mechanical properties  
In particular, the characterisation of heterogeneous properties using traditional techniques is limited by spatial resolution or insufficient reliability. Heterogeneity arises in many engineering applications. It can be introduced intentionally, in  composite materials for example, or it may arise as a consequence of a component’s manufacturing process or use. Some examples include: welding, additive manufacturing, irradiation damage, and many others.

Benefit of mechanical property identification...
Comparison to current tools....

## Relevant theory and resources

###### Development background
This work originated during the below PhD project.

Robert Hamill (2025) ‘Development of a Methodology for the Automated Spatial Mapping
of Heterogeneous Elastoplastic Properties of Welded Joints’, University of Southampton, Faculty of Engineering and Physical Sciences, PhD Thesis. Supervised by: Prof. Fabrice Pierron (University of Southampton, MatchID), Dr. Aleksander Marek (University of Southampton), and Dr. Allan Harte (UK Atomic Energy Authority)
https://eprints.soton.ac.uk/502239/

The goal of the project was to develop a novel methodology for the characterisation of heterogeneous mechanical properties by extending the virtual fields method through the automated spatial parameterisation of constitutive  parameters. Collaboration with the United Kingdom Atomic Energy Authority provided the  project with an application focus on the characterisation of the spatially-varying, elastoplastic  mechanical properties of welded joints. The developed methodology enables the novel characterisation of welds with assorted geometries, varied loading configurations and dissimilar  materials.

link to paper
###### Key resources / references
- List of key papers, attribution, link to some key theory pages simply explaining core components of stress recon, VFM, optimisation.
- For an overview of Material Testing 2.0 and inverse identification techniques see 'Material Testing 2.0: A brief review' by  Fabrice Pierron (2023).
- Alex's papers on SBVF
- Rory's paper

---

## Aims & Objectives
This project aims to translate the prototype Matlab codebase to clean, extensible and open-source Python, forming a user-friendly toolkit for inverse identification. Once the core codebase is established, further development will target robustness, performance and functionality. 

The module aims to support smooth collaboration with others, transparency, computational performance and extensibility. Some target characteristics are outlined below:
- Computational performant (vectorisation and parallelisation where feasible)
- Transparency, abstraction and modularity to enable extensibility
- Release the software openly, with example data and comprehensive documentation to encourage utilisation, dissemination and collaboration
- Unit and system-level testing to ensure robustness and support collaborative development

Some of the key functionality is listed below.
##### Inverse identification tool 
- Input full-field strains, output maps of mechanical properties
###### Constitutive laws 
- for a range of constitutive laws
	- the scope is initially limited to:
		- elastic 
		- isotropic von Mises elastoplastic model with linear hardening
		- plans to extend to other models in due course

###### Spatial parameterisation of properties
- homogeneous properties
- heterogeneous properties
	- spatial parameterisation is initially limited to:
		- manually discretised regions
		- zero-order mesh 
		- Gaussian radial basis functions 
		- plans to extend to other parameterisation techniques in due course

###### Stress equilibrium cost function metrics
- sensitivity based virtual fields
- equilibrium gap indicator
- force reconstruction error

###### Optimisation routines
- there is vast flexibility in how the above components (constitutive law, spatial parameterisation and cost metrics) are combined to perform identificatio
- various optimisers
- robustness 
- computational efficiency
- extensibility

##### Stress equilibrium assessment tools
- allow the user to evaluate equilibrium metrics for a given stress field


---

A non-exhaustive list of sub-modules is provided below including the required inputs, processing and outputs for each. Note that the specific organisation of sub-modules will change as the project develops, it most important that the key functionality and workflows are supported for the objectives above.


## Sub-Module: Stress reconstruction (unsure if best as a submodule or not?)
This sub-module computes full-field stresses from full-field strains using a defined constitutive laws and set of constitutive parameters. 

Two main approaches are implemented in this package:

.1. Radial return
- ref: "Computational Methods in Plasticity - de Souza Neto, EA, et al.
- give credit to Aleksander Marek (ref SBVF paper)

- theory (can maybe have separate theory pages to go with module)
	- "Computational Methods in Plasticity - de Souza Neto, EA, et al.
	- https://www.youtube.com/watch?v=InJMSYwV4P8&list=PLaDWa6xI4zefhDwPmTR1L6z8c8rtyRDVH&index=25
	- https://www.youtube.com/watch?v=1ydR6LFFbhA&list=PLaDWa6xI4zefhDwPmTR1L6z8c8rtyRDVH&index=26
	- www.youtube.com/watch?v=HJqECZXlEas&t=114s

.2. Stress reconstruction using NEML 
- ref neml2  https://github.com/applied-material-modeling/neml2
- give credit to Rory Spencer (ref paper)


### Inputs
- strain ()
	- convention of strain variable to be defined consistently.
	- options include:
		- 4d array: x,y,component,time 
		- multiple 2d, 3d arrays e.g.  strain.c11 is 3d array
	- think 4d array is probably best for vectorisation etc. Need to ensure consistent order of dimensions throughout.
	- number of components could possibly change in future
		- c11 / xx
		- c22 / yy
		- c12 / xy
		- von mises?
		- others?
	- several strain conventions exist. Two main possibilities for this work: tensorial and engineering strain. Changing the convention changes the definition (and hence values) of the strains - hence - any calculations must correspond accordingly
- material properties
	- constitutive law
		- this should be a string defined by the user that corresponds to a 'incarnation' of the material hardening calculation
		- over time new material laws can be added, so want this to be easily extensible
	- constitutive parameters
		- can be scalars
		- or if heterogeneous: can be 2d arrays which define value for each point in space (possibly in future - 3d if 3d space, or 4d if vary in time). For now stick to scalar or 2d.
		- the quantity and type of parameters should correspond to the defined constitutive law
		- should always contain: elastic_modulus, poissons_ratio
- options
	- should have defaults
	- should have checks 
	- stress recon. method (string)
		- radial return
			- doc should ref: "Computational Methods in Plasticity - de Souza Neto, EA, et al.
			- give credit to Aleksander Marek (ref SBVF paper)
		- neml2
			- https://github.com/applied-material-modeling/neml2
			- ref neml2
			- give credit to Rory Spencer (ref paper)
	- radial return stress tolerance
	- radial return max iterations
	- compute stiffness boolean
	- how to handle unloading for radial return
	- non linear geometry option
	- rotation matrix (for nl geometry)

### Outputs
- stress
	- convention of stress variable to be defined consistently.
		- should correspond with strain (see inputs)
		- 4d? order of dimensions? what components to include (von mises?)?
- yield map
	- optional 3d array of which datapoints (in space and time) have yielded#

### Workflow
###### Radial return

Key steps (rough outline of the process):
- Compute trial stress (elastic predictor)
- Compute effective stress 
- Compute flow stress from hardening
- Check yield criterion
- Perform optimisation (Newton–Raphson) to evaluate plastic multiplier for plastic points
- Update of plastic strain, elastic strain, stress, and internal variables
- Optionally: compute of the consistent tangent (Jacobian)

- The current implementation isn't very clearly structured
- Add comments to make above steps clearer
- Change variable names for clarity (below are suggestions only)
	- flyt to yield_criterion?
	- dlambda to plastic_multiplier_increment
	- PEEQ to equivalent_plastic_strain
	- Hbar to plastic_modulus
	- ksi to j2_stress_invariant
	- flytprime to yield_criterion_derivative
	- dksiDlam to d_j2_stress_invariance_d_plastic_multiplier

###### Stress reconstruction using NEML
- Leave as placeholder for now. Can implement at later date.


### Deliverables
- A module with the following sub-modules (note these are just suggested names and are not binding, use whatever structure makes most sense during development to achieve the desired functionality):
    - StressRecon.RadialReturn
    - StressRecon.Neml  (better name?)
- Full doc-strings and auto generated documentation for all modules and sub-modules
- A pragmatic suite of software tests including unit and regression tests for all modules and sub-modules
- Example/tutorial scripts demonstrating the functionality 
- A short markdown report comparing radial return and neml
- A short markdown theory document explaining core ideas 


## Sub-Module: Stress equilibrium evaluation

This sub-module computes various stress equilibrium metrics. These metrics can be used to assess how well a given stress field satisfies stress equilibrium. The existing metrics use various formulations of virtual fields to assess different aspects of the stress field. The key metrics as of now are: sensitivity based virtual fields (SBVFs), Equilibrium Gap Indicator (EGI) and Force Reconstruction Error (FRE).These metrics can be combined into a cost function for the purpose of mechanical property identification using the virtual fields method.

### Inputs
- stress
	- convention of stress variable to be defined consistently.
		- should correspond with strain (see inputs)
		- 4d? order of dimensions? what components to include (von mises?)?
- applied force / boundary conditions on specimen surface
	- need convention of boundary conditions
	- which surfaces are fixed, free, have a traction etc
- options
	- TBD

### Outputs
- equilibrium metrics
	- could be 3d maps with spatial-temporal data
	- could be single scalar value for optimisation
	- could be 2d map showing spatial variation throughout specimen

### Workflow
- will vary slightly for each metric

#### SBVF
- compute mesh for sbvf (see generateVirtualMesh.m)
- compute stress sensitivities (see computeStressSensitivity_dof.m)
	- should work with homogeneous parameters and het parameters (DOF level)
- compute SBVFs (see sensitivityVFs.m)
- evaluate SBVF cost function (see globalVirtualFieldCostFunction.m)


#### FRE
...

#### EGI
...



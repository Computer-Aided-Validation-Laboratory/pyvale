# Example VFM user flow 
The below flow is deliberating complex to capture the breadth of possibilities surrounding the identification procedure.

## Context
The user has performed experimental DIC testing on a welded joint. They know the elastic properties (elastic modulus and Poisson's ratio) and want to identify the spatially-varying maps of yield strength and hardening modulus assuming a linear hardening law.

## Inputs

| Group               | Field              | Type                            | Dimensions                                      |
|---------------------|--------------------|---------------------------------|-------------------------------------------------|
| Specimen Geometry   | x (mm)             | 2D array (float64)              | (y_size, x_size)                                |
|                     | y (mm)             | 2D array (float64)              | (y_size, x_size)                                |
|                     | thickness (mm)     | float                           | -                                               |
|                     | region of interest | 2D array (bool) OR polygon      | (y_size, x_size)                                |
|                     | area (mm^2/px)     | 2D array (float64)              | (y_size, x_size)                                |
| Boundary Conditions | edge conditions    | dict[edge, condition]           | -                                               |
|                     | force (N)          | 2D array (float64)              | (num_timesteps, num_components)                 |
| Strain Data         | strain field       | 4D array (float64)              | (num_timesteps, num_components, y_size, x_size) |

- Boundary Condition options
	- Edge: Lower, Upper, Left, Right
		- Lower is minimum y value
		- Left is minimum x value 
	- Conditions: Free, Fixed, Traction

## Outputs

| Group                             | Field                          | Type                                   | Dimensions                                      |
|-----------------------------------|--------------------------------|----------------------------------------|-------------------------------------------------|
| Parameter Maps                    | maps                           | 2D array (float64) per parameter       | (y_size, x_size)                                |
| Identified Stress                 | stress                         | 4D array (float64)                     | (num_timesteps, num_components, y_size, x_size) |
| Identification Diagnostics (TODO) | duration (seconds)             | float                                  |                                                 |
|                                   | number of iterations           | int                                    |                                                 |
|                                   | degree of freedom updates      | time-series data structure             |                                                 |
|                                   | per phase, per parameterisation|                                        |                                                 |
|                                   | cost function history          | 1D or 2D array (float over iterations) |                                                 |
|                                   | convergence statistics         | struct/dict                            |                                                 |

## Example 1

### Identification

#### Mechanical Properties

| Constitutive Law | Parameter         | Initial Value      | Lower Bound | Upper Bound |
|------------------|-------------------|--------------------|-------------|-------------|
| Linear Hardening | Elastic modulus   | scalar or 2D array | scalar      | scalar      |
|                  | Poisson's ratio   | scalar or 2D array | scalar      | scalar      |
|                  | Yield strength    | scalar or 2D array | scalar      | scalar      |
|                  | Hardening modulus | scalar or 2D array | scalar      | scalar      |


#### Phases

| Phase | Parameter         | Parameterisation | Options                             | Cost function                 | Weight | Optimiser      |
|-------|-------------------|------------------|-------------------------------------|-------------------------------|--------|----------------|
| 1     | Elastic modulus   | Known            |                                     | UDVF (uniform extension in y) | 1.0    | LM             |
|       | Poisson’s ratio   | Known            |                                     |                               |        |                |
|       | Yield strength    | Homogeneous      |                                     |                               |        |                |
|       | Hardening modulus | Homogeneous      |                                     |                               |        |                |
| 2     | Elastic modulus   | Known            |                                     | Slice-wise cost function      | 1.0    | fsolve (GB)    |
|       | Poisson’s ratio   | Known            |                                     |                               |        |                |
|       | Yield strength    | Slice-wise       | 30 slices                           |                               |        |                |
|       | Hardening modulus | Slice-wise       | 15 slices                           |                               |        |                |
| 3     | Elastic modulus   | Known            |                                     | SBVF                          | 1.0    | LM             |
|       | Poisson’s ratio   | Known            |                                     |                               |        |                |
|       | Yield strength    | Mesh             | 2 rows x 3 cols                     |                               |        |                |
|       |                   |                  | initial element order: 0            |                               |        |                |
|       |                   |                  | h-refinement: split/merge elements  |                               |        |                |
|       | Hardening modulus | Mesh             | 2 rows x 3 cols                     |                               |        |                |
|       |                   |                  | initial element order: 0            |                               |        |                |
|       |                   |                  | h-refinement: change element order  |                               |        |                |
| 4     | Elastic modulus   | Known            |                                     | EGI (n windows, window sizes) | 0.5    | Pattern search |
|       |                   |                  |                                     | FRE (npts per slice)          | 0.5    |                |
|       | Poisson’s ratio   | Known            |                                     |                               |        |                |
|       | Yield strength    | BFS              | bivariate                           |                               |        |                |
|       |                   |                  | initial number                      |                               |        |                |
|       | Hardening modulus | Known            | from previous phase                 |                               |        |                |
| 5     | Elastic modulus   | Known            |                                     | EGI                           | 0.5    | Pattern search |
|       |                   |                  |                                     | FRE                           | 0.5    |                |
|       | Poisson’s ratio   | Known            |                                     |                               |        |                |
|       | Yield strength    | Known            |                                     |                               |        |                |
|       | Hardening modulus | BFS              | phase 4 yield strength distribution |                               |        |                |
| 6     | Elastic modulus   | Known            |                                     | SBVF                          | 1.0    | LM (GB)        |
|       | Poisson’s ratio   | Known            |                                     |                               |        |                |
|       | Yield strength    | BFS              |                                     |                               |        |                |
|       | Hardening modulus | BFS              |                                     |                               |        |                |

- Parameterisations:
	- BFS: basis functions
- Cost Functions:
	- UDVF: Uniform Displacement Virtual Fields
	- EGI:
	- SBVF: sensitivity based virtual fields
	- FRE:
- Optimisers:
	- LM: Levenberg-Marquardt
	- GB: gradient based

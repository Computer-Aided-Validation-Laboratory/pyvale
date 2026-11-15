# `pyvale.designquality`: Design Specification v1.0

## 1. Motivation

`PyVale` is being developed into a general experimental design toolbox for fusion engineering. In addition to generating candidate designs and simulating their measurements, engineers need to assess whether a proposed design explores its input parameter space adequately, which parameters influence the resulting measurements, and whether those measurements contain sufficient information to identify the parameters of interest.

The proposed `pyvale.designquality` module will provide reusable numerical tools for **sampling quality**, **sensitivity analysis**, and **information quality / parameter identifiability**. These answer related but different questions:

1. **Sampling quality:** How well does a collection of input points explore a specified parameter space?
2. **Sensitivity:** How do measured or simulated responses depend on input parameters, locally or across a parameter distribution?
3. **Information quality:** How precisely, and how independently, could model parameters be inferred from the proposed measurements, under stated measurement noise assumptions?

These capabilities will ultimately support `pyvale.engine` workflows, sensor placement optimisation and other experimental design applications in the future. 

> **Design principle:** This specification defines scientific capabilities, example workflows, numerical verification and deliverables. Suggested filenames and APIs are illustrative, **not prescriptive**. The developer is expected to investigate suitable implementations and propose hanges while maintaining correctness, performance, clarity and the required examples/tests.

## 2. Aims and objectives

Develop a `pyvale.designquality` module that:

- Analyses existing sets of input design points without needing to generate or execute simulations.
- Calculates local sensitivities from supplied parameter perturbations and model responses, and analyses a user supplied sensitivity matrix.
- Provide both Morris and Sobol global sensitivity analyses using appropriate, explicitly documented sampling designs and parameter distributions.
- Calculates Fisher information and associated optimality / identifiability diagnostics for independent and correlated measurement uncertainties.
- Supports reusable, interpretable visualisations and textual/numerical outputs.
- Works primarily on NumPy arrays and can be used independently of other `pyvale` modules.
- Is verified through analytical tests and reusable regression tests based on the user-facing examples.

## 3. Scientific capabilities

The following describes expected functionality, not required class/function structures. Select methods and conventions that are mathematically correct and practical for engineers. Document input shapes, parameter conventions, units, required sample arrangements and assumptions - this is especially important for Numpy arrays where array axis meaning and data types are often opaque.

### 3.1 Sampling quality

**Inputs:** An array of sampled input parameter values, with parameter ranges or another explicit normalisation convention. A typical logical shape is `(samples, parameters)`.

**Required analyses:**

- Minimum pairwise separation / maximin quality (handling repeated points and the effect of parameter scaling).
- A discrepancy / space-filling quality measure suitable for designs normalised to the unit hypercube.
- Pairwise parameter correlation or equivalent projection diagnostics that expose poor coverage in low-dimensional projections.
- Meaningful diagnostics for degenerate and badly scaled designs.

Sampling quality is assessed in **input parameter space only**, not in the output response space. Explain which measures are invariant to permutation, what parameter scaling means, and why different quality criteria can rank designs differently. Do not invent one universal composite design-quality score.

Potential extension methods: nearest-neighbour distributions, coverage / fill distance, and additional discrepancy measures.

### 3.2 Local sensitivity analysis

**Inputs:** Parameter values and model responses for defined perturbations about a nominal state, or a user supplied sensitivity / Jacobian matrix. Logical shapes and axis interpretations must be documented (for example, `(measurements, parameters)` for a Jacobian).

**Required analyses:**

- Derivatives estimated from supplied finite perturbations, with appropriate one-sided and/or central finite differences and clear assumptions about perturbation size.
- Analysis of an existing Jacobian without recalculating derivatives.
- Normalised / dimensionless sensitivities, with careful handling of zero nominal values, differing units, and parameter scalings.
- Simple diagnostics and visualisations identifying influential parameters and measurements.

The module does not need to call a model repeatedly to obtain perturbations. It should be able to analyse previously calculated user supplied responses. 

### 3.3 Global sensitivity analysis

**Required:** Morris elementary effects and variance based Sobol indices.

- Morris: report a meaningful screening summary such as mean, mean absolute effect and spread of elementary effects, with clear parameter scaling and trajectory assumptions.
- Sobol: first order and total order indices; uncertainty / convergence estimates where reasonably supported by the chosen estimator; document estimator assumptions and computational cost.
- Support model responses that are already available in memory, and identify the required design/sample structure for each estimator. 
- State the parameter ranges/distributions and independence assumptions. Standard Sobol estimators typically assume independent inputs; correlated input global analysis is outside of scope for v1.0.
- Clearly distinguish local derivatives around a nominal state from global influence over a parameter distribution. This distinction must be evident in both the documentation and examples.

Implementations may leverage published algorithms for guidance or validation. Research `SALib` and its numerical/performance trade-offs, but prefer an appropriate in-house NumPy/SciPy implementation where it remains readable, testable and maintainable. If a new dependency is proposed, explain why it offers sufficient value to justify its maintenance cost. 

### 3.4 Information quality and identifiability

For a model with local sensitivity matrix `J` and measurement noise covariance `Sigma`, a central method is the local Fisher information matrix:

```text
F = J.T @ inv(Sigma) @ J
```

**Required analyses:**

- Support both independent observation uncertainties (diagonal covariance) and correlated observation uncertainties (full covariance).
- Information-based comparison criteria, including D-, A- and E-optimality, with precisely documented conventions (e.g. log-determinant, trace of inverse, minimum eigenvalue), units/normalisation, and behaviour when the information matrix is singular.
- Numerical rank, singular values / eigenvalues, condition diagnostics, and identification of indistinguishable parameter combinations.
- Approximate parameter covariance and parameter correlation where mathematically justified.
- A clear distinction between measurement noise covariance (input) and inferred parameter covariance (derived quantity).
- Correct treatment of mixed physical measurement units and explicit parameter scaling / reparameterisation conventions. Information metrics are not all invariant under parameter reparameterisation.

The first release is local information analysis under stated model linearisation and noise assumptions, not a full nonlinear or Bayesian inverse problem framework. A singular or indefinite measurement covariance must produce a clear diagnostic/error rather than a misleading result. Handling positive-semidefinite covariance through specialised methods is optional.

High sensitivity alone does not imply identifiability. Two parameters may both strongly influence responses yet be indistinguishable if their sensitivity columns are nearly dependent. Examples must demonstrate this distinction.

### 3.5 Outputs and visualisation

- Numerical outputs must be available directly for further Python analysis as numpy arrays or defined dataclasses.
- Reusable plotting utilities should cover the main use cases (e.g. sample distributions/projections, sensitivity plots/heatmaps, Fisher-information diagnostics and approximate confidence regions where valid).
- Plotting utilities should return Matplotlib figure and axes handles, allowing user modification. They should not implicitly call `show()` or save files as a side effect.
- Examples should write clear summaries and save figures to disk by default, with an obvious optional switch for interactive plotting.
- Document conventions for relative/absolute metrics, parameter names, response names, units, and axes.

## 4. Suggested module organisation

```text
src/pyvale/designquality/
        __init__.py
        sampling.py
        sensitivity.py
        information.py
        plotting.py
```

It may be reasonable to separate local and global sensitivity, or to split tests/visualisations further as the code evolves. The developer should choose a small, consistent public interface without excessive framework code. Avoid introducing interfaces, inheritance or configuration classes without a demonstrated use case.

**Potential dependencies:** NumPy, SciPy, Matplotlib; investigate SALib as a reference implementation and/or selective dependency for Morris/Sobol. The first implementation should be reasonably vectorised with NumPy and mindful of memory layout, axis ordering and unnecessary intermediate/copies of arrays, while prioritising correctness and maintainability. Cython, JAX, PyTorch or other accelerated implementations may be explored in future work.

Use PyVale's Python coding/style conventions and existing test, documentation and example practices.

## 5. User-facing example requirements

**Eight examples are required**: two focused examples for each of the three analysis families, followed by one integrated analytical example and one example based on existing PyVale thermomechanical simulation data.

Examples should be comprehensible in isolation, build in complexity, avoid long running physics calculations, and clearly state the question being asked, the numerical inputs, the physical interpretation, and the conclusion supported by the results.

### Common example conventions

`pyvale` examples for this module should be located in ./src/pyvale/examples/designquality/. Have a look at some of the other `pyvale` examples for other modules to learn how to format examples. 

- Suggested script names begin `exN_<description>.py` (or follow the established PyVale gallery convention as appropriate).
- Output directory: `pyvale-output/designquality/designquality_exN_NAME/` for each example, under PyVale's standard output root. `N` and `NAME` should match the example being run.
- Each example must save interpretable numerical results/summary and figures where appropriate. Figures default to non-interactive saving; a simple user-selectable option may enable interactive display.
- Use deterministic inputs and documented random seeds where sampling is random.
- Examples should be runnable independently and should flow narratively teaching the user a specific concept.
- The examples should be used as regression tests. Tests may invoke shared example functions to avoid launching a new process or creating figures unnecessarily. Assertions should be used to check numerical scientific outputs.
- Keep the core analytical examples fast and independent of MOOSE or any external solver. Existing example data may be loaded for the FE based case; no new physics solve is required.


### Example 1 — Sampling quality: 2D plate with a hole

**Purpose:** Introduce design space sampling quality using parameters from a recognisable PyVale mechanical problem.

**Suggested scenario:** Select two or three "plate with hole" inputs (for example Young's modulus, Poisson's ratio and/or an applied load) with meaningful bounds. Generate several designs of comparable sample count using existing sampling utilities or a small amount of example local code: random, Latin hypercube and Cartesian grid where sample counts and boundary conditions permit fair comparisons. You can find the "plate with hole" example [here]( https://github.com/Computer-Aided-Validation-Laboratory/pyvale-data/blob/main/Data/simulation/simcases/case17.i.)

**User workflow:**

1. Define input parameter ranges and generate candidate sample sets.
2. Normalise all sample sets consistently.
3. Compute minimum separation, discrepancy and pairwise projection/correlation diagnostics.
4. Save sample/projection figures and a numeric comparison table.
5. Explain why the rankings differ and why this says nothing by itself about measurement informativeness.


### Example 2 — Sampling quality: 3D monoblock parameter study

**Purpose:** Demonstrate that design quality assessment works in a more realistic, higher dimensional engineering parameter space and that attractive 2D projections do not guarantee full dimensional coverage.

**Suggested scenario:** Define physically sensible ranges for at least four monoblock thermal/structural parameters (for instance conductivity, heat-transfer or applied heat-flux parameters, expansion coefficient, elastic properties), referenced to [PyVale's 3D monoblock physics case](https://github.com/Computer-Aided-Validation-Laboratory/pyvale-data/blob/main/Data/simulation/simcases/case16.i).

**User workflow:**

1. Define the multi-parameter bounds, generate then compare at least two candidate designs with comparable numbers of sample points.
2. Analyse separation/discrepancy in the full normalised space and correlation/projections in selected parameter pairs.
3. Demonstrate at least one deliberately deficient design (e.g. duplicated samples or correlated parameter columns).
4. Save a numerical quality summary and representative pairwise projection figure(s).


### Example 3 — Local sensitivity: thermal expansion of a 1D rod

**Purpose:** Verify numerical finite difference derivatives against a closed form thermo-mechanical calculation and introduce dimensionless sensitivity.

**Suggested model:** An unconstrained uniform rod of initial length `L`, subjected to uniform temperature change `Delta_T`, with axial expansion `u = alpha * L * Delta_T`. An alternative comparably simple model may be used if its derivatives are exact and transparent.

**User workflow:**

1. Evaluate an analytical response at a nominal state and independently perturbed input parameters.
2. Compute finite difference sensitivities and compare with closed form derivatives.
3. Compute normalised sensitivities and explain dependence on nominal parameters and units.
4. Plot analytic versus numerical sensitivities or finite difference error against step size.


### Example 4 — Global sensitivity: combined axial and bending load on a cantilever

**Purpose:** Demonstrate Morris and Sobol global sensitivity analysis on a physically interpretable, multi-parameter, multi-response analytical mechanics model.

**Suggested model:** Small deflection Euler–Bernoulli cantilever with combined axial force and transverse end load. Parameters may include `E`, beam dimensions, length and loads. Candidate responses could include axial extension, tip transverse deflection and bending stress. Retain a domain in which the underlying linear beam assumptions remain valid.

**User workflow:**

1. Define parameter distributions/bounds, model responses and analysis specific sample designs.
2. Evaluate the analytical model with compatible Morris and Sobol samples (without building a generic sampler in the new module).
3. Report Morris summary statistics and Sobol first-order/total-order indices for at least two responses.
4. Compare the response-specific parameter rankings, explain interaction or nonlinearity where present, and distinguish global results from a local derivative at one nominal point.
5. Plot clearly labelled global sensitivity indices and their uncertainty/convergence diagnostics if provided.


### Example 5 — Information quality: identifiable and unidentifiable analytical systems

**Purpose:** Demonstrate Fisher information, uncertainty weighting and the difference between parameter sensitivity and identifiability.

**Suggested scenario:** A simple thermal, structural or coupled analytical model with two unknown parameters and several possible observations. Construct at least one configuration with independent sensitivity directions and one where measurement sensitivities are proportional (or nearly proportional).

**User workflow:**

1. Supply or calculate Jacobians for alternative measurement configurations.
2. Define independent measurement variances and construct Fisher information matrices.
3. Compare D-, A- and E-optimality (using clearly stated conventions), singular values, rank/conditioning and parameter correlations/covariances where valid.
4. Show a well-identified configuration and an unidentifiable/poorly identified configuration.
5. Save comparison tables and, where meaningful, local parameter confidence ellipse(s).


### Example 6 — Information quality: correlated measurement uncertainties

**Purpose:** Demonstrate how correlation and differing noise levels alter information quality.

**Suggested scenario:** Reuse the analytical model from Example 5 or an equally simple thermo-mechanical model. Compare several measurement arrangements under equal uncorrelated noise, unequal noise, and non-zero cross correlation.

**User workflow:**

1. Define multiple measurement covariance matrices, including a non-diagonal positive-definite matrix.
2. Compute information and conditional local parameter covariance for comparable measurement configurations.
3. Demonstrate how correlations and heterogeneous measurement units/precision affect the conclusions.
4. Save covariance/correlation or confidence region visualisations and a table of diagnostics.


### Example 7 — Integrated analytical experimental-design analysis

**Purpose:** Show a complete analysis-only progression, from an input DOE through response sensitivity to local information quality, in a single reproducible study.

**Suggested model:** An analytically solvable coupled thermo-mechanical experiment, such as a heated rod under axial loading with measurements of temperature and displacement, or a similarly simple model with independently distinguishable parameters and known derivatives. The example should be rich enough to investigate measurement choice but not require numerical PDE solving.

**User workflow:**

1. Define a candidate input design and evaluate sampling quality.
2. Evaluate analytical responses for the design and perturbed parameter states.
3. Compute local and global sensitivity information using appropriately constructed sample sets for each method..
4. Construct alternative candidate measurement sets and reasonable independent/correlated noise models.
5. Compare information/identifiability and explain which conclusions are local, global or dependent on measurement noise assumptions.
6. Save a compact collection of figures and a human readable summary of the design comparison.


### Example 8 — Existing PyVale 2D thermomechanical sensor-simulation results

**Purpose:** Demonstrate the new analysis toolbox on realistic  pre-computed PyVale data.

**Starting reference:** [Extended sensor simulation — Experiment simulation in 2D (`ex5a_expsim_thermmech2d`)](https://computer-aided-validation-laboratory.github.io/pyvale/examples/extsensorsim/ex5a_expsim_thermmech2d.html). This existing example reads nominal and perturbed 2D thermomechanical simulations and uses virtual temperature and displacement sensors.

**User workflow:**

1. Load relevant existing simulation/sensor response data as package data from the pyvale-data repo.
2. Verify and document the physical input parameters and exact perturbation values associated with each stored simulation. Check against the source data: approximately 10% perturbations in thermal conductivity and thermal expansion coefficient.
3. Extract responses with a documented ordering across sensor location, field component and time. Form suitable sensitivity information only for parameter perturbations that can actually be resolved independently from the available data.
4. Compare the sensitivity of temperature only and strain only measurements.
5. If the available data span enough independent perturbation directions, compare local identifiability under specified noise covariance. Otherwise perform the defensible lower-dimensional / directional information analysis, explain why a two-parameter Fisher analysis is not identifiable from the present dataset, and identify the additional input data that would be needed.
6. Save readable plots and numerical summaries, interpreting physics and measurement assumptions.

## 6. Verification and test strategy

The tests are a core deliverable and ensure the first key principle of our style guides: 1. Make it correct. Prefer numerical checks against analytical models, hand-calculated small cases, or explicitly cited/reference calculations. Independent implementations used only for verification should be short and understandable. `SALib` may serve as an additional comparison for Morris/Sobol, but matching its output is not by itself sufficient verification.

### 6.1 Analytical numerical tests

| Capability | Required checks |
| --- | --- |
| Input sampling geometry | Exact distances on small sets, duplicate points, unit-cube normalisation, changes of physical units |
| Discrepancy / projection diagnostics | Trusted small-design reference values and deliberately correlated/degenerate inputs |
| Finite-difference sensitivity | Analytic derivatives for rod and beam models; accuracy vs step size; one-sided/central schemes as implemented |
| Supplied Jacobians | Correct axes, ranks and consistency with derivative-based results |
| Normalised sensitivity | Scaling, parameter units, zero nominal responses/parameters and explicit undefined cases |
| Morris | Additive/linear analytical function with exact elementary effects; multiple trajectories; reproducibility |
| Sobol | Additive independent-input model with known first- and total-order indices; a nonlinear benchmark such as Ishigami for known reference indices |
| Fisher information | Hand-calculable matrices for independent and correlated covariance; expected covariance scaling |
| Optimality criteria | Known determinant/log-determinant, trace, eigenvalues and singular-matrix conventions |
| Identifiability | Full rank, exact rank deficiency, nearly dependent sensitivities, mixed scales and covariance validation |
| Plotting utilities | Construct expected figure/axes and plotted numerical data without interactive display |

When using sampled global estimators, document statistical tolerances, seed policies, sample sizes and the balance between CI runtime and estimator convergence. Test structural invariants and exact low-dimensional cases wherever possible; avoid flaky, overly tight stochastic assertions.

### 6.2 Regression testing using examples

- The **eight examples must be runnable through the test suite**, possibly by calling shared analysis functions instead of executing notebook/gallery rendering.
- Assert selected numeric outputs, shapes and scientific invariants for **every example**, not only exit status or image creation.
- Use stable reference datasets, reasonable relative/absolute tolerances, and fixed seeds when randomness is essential. Explain why tolerances are appropriate.
- Add separate focused tests for methods, invalid inputs and edge cases that examples do not cover.
- The core examples must be suitable for normal continuous integration; if large FE input data require a separate data download, mark or organise that regression test in accordance with existing PyVale data-test conventions, while keeping an always-available lightweight numerical test for the relevant analysis method.
- No pixel-level or image-comparison regression tests are required. Tests may verify that plotting functions create usable figure/axis handles and that examples save the expected files.
- Clear error handling is required for malformed shapes, missing/nonfinite values where unsupported, duplicated/invalid perturbation coordinates, incompatible Sobol/Morris sampling layouts, invalid covariance and impossible/undefined metrics. Avoid blanket silent fallbacks.

## 7. Documentation, integration and engineering practices

- Provide public API documentation/docstrings describing scientific assumptions, required shapes, units and return values.
- Provide concise user guidance on each family of analysis and cross-references to the eight examples.
- Keep API choices and layout flexible, but document any consequential design decisions and rationale, including the decision to add or avoid SALib as a dependency.
- Follow the CAV Lab / PyVale Python style guidance, keep code readable, and use vectorised NumPy where reasonable. Be attentive to NumPy axis ordering, memory layout and avoidable large intermediate arrays.
- Keep the analysis module independent of an engine implementation; later it may be wrapped by `IEngineStage` / `pyvale.engine` drivers or used as objective calculations. Liaise with the engine developers as the project matures to ensure sensible data boundaries.
- Prefer lossless, reproducible numerical outputs for reference data (for example NPZ with accompanying metadata where useful); do not introduce Python pickle for persistence.
- If an existing numerical library supplies a robust, fast, well-tested method, evaluate reuse versus in-house maintenance rather than duplicating it automatically. Internal implementation is preferred when the complexity is manageable and the method is testable.

## 8. Deliverables — in priority order

1. **Core numerical module:** `pyvale.designquality` with the three analysis families and a small documented, NumPy-oriented public API.
2. **Sampling-quality features and Examples 1–2**, with analytical tests and regression tests.
3. **Local sensitivity and global Morris/Sobol features and Examples 3–4**, with analytical and statistical verification.
4. **Information/identifiability features and Examples 5–6**, including diagonal and full measurement covariance, with small-matrix verification tests.
5. **Integrated analytical Example 7**, exercising all three families and demonstrating their different physical interpretations.
6. **Real-data Example 8**, grounded in the existing 2D thermal-mechanical results, with verified perturbation metadata and defensible dimensionality of the inferred sensitivities.
7. **Reusable Matplotlib visualisations**, used in the examples and returning figure/axes handles without implicit display or saving.
8. **Complete automated testing and documentation:** example based regression tests plus focused numerical and invalid-input tests; API documentation; example descriptions and interpretation; summary of dependency choices.

Deliverables can be developed in an order that makes engineering sense (for example, plotting alongside examples, tests alongside each method). The priorities express the intended project emphasis, not a mandated coding sequence. The project has **no fixed performance target** and this specification does **not** mandate a completion timeline.

## 9. Useful references

- Existing CAV Lab Python coding guide and othe relecant style guides: https://github.com/Computer-Aided-Validation-Laboratory/styleguides
- PyVale extended sensor simulation, 2D thermomechanical example: https://computer-aided-validation-laboratory.github.io/pyvale/examples/extsensorsim/ex5a_expsim_thermmech2d.html
- SALib documentation and method descriptions: https://salib.readthedocs.io/en/stable/
- SciPy QMC / discrepancy tools: https://docs.scipy.org/doc/scipy/reference/stats.qmc.html

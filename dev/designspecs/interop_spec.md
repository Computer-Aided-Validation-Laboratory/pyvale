# Design Specification for Pyvale orchestration

## Module name suggestions?

* `pyvale.orchestrate`
* `pyvale.pipeline`
* `pyvale.engine`
* `pyvale.workflow`

Suggestions welcome!

## Motivation

With the independent components of PyVale (image deformation, rendering, DIC, VVUQ) taking shape, the key challenge is integrating them into a coherent, automated workflow orchestration toolkit. Without this, the tool cannot deliver its central aim of enabling experimental design and optimised sensor placement for component level validation. Currently, these capabilities operate in isolation, with each component having its own inputs, outputs, assumptions and computational requirements. By Developing a HPC-compatible workflow, we will be able to maximise its impact in the experimental design space.

The workflow needs to provide a mechanism for passing information between the different stages without requiring significant manual intervention. In particular, the output from one stage should be directly usable as the input to the next, allowing a complete chain from simulation and image generation through to image analysis, strain extraction and uncertainty quantification. This will also provide a more consistent framework for assessing how changes to the experimental configuration, material response or sensor arrangement propagate through the overall validation process.

A further motivation is the computational cost associated with these workflows. A single end-to-end run can involve FEA (with rabbit-fem) or image rendering and deformation (riley), DIC processing (pyvale.dic) and post processing/validation (pyvale.vvuq??). When these calculations are repeated over a large parameter space, the cost can quickly become prohibitive. An HPC implementation provides an initial solution to this problem by allowing many independent workflow runs to be executed in parallel. However, a simple parameter sweep is unlikely to be the most efficient approach when only a relatively small number of simulations may be required to characterise the behaviour of the system. The aim itherefore to move beyond a purely brute-force approach towards an adaptive workflow. This should use information from previous simulations and experiments to determine where additional sampling is most useful. Surrogate models, such as Gaussian processes, provide one potential approach for doing this by approximating the response of the full workflow and identifying regions of the parameter space where further information would have the greatest value.

The resulting workflow should therefore provide a common framework for both straightforward high-throughput studies and more advanced adaptive experimental design. This will also establish a foundation for future integration of additional PyVale components without requiring the overall workflow architecture to be redesigned.

## Aims & Objectives

The aim will be to integrate Pyvale’s independently developed modules into a coherent, interoperable workflow despite their differing architectures and coding styles. The initial focus will be on establishing a consistent interface between the existing components, defining the data structures passed between them and identifying where additional wrappers or conversion steps are needed.

Initially, it may be worth attempting a naïve HPC deployment that chains components together within a large Slurm-driven parameter sweep, which wiill hopefully be a useful baseline for understanding the computational cost and behaviour of the complete workflow. Each workflow instance should be capable of running independently, with the relevant parameters passed through the full chain and the outputs stored in a consistent format. This will also allow the individual stages to be profiled and potential computational bottlenecks to be identified in a multimodule pyvale call.

The main goal will then be to develop a 'smarter' and more adaptive workflow. This will link image deformation/FEA, rendering, DIC, strain extraction and VVUQ into a single process, with machine learning methods used to determine which workflow runs are most informative. Gaussian process-based surrogate models are one potential approach, with relationships between input and output can to be approximated without requiring every possible configuration to be evaluated. completed passes can be used to update the surrogate model and determine where additional samples should be taken. This could be based on uncertainty in the surrogate prediction, expected changes in the output, or a combination of both. Using this kind of approach the workflow can progressively focus computational effort on regions of an extensive design space that are required to imrpove inderstanding of the validation problem.

The overall objectives are therefore to:

* Establish a common interface between the existing PyVale modules.
* Define consistent input and output data structures across the workflow.
* Demonstrate an end-to-end workflow running on HPC infrastructure and understand the bottlenecks.
* Develop an adaptive workflow capable of selecting informative simulations.
* Investigate the use of Gaussian process-based surrogate models for reducing the number of expensive workflow evaluations.
* Use the workflow to support optimisation of experimental design and sensor placement.

## Deliverables

* A clearly defined API and backend interoperability layer enabling consistent data exchange and function calls across  modules.

* Defined workflow data model describing the parameters, intermediate results and final outputs passed between image deformation/FEA, rendering, DIC, strain extraction and VVUQ.

* Naïve end-to-end HPC workflow (Slurm-based parameter sweep) demonstrating the module chain.

* Parameter studies demonstrating that multiple independent workflow instances can be executed and analysed consistently.

* Performance and scalability information for the naïve workflow, including identification of the main computational bottlenecks.

* An ML-driven workflow prototype using Gaussian processes, or similar, to identify the most informative workflow runs and reduce computational cost.

* A mechanism for updating the surrogate model as new workflow results become available and using this information to select subsequent simulations.

* A demonstration of the adaptive workflow on an experimental design or sensor placement problem, showing how the selected configurations compare with a conventional parameter sweep.

* Documentation describing the workflow architecture, interfaces, data structures and HPC deployment requirements, providing a basis for future development and integration of additional PyVale modules.

**### Example API**

some pseudo-code for a camera placement study? 

```python
import pyvale

workflow = pyvale.Workflow()

workflow.add_simulation(sim_data) # add some simdata stuff

workflow.add(
    pyvale.render(...)
    pyvale.dic(...)
    pyvale.vvuq(...)
)

workflow.add_parameter(
    "camera_position",
    candidate_camera_positions
)

best_camera = results.best(
    parameter="camera_position", # free parameters
    metric="vvuq.dic_correlation_error" # Optimize the ZNSSD
)

optimiser = pyvale.AdaptiveWorkflow(
    workflow=workflow,
    objective="vvuq.dic_correlation_error",
    method="gaussian_process"
)

result = optimiser.run(
    initial_samples=10,
    max_runs=50,
    ...
)
```
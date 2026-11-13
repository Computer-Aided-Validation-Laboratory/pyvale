.. _guide_mooseherder:

Physics simulation with Mooseherder
=======================================

Run a MOOSE input from Python, inspect its output, then repeat the solve for a
set of parameter choices. You do not need a MOOSE source checkout for the
packaged Rabbit FEM workflows.

Before you start
--------------------

MOOSE is the finite element framework. Rabbit FEM supplies a precompiled
MOOSE application for thermo-mechanical problems. A MOOSE application selects
which physics objects are available; Rabbit packages those needed for its
supported problems. Mooseherder is PyVale's runner and orchestration layer: it
stages inputs, launches meshing/solving and returns paths to results.

The contents of a ``.i`` file define the physics. Included files can hold shared
model definitions. A Gmsh ``.geo`` file instead defines geometry and mesh
construction; named mesh regions let the solver select material blocks and
boundaries but do not assign material laws themselves.

1. Run a small model
------------------------

Start with the registered thermo-mechanical cube. Its native units are metres,
pascals, seconds and degrees Celsius; strains are dimensionless.

.. code-block:: python

   from pathlib import Path
   import pyvale.data as dataset
   import pyvale.mooseherder as mh

   input_file = dataset.element_case_input_path(dataset.EElemTest.HEX8)
   output_dir = mh.get_next_run_path(
       Path.cwd() / "pyvale-output/guide_solve"
   )
   result = mh.MooseRunner().run(input_file, output_dir=output_dir)
   print(result.output_file)
   print(result.output_dir / "run.log")


``RunResult.output_file`` identifies the output, while ``run.log`` records the
solver run. The output folder is newly allocated so a repeated run does not
replace an earlier experiment.

2. Find and inspect input files
-----------------------------------

PyVale's data API downloads registered ``.i``, include and geometry files on
demand. Rabbit also has its own installed example library. These are distinct
collections; do not assume case names, refinements or results are identical.
Read the selected deck and its includes before changing parameters.

.. code-block:: python

   import pyvale.data as dataset
   import rabbit.sims as rabbit_sims

   # PyVale's registered inputs: downloaded on demand and cached.
   print(dataset.sim_case_input_file_path(16))
   print(dataset.sim_case_gmsh_file_path(16))
   print(dataset.sim_case_directory_path())  # fetch the full input tree

   # Rabbit's own installed simulation library: a separate source of inputs.
   print(rabbit_sims.monoblock_input_path())
   print(rabbit_sims.monoblock_geo_path())


A MOOSE input normally names the mesh, unknowns, materials, equations, boundary
conditions, execution settings and outputs. Upstream MOOSE tutorials explain
these concepts. Some upstream examples need a custom application, so they are
not all runnable with Rabbit unchanged.

3. Repeat the solve with different inputs
---------------------------------------------

Use native parameter names exposed by the chosen input. The example cube
exposes ``EMod`` in pascals. ``SimStep.name_subdir`` names a step and the matching
key in each case's parameter dictionary. Each list entry below is one case;
``solve`` names its single step. Returned results follow case order, then step
order within each case.

.. code-block:: python

   from pathlib import Path
   import pyvale.data as dataset
   import pyvale.mooseherder as mh

   input_file = dataset.element_case_input_path(dataset.EElemTest.HEX8)
   steps = [mh.SimStep("solve", mh.MooseRunner(), input_file)]
   herd = mh.MooseHerd(
       steps,
       output_dir=Path.cwd() / "pyvale-output/guide_sweep",
       parallel_runners=2,
   )
   results = herd.run([
       {"solve": {"EMod": 100e9}},
       {"solve": {"EMod": 120e9}},
   ])
   for case_results in results:
       print(case_results[-1].output_file)


When the mesh must be generated first, put a ``GmshRunner`` step before the
solver and pass its output with ``mesh_from``. See the existing mesh-and-solve
tutorial rather than manually copying a mesh between case folders.

Analyse the result
------------------

Use the solver log and output fields to establish whether each case completed
as intended. Then assess convergence, units, boundary conditions and mesh
sensitivity before treating the result as the truth for a virtual experiment.
Preserve the staged input snapshot, parameter choices and logs with the results.

Compare cases through their recorded parameter choices. Completed archived
sweeps have ``sweep.json`` manifests that ``SweepLoader`` can reopen, preserving
the relationship between inputs and outputs.

``parallel_runners`` runs independent cases concurrently; it is separate from
MPI ranks and threads within one solve. MPI requires an MPI-capable executable
and suitable input. Allocate resources across these layers so that concurrent
cases do not each request all available cores.

Continue with these examples and guides:

:doc:`Mooseherder tutorials </examples/examples_mooseherder>` · `MOOSE tutorials <https://mooseframework.inl.gov/getting_started/examples_and_tutorials/index.html>`_ · `Rabbit FEM <https://github.com/Computer-Aided-Validation-Laboratory/rabbit-fem>`_

:doc:`Geometry and meshing <guide_meshing>` · :doc:`Working with SimData <guide_simdata>`

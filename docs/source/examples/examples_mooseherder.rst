.. _examples_mooseherder:

Mooseherder
===========

Packaged Rabbit FEM and Python Gmsh provide the default runners. Start with a
single SMP solve and SimData exploration, then progress through runner options,
mesh generation, simulation chains and optimisation. All examples run headlessly
with packaged defaults. MPI requires explicit opt-in; custom executables are
optional.

Getting started
---------------

.. toctree::
   :maxdepth: 1

   mooseherder/ex01a_rabbit_quickstart
   mooseherder/ex01b_inspect_simdata

MOOSE runner
------------

.. toctree::
   :maxdepth: 1

   mooseherder/ex02a_moose_cli_parameters
   mooseherder/ex02b_moose_included_inputs
   mooseherder/ex02c_moose_multiple_inputs
   mooseherder/ex02d_modify_input_text
   mooseherder/ex02e_mpi_solve
   mooseherder/ex02f_custom_moose_installation

Gmsh runner
-----------

.. toctree::
   :maxdepth: 1

   mooseherder/ex03a_gmsh_geo
   mooseherder/ex03b_gmsh_python
   mooseherder/ex03c_custom_gmsh_installation

Simulation chains and sweeps
----------------------------

.. toctree::
   :maxdepth: 1

   mooseherder/ex04a_mesh_and_solve
   mooseherder/ex04b_sequential_sweep
   mooseherder/ex04c_parallel_sweep
   mooseherder/ex04d_read_sweep_results

Optimisation and FEMU
---------------------

.. toctree::
   :maxdepth: 1

   mooseherder/ex05a_reuse_particle_directories
   mooseherder/ex05b_pymoo_femu

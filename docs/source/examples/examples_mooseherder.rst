.. _examples_mooseherder:

Mooseherder
===========

Packaged Rabbit FEM and Python Gmsh provide the default runners. Start with a
single SMP solve and SimData exploration, then progress through runner options,
mesh generation, simulation chains and optimisation. All examples run headlessly
with packaged defaults. MPI requires explicit opt-in; custom executables are
optional. Outputs are saved in ``pyvale-output/mooseherder_*`` subdirectories.

Getting started
---------------

.. toctree::
   :maxdepth: 1

   mooseherder/ex01a_quickstart_rabbitfem
   mooseherder/ex01b_quickstart_simdata

MOOSE runner
------------

.. toctree::
   :maxdepth: 1

   mooseherder/ex02a_moose_cli_parameters
   mooseherder/ex02b_moose_included_inputs
   mooseherder/ex02c_moose_multiple_inputs
   mooseherder/ex02d_moose_modify_input_text
   mooseherder/ex02e_moose_mpi_solve
   mooseherder/ex02f_moose_custom_installation

Gmsh runner
-----------

.. toctree::
   :maxdepth: 1

   mooseherder/ex03a_gmsh_geo
   mooseherder/ex03b_gmsh_python
   mooseherder/ex03c_gmsh_custom_installation
   mooseherder/ex03d_gmsh_box3d
   mooseherder/ex03e_gmsh_monoblock

Sim Sweep
---------

.. toctree::
   :maxdepth: 1

   mooseherder/ex04a_simsweep_mesh_and_solve
   mooseherder/ex04b_simsweep_sequential_sweep
   mooseherder/ex04c_simsweep_parallel_sweep
   mooseherder/ex04d_simsweep_read_sweep_results

FEMU
----

.. toctree::
   :maxdepth: 1

   mooseherder/ex05a_femu

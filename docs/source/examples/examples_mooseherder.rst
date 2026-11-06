.. _examples_mooseherder:

Mooseherder
===========

Packaged Rabbit FEM and Python Gmsh provide the default runners. Start with a
single SMP solve, then explore CLI parameters, included decks, meshing and sweeps.
All examples run headlessly with packaged defaults. The MPI example requires an
explicit opt-in before launching multiple ranks; custom executables are optional.

.. toctree::
   :maxdepth: 1

   mooseherder/ex01_rabbit_quickstart
   mooseherder/ex02_moose_cli_parameters
   mooseherder/ex03_moose_included_inputs
   mooseherder/ex04_moose_multiple_inputs
   mooseherder/ex05_gmsh_geo
   mooseherder/ex06_gmsh_python
   mooseherder/ex07_mesh_and_solve
   mooseherder/ex08_sequential_sweep
   mooseherder/ex09_parallel_sweep
   mooseherder/ex10_mpi_solve
   mooseherder/ex11_custom_installations
   mooseherder/ex12_read_sweep_results
   mooseherder/ex13_modify_input_text

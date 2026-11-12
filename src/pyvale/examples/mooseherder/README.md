Mooseherder examples
===================

All examples default to serial/SMP execution and write into ``mooseherder_*``
subdirectories beneath ``pyvale-output``.
The filenames group related workflows; letters give the order within each group.
Examples run independently and print input paths fetched from the PyVale data
repository and cached locally.

Getting started
---------------

* ``ex01a_quickstart_rabbitfem.py``
* ``ex01b_quickstart_simdata.py``

MOOSE runner
------------

* ``ex02a_moose_cli_parameters.py``
* ``ex02b_moose_included_inputs.py``
* ``ex02c_moose_multiple_inputs.py``
* ``ex02d_moose_modify_input_text.py``
* ``ex02e_moose_mpi_solve.py``
* ``ex02f_moose_custom_installation.py``

Gmsh runner
-----------

* ``ex03a_gmsh_geo.py``
* ``ex03b_gmsh_python.py``
* ``ex03c_gmsh_custom_installation.py``

Sim Sweep
---------

* ``ex04a_simsweep_mesh_and_solve.py``
* ``ex04b_simsweep_sequential_sweep.py``
* ``ex04c_simsweep_parallel_sweep.py``
* ``ex04d_simsweep_read_sweep_results.py``

FEMU
----

* ``ex05a_femu.py``

Downloaded simulations
--------------------

* ``case00_HEX8.i`` is a 3D thermo-mechanical cube using m, Pa, seconds and
  degrees Celsius. Fetch it with
  ``dataset.element_case_input_path(dataset.EElemTest.HEX8)``. The getting-started
  and MOOSE runner groups use this case, as do the parallel sweep and sweep reader examples.
* ``case17.i`` and ``case17.geo`` describe a 2D plate with a hole using weak plane
  stress, mm, MPa and seconds. Fetch them with
  ``dataset.sim_case_input_file_path(17)`` and
  ``dataset.sim_case_gmsh_file_path(17)``. The Gmsh group, mesh/solve chain,
  sequential sweep and FEMU examples use this case. ``case17_mesh.py`` lives
  in the simulation input data set and supplies an alternative Python Gmsh mesh.

The MPI example requires an MPI-capable executable and launcher when ``n_tasks``
is greater than one. Custom installations are optional. Specialised analytic
fixtures are test-only data.

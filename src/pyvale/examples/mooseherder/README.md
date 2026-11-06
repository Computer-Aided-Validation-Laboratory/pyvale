Mooseherder examples
===================

Run packaged Rabbit and Gmsh, pass native parameters, stage included decks, and
chain isolated mesh/solve steps into reproducible sweeps. All tutorials default
to serial/SMP execution and write beneath ``pyvale-output``. Each tutorial prints
the installed input paths so its original .i scripts are easy to find.

The tutorials reuse ``pyvale/data/simulation/simcases``:

* ``case00_HEX8.i``: 3D thermo-mechanical cube, m/Pa/seconds/degrees Celsius;
  get its path with ``dataset.element_case_input_path(dataset.EElemTest.HEX8)``.
  Tutorials 1-4, 9-10 and 12-14 use this generated-mesh case.
* ``case17.i`` and ``case17.geo``: 2D plate with a hole, weak plane stress,
  mm/MPa/seconds; use ``dataset.sim_case_input_file_path(17)`` and
  ``dataset.sim_case_gmsh_file_path(17)``. Tutorials 5-8, 11 and 15 use this case.
  ``case17_mesh.py`` lives beside the .i file and provides an alternative Python
  Gmsh mesh of the same geometry with matching boundary names.

The MPI tutorial requires an explicit flag before launching multiple ranks.
Custom installations are optional. Tutorial 13 demonstrates deliberate textual
editing. The final tutorials cover reusable particle directories and
noise-normalised plate-with-hole FEMU with pymoo, identifying modulus and
Poisson's ratio. Specialised analytic fixtures are test-only data.

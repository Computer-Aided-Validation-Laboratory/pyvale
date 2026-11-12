.. _examples_sensorsim_extended:

Extended Sensor Simulation
==========================

Extend the workflows in :doc:`examples_basics_sensorsim` with your own
simulation data, mesh-free sampling and configurable error models. The examples
progress from individual field measurements to repeated thermo-mechanical
experiments.

Simulation data and sampling
----------------------------

Supply your own simulation data or sample fields without a mesh.

.. toctree::
   :maxdepth: 1

   extsensorsim/ex1_byosimdata
   extsensorsim/ex2_meshfreesensors

Scalar, vector and tensor fields
--------------------------------

Compare sensor configuration for each field type in 2D and 3D.

.. toctree::
   :maxdepth: 1

   extsensorsim/ex3a_scal2d
   extsensorsim/ex3b_scal3d
   extsensorsim/ex3c_vec2d
   extsensorsim/ex3d_vec3d
   extsensorsim/ex3e_tens2d
   extsensorsim/ex3f_tens3d

Sensor error models
-------------------

Add basic, field-based, calibration and spatial-averaging errors.

.. toctree::
   :maxdepth: 1

   extsensorsim/ex4a_basicerrs_scal2d
   extsensorsim/ex4b_fielderrs_scal3d
   extsensorsim/ex4c_angleerrs_vec2d
   extsensorsim/ex4d_fieldlockerrs_vec3d
   extsensorsim/ex4e_chainfielderrs_vec2d
   extsensorsim/ex4f_caliberrs_scal2d
   extsensorsim/ex4g_spatavgerrs_scal2d

Thermo-mechanical experiments
-----------------------------

Combine sensor arrays and repeat experiments on coupled simulations.

.. toctree::
   :maxdepth: 1

   extsensorsim/ex5a_expsim_thermmech2d
   extsensorsim/ex5b_expsim_thermmech3d

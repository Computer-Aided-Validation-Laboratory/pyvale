Validation examples
===================

Follow the validation workflow from explicit NumPy observations to registered
sensor data, epistemic scenarios and three-dimensional simulation overlays.
Examples save figures under their own ``pyvale-output/valid_ex*`` directories
and disable interactive plotting by default.

Start with ``ex0`` for the complete workflow, then follow the tutorials in order.
Examples 1–5 introduce the data model, spatial observables, epistemic scenarios,
calibration offsets and time. Example 6 explains diagnostics; example 7 connects
SensorSim and field data. Example 8 uses packaged 3D tensor fields, and example 9
runs a MOOSE cube sweep with 10% parameter perturbations. Example 10 reproduces
the Pulse25X analysis from
`Horne-Jones et al. (2026) <https://arxiv.org/abs/2605.11886>`_.

.. toctree::
   :maxdepth: 1

   valid/ex0_val_quickstart
   valid/ex1_val_metric
   valid/ex2_val_steady
   valid/ex3_val_epistemic
   valid/ex4_val_bounds
   valid/ex5_val_transient
   valid/ex6_val_diagnostics
   valid/ex7_val_field
   valid/ex8_val_cube_tensor
   valid/ex9_val_cube_sweep
   valid/ex10_val_pulse25x

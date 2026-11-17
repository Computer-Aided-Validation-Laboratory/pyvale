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

   valid/ex00_valid_quickstart
   valid/ex01_valid_metric
   valid/ex02_valid_steady
   valid/ex03_valid_epistemic
   valid/ex04_valid_bounds
   valid/ex05_valid_transient
   valid/ex06_valid_diagnostics
   valid/ex07_valid_field
   valid/ex08_valid_cube_tensor
   valid/ex09_valid_cube_sweep
   valid/ex10_valid_pulse25x

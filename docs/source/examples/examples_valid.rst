Validation examples
===================

Follow the validation workflow from explicit NumPy observations to registered
sensor data, epistemic scenarios and three-dimensional simulation overlays.
Examples save figures under their own ``pyvale-output/valid_ex*`` directories
and disable interactive plotting by default.

The plate examples and ``ex7`` use packaged data. ``ex8`` runs a MOOSE cube sweep
with the packaged Rabbit application, perturbing heat flux and thermal expansion
by 10%. ``ex9`` reproduces the full Pulse25X point-sensor analysis from
`Horne-Jones et al. (2026) <https://arxiv.org/abs/2605.11886>`_, with a frozen
reference table and explicit notes on the original preprocessing choices.

.. toctree::
   :maxdepth: 1

   valid/ex1_val_metric
   valid/ex2_val_steady
   valid/ex3_val_epistemic
   valid/ex4_val_bounds
   valid/ex5_val_transient
   valid/ex6_val_field
   valid/ex7_val_cube_tensor
   valid/ex8_val_cube_sweep
   valid/ex9_val_pulse25x

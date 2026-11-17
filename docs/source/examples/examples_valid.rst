Validation examples
===================

Run the tutorials from the project root with
``python scripts/run_all_valid_examples.py``. Add ``--data-dir`` for a local
unpublished data checkout. Figures and reference tables are saved beneath
``pyvale-output/valid_ex*``; interactive windows are disabled by default.

The unchanged quickstart is followed by analytic metrics, steady spatial bias,
transient drift, transferable bounds, nuisance views and profiling. Later examples
separate coherent offsets from confidence bounds, sample 3D fields, run a MOOSE
parameter sweep and reproduce the Pulse25X analysis. The example README contains
the feature coverage matrix and the equations defining each synthetic bias.

.. toctree::
   :maxdepth: 1

   valid/ex00_valid_quickstart
   valid/ex01_valid_metric
   valid/ex02_valid_steady
   valid/ex03_valid_transient
   valid/ex04_valid_transferable_bounds
   valid/ex05_valid_epistemic_axes
   valid/ex06_valid_nuisance_profiles
   valid/ex07_valid_offsets
   valid/ex08_valid_sampling
   valid/ex09_valid_cube_sweep
   valid/ex10_valid_pulse25x

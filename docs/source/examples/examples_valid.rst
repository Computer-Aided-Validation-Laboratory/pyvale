Validation examples
===================

Run the tutorials from the project root with
``python scripts/run_all_valid_examples.py``. Add ``--data-dir`` for a local
unpublished data checkout. Figures and validation archives are saved beneath
``pyvale-output/valid_ex*``; interactive windows are disabled by default.

The quickstart is followed by scalar metrics, plate validation, epistemic
bounds, nuisance profiling and coherent offsets. Later examples sample fields
at sensors, run a MOOSE sweep, reproduce Pulse25X and compare serial and parallel
IR camera validation. The example README gives an overview of each workflow.

.. toctree::
   :maxdepth: 1

   valid/ex00_valid_quickstart
   valid/ex01_valid_scalar_metrics
   valid/ex02_valid_steady_state
   valid/ex03_valid_transient
   valid/ex04_valid_epistemic_bound_metrics
   valid/ex05_valid_nuisance_ecdf_profiles
   valid/ex06_valid_nuisance_metric_profiles
   valid/ex07_valid_epistemic_offsets
   valid/ex08_valid_sim_sensor_sampling
   valid/ex09_valid_sim_sweep_sampling
   valid/ex10_valid_pulse25x_benchmark
   valid/ex11_valid_sensorsim_import
   valid/ex12_parallel_valid_ir_camera

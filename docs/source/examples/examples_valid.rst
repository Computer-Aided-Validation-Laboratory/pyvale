.. _examples_validation:

Validation examples
===================

Run the tutorials from the project root with
``python scripts/run_all_valid_examples.py``. Add ``--data-dir`` for a local
unpublished data checkout. Figures and validation archives are saved beneath
``pyvale-output/valid_ex*``; interactive windows are disabled by default.

Getting started
---------------

Build a validation case, compare scalar metrics, and analyse steady-state and
transient data.

.. toctree::
   :maxdepth: 1

   validation/ex00_valid_quickstart
   validation/ex01_valid_scalar_metrics
   validation/ex02_valid_steady_state
   validation/ex03_valid_transient

Epistemic uncertainty
---------------------

Compare distribution bounds, profile nuisance uncertainty, and apply coherent
epistemic offsets.

.. toctree::
   :maxdepth: 1

   validation/ex04_valid_epistemic_bound_metrics
   validation/ex05_valid_nuisance_ecdf_profiles
   validation/ex06_valid_nuisance_metric_profiles
   validation/ex07_valid_epistemic_offsets

Data preparation and benchmarks
-------------------------------

Sample simulation fields at sensors, analyse a MOOSE sweep, reproduce the
Pulse25X benchmark, import SensorSim measurements, and run parallel validation.

.. toctree::
   :maxdepth: 1

   validation/ex08_valid_sim_sensor_sampling
   validation/ex09_valid_sim_sweep_sampling
   validation/ex10_valid_pulse25x_benchmark
   validation/ex11_valid_sensorsim_import
   validation/ex12_parallel_valid_ir_camera

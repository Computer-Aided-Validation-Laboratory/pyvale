Validation examples
===================

All validation examples save diagnostic figures in ``pyvale-output``. They use
prepared data and do not run MOOSE. ``ex6`` loads a packaged Exodus file to
show actual ``IField`` extraction and the SensorSim-to-``ExpData`` bridge.

.. list-table::
   :header-rows: 1

   * - Script
     - Diagnostic
   * - ``ex1_val_metric.py``
     - Standalone metrics and empirical CDFs
   * - ``ex2_val_steady.py``
     - Plate thermocouple CDFs and AVM map
   * - ``ex3_val_epistemic.py``
     - Sampled epistemic CDF envelope and metric distribution
   * - ``ex4_val_bounds.py``
     - Offset CDF envelope and directional MAVM map
   * - ``ex5_val_transient.py``
     - Metric versus time and selected CDFs
   * - ``ex6_val_field.py``
     - Strain gauge grid map and temperature time series

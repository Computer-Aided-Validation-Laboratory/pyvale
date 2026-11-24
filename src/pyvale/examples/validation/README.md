Validation examples
===================

Run from the project root, using the project Python environment:

```bash
python scripts/run_all_valid_examples.py
```

For an unpublished local data checkout, add `--data-dir ../pyvale-data/Data`.
Figures and validation archives are written to the git-ignored
`pyvale-output/valid_ex*` directories. Interactive plotting is opt-in in the
examples that expose a `SHOW_PLOTS` setting.

| Example | What it shows |
| --- | --- |
| `ex00_valid_quickstart` | End-to-end transient plate validation with ECDFs, a MAVM time history and mesh overlays. |
| `ex01_valid_scalar_metrics` | An analytic scalar case for comparing metric behaviour and integration areas. |
| `ex02_valid_steady_state` | Temperature and strain at a single steady state, with spatial AVM and MAVM support plots. |
| `ex03_valid_transient` | Temperature and strain over time, with ECDF snapshots and AVM/MAVM histories. |
| `ex04_valid_epistemic_bound_metrics` | AVM and MAVM calculated from transferable ECDF bounds versus original sim–exp case pairs. |
| `ex05_valid_nuisance_ecdf_profiles` | Mean, median and 25th-percentile nuisance profiles against all-case ECDF envelopes, with KS and MAVM overlays. |
| `ex06_valid_nuisance_metric_profiles` | Metric extrema and their controlling ECDF pairs, MAVM support ranking and NPZ round-trips. |
| `ex07_valid_epistemic_offsets` | Nominal and offset ECDFs with outer experimental confidence limits and a MAVM support envelope. |
| `ex08_valid_sim_sensor_sampling` | Scalar, vector, tensor and global sampling with synthetic experimental acquisitions. |
| `ex09_valid_sim_sweep_sampling` | Run a MOOSE parameter sweep, sample each run at virtual sensors and compare strain histories. |
| `ex10_valid_pulse25x_benchmark` | Reproduce the Pulse25X benchmark's MAVM endpoint comparisons and diagnostics. |
| `ex11_valid_sensorsim_import` | Run SensorSim ex3 and validate its exported temperature repeats at the stored nominal positions and times. |
| `ex12_parallel_valid_ir_camera` | Compare serial and four-worker AVM on a 640×480 plate image with noisy frames and emissivity sweeps. |

Recognisable discrepancy patterns
---------------------------------

Examples 02 and 03 construct new experimental arrays in memory from the sim
sample grid. They do not modify packaged data. This makes their AVM equal to the
imposed translation rather than a random-looking difference between sampling
realisations:

- ex02: `bias = 3*x_fraction` degC, rising from zero at the left sensor edge to
  3 degC at the right edge.
- ex03: `bias = (2 + 3*x_fraction)*time_fraction` degC, zero initially and rising
  to 2–5 degC at the final time.
- ex05: asymmetric nuisance shifts include one 178 degC nuisance shift, making the
  sensitivity of means and pointwise quartiles clear.
- ex06: position/time-dependent bias and prescribed nuisance spreads produce
  structured, sometimes piecewise metric trends as controlling cases change.
  KS and CvM can saturate while physical distances continue increasing.
- ex08: sensor-frame bias grows with time and x, with symmetric repeat offsets
  much smaller than the final discrepancy.
- ex09: metric structure comes from real heat-flux and thermal-expansion changes;
  seeded measurement noise is deliberately small.

MAVM time histories can show d+ and d− together. Experimental confidence shifts can
produce a nonzero floor even when the prescribed mean bias is zero. Epistemic
samples are never pooled into aleatory observations. Mean/median/quantile curves
are labelled as synthetic summaries; ranked nuisance cases retain actual joint
sample indices.

Ex10 retains the numerical conventions of Horne-Jones et al. (2026),
https://arxiv.org/abs/2605.11886, including its frozen reference checks. The general
bounds/profiling algorithms in ex04–ex06 are deliberately distinguished from that
paper-specific endpoint calculation. The larger performance benchmark is in
`scripts/perf_valid.py`.

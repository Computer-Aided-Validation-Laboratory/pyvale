Validation examples
===================

Run from the project root, using the project Python environment:

```bash
python scripts/run_all_valid_examples.py
```

For an unpublished local data checkout, add `--data-dir ../pyvale-data/Data`.
Figures and numerical reference tables are written to the git-ignored
`pyvale-output/valid_ex*` directories. Interactive plotting is opt-in through
`SHOW_PLOTS` in each tutorial. The gallery is limited to ex00–ex10; ex00 is
unchanged by this reorganisation.

| Example | Physical regime | Main features |
| --- | --- | --- |
| ex00_valid_quickstart | Existing transient case | Original end-to-end quickstart, unchanged |
| ex01_valid_metric | Steady global scalar | All metrics, deterministic/aleatory sample counts, paired versus distributional comparisons, metric classifications, composable overlays, constant normalisation and U-pooling |
| ex02_valid_steady | Steady 2D plate | Multiple observables/components, registered spatial points and metric maps with a prescribed edge bias |
| ex03_valid_transient | Transient 2D plate | Time and repeat axes, drift histories, spatial maps and explicit snapshot selection |
| ex04_valid_transferable_bounds | Steady transferable alternatives on both sources | ECDF bounds, AVM/MAVM over boundaries versus original scenario comparisons, crossing curves and provenance |
| ex05_valid_epistemic_axes | Steady transferable/nuisance alternatives on both sources | All-case envelopes, fixed/subset nuisance selections, mean/median/quartile reductions, selected joint curves and reference-quantile normalisation |
| ex06_valid_nuisance_profiles | Transient, both epistemic axes | All-metric nuisance extrema, outer shifted supports, retained transferable axes, original indices and ranking coherent nuisance cases by total or additional width |
| ex07_valid_offsets | Transient spatial observations | Asymmetric coherent offsets, nominal/lower/upper selections and experimental confidence construction, kept distinct from epistemic envelopes |
| ex08_valid_sampling | Transient 3D cube | Reusable scalar/vector/tensor fields, rotated averaging footprints, globals, single/sweep sampling, SensorSim-shaped acquisitions, ExpData assembly, repeated cases and derived backgrounds |
| ex09_valid_cube_sweep | Transient 3D MOOSE sweep | Real +10% parameter perturbations, transferable alternatives, tensor histories, stored/derived backgrounds and engine execution options |
| ex10_valid_pulse25x | Existing experimental dataset | Frozen paper reproduction, legacy endpoint calculation, provenance and numerical parity |

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
- ex05: asymmetric nuisance shifts include one 18 degC alternative, making the
  sensitivity of means and pointwise quartiles clear.
- ex06: position/time-dependent bias and prescribed nuisance spreads produce
  structured, sometimes piecewise metric trends as controlling cases change.
  KS and CvM can saturate while physical distances continue increasing.
- ex08: sensor-frame bias grows with time and x, with symmetric repeat offsets
  much smaller than the final discrepancy.
- ex09: metric structure comes from real heat-flux and thermal-expansion changes;
  seeded measurement noise is deliberately small.

MAVM d+ and d− are plotted separately. Its experimental confidence shifts can
produce a nonzero floor even when the prescribed mean bias is zero. Epistemic
samples are never pooled into aleatory observations. Mean/median/quantile curves
are labelled as synthetic summaries; ranked nuisance cases retain actual joint
sample indices.

Ex10 retains the numerical conventions of Horne-Jones et al. (2026),
https://arxiv.org/abs/2605.11886, including its frozen reference checks. The general
bounds/profiling algorithms in ex04–ex06 are deliberately distinguished from that
paper-specific endpoint calculation. Parallel performance measurements remain in
`scripts/perf_valid.py`; the small gallery cases use serial execution.

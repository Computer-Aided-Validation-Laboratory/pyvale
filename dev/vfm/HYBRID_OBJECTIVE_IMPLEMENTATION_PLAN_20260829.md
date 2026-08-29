# Hybrid EGI/FRE objective implementation plan

Date: 29 August 2026

## Decision being implemented

Develop and validate a material-information objective for the notched-EBW
identification. Multiscale-equal RMS remains a control, not the intended final
objective. The new objective will combine global mechanical closure with
noise-normalised EGI tails, coherent FRE, and an optional frozen native-DOF
yield projection.

The objective must be assessed at every basis-function count against a direct
map-fitting representability curve. A separate BF7-to-BF8 stopping study is not
a prerequisite. Investigation runs may proceed to a fixed eight-BF cap and
retain the complete trajectory.

## Campaign sequence

The three rounds are:

1. Offline feature, window, and objective screen.
2. First online identification round using controls and the two leading hybrid
   formulations.
3. Matched-seed confirmation of the two leading online formulations.

Do not launch new hybrid online cases until Round 1 has frozen the window set,
feature definitions, noise normalisation, projection choice, and alpha. The
following independent work may run concurrently with Round 1:

- direct-fit SPD Gaussian reference curves for BF0-BF8;
- existing current-objective controls;
- existing 15/29/57 multiscale-equal controls;
- short non-scientific hybrid smoke tests used only to verify the software.

## Shared definitions

### Load regimes

Derive regimes from Phase-0 predicted yielded fraction, then freeze them for
the refined phase. Do not use fixed frame indices in production code.

- pre-yield: noise/model-consistency only;
- yield onset: primary yield-information block;
- developed plasticity: shared yield/hardening information;
- late plasticity: hardening and compensation-sensitive information.

The regime resolver must store both its thresholds and resolved frame indices
in result metadata.

### Candidate EGI supports

Window candidates are declared in physical side length. Pixel dimensions are
resolved from the prepared grid, rounded to valid odd dimensions, and recorded.

For the current synthetic screen, evaluate 1.4, 3.0, 5.8, and 11.4 mm
(approximately 7, 15, 29, and 57 points). Screen these candidate sets:

- two-scale fine/broad;
- three-scale fine/middle/broad, allowing either middle candidate;
- all four scales as an information-reference case only.

The third scale is retained only when its validation improvement exceeds
bootstrap uncertainty or it adds material conditional information through
subspace angle, Fisher log-determinant, or minimum singular-value improvement.

### Residual features

Implement these reusable scalar reductions over a signed residual block:

- weighted RMS;
- weighted CVaR of absolute residual, initially at 90% and 95%;
- coherent RMS after mask-normalised Gaussian smoothing in physical units;
- full material-projected RMS;
- yield-unique projected RMS after residualising yield sensitivities against
  hardening sensitivities.

P90/P95 remain available as diagnostics, but CVaR is preferred inside the
optimisation because it averages the tail rather than selecting one quantile.

### Noise and stage normalisation

For feature `f_j`, resolve a propagated noise floor `n_j` and a fixed-BF
stage-start reference `f_ref_j`:

```
u_j(theta) = positive_part(f_j(theta) - n_j)
             / max(f_ref_j - n_j, epsilon)
```

Use a smooth positive part in the objective and retain the unclipped values in
diagnostics. Initial implementation may use diagonal residual whitening after
metric propagation, matching the native-noise study. Full spatial/cross-metric
covariance is a later extension and must not be implied by the v1 metadata.

### Candidate objective family

Define:

```
J_alpha = (1 - alpha) * J_global + alpha * J_material
```

`J_global` is noise/stage-normalised multiscale EGI/FRE RMS closure.
`J_material` is a smooth maximum plus a small mean contribution over a sparse
semantic feature vector:

1. fine EGI onset CVaR;
2. one fine/middle developed-or-late raw/projected EGI term;
3. broad EGI onset CVaR;
4. developed/late coherent FRE.

Do not include both full projection and yield-unique projection in the same
candidate unless Round 1 disproves their observed redundancy. Do not include a
pre-yield hardening term.

Round 1 screens alpha in 0.25, 0.50, and 0.75. Candidate feature weights are
equal after noise/stage normalisation unless a sparse non-negative weight fit
improves held-out families. Any fitted weights must be frozen from development
data and reported explicitly.

## Core implementation

### 1. Residual blocks and reductions

Add a core module, provisionally:

`src/pyvale/vfm/residualfeatures.py`

Responsibilities:

- split EGI/FRE results into metric x physical scale x load-regime blocks;
- construct correctly normalised observation weights;
- compute RMS, CVaR, coherent RMS, and projected reductions;
- smooth signed fields with mask normalisation and a physical smoothing length;
- return immutable diagnostic records with valid/effective observation counts.

The module must not contain WDBN1-specific paths, frame indices, or truth-map
logic.

### 2. Load-regime resolver

Add `src/pyvale/vfm/loadregimes.py` with a configuration and resolved result.
It should accept yielded-fraction thresholds, resolve from a Phase-0 parameter
state, freeze indices, and serialise its decision.

### 3. Prepared-objective lifecycle

Replace objective-specific `isinstance` branches in `identification.py` with a
small optional lifecycle protocol. The required hooks are provisionally:

- `prepare_phase(context)` for prior-phase baselines, noise floors, regimes,
  and any phase-level state;
- `prepare_solve(context)` after each refinement and before optimisation, for
  fixed-BF feature references and native-DOF projection bases;
- `diagnostics()` for durable result metadata.

Existing combined-objective behaviour must be preserved through the same
hooks. Objectives that do not implement the protocol require no changes.

### 4. Hybrid objective

Add `src/pyvale/vfm/objectivefuncmaterialinformation.py` containing:

- typed feature-term configuration;
- global closure configuration;
- alpha and smooth-maximum configuration;
- resolved feature baselines/noise floors;
- frozen projection bases for selected terms;
- decomposed last-evaluation diagnostics;
- validation that every configured feature has a matching metric, scale, and
  resolved regime.

The optimiser-facing method remains scalar `evaluate(metric_results)`. All
expensive sensitivities are prepared once per fixed-BF solve.

### 5. Native-DOF projection preparation

Implement projection construction as a reusable service rather than importing
from a development report script. It must:

- perturb actual normalised active DOFs;
- restore the exact accepted stage-start state after every perturbation;
- form residual derivatives after the configured whitening;
- group directions into yield and hardening;
- use rank-revealing SVD/QR with a recorded tolerance;
- construct at most the projection bases required by configured features;
- record rank, singular values, condition estimates, group correlations, and
  runtime.

Projection arrays may use float32 storage after numerical equivalence is
tested, but calculations and reported diagnostics remain float64.

### 6. Result/checkpoint support

Every accepted BF solve already has a phase snapshot. Extend objective
diagnostics so each solve additionally records:

- resolved load regimes;
- physical and pixel window sizes;
- global and material objective contributions;
- every normalised and raw feature value;
- noise floors and stage references;
- projection ranks/conditioning;
- alpha and smooth-max settings.

Campaign launchers must write a manifest before work begins, update it after
each completed case, skip complete cases on resume, and keep one log per case.

## Verification tests

Add focused tests before campaign execution:

### Residual feature tests

- weighted CVaR against hand-calculated examples;
- monotonicity when a tail residual increases;
- coherent RMS suppresses alternating noise but retains coherent signed error;
- physical smoothing gives equivalent results after controlled grid resampling;
- masks and NaNs do not leak zeros into valid support;
- projected RMS agrees with direct orthogonal projection.

### Objective tests

- alpha zero reproduces the configured global objective;
- component normalisation is one at stage start and zero at the noise floor;
- worsening one dominant material block increases the smooth maximum;
- global closure cannot be silently omitted;
- decomposed diagnostics reconstruct the returned scalar;
- full/yield-unique projection configuration validation.

### Lifecycle/projection tests

- prepare hooks run once per phase and once per fixed-BF solve as intended;
- DOFs and parameter maps are exactly restored after sensitivity preparation;
- projection is refreshed following BF addition;
- inactive hardening produces no hardening-unique block;
- existing combined-objective regression tests remain unchanged.

### Integration tests

- compact BF0-BF2 synthetic run using the hybrid objective;
- save/load round trip preserves all objective configuration and diagnostics;
- resumed campaign does not repeat completed cases;
- numpy and Cython stress backends agree within existing tolerances.

## Round 1: offline screen

### Tools

Add these development entry points:

1. `dev/vfm/build_notched_ebw_direct_fit_reference.py`
2. `dev/vfm/screen_notched_ebw_hybrid_objective.py`
3. `dev/vfm/report_notched_ebw_hybrid_objective_screen.py`
4. `dev/vfm/run_notched_ebw_hybrid_objective_screen.sh`

The direct-fit tool must use the same SPD Gaussian parameterisation, bounds,
homogeneous floor, fit mask, and BF count as online identification. It should
run multiple starts per count and retain the best plus spread, not only one
sequential conventional-Gaussian fit.

The screen merges:

- optimiser trajectories at every available BF count;
- independent development and validation perturbation families;
- direct-fit maps/reference errors;
- native-DOF sensitivities;
- WDBN1-calibrated noise realisations;
- all candidate window supports and feature definitions.

### Outputs

Write a self-contained Round-1 directory containing:

- `screen_manifest.json`;
- `direct_fit_reference.csv` and `.npz`;
- `state_feature_rows.csv` or a chunked equivalent;
- `candidate_objective_scores.csv`;
- `window_information_scores.csv`;
- `selected_windows.json`;
- `selected_objectives.json` containing the leading raw and projected hybrids;
- `ROUND1_DECISION.md` and a concise PDF.

### Evaluation

Evaluate candidates using:

- Spearman ranking within each BF count, never only pooled BF counts;
- adjacent-state and arbitrary-pair decision accuracy;
- yielded and high-plastic targets separately;
- direct-fit recovery gap at each BF count;
- area under the BF1-BF6 recovery-gap curve where trajectories permit;
- clean-to-noisy degradation;
- leave-one-seed-out and leave-one-perturbation-family-out validation;
- feature redundancy and projection conditioning.

### Freeze gate

Round 1 freezes exactly:

- two or three physical EGI supports;
- load-regime thresholds;
- feature definitions and smoothing lengths;
- raw versus projected developed/late term;
- alpha and smooth-max temperature;
- noise/stage normalisation method;
- the two hybrid configurations entering Round 2.

Do not select a third window or additional feature unless its held-out benefit
exceeds bootstrap uncertainty and it is not redundant with retained terms.

## Round 2: first online identification round

### Tools

Add:

1. `dev/vfm/run_notched_ebw_hybrid_identification_campaign.py`
2. `dev/vfm/analyse_notched_ebw_hybrid_identification_campaign.py`
3. `dev/vfm/report_notched_ebw_hybrid_identification_campaign.py`
4. `dev/vfm/run_notched_ebw_hybrid_identification_campaign.sh`

Use one launcher for Rounds 2 and 3, driven by a round configuration/manifest.
Do not fork near-duplicate campaign implementations.

### Cases

Run four objectives:

- current 29/57 control;
- frozen multiscale-equal control;
- frozen raw tail/coherence hybrid;
- frozen projected hybrid.

Use two matched optimiser seeds and clean/1x WDBN1-noise conditions: 16 cases.
Use SPD geometry, sensitivity-correction growth, zero percentage gate, and a
fixed BF7 cap for this first round. Store BF1-BF7 snapshots and metrics.

### Round-2 decision

The analysis must compare objective trajectories to the direct-fit reference.
A hybrid advances only if it improves recovery-gap AUC and late-region map
error relative to multiscale equal without unacceptable mechanical-closure or
seed-stability regression. Select two objectives for confirmation; a control
may be one of the two if neither hybrid wins credibly.

## Round 3: matched-seed confirmation

Run the two Round-2 finalists over six additional matched seeds, clean and 1x
noise: 24 new cases and eight total seeds per finalist. Use a fixed BF8 cap so
the complete late trajectory is visible; BF8 is evidence about objective
behaviour, not a prerequisite stopping-rule study.

Report:

- recovery-gap trajectory and AUC;
- yielded/high-plastic RMSE and MAPE at every BF count;
- weld/HAZ gradient and upper-tail error;
- common mechanical closure components;
- seed/noise median, IQR, and adverse tails;
- projection rank/conditioning and BF novelty;
- map/objective stabilisation across successive additions.

Freeze a production candidate only if the predeclared synthetic accuracy and
robustness criteria are met. Otherwise retain the best formulation as the next
development baseline and identify the failing component explicitly.

## Planned CLI contract

Final copy-paste commands will be issued only after the entry points and their
`--help` output exist and have passed smoke tests. The intended command order
is:

1. direct-fit reference and reusable online controls may start together;
2. run/resume the offline screen;
3. generate and inspect `ROUND1_DECISION.md`;
4. launch Round 2 using `selected_objectives.json` and
   `selected_windows.json`;
5. analyse/report Round 2 and write the finalists manifest;
6. launch Round 3 from that finalists manifest;
7. analyse/report the final confirmation.

Workstation-specific SSH, detached-session, archive, checksum, and transfer
commands will follow the conventions in `dev/vfm/WORKSTATION_USAGE.md` once
that file is available on disk. Local launchers will continue to support
resumption and explicit dataset/output paths so the same scientific commands
can be used locally or remotely.

## Immediate build order

1. Residual features and tests.
2. Objective lifecycle hooks and regression tests.
3. Hybrid objective without projection and compact integration test.
4. Reusable native-DOF projection preparation and tests.
5. Direct-fit reference tool.
6. Round-1 screen, decision artifacts, and report.
7. Generic Round-2/Round-3 campaign launcher and analysis.
8. Workstation wrapper and final verified terminal commands.

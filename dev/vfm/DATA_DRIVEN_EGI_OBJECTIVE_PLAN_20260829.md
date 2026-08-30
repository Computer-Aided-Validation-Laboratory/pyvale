# Data-driven EGI support and sensitivity-objective plan

Date: 29 August 2026

Status: focused implementation plan. This supersedes the broad objective
variant search as the immediate development priority.

Detailed code interfaces, tests, persistence, and milestone gates are specified
in `DATA_DRIVEN_EGI_OBJECTIVE_IMPLEMENTATION_PLAN_20260829.md`.

## Decision

Build and test one algorithm with two data-driven preparations:

1. after toolkit Phase 0 (homogeneous identification), select a target of
   three EGI support lengths: fine, complementary middle, and broad;
2. before every fixed-parameterisation solve in toolkit Phase 1 (adaptive
   spatial identification), calculate the native active-DOF sensitivity of
   the three EGI fields and FRE, construct one frozen material-information
   objective, then optimise without changing that objective.

Do not run more hand-designed EGI-window or scalar-weight combinations until
this path has passed its staged gates. Keep pattern search, basis growth, and
the fixed-BF exploration policy unchanged while evaluating the objective.

## Important phase clarification

Toolkit Phase 0 identifies homogeneous yield strength and, when released,
homogeneous hardening. It supplies an approximate **mechanical state**:
stress, yielded progression, and plastic-strain progression. It does not
supply a rough spatial property map.

The Phase-0 EGI support sweep must therefore use controlled local material
probes around the homogeneous state. This lets it ask which EGI supports can
observe plausible spatial property changes before the adaptive basis exists.

## Algorithm lifecycle

```text
Phase 0 homogeneous solve
        |
        v
Phase-0 stress/plastic state + residual-noise model
        |
        v
Local material-probe bank -> EGI support sweep -> freeze fine/middle/broad
        |
        v
Create Phase-1 metrics: FRE + three selected EGI supports
        |
        v
Before each fixed-BF solve:
  canonical signed/noise-whitened residual
  -> native active-DOF Jacobian
  -> noise-resolved material subspace
  -> frozen information objective + FRE/broad guards
        |
        v
Optimise with unchanged pattern search
        |
        v
Add/refine basis, then repeat sensitivity preparation
```

The selected support bank is frozen for the whole Phase-1 identification in
version 1. A support rescreen after each solve may be recorded diagnostically,
but it must not switch the training metrics during the first validation.

## 1. Phase-0 EGI support sweep

### Candidate bank

Work in physical length and resolve to unique odd pixel windows.

- lower bound: the smallest valid side length, three datapoints, subject to
  the known DIC correlation/filter footprint;
- upper bound: half the smaller specimen bounding-box dimension;
- candidate count: 10 approximately logarithmically spaced physical lengths;
- conversion: resolve rows and columns separately from grid spacing, round to
  odd sizes, and deduplicate coincident pixel windows;
- eligibility: retain adequate valid-centre coverage around the ROI and do
  not let boundary/mask loss silently define a scale.

The candidate count is an engineering default, not a scientific parameter to
sweep. The resolved physical and pixel sizes must both be persisted.

### Probe bank

Evaluate information about spatial properties using a compact, fixed local
probe bank around the accepted homogeneous maps:

- one homogeneous yield-strength perturbation;
- local positive and negative yield-strength probes distributed over the
  yielded or near-yield ROI, including the weld/HAZ, notch shoulders, parent
  material, and boundary-adjacent regions;
- one homogeneous hardening perturbation and a smaller set of local hardening
  probes only if Phase 0 contains developed plastic strain and hardening is to
  be identified in Phase 1.

Use dimensionless perturbations tied to parameter bounds. A provisional
amplitude is 1% of the parameter range, verified at 0.5% for a subset. Probe
widths should be tied to the smallest physically resolvable material length,
not to any candidate EGI window.

Stress reconstruction is performed once per perturbed material map. Every EGI
support is then evaluated from the same stress response, so increasing the
support bank does not multiply constitutive evaluations.

### Common whitening contract

For every support, build a signed response using the same observation mask,
bias estimate, and noise standard deviation that will later be used by the
objective:

```text
z_L = W_L (r_L - mu_L)
S_L = d z_L / d p
```

Here `p` labels the material probes. Version 1 may use diagonal propagated
noise standard deviations, explicitly labelled as an approximation. Noise
must be propagated to each EGI support; a scalar feature floor is not residual
whitening.

Within-support row weights are normalised so a support is not rewarded merely
for containing more valid pixels. Record coverage, effective observation
count, response-to-noise for every probe, singular spectrum, and runtime.

### Select fine, broad, and middle

Use distinct physical roles rather than equal spacing.

1. **Fine:** choose the smallest eligible support for which a predeclared
   fraction of the relevant local yield probes produces a response above the
   noise-resolved threshold. The initial threshold is one predicted residual
   standard deviation for a 1%-range probe; calibrate it once against the
   null/noise simulation rather than sweeping it against truth error.
2. **Broad:** choose the largest eligible support that retains the minimum
   spatial coverage and has a resolved homogeneous/coherent yield response.
   Its role is coarse equilibrium redistribution, not fine localisation.
3. **Middle:** for each remaining candidate, stack its sensitivity with the
   selected fine and broad matrices and measure incremental parameter-space
   information. Select the candidate with the largest regularised Fisher
   log-determinant gain; use improvement in the smallest retained singular
   value as a reported cross-check.

The middle calculation must use the full multi-probe sensitivity, not only a
pairwise cosine between one aggregated response field. If no candidate adds a
noise-resolved direction, retain the best candidate for the three-window
development experiment but report that the evidence supports only two unique
scales. Do not hide that result.

### Phase-0 sweep gate

Proceed only when:

- all selected supports meet the declared coverage rule;
- the fine support is above the calibrated response-to-noise threshold;
- the middle support has positive incremental information after fine+broad;
- the decision is stable to the finite-difference step check;
- clean and 1x-noise preparations resolve the same support or adjacent
  candidates in the bank.

This is a method gate, not a truth-error optimisation. Synthetic truth is used
after selection to assess whether the data-driven choice generalises.

## 2. Sensitivity-prepared Phase-1 objective

### Canonical residual vector

Before each fixed-BF solve, assemble the signed residuals from the selected
three EGI supports and FRE. Use all valid load steps except observations that
are physically undefined (for example relative FRE at zero force):

```text
z(theta) = block_balance( W [r(theta) - mu] )
```

Noise whitening, masks, block balance, supports, and the accepted reference
state are frozen for the solve. Yield-onset/developed/late labels remain
diagnostics; version 1 lets sensitivity determine useful time observations
rather than imposing manually tuned temporal windows.

### Native active-DOF sensitivity

Differentiate with respect to the actual normalised active optimiser DOFs:

```text
S = d z / d q
```

Use bound-aware central/one-sided differences, exact restoration, and a second
step-size check on the weakest and strongest columns. Record DOF roles,
column response-to-noise, rank, singular values, conditioning, and
yield-hardening correlation.

Retain singular directions only when a declared meaningful DOF movement would
produce a response above propagated noise. A relative SVD tolerance alone is
not an observability rule.

### One objective, with explicit physical roles

Let `Q` span the retained, noise-resolved columns of `S`. Define three terms:

1. material-explainable information from all three EGI supports and FRE;
2. full FRE residual as the absolute stress/resultant guard;
3. full broad-EGI residual as the coarse spatial-equilibrium guard.

For the material term, project the frozen canonical residual into the
sensitivity subspace:

```text
u(theta) = Q^T z(theta)
```

Because block balancing changes the projected noise covariance, whiten `u`
again using the propagated covariance `Q^T C_z Q`. The optimiser loss is the
equal-role mean:

```text
J(theta) = [
    mean rho(whitened u(theta))
  + mean rho(z_FRE(theta))
  + mean rho(z_EGI_broad(theta))
] / 3
```

Use one fixed robust quadratic-to-linear loss `rho`, with its transition set
in standard-noise units. This is the version-1 objective. It has no alpha or
window-weight sweep:

- sensitivity decides which scale/space/time combinations are materially
  explainable;
- FRE prevents loss of absolute stress scale;
- broad EGI prevents local features from satisfying the objective while
  violating coarse equilibrium.

Fine and middle EGI observations influence the material term when they carry
noise-resolved parameter information. Large unexplained residuals do not
automatically receive large material-identification weight.

The sensitivity basis and all transforms remain frozen until the optimiser
finishes. Costs from different BF solves are not model-order evidence because
the sensitivity subspace changes. Fixed common diagnostics remain responsible
for later acceptance/stopping work.

### Phase-1 objective gate

First run fixed trajectories through BF3. Proceed only when:

- enabling preparation restores maps, DOFs, metrics, and scalar evaluations
  exactly;
- repeated preparation at the same state produces the same selected supports,
  rank, objective, and diagnostics;
- the objective and its sensitivity basis remain unchanged inside a solve;
- the retained observation leverage concentrates in yielded/near-yield space
  and plausible load frames, while pre-yield hardening response remains below
  noise;
- FRE and broad-EGI guards do not regress materially;
- clean and 1x-noise identification outperform or match the raw 7/57 control
  on the common yielded/high-plastic/ROI trajectory metrics without a severe
  adverse seed.

Do not require the final 20/30 MPa targets at BF3. This gate tests objective
mechanics and early recovery, not release performance.

## 3. Minimal implementation work

### Already available

- physical-length to odd-pixel support resolution and deduplication;
- coverage, response-to-noise, and simple redundancy diagnostics;
- copied fixed-BF solve-preparation context with semantic DOF descriptors;
- canonical signed residual blocks with diagonal whitening and block balance;
- bound-aware, restoration-safe native-DOF finite differences;
- rank-revealing yield/hardening projection diagnostics;
- opt-in per-solve audit persistence.

### Missing for the focused algorithm

1. candidate-bank generation from grid spacing and specimen bounding box;
2. Phase-0 local material-probe generation and batched multi-support response;
3. propagated residual-noise arrays for every EGI support and FRE;
4. multi-probe Fisher/singular-gain fine-middle-broad selector;
5. a lifecycle hook that freezes the selected support bank before Phase 1
   metrics are instantiated;
6. absolute noise-resolved rank selection and projected-noise rewhitening;
7. the equal-role projected-information/FRE/broad-EGI objective;
8. durable support-selection and per-solve leverage diagnostics;
9. one compact BF0-BF3 campaign and report.

The current `MaterialInformationObjective` remains a historical control. Do
not keep extending its manually selected feature/alpha family for this work.

## 4. Build and experiment order

### Milestone A — automatic support selection

Implement items 1–4 and run the support sweep once on the accepted Phase-0
WDBN1 synthetic state. Produce one report containing the candidate bank,
coverage, probe response-to-noise, singular spectra, fine/broad eligibility,
middle incremental gain, and the final three supports.

### Milestone B — lifecycle integration

Implement item 5. Re-run the ordinary identification with the selected
supports but the existing objective. Assert that a fixed external support
configuration reproduces the automatic decision and that preparation alone
does not change scientific results.

### Milestone C — sensitivity objective

Implement items 6–8. Validate the objective on saved states and synthetic
parameter perturbations before an online solve. The known map may label
ranking performance but must not enter objective construction.

### Milestone D — BF0-BF3 online gate

Run only these matched cases:

- new data-driven sensitivity objective: clean and 1x noise, seeds 0 and 1;
- raw 7/57 control: reuse existing results where configuration matches;
- fixed selected-support full-residual control: clean and 1x noise, seeds 0
  and 1 only if needed to attribute projection benefit.

Keep basis growth, optimiser, starts, and BF trajectory identical. If the gate
passes, extend the four new cases to BF7. Do not launch a new objective
factorial.

## 5. Explicitly deferred work

Until Milestone D passes, do not spend investigation time on:

- more hand-selected EGI combinations or alpha sweeps;
- BF7/BF8 selector optimisation;
- new basis families or independent hardening geometry;
- sensitivity-informed poll directions;
- trust-region least squares;
- full covariance models beyond the declared diagonal/separable v1;
- experimental-data identification.

Model-order selection remains necessary, but fixed-BF trajectories are enough
to validate the support choice and training objective first. Selector work
resumes only after the objective mechanics are credible.

## 6. Definition of success

This focused work succeeds when the code can, without truth information:

1. select and persist fine/middle/broad EGI supports from a homogeneous
   accepted state and a declared noise model;
2. explain the choice through coverage, probe response-to-noise, and
   incremental information;
3. prepare a fixed native-DOF material-information subspace before every
   adaptive solve;
4. minimise one stable equal-role EGI/FRE objective with explicit FRE and
   broad-EGI guards;
5. improve early clean/noisy recovery relative to the raw 7/57 control without
   introducing an adverse seed or mechanical-closure regression.

Only after those five conditions are met should the project return to basis
acceptance/stopping, proposal novelty, and solver acceleration.

# Identification algorithm convergence review

Date: 29 August 2026

Status: recommended implementation direction after reviewing the supplied
investigation PDFs, the Hamill paper, the current hybrid-objective plan, the
latest raw-objective BF3 gate, and the implementation in `src/pyvale/vfm`.
The supplied documents are treated as evidence and prior analysis, not as
instructions.

## Implementation progress

The first non-scientific-change milestone has now started in
`src/pyvale/vfm`:

- the fixed-BF preparation hook receives a typed context containing copied
  state, current maps and stress, metric results, normalised DOFs, and
  semantic DOF descriptors;
- candidate metric evaluation is restoration-safe because every perturbation
  is evaluated on a copy of the accepted state;
- EGI supports can be specified in physical length, resolved to odd pixel
  windows, and deduplicated when multiple requested lengths map to the same
  grid window;
- support evidence now has explicit coverage, raw/whitened residual,
  response-to-noise, and cross-support redundancy representations, with a
  sparse fine/middle/broad selector;
- the native sensitivity helper is bound-aware, records central/one-sided
  schemes and actual step sizes, and restores the accepted point even when a
  perturbation fails;
- canonical residual blocks now freeze metric/support/regime identity,
  observation masks, bias, diagonal noise whitening, and within-block weights
  at the accepted state; block influence is normalised independently of pixel
  count;
- an opt-in native-DOF audit differentiates that exact canonical vector,
  records response-to-noise column norms, rank/conditioning, yield/hardening
  subspaces and correlation, runtime, and the residual-layout metadata;
- the material-information objective can run this audit once before each
  fixed-BF solve and persists its compact diagnostics through the existing
  per-solve objective diagnostics without changing the training scalar.

These facilities are audit infrastructure only: they do not yet change the
training objective, EGI support set, refinement decision, or optimiser. The
full VFM regression suite passes with the changes (176 passed, 8 skipped).
Wiring frozen homogeneous load-regime resolution and calibrated noise arrays
into a first BF0-BF3 audit configuration is the next implementation step; the
projected terms remain diagnostic rather than part of the online objective.

## Repaired raw BF7 confirmation

The repaired fixed-trajectory campaign completed all eight requested cases:
raw 1.4/11.4 mm and raw 1.4/5.8/11.4 mm, clean and 1x noise, with two matched
seeds through BF7. All 56 adaptive states were retained, so this campaign can
be used for parent-to-child selector development.

At BF7, median yielded/high-plastic RMSE was 20.85/28.96 MPa for raw 7/57 and
19.23/28.51 MPa for raw 7/29/57 on clean data. The information-rich objective
therefore met the provisional 20/30 MPa targets cleanly, while the parsimonious
objective was marginally above the yielded target. Under 1x noise, the same
medians were 37.39/38.98 MPa and 53.59/45.85 MPa respectively. Neither noisy
configuration met the targets, but raw 7/57 materially outperformed raw
7/29/57 and both reusable BF7 controls.

The noisy trajectories expose the stopping problem directly. Raw 7/57 had its
best median yielded result at BF5 and whole-ROI result at BF4; raw 7/29/57 had
both at BF4. Continuing to BF7 increased whole-ROI RMSE by 9.83 and 13.82 MPa
respectively, even though high-plastic RMSE continued to improve. Across all
eight paired BF6-to-BF7 transitions, yielded error improved in 6/8 and
high-plastic error in 7/8, but whole-ROI error improved in only 2/8. Training
progress is therefore not a sufficient model-order rule.

Decision: nominate raw 7/57 as the leading noisy training control, retain raw
7/29/57 and equal 7/29/57 as matched controls, and freeze a separate common,
noise-calibrated selector before testing more training variants. Two seeds are
not sufficient to release an objective, and the experimental identification
remains on hold.

## Immediate focus reset

The immediate priority is now the bounded algorithm in
`DATA_DRIVEN_EGI_OBJECTIVE_PLAN_20260829.md`:

1. select fine, complementary-middle, and broad EGI supports from controlled
   local material probes around the accepted homogeneous state;
2. before each adaptive solve, construct and freeze one noise-resolved native-
   DOF material subspace from those three EGI fields plus FRE;
3. optimise an equal-role projected-information objective with full FRE and
   broad-EGI guards.

Further manual objective/window sweeps, selector tuning, basis-family work,
and solver changes are deferred until this path passes a BF0-BF3 gate. The
existing BF1-BF7 trajectories remain validation evidence, not the next source
of objective variants.

## Executive decision

Start converging on the implementation now. Do not wait for another broad
objective-weight sweep, but also do not promote the current raw or projected
hybrid to a production objective yet.

The most defensible architecture is not one all-purpose scalar. It has three
separate decision layers:

1. a fixed-parameterisation training objective that is stable within a solve;
2. sensitivity-informed basis proposal and conditioning checks;
3. a common, truth-free acceptance and stopping rule evaluated across basis
   counts.

For the fixed-parameterisation objective, the leading production design should
be a block-balanced, noise-whitened EGI/FRE residual loss. Metric, physical
scale, and load regime should remain separate until the final block
aggregation. Raw tail/coherence terms can be tested as a bounded addition.
Projected residuals should initially be diagnostics and growth/identifiability
evidence, not the sole training objective or BF acceptance gate.

Evaluating sensitivities after every refinement and before every fixed-BF
solve is sensible and affordable. At BF6 the current problem has roughly 38
active DOFs, so a central-difference Jacobian costs about one complete
pattern-search poll. The sensitivity calculation can then improve DOF scaling,
poll directions, observability reporting, basis novelty testing, and later a
trust-region least-squares solver.

For the current WDBN1 synthetic case, treat BF6 as the default reportable
model and BF7 as a routinely evaluated but guarded extension. That is a
dataset-specific temporary policy, not a general algorithmic constant. Use
the stored BF8 states as adverse/overfit examples when freezing the selector.

## Why this is the right point to converge

The accumulated evidence now distinguishes the main failure modes well enough
to design the algorithm around them.

| Evidence | Current conclusion | Implementation consequence |
|---|---|---|
| The latest direct-fit SPD reference reaches 13.81 MPa yielded RMSE at BF3, 6.11 MPa at BF5, and 3.66 MPa at BF6. | The basis family can represent this synthetic map much better than the inverse solves recover it. | Do not change representation solely because current inverse error is high. Keep a same-BF representability reference in synthetic validation. |
| Mature inverse trajectories remain near 39 MPa yielded RMSE and 56-57 MPa high-plastic RMSE. | The dominant gap is inverse recoverability, geometry/path selection, and objective alignment rather than raw capacity. | Prioritise residual information, native-DOF conditioning, and proposal quality. |
| Incorrect maps can have lower current scalar cost than the known map; unyielded observations dominate the closure floor. | The current global RMS scalar is useful mechanical closure but is not a reliable late-stage map selector. | Preserve block structure and never use training-cost decrease alone for model order. |
| BF6 improved all 15 mature transitions; BF7 improved 13/14; BF8 improved only 4/14 while training cost continued to fall. | Late overfit is real and a percentage cost gate cannot stop it. | Calibrate a common evidence-based selector on BF7-to-BF8 and controlled adverse transitions. |
| In the dedicated BF6-to-BF7 study, BF7 improved yielded and high-plastic error in 7/8 seeds, but the median changes were only -0.93 and -1.68 MPa. | BF7 is usually beneficial but the signal is small relative to the overall recovery gap. | Evaluate BF7 routinely, but require evidence stronger than a small cost decrease. |
| Raw EGI7 late and raw EGI15 developed gave the best BF6-to-BF7 paired decisions at 1x noise. | Fine-scale raw evidence survives realistic noise and is useful for adjacent model decisions. | Carry these components in the selector even if the training objective uses a different middle scale. |
| Projected/yield-unique EGI7 retained ranking power but accepted the one marginally adverse BF7 seed almost universally. | Projection is useful for identifiability, but specificity as a stopping gate is unproven. | Use projection for diagnosis, proposal novelty, and bounded objective experiments; do not make it the only acceptance criterion. |
| The one-seed objective/noise screen favoured equal 15/29/57 cleanly and sensitivity-equal under noise. | Multiscale balance helps, but unsigned frozen weighting and one seed are not production evidence. | Keep multiscale equal as a strong control; replace unsigned sensitivity magnitude with native-DOF decomposition. |
| Corrected offline Round 1 selected raw 7/29/57 at alpha 0.5 and raw 7/57 at alpha 0.25. | These feature families rank broad perturbations well enough for an online pilot. Alpha 0.5 was pragmatic, not uniquely identified. | Finish the small raw BF7 pilot; do not freeze alpha from offline ranking alone. |
| The corrected raw BF3 gate completed 8/8 cases and accepted all 24 stages. Clean data favoured 7/29/57, while two noisy seeds were mixed. | The lifecycle/scaling repair works; objective quality at late BF remains unresolved. | Continue only the eight raw hybrid cases to BF7 before a larger confirmation campaign. |
| Best noisy results are still roughly 40 MPa yielded and 70 MPa high-plastic RMSE. | The current method does not meet the provisional 20/30 MPa synthetic targets. | Do not process the experiment as a definitive identification yet. |

The apparent disagreement over middle spatial scale is informative rather
than problematic. The broad perturbation screen selected 5.8 mm (29 points),
whereas adjacent BF6-to-BF7 decisions favoured 3.0 mm (15 points). Training
supports and acceptance supports need not be identical. The implementation
should allow extra diagnostic supports that do not contribute to the training
objective.

## Recommended algorithm

### 1. Canonical residual blocks

Create one canonical signed residual representation used by the objective,
sensitivity analysis, basis proposal, diagnostics, and acceptance policy. A
block is identified by:

- metric: EGI or FRE;
- physical support, where applicable;
- frozen load regime;
- optional training/guard role.

For block `b`, form

```text
z_b(theta) = W_b [r_b(theta) - mu_b]
```

where `mu_b` is the expected residual bias and `W_b` is a noise-whitening
operator. The first implementation may use diagonal or separable whitening,
but it must call this an approximation and retain enough metadata to replace
it with spatial/component covariance later. A scalar propagated feature floor
is not equivalent to residual whitening.

Normalise total block influence, not individual observation count. Otherwise
a block with many low-information pixels again overwhelms a smaller
material-sensitive block. A suitable initial training loss is

```text
J_closure(theta) = sum_b w_b mean_i rho(z_bi(theta))
```

with equal total weight across predeclared informative blocks and a robust
quadratic-to-linear loss `rho`. Force-squared temporal weighting may remain
within a block, but onset, developed, and late regimes must have independent
normalisation.

Recommended initial roles for WDBN1 are:

- fine EGI near yield onset and in developed/late plasticity: primary local
  yield evidence;
- a middle EGI support: candidate additional gradient information;
- broad EGI: coherent equilibrium guard;
- FRE: global force/cross-section guard, especially after developed
  plasticity;
- pre-yield blocks: noise/model-consistency diagnostics, not hardening
  identification evidence.

The physical supports currently under consideration are 1.4, 3.0, 5.8, and
11.4 mm. Resolve these to odd pixel dimensions for each prepared grid and save
both forms.

### 1a. EGI physical-support sweep

EGI support selection should be an explicit part of the algorithm, not a
one-off WDBN1 tuning exercise. The sweep should operate in physical length,
with pixel dimensions treated only as a grid-specific resolution detail.

Run the sweep at two levels:

1. **initial selection after homogeneous identification**: select the two or
   three EGI supports allowed to influence the adaptive training objective;
2. **diagnostic rescreen after each accepted BF state**: check whether the
   informative length scale has shifted as the remaining residual becomes
   more local, without automatically changing the training objective.

Do not sweep or switch EGI sizes inside optimiser evaluations. The training
support set must remain frozen throughout each fixed-BF solve. In the first
production version it should remain frozen for the whole adaptive-yield phase;
the between-BF rescreen is diagnostic. A later adaptive-window version may
change supports between BF stages only after that policy has been validated,
with model selection still performed using a separate common support bank.

Construct the candidate bank from:

- a lower limit safely above the measured DIC correlation/filter footprint
  and the minimum valid EGI stencil;
- an upper limit set by specimen dimensions and acceptable traction-boundary,
  notch, and mask coverage;
- approximately logarithmic intermediate physical sizes, augmented by any
  known process length scale worth testing.

The current 1.4, 3.0, 5.8, and 11.4 mm supports are a useful coarse WDBN1
bank, but the implementation should accept a denser physical sweep rather
than hard-coding those four values.

For every support and load regime, record:

- valid and effective observation count, including edge/mask loss;
- propagated noise mean, variance, and spatial correlation;
- raw and whitened residual magnitude;
- native yield and hardening response-to-noise;
- singular values and numerical rank contributed by that support;
- principal angles/redundancy relative to already selected supports;
- incremental minimum-singular-value or Fisher log-determinant gain;
- runtime and memory cost.

Selection should be sparse and role-based:

- choose the finest support whose material response is resolved above noise
  and whose valid spatial coverage is acceptable;
- choose one broad support that supplies stable coherent/global equilibrium
  information;
- add a middle support only when it provides independent parameter
  information beyond the fine and broad pair.

Optimisation supports and selector supports need not be identical. Keep a
small common diagnostic bank across all BF counts so the acceptance rule can
compare parent and child models on fixed evidence. This permits, for example,
5.8 mm to enter a training objective while 3.0 mm remains a stronger
BF6-to-BF7 decision channel.

During synthetic development, validate the automatic selection against
held-out map families, noise realisations, and grid resampling. A support is
not justified merely because it ranks the current idealised WDBN1 truth well.

### 2. Homogeneous initialisation and frozen load regimes

Use the current homogeneous SBVF solve to identify homogeneous yield strength
and, when data permit, homogeneous hardening. Known elastic properties remain
fixed.

From this accepted homogeneous state:

1. calculate the predicted yielded fraction for every load step;
2. resolve disjoint pre-yield, onset, developed, and late regimes;
3. freeze and serialise the thresholds, yielded fractions, and frame indices;
4. estimate or load the residual noise model for every configured block.

The relative monotone-yield-progress resolver is preferable to specimen-
independent absolute thresholds for the current data. It should be a core
phase-preparation decision rather than an offline JSON-only convention.

### 3. Two sensitivity preparations per adaptive cycle

There are two different sensitivity questions and they should not be
conflated.

#### A. At the accepted BF state: what structure should be added?

After a fixed-BF solve converges, evaluate the signed, whitened residual and
the response to a local yield-strength correction. Use the objective/residual
cotangent to form a signed correction field. Fit an SPD Gaussian to the
dominant connected correction feature rather than simply placing it at the
largest unsigned EGI peak.

For a proposed kernel, calculate its residual response column and remove the
span of the active model:

```text
s_new_perp = (I - Q_active Q_active^T) s_new
novelty = ||s_new_perp|| / ||s_new||
```

Do not launch an expensive joint solve when the new response is below the
noise floor, nearly contained in the active span, or makes conditioning
unacceptable. This is a pre-solve eligibility test, not final acceptance.

#### B. After adding/refining the representation: can the active DOFs be solved?

Immediately after the structural refinement and before optimisation, compute
the Jacobian with respect to the actual normalised active DOFs:

```text
S_b = d z_b / d q
```

Group its columns into homogeneous yield, BF amplitude, BF geometry, and
hardening directions. Build a rank-revealing SVD/QR and record:

- singular spectrum, numerical rank, and condition estimate;
- column norms and near-null DOFs;
- yield/hardening principal correlations;
- full and yield-unique residual projections;
- response-to-noise for every semantic block;
- change from the preceding BF model.

Finite differences must be bound-aware. Use central differences in the
interior and one-sided differences near `[0, 1]` bounds, and verify selected
columns at a second step size. The existing generic central-difference helper
does not yet implement this bounded contract.

The accepted state must be restored exactly after every perturbation. Freeze
the resulting snapshot throughout one fixed-BF solve. Refresh it after every
basis addition; optionally refresh within a solve only when a declared trust-
region displacement is exceeded.

### 4. Fixed-BF optimisation

Do not change objective, noise transform, load regimes, stage references, or
projection basis during one optimiser call.

The first serious implementation should retain pattern search as the control,
but use the prepared Jacobian more effectively:

- scale or freeze DOFs with negligible observable response;
- poll along right-singular-vector directions before random orthogonal
  directions;
- choose a bounded initial step from predicted residual change in noise units;
- retain coordinate polling as a fallback;
- report convergence in both objective and parameter-map change.

This is a low-risk use of sensitivity because it changes search directions,
not the scientific objective. One native Jacobian is likely cheaper than many
uninformed pattern polls.

In parallel, prototype a vector residual objective with bounded trust-region
least squares. The EGI/FRE operators and constitutive reconstruction are
piecewise smooth enough to justify a controlled comparison, but do not change
the objective, basis growth, and optimiser in the same scientific experiment.
Compare pattern search, sensitivity-informed pattern search, and least squares
on identical fixed geometries and starts before changing the online default.

For the current raw hybrid, stage references may continue to be refreshed for
within-stage optimisation. Such costs are not comparable across BF counts.
Only the common residual/evidence evaluator may make an acceptance decision.

### 5. Basis acceptance and stopping

Split basis proposal from model selection. The current refinement policy both
proposes structure and accepts/restores solved candidates; the need for
`fixed_basis_trajectory` demonstrates why these responsibilities should be
separate.

For every candidate, retain both the accepted parent and solved child. Evaluate
the same common, frozen evidence on both. A provisional evidence vector is:

1. fine EGI yield-onset raw tail/RMS;
2. fine or middle EGI developed/late raw score;
3. one projected or yield-unique developed/late diagnostic;
4. broad EGI closure guard;
5. FRE/global force guard;
6. active-model novelty and conditioning;
7. parameter-map and force-reconstruction stability under a local restart or
   matched perturbation.

The raw fine-scale terms should lead the BF decision for now. Projection
should be a required identifiability diagnostic or one bounded vote, not the
sole gate. Broad EGI and FRE should be non-regression guards rather than
dominant map-discrimination terms.

Thresholds should be noise-calibrated deltas, not percentages of a changing
scalar. For example, label a component improvement only when its paired
parent-to-child change is larger than its propagated noise/restart uncertainty;
apply the corresponding one-sided non-regression interval to guards. Freeze
the logical rule and tolerances on development states, then report performance
on held-out seeds, perturbation families, and BF8 adverse states.

For WDBN1 only:

- BF6 is the default returned model while the selector is under validation;
- BF7 is always evaluated and returned only when the frozen selector passes;
- BF8 is evaluated in validation campaigns to challenge the stopping rule,
  not used as a routine experimental endpoint;
- counts above BF8 currently have no justification.

A general implementation must stop from evidence, not from the number six.

### 6. Parameter staging

Use scientific roles rather than relying on ambiguous phase numbering:

1. **homogeneous initialisation**: homogeneous yield and homogeneous
   hardening;
2. **adaptive yield mapping**: spatial yield map with hardening remaining
   homogeneous;
3. **optional hardening release**: only if developed/late plasticity provides
   a hardening direction above noise and sufficiently independent of yield.

When hardening is released spatially, first reuse the accepted yield geometry
and release hardening amplitudes. Add independent hardening geometry only if
its residual response has demonstrable novelty. A strong pre-yield
hardening-unique direction is a diagnostic of compensation or numerical
structure, not physical evidence.

## Assessment of the current implementation

The repository is closer to this architecture than the earlier plan implies.

### Foundations already present

- immutable residual reductions for RMS, CVaR, coherent RMS, and projection;
- physical-length to odd-pixel conversion;
- absolute and relative load-regime resolvers;
- rank-revealing projection utilities and a restoration-aware finite-
  difference helper;
- a hybrid material-information objective with corrected dimensionless
  positive-part scaling;
- a per-solve preparation call in the correct location;
- SPD basis support and signed sensitivity-correction growth;
- fixed-cap trajectory mode, durable per-solve snapshots, decomposed objective
  diagnostics, and resumable campaign manifests;
- direct-fit and offline objective-screen tooling.

The focused objective, residual, load-regime, projection, refinement, and
pattern-search tests currently pass (49 tests).

### Gaps to close before a projected online objective

| Current seam | Required change |
|---|---|
| `_prepare_objective_solve` supplies only `metric_results`. | Introduce a typed solve-preparation context containing current normalised DOFs, DOF descriptors/groups, parameter maps, metrics, a residual evaluator, experiment/noise metadata, and phase/BF identity. |
| `MaterialInformationObjective.prepare_solve` refreshes only scalar stage references. | Let a prepared objective consume a canonical residual layout and optional projection snapshot without owning the sensitivity calculation. |
| `materialprojection.py` is not integrated online. | Add a reusable stage-sensitivity service with bound-aware differences, exact restoration, block layouts, group residualisation, timing, and serialisable diagnostics. |
| Projection bases and residual weights have no single whitening contract. | Build both from the same canonical whitened vector to prevent double weighting or inconsistent observation masks. |
| Sensitivity spatial weighting is resolved once at phase start from parameter names. | Retain it as a historical control only; native active-DOF analysis should refresh after each structural change. |
| Sensitivity-correction growth differentiates the wrapped global scalar through a local parameter perturbation. | Preserve this useful proposal seed, then add candidate response novelty and conditioning checks using the active native-DOF span. |
| `EquilibriumGapBasisGrowthRefinement` owns proposal, acceptance, restoration, and stopping. | Separate `BasisProposalPolicy`, `CandidateEligibilityPolicy`, and `ModelSelectionPolicy`; add an explicit exploration policy for fixed trajectories. |
| Feature configs contain frozen frame indices from an offline script. | Resolve regimes through the core lifecycle and save thresholds plus indices in each result. Allow a frozen external config to reproduce the same decision. |
| Caller and metrics are configured mainly in pixel counts. | Make physical support length the scientific configuration and pixel size a resolved detail. |
| Campaign resumption skips only completed result files. | Add an atomic accepted-BF checkpoint if long online runs need true within-case continuation. |

The lifecycle refactor should preserve current combined-objective behaviour and
the corrected raw BF3 trajectories before it changes any scientific result.

## Implementation and experiment sequence

### Workstream A: close the current raw-objective question

Run only the eight existing raw hybrid cases through BF7:

- raw 1.4/11.4 mm, alpha 0.25;
- raw 1.4/5.8/11.4 mm, alpha 0.5;
- two matched seeds;
- clean and 1x WDBN1 noise.

Reuse the eight completed current and multiscale-equal controls. Compare the
whole BF trajectory against the direct-fit curve using recovery-gap AUC,
yielded/high-plastic error, common closure, seed/noise spread, feature margins
above noise, and successive map change. This pilot can nominate a training
objective, but two seeds cannot release one.

Do not add projected terms to these runs. That would mix the raw-objective
question with unvalidated online projection preparation.

### Workstream B: implement sensitivity audit mode without changing results

1. Add the typed solve-preparation context and canonical residual layout.
2. Add the physical EGI-support sweep and persist its coverage, noise,
   response-to-noise, and redundancy diagnostics.
3. Integrate bound-aware native-DOF Jacobian preparation after every basis
   change.
4. Persist singular spectra, group correlations, projected scores, response-
   to-noise, and runtime.
5. Assert exact restoration and unchanged objective results when audit mode is
   enabled.
6. Run BF0-BF3 smoke tests and one stored BF6 state.

This is the first implementation milestone. It creates the infrastructure
needed by every plausible next objective without committing to one.

### Workstream C: freeze a rejection-capable selector

Evaluate the nominated raw and projected components on existing BF7-to-BF8
states first. Add controlled adverse candidates if needed:

- a duplicate/overlapping Gaussian;
- a basis concentrated in unyielded material;
- a boundary/noise feature;
- a yield-hardening compensation direction;
- a small training-cost improvement with poor response novelty.

Use known map error only to label development and validation transitions.
Freeze a truth-free rule from the residual deltas, noise uncertainty,
conditioning, and restart stability. Do not claim specificity from the single
marginally adverse BF6-to-BF7 seed.

### Workstream D: improve basis growth and solver efficiency

Add candidate response residualisation and pre-solve eligibility to the signed
correction growth. Then compare current pattern search with sensitivity-SVD
polling on matched fixed-BF states. Only after that comparison, test the
canonical residual vector with trust-region least squares.

The order matters: it attributes any gain to proposal, search, or objective
rather than changing all three together.

### Workstream E: matched confirmation and release suite

Advance at most two training objectives, with multiscale equal allowed to win.
Use at least eight matched seeds, clean and 1x noise, and retain BF trajectories
through the negative-control count. Then stress-test:

- multiple synthetic map widths, locations, orientations, contrasts, and
  asymmetries;
- strain/force bias and realistic spatial/component correlation;
- strain-force temporal shifts and load-window omission;
- mask, edge, and notch-boundary perturbations;
- thickness and boundary-condition errors;
- modest constitutive mismatch, including non-linear hardening data fitted by
  the assumed linear-hardening model.

Release to experimental use only when the frozen method achieves acceptable
median and adverse-tail errors, does not degrade at late BF counts, preserves
force/mechanical closure, and reports stable uncertainty/identifiability. The
current provisional targets of at most 20 MPa yielded RMSE and 30 MPa
high-plastic RMSE remain useful, but should be confirmed as engineering
requirements rather than tuned to one map.

## Immediate build order

1. Finish the eight-case raw BF7 pilot and analyse it against the reusable
   controls.
2. Refactor the per-solve lifecycle around a typed preparation context.
3. Implement the physical EGI-support sweep and selection diagnostics.
4. Implement the canonical block residual/noise transform.
5. Integrate native-DOF sensitivity audit mode with exact restoration tests.
6. Split basis proposal from model selection.
7. Replay a compact selector on BF7-to-BF8 and controlled adverse states.
8. Add novelty-gated sensitivity-correction growth.
9. Test sensitivity-informed pattern polling.
10. Only then run a projected/bounded objective and matched confirmation.

## Bottom line

The project should now move from “find a better weighted scalar” to “build an
adaptive, information-aware inverse solver.” EGI and FRE remain the correct
physics residuals, but they need a consistent noise model, preserved semantic
blocks, and native-DOF sensitivities. Sensitivities should be recomputed after
each refinement, used first to expose identifiability and guide the search,
and only later allowed a bounded role in the optimisation scalar. A separate,
noise-calibrated model-order selector is essential because no training scalar
examined so far can both drive gross convergence and reliably stop late-stage
overfit.

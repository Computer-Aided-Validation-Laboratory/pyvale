# Data-driven EGI objective: implementation specification

Date: 29 August 2026

Status: implementation-ready plan for Milestones A-D. The scientific scope
and algorithm are fixed by
`DATA_DRIVEN_EGI_OBJECTIVE_PLAN_20260829.md`; this document defines the code
changes, interfaces, tests, artefacts, and gates needed to implement it.

## 1. Scope and version-1 decisions

Implement one path only:

1. use the accepted homogeneous Phase-0 state to screen an automatically
   generated EGI support bank with controlled local material probes;
2. select and freeze fine, middle, and broad EGI supports before Phase 1;
3. before every fixed-basis-function Phase-1 solve, calculate the native
   active-DOF sensitivity and freeze a noise-resolved objective;
4. optimise the equal-role material-information/FRE/broad-EGI objective;
5. validate the mechanics and early recovery through BF3 before any BF7 run.

The following choices are fixed for version 1 and are not tuning dimensions:

- approximately 10 logarithmically spaced requested support lengths;
- odd resolved windows from 3 datapoints to half the smaller specimen extent;
- one support bank for the whole Phase-1 run;
- diagonal residual-noise whitening with an explicit noise-model identity;
- finite differences in normalised native active DOFs;
- one noise-resolved rank rule, calibrated from null/noise responses;
- one robust loss transition expressed in whitened-noise units;
- equal scalar contribution from material information, FRE guard, and broad
  EGI guard;
- existing optimiser, refinement policy, basis trajectory, and starts.

No synthetic truth map may be passed to support selection, sensitivity
preparation, or objective evaluation. Truth is restricted to post-run scoring.

## 2. Target architecture and ownership

Keep the two preparation operations separate because they have different
lifetimes:

```text
IdentificationPhase configuration
        |
        | copy mutable phase components
        v
Phase preparation (once, before PhaseRuntime.prepare)
  Phase-0 accepted maps/state + noise model
  -> candidate EGI metrics
  -> probe sensitivities
  -> selected fine/middle/broad metrics
  -> persisted PhasePreparationResult
        |
        v
PhaseRuntime.prepare(experiment_data)
        |
        | before every fixed-BF optimiser call
        v
Objective.prepare_solve(SolvePreparationContext)
  -> frozen residual layout
  -> native-DOF Jacobian
  -> frozen projection and guard transforms
  -> persisted solve-preparation diagnostics
        |
        v
optimiser evaluations (read-only prepared objective)
```

Phase preparation owns support selection and metric construction. The
objective owns only the per-solve residual transform and sensitivity
projection. This prevents an objective from mutating the metric bank or
changing supports inside an optimiser solve.

### Proposed source modules

| Module | Responsibility |
| --- | --- |
| `egisupports.py` | Candidate-bank generation, multi-probe evidence, Fisher selection |
| `materialprobes.py` | Deterministic homogeneous/local probe definitions and perturbed maps |
| `residualnoise.py` | Immutable bias/standard-deviation arrays and semantic lookup |
| `phasepreparation.py` | Generic phase-preparation protocol, context, result, and EGI implementation |
| `residualblocks.py` | Frozen canonical block layout; extend semantic addressing where needed |
| `materialprojection.py` | Native-DOF finite differences, absolute rank rule, projection covariance |
| `objectivefuncsensitivityinformation.py` | New equal-role prepared objective |
| `identificationconfig.py` | Optional phase-preparation field |
| `identification.py` | Invoke phase preparation once and persist both preparation records |
| `identificationresult.py` | Backward-compatible phase/solve diagnostics schema |

Do not add the new behaviour to `MaterialInformationObjective`. Retain that
class as a historical control.

## 3. Shared data contracts

Define these contracts before implementing the numerical paths. Names may be
adjusted for local style, but responsibilities and lifetimes should not move.

### Residual-noise snapshot

Add an immutable `ResidualNoiseSnapshot` with:

- `model_name`, `model_version`, and optional source checksum;
- semantic block key, metric kind, residual field, and support window;
- bias array or scalar;
- positive standard-deviation array or scalar;
- mask/shape metadata and sample count;
- a compact diagnostics method suitable for YAML.

Lookup must use a semantic key such as `(metric_kind, residual_field,
window_size)`, not a mutable list index. A missing or shape-incompatible entry
is a hard error. Version 1 accepts diagonal scales only and records that
limitation explicitly.

Provide a small constructor for empirical residual samples that calculates
bias and standard deviation with a declared floor. Experiment-specific noise
generation remains outside the core selector. The WDBN1 trial configuration
will supply the already calibrated synthetic noise model and seed.

### Phase-preparation contract

Add:

```python
@dataclass(slots=True, frozen=True)
class PhasePreparationContext:
    phase_index: int
    experiment_data: ExperimentData
    constitutive_law: IConstitutiveLaw
    parameter_map_size: NDArray[np.uint32]
    accepted_parameter_maps: dict[str, NDArray[np.float64]]
    accepted_stress: NDArray[np.float64]
    configured_metrics: tuple[IMetric, ...]

@dataclass(slots=True)
class PhasePreparationResult:
    metrics: list[IMetric]
    diagnostics: Summary

class IPhasePreparation(Protocol):
    def prepare(self, context: PhasePreparationContext) -> PhasePreparationResult: ...
```

Arrays supplied to the preparation service must be copies or read-only views.
The result returns a replacement runtime metric list and serialisable
diagnostics. It must not modify `IdentificationPhase.metrics`.

Phase 0 has no predecessor and therefore cannot use predecessor-dependent
preparation. An EGI preparation configured there is a validation error.

### Selected-support result

Replace/extend the current pairwise-only result with a durable
`EgiSupportSelectionResult` containing:

- the complete resolved candidate bank;
- one evidence record per candidate;
- role-to-index mapping for `fine`, `middle`, and `broad`;
- selection status (`three_resolved` or `two_unique_directions`);
- declared thresholds and noise-model identity;
- finite-difference step-check outcome;
- reasons for every ineligible candidate.

Each candidate evidence record contains coverage, effective observation
count, homogeneous response-to-noise, resolved local-probe fraction, the
multi-probe singular spectrum, regularised Fisher score, incremental Fisher
gain after fine+broad, minimum-retained-singular-value gain, and runtime.

Large residual and sensitivity arrays remain transient. Store compact
diagnostics in YAML/JSON and optionally write arrays to a named NPZ sidecar for
analysis. The result must not contain Python object identities.

### Per-solve prepared-objective result

Have `prepare_solve` return a serialisable diagnostics summary. Change
`_prepare_objective_solve` to return that summary and attach it to
`SolveResult.details["solve_preparation"]` after the optimiser result is
created.

The summary includes:

- phase and solve indices and reference DOF checksum;
- residual block/mask/noise configuration;
- difference schemes and steps for every DOF;
- column norms and semantic DOF labels;
- retained rank, singular values, absolute threshold, and condition estimate;
- yield/hardening subspace correlation;
- projected-noise covariance eigenvalues and applied floor;
- leverage summaries by metric, support, load frame/regime, and parameter
  group;
- preparation runtime and repeatability checksum.

## 4. Milestone A — automatic EGI support selection

### A1. Generate the candidate bank

Extend `egisupports.py` with a configuration and generator that:

1. obtains robust positive row/column grid spacings from specimen coordinates;
2. calculates the smaller finite specimen bounding-box extent;
3. sets the physical lower request to the length represented by a 3-point
   window and the upper request to `0.5 * min_bbox_extent`;
4. creates 10 logarithmically spaced requested lengths including both ends;
5. resolves row and column sizes separately to odd pixel counts;
6. deduplicates coincident resolved windows while preserving all requested
   lengths in each `PhysicalEgiSupport`;
7. rejects a resolved window that cannot be evaluated or has insufficient
   valid-centre coverage.

The candidate generator must work on anisotropic grids. Ordering is by
nominal physical side length, then pixel tuple for a deterministic tie-break.

### A2. Build a deterministic material-probe bank

Create `materialprobes.py` with immutable `MaterialProbeSpec` records and a
`MaterialProbeBankConfig`. A probe records parameter name/group, centre,
physical width, signed normalised amplitude, bound-limited actual amplitude,
and spatial role.

Version-1 construction is:

- one homogeneous yield-strength derivative;
- local yield probes centred by deterministic farthest-point sampling over
  finite specimen points in the yielded or near-yield Phase-0 state;
- ensure the eligible set includes spatial coverage across the specimen,
  rather than hand-labelled weld/notch truth regions;
- one homogeneous hardening derivative and fewer local hardening probes only
  when the accepted Phase-0 state has a declared minimum developed-plastic
  fraction and hardening is active in Phase 1;
- local probe width fixed in physical units from the smallest resolvable
  material length, independent of candidate EGI support length;
- default derivative amplitude 1% of parameter range, with a 0.5% repeat for
  the declared step-check subset.

Use smooth compact or Gaussian-like probe masks normalised to unit peak. Clip
perturbations through the existing parameter bounds and record whether each
derivative used central, forward, or backward differences. Never use the
known heterogeneous map to place or label probes.

The yielded/near-yield mask must be derived from the accepted homogeneous
stress/plastic history. Reuse the constitutive state calculation behind
plasticity diagnostics, but move/share its numerical core so production
identification does not import a reporting module.

### A3. Evaluate responses efficiently

For every unique perturbed parameter map:

1. reconstruct stress once;
2. evaluate every candidate EGI support with the existing batched EGI path;
3. subtract plus/minus responses to form the derivative for each probe;
4. align with the corresponding residual-noise snapshot and frozen valid mask;
5. whiten and apply within-support observation normalisation.

Use the same load-frame set for every support. Undefined and invalid values are
excluded through a frozen common mask per support; do not silently replace
them by zero.

For support `L`, assemble `S_L` with observations in rows and material probes
in columns. A probe's response-to-noise is the RMS whitened response for the
declared 1%-range movement. Singular and Fisher calculations use the complete
matrix, not parameter-group RMS summaries.

### A4. Select the three roles

Implement the following deterministic selection order:

1. Mark coverage-eligible candidates.
2. **Fine:** first eligible candidate whose fraction of resolved local yield
   probes meets the configured requirement. A probe is resolved when its
   response-to-noise exceeds the threshold calibrated from the null ensemble.
3. **Broad:** last eligible candidate with resolved homogeneous/coherent yield
   response.
4. **Middle:** for every remaining candidate, stack its normalised `S_L` with
   fine and broad, then calculate the regularised Fisher log-determinant gain.
   Select maximum gain; break ties by minimum-retained-singular-value gain,
   then smaller physical support.

Regularisation and absolute rank thresholds are stored in the result. If no
middle candidate contributes a noise-resolved direction, retain the best
three-window development candidate but set status to
`two_unique_directions`; the Milestone-A scientific gate then fails visibly.

### A5. Milestone-A tests

Add focused unit tests, principally in `tests/vfm/test_egisupports.py` and
`tests/vfm/test_materialprobes.py`:

- physical bounds, endpoint inclusion, odd conversion, deduplication, and
  deterministic ordering;
- anisotropic and masked grids;
- coverage rejection and clear failure when fewer than three usable windows
  exist;
- probe placement reproducibility, specimen containment, bound-aware
  differences, and no truth-map input;
- batched and individual EGI evaluations agree;
- constructed sensitivity matrices choose known fine/broad/middle supports;
- observation duplication and uniform unit rescaling do not change selection;
- middle selection uses incremental multi-probe information rather than one
  response cosine;
- NPZ/diagnostic round trip and deterministic checksum;
- 1% versus 0.5% step check flags intentionally nonlinear test responses.

### A6. Milestone-A artefact and gate

Add a development runner that loads the accepted WDBN1 Phase-0 state and
writes:

```text
<campaign>/support_selection/
  selection.json
  support_evidence.npz
  SUPPORT_SELECTION_REPORT.md
  support_selection.png
```

Milestone A passes only if coverage, fine response threshold, positive middle
incremental information, step stability, and clean/noisy adjacent-candidate
stability all pass. Do not proceed by manually substituting preferred window
sizes if this gate fails.

## 5. Milestone B — phase lifecycle integration

### B1. Extend phase configuration without breaking existing callers

Add `phase_preparation: IPhasePreparation | None = None` to
`IdentificationPhase`. Add the matching optional `ObjectSnapshot` to
`PhaseConfigSnapshot` and include it in `snapshot_phase_config`. Existing
configs and saved results must continue loading with `None`.

Add `preparation: Summary = field(default_factory=dict)` to `PhaseResult` for
the executed result. This is separate from the declarative config snapshot.

### B2. Split copying from data preparation

Refactor `prepare_phase_runtime` into two internal operations while retaining
the existing public helper as a compatibility wrapper:

1. `copy_phase_runtime(phase)` deep-copies parameterisations, metrics,
   objective, optimiser, refinement policy, and phase preparation together;
2. optional phase preparation replaces only the copied runtime metrics;
3. `PhaseRuntime.prepare(experiment_data)` initializes supports and metrics.

In `run_identification`, invoke phase preparation after the predecessor has
completed and before `PhaseRuntime.prepare`. Build accepted parameter maps and
stress from `completed_phase_maps[phase_index - 1]`. Phase-0 preparation must
not be re-run during Phase-1 refinement.

Construct the final Phase-1 metric order deterministically as:

```text
FRE (and any explicitly preserved non-EGI metrics), EGI-fine,
EGI-middle, EGI-broad
```

The objective must resolve blocks through semantic metric descriptors. If a
metric-index field remains in `ResidualBlockSpec` for compatibility, resolve
semantic keys to indices once during preparation and persist the mapping.

### B3. Automatic and fixed reproduction modes

The EGI phase preparation supports:

- `auto`: execute the support sweep and select roles;
- `fixed`: accept three physical/pixel supports plus a saved selection record,
  validate them against the current grid/noise model, and construct identical
  metrics without rerunning probes.

`fixed` is required for exact reproduction and campaign controls. It must not
be a hidden fallback when `auto` fails.

### B4. Lifecycle and persistence tests

Add tests in `tests/vfm/test_phasepreparation.py`, plus integration tests for
identification history:

- preparation runs once and before metric initialization;
- the declarative `IdentificationPhase` remains unchanged;
- shared copied objects retain intended identity relationships;
- predecessor maps/stress are copies and mutation attempts cannot alter
  accepted state;
- automatic preparation creates exactly three role-labelled EGI metrics;
- refinement and repeated solves do not rerun support selection;
- fixed mode reproduces automatic metric definitions and common metric values;
- preparation diagnostics survive YAML save/load;
- old configurations/results with no preparation still round trip and produce
  unchanged scientific results;
- invalid Phase-0 predecessor-dependent preparation fails validation.

### B5. Milestone-B gate

Run one short existing-objective identification in both `auto` and `fixed`
modes. The resolved supports, initial metrics, final metrics, DOFs, and maps
must agree to numerical tolerance. A no-preparation legacy smoke test must
also remain bitwise or tolerance-equivalent to its prior result.

## 6. Milestone C — sensitivity-prepared objective

### C1. Configuration and prepared state

Create `SensitivityInformationObjectiveConfig` containing only declared
method constants:

- semantic residual block definitions for FRE and the three EGI roles;
- noise snapshot;
- primary and check finite-difference steps;
- meaningful normalised DOF movement used for the absolute observability rule;
- null-calibrated response threshold;
- projection covariance eigenvalue floor;
- robust-loss transition in standard-noise units;
- parameter-name to yield/hardening group mapping.

Create a private immutable prepared-state object holding the frozen canonical
layout, reference checksum, sensitivity matrix, retained basis, projected
noise whitening transform, FRE and broad block slices, and diagnostics.
Calling the objective before `prepare_solve` is an error.

### C2. Canonical residual and sensitivity

During `prepare_solve(context)`:

1. resolve semantic block specifications to the actual metric list;
2. freeze masks, signed bias correction, diagonal noise whitening, and
   within-block balance using the reference metric results;
3. evaluate the canonical residual at the accepted DOFs;
4. calculate every Jacobian column with the existing bound-aware native-DOF
   finite-difference machinery;
5. verify exact restoration by checksum and a repeat scalar evaluation;
6. run the secondary step check for weakest/strongest columns and record
   disagreement;
7. determine retained noise-resolved material directions;
8. construct the projected-noise whitening transform;
9. freeze the prepared state and return compact diagnostics.

Use derivatives with respect to normalised DOFs so columns have a common
interpretation. The absolute rank rule retains a singular direction only when
the configured meaningful DOF movement produces a response above the
null-calibrated noise threshold. A small relative numerical tolerance may be
used only after this physical threshold.

The implementation must explicitly propagate the canonical transformed-noise
covariance into projected coordinates. Form the symmetric projected
covariance, eigendecompose it, apply the declared eigenvalue floor, and store
the resulting inverse square-root. Validate its calibration with Monte Carlo
null residuals: retained projected coordinates should have RMS near one.

### C3. Objective evaluation

For every optimiser metric result:

1. evaluate the frozen canonical vector;
2. calculate and projected-noise-whiten `Q.T @ z` for the material term;
3. extract the full whitened FRE guard;
4. extract the full whitened broad-EGI guard;
5. apply the same fixed robust loss to each coordinate;
6. take the mean inside each role and the arithmetic mean of the three roles.

No runtime scalar alpha is exposed. Fine and middle EGI enter through the
material projection; FRE and broad EGI appear both in that projection when
informative and as explicit mechanical guards.

`diagnostics()` returns the frozen preparation summary plus current component
costs. It must not rebuild sensitivities. Add an internal preparation counter
and checksum so tests can prove no transform changes during optimiser calls.

### C4. Leverage diagnostics

Calculate diagnostics from the retained projection without using truth:

- total squared leverage by semantic residual block and EGI role;
- leverage by load frame and the diagnostic load regimes;
- spatial leverage maps or compact quantiles for each EGI support;
- contribution by yield/hardening-labelled DOF groups;
- unresolved column list and yield-hardening maximum subspace correlation.

Persist compact values in YAML. Optional full maps go to an NPZ sidecar named
by phase/solve index. Pre-yield hardening leverage is a diagnostic warning,
not a manually zeroed weight in version 1.

### C5. Milestone-C tests

Add tests in `tests/vfm/test_objectivefuncsensitivityinformation.py` and extend
`test_materialprojection.py`:

- hand-calculated linear residual/Jacobian/projected objective;
- material-null residual is excluded from the material term but remains in an
  applicable guard;
- FRE and broad guards cannot be removed by projection;
- equal-role means are invariant to duplicating observations inside a block;
- diagonal noise scaling and unit changes leave the standardised result
  invariant;
- absolute rank changes at the declared noise threshold, not merely at a
  relative singular-value threshold;
- projected Monte Carlo null RMS is approximately one;
- bound-aware one-sided differences and exact accepted-state restoration;
- deterministic repeat preparation and checksums;
- prepared state and counter remain fixed over many objective evaluations;
- Huber/quadratic-to-linear transition has correct values and continuity;
- missing semantic metrics, noise entries, empty rank, and changed residual
  shapes fail with actionable messages;
- solve-preparation diagnostics persist in `SolveResult.details` and survive
  YAML round trip;
- legacy objectives with a `prepare_solve` method returning `None` continue to
  work.

Before online optimisation, run saved-state tests at BF0-BF3 and controlled
synthetic DOF perturbations. Confirm that predicted linear response and actual
metric response agree locally and that the weakest/strongest step checks are
credible.

### C6. Milestone-C gate

Milestone C passes when all restoration/repeatability tests pass, null noise
is calibrated, retained leverage occurs in mechanically plausible frames and
regions, and the objective responds in the predicted direction to controlled
yield/hardening perturbations. This gate does not use final truth error.

## 7. Milestone D — matched BF0-BF3 online gate

### D1. Trial matrix

Run only four new primary cases:

| Objective | Noise | Seed | Maximum BF |
| --- | --- | ---: | ---: |
| data-driven sensitivity | clean | 0 | 3 |
| data-driven sensitivity | clean | 1 | 3 |
| data-driven sensitivity | 1x | 0 | 3 |
| data-driven sensitivity | 1x | 1 | 3 |

Clean seeds are retained to exercise deterministic campaign wiring; their
scientific results should coincide. Reuse the matching raw 7/57 controls
already available. Run the fixed selected-support/full-residual controls only
if the primary comparison cannot distinguish support-selection benefit from
projection benefit.

Every case must share Phase-0 state, initial Phase-1 maps, optimiser settings,
basis proposals, refinement trajectory through BF3, maximum evaluations, and
common postprocessing. Only objective construction/noise realization may
differ as declared.

### D2. Campaign files and resumability

Add one manifest generator and one runner using the existing campaign
conventions. Each case records:

- git commit and dirty-worktree status;
- input, Phase-0 result, noise-model, and support-selection checksums;
- full command and environment metadata;
- automatic/fixed support record;
- progress, completion status, and case runtime;
- one result YAML and optional preparation NPZ sidecars.

The runner must skip only cases with a valid completion marker and matching
configuration checksum. A partial output is resumed or quarantined according
to existing campaign behaviour; it is never silently treated as complete.

### D3. Common analysis

Produce a single matched report with, at each BF0-BF3 state:

- yielded, high-plastic, whole-ROI, weld/HAZ, and parent recovery errors;
- common raw EGI 7/57, selected fine/middle/broad EGI, and FRE diagnostics;
- material, FRE-guard, broad-guard, and total training costs;
- retained rank, singular spectrum, condition estimate, unresolved DOFs, and
  yield-hardening correlation;
- leverage by support, load regime, and parameter group;
- optimiser evaluations, success/status, and runtime;
- clean/noise and seed summaries with individual trajectories visible.

Truth-derived recovery metrics are analysis columns only. They cannot be read
by the selection or objective code.

### D4. Predeclared decision gate

The BF3 gate passes only when all of the following hold:

1. all four primary cases finish all requested fixed-BF solves with no
   non-finite metrics, state-restoration failure, or preparation drift;
2. support selection satisfies Milestone A and clean/noisy decisions are the
   same or adjacent candidate-bank entries;
3. FRE and broad-EGI common diagnostics do not regress beyond their propagated
   noise uncertainty relative to the matched raw control;
4. median yielded and high-plastic recovery trajectories through BF1-BF3
   match or improve the raw 7/57 control in both clean and noisy conditions;
5. no individual noisy seed shows a severe adverse recovery or mechanical
   closure regression hidden by the median;
6. retained rank and principal leverage are stable under the declared finite-
   difference step check.

Quantify `noise uncertainty` and `severe adverse` from the existing null/noise
ensemble before opening the trial results; write those numerical limits into
the campaign analysis config. Do not choose them after seeing truth errors.

If the gate fails, attribute the failure to one of support selection, noise
calibration, sensitivity/rank preparation, objective guards, or optimiser
interaction. Run only the fixed-support full-residual control needed to make
that attribution. Do not start another window/weight factorial.

If the gate passes, extend the four primary cases unchanged to BF7. Model-
order stopping and selector work remains a later task.

## 8. Implementation sequence for the coding pass

Use this order so every commit leaves a testable seam:

1. Add noise and result data contracts with serialization tests.
2. Add candidate-bank generation and tests.
3. Add material-probe construction and accepted-state/plastic-state helper.
4. Add batched multi-support sensitivity evidence and Fisher selector.
5. Add the offline WDBN1 Milestone-A runner/report and pass its gate.
6. Add phase-preparation protocol, config/result schema, and lifecycle tests.
7. Add automatic/fixed EGI phase preparation and legacy-equivalence tests.
8. Add absolute rank and projected covariance utilities with numerical tests.
9. Add the new prepared objective and solve-diagnostic plumbing.
10. Run saved-state BF0-BF3 objective tests and the full `tests/vfm` suite.
11. Add the four-case campaign manifest/runner and matched analysis.
12. Run a one-case smoke test locally, then generate workstation commands.

Suggested verification after each relevant step:

```bash
uv run pytest tests/vfm/test_egisupports.py -q
uv run pytest tests/vfm/test_materialprobes.py -q
uv run pytest tests/vfm/test_phasepreparation.py -q
uv run pytest tests/vfm/test_materialprojection.py -q
uv run pytest tests/vfm/test_objectivefuncsensitivityinformation.py -q
uv run pytest tests/vfm -q
```

The implementation pass is complete only after the full VFM suite passes and
the Milestone-A offline selection artefact has been inspected. Trial commands
should be generated from the committed campaign manifest after that point,
not hand-composed before filenames and CLI options exist.

## 9. Definition of done by milestone

- **A:** automatic selection produces an explainable persisted
  fine/middle/broad decision from Phase-0 state and declared noise only.
- **B:** the selection is invoked once at the correct lifecycle point,
  automatic and fixed modes reproduce, and legacy runs remain unchanged.
- **C:** every fixed-BF solve uses a frozen, restoration-safe,
  noise-calibrated sensitivity objective with explicit FRE/broad guards and
  durable diagnostics.
- **D:** the four matched BF0-BF3 cases complete and meet the predeclared
  scientific/mechanical gate, or fail with enough evidence to isolate one
  component without reopening a broad search.


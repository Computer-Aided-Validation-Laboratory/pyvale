"""One-shot preparation hooks that construct a runtime phase metric bank."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.metric import IMetric
from pyvale.vfm.metricequilibriumgap import EquilibriumGapMetric
from pyvale.vfm.egisupports import PhysicalEgiSupport
from pyvale.vfm.egisupports import (
    EgiSupportBankConfig,
    EgiSupportInformationSelectionConfig,
    EgiSignalSelectionConfig,
    analyse_egi_signal_sweep,
    analyse_egi_support_information,
    generate_physical_egi_support_bank,
    generate_odd_pixel_egi_support_bank,
    select_log_spaced_egi_supports,
    select_information_egi_supports,
)
from pyvale.vfm.metricequilibriumgap import evaluate_equilibrium_gap_batch
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.slicewise_utils import SliceConfig


Summary = dict[str, object]


@dataclass(slots=True, frozen=True)
class PhasePreparationContext:
    """Immutable predecessor state supplied before runtime metrics initialise."""

    phase_index: int
    experiment_data: ExperimentData
    constitutive_law: IConstitutiveLaw
    parameter_map_size: npt.NDArray[np.uint32]
    accepted_parameter_maps: dict[str, npt.NDArray[np.float64]]
    accepted_stress: npt.NDArray[np.float64]
    configured_metrics: tuple[IMetric, ...]


@dataclass(slots=True)
class PhasePreparationResult:
    """Replacement metric list plus durable, non-array diagnostics."""

    metrics: list[IMetric]
    diagnostics: Summary


class IPhasePreparation(Protocol):
    """Build data-dependent runtime metrics exactly once before a phase."""

    def prepare(self, context: PhasePreparationContext) -> PhasePreparationResult:
        """Return copied runtime metrics and serialisable diagnostics."""


@dataclass(slots=True, frozen=True)
class CombinedPhasePreparation:
    """Apply multiple one-shot preparations to the same accepted Phase-0 state."""

    preparations: tuple[tuple[str, IPhasePreparation], ...]

    def prepare(self, context: PhasePreparationContext) -> PhasePreparationResult:
        metrics = list(context.configured_metrics)
        diagnostics: dict[str, object] = {"mode": "combined"}
        for name, preparation in self.preparations:
            stage_context = PhasePreparationContext(
                phase_index=context.phase_index,
                experiment_data=context.experiment_data,
                constitutive_law=context.constitutive_law,
                parameter_map_size=context.parameter_map_size,
                accepted_parameter_maps=context.accepted_parameter_maps,
                accepted_stress=context.accepted_stress,
                configured_metrics=tuple(metrics),
            )
            prepared = preparation.prepare(stage_context)
            metrics = list(prepared.metrics)
            diagnostics[name] = prepared.diagnostics
        diagnostics["metric_order"] = [type(metric).__name__ for metric in metrics]
        return PhasePreparationResult(metrics, diagnostics)


@dataclass(slots=True, frozen=True)
class FreResolutionSelectionConfig:
    """Truth-free Monte-Carlo qualification for longitudinal FRE resolution."""

    candidate_start: int = 9
    candidate_step: int = 4
    minimum_rows_per_slice: float = 2.0
    minimum_points_per_slice: int = 1
    minimum_correlation_p10: float = 0.95
    maximum_nrmse_p50: float = 0.25
    replicates: int = 16
    random_seed: int = 20260901
    strain_noise_sigmas: tuple[float, float, float] = (
        172.51419993963228e-6,
        174.86979476446461e-6,
        123.49056061371802e-6,
    )
    strain_filter_sigmas_mm: tuple[tuple[float, float], ...] = (
        (0.09500277042388916, 0.03187108039855957),
        (0.03166759014129639, 0.09561324119567871),
        (0.052779316902160645, 0.05311846733093262),
    )
    force_noise_sigma: float = 0.3216528022847312

    def __post_init__(self) -> None:
        if self.candidate_start < 1 or self.candidate_start % 2 == 0:
            raise ValueError("candidate_start must be a positive odd integer.")
        if self.candidate_step < 1 or self.candidate_step % 2 != 0:
            raise ValueError("candidate_step must be a positive even integer.")
        if self.minimum_rows_per_slice < 1.0:
            raise ValueError("minimum_rows_per_slice must be at least one.")
        if self.minimum_points_per_slice < 1 or self.replicates < 2:
            raise ValueError("FRE selection requires positive coverage and at least two replicates.")
        if not -1.0 <= self.minimum_correlation_p10 <= 1.0:
            raise ValueError("minimum_correlation_p10 must lie in [-1, 1].")
        if self.maximum_nrmse_p50 < 0.0:
            raise ValueError("maximum_nrmse_p50 must be non-negative.")
        if len(self.strain_noise_sigmas) != 3 or len(self.strain_filter_sigmas_mm) != 3:
            raise ValueError("FRE strain-noise configuration requires three components.")
        if any(not np.isfinite(value) or value < 0.0 for value in self.strain_noise_sigmas):
            raise ValueError("strain_noise_sigmas must be finite and non-negative.")
        if any(
            len(pair) != 2 or any(not np.isfinite(value) or value < 0.0 for value in pair)
            for pair in self.strain_filter_sigmas_mm
        ):
            raise ValueError("strain_filter_sigmas_mm must contain finite non-negative y/x pairs.")
        if not np.isfinite(self.force_noise_sigma) or self.force_noise_sigma < 0.0:
            raise ValueError("force_noise_sigma must be finite and non-negative.")


@dataclass(slots=True, frozen=True)
class FinestStableFrePreparation:
    """Install the finest noise-stable, adequately sampled FRE slicing.

    Independent measurement-noise realisations are added to the observed
    dataset at the accepted homogeneous material state. Candidate profiles
    are compared with the unperturbed observed profile; synthetic material
    truth is never consulted. The finest candidate passing sampling,
    coverage, repeatability and NRMSE limits is frozen before BF1.
    """

    config: FreResolutionSelectionConfig = FreResolutionSelectionConfig()
    diagnostic_callback: Callable[[str, dict[str, object]], None] | None = None

    def prepare(self, context: PhasePreparationContext) -> PhasePreparationResult:
        force_indices = [
            index for index, metric in enumerate(context.configured_metrics)
            if isinstance(metric, SliceWiseForceReconstructionMetric)
        ]
        if len(force_indices) != 1:
            raise ValueError("FRE resolution preparation requires exactly one slice-force metric.")
        template = context.configured_metrics[force_indices[0]]
        if template.slice_config is None:
            raise ValueError("FRE resolution preparation requires a declarative SliceConfig.")
        axis = template.slice_config.axis
        experiment = context.experiment_data
        axis_points = experiment.strain.shape[2 if axis == "y" else 3]
        maximum = int(np.floor(axis_points / self.config.minimum_rows_per_slice))
        if maximum % 2 == 0:
            maximum -= 1
        candidates = list(range(self.config.candidate_start, maximum + 1, self.config.candidate_step))
        for value in (33, 63, 105, maximum):
            if self.config.candidate_start <= value <= maximum and value % 2 == 1:
                candidates.append(value)
        candidates = sorted(set(candidates))
        if not candidates:
            raise RuntimeError("No FRE slice candidate satisfies the sampling limit.")

        replicate_experiments, replicate_stresses = self._replicates(context, axis)
        rows: list[dict[str, object]] = []
        for count in candidates:
            metric = SliceWiseForceReconstructionMetric(
                slice_config=SliceConfig(axis=axis, num_slices=count)
            )
            metric.initialise(experiment)
            reference = metric.evaluate_force_recon_error(context.accepted_stress, experiment)
            reference_residual = np.asarray(
                reference.metric_result.additional_fields["normalised_residual"],
                dtype=np.float64,
            )
            temporal_weights = np.asarray(
                reference.metric_result.additional_fields["temporal_weights"],
                dtype=np.float64,
            )
            reference_profile = np.sqrt(
                np.nansum(temporal_weights[:, None] * reference_residual**2, axis=0)
            )
            correlations: list[float] = []
            nrmses: list[float] = []
            for replicate_experiment, replicate_stress in zip(
                replicate_experiments, replicate_stresses, strict=True,
            ):
                replicate = metric.evaluate_force_recon_error(
                    replicate_stress, replicate_experiment
                )
                residual = np.asarray(
                    replicate.metric_result.additional_fields["normalised_residual"],
                    dtype=np.float64,
                )
                profile = np.sqrt(
                    np.nansum(temporal_weights[:, None] * residual**2, axis=0)
                )
                valid = np.isfinite(reference_profile) & np.isfinite(profile)
                if np.count_nonzero(valid) < 3:
                    correlations.append(float("nan"))
                elif np.std(reference_profile[valid]) <= np.finfo(float).eps:
                    correlations.append(float("nan"))
                else:
                    correlations.append(float(np.corrcoef(
                        profile[valid], reference_profile[valid]
                    )[0, 1]))
                scale = max(
                    float(np.linalg.norm(reference_profile[valid])),
                    np.finfo(float).eps,
                )
                nrmses.append(float(np.linalg.norm(
                    profile[valid] - reference_profile[valid]
                ) / scale))
            partition = metric.slice_partition
            assert partition is not None
            correlation_p10 = float(np.nanquantile(correlations, 0.10))
            nrmse_p50 = float(np.nanmedian(nrmses))
            row = {
                "num_slices": count,
                "approx_rows_per_slice": float(axis_points / count),
                "mean_span_mm": float(np.mean(partition.spans)),
                "minimum_points": int(np.min(partition.point_counts)),
                "correlation_p10": correlation_p10,
                "nrmse_p50": nrmse_p50,
            }
            row["passes"] = bool(
                row["approx_rows_per_slice"] >= self.config.minimum_rows_per_slice
                and row["minimum_points"] >= self.config.minimum_points_per_slice
                and np.isfinite(correlation_p10)
                and correlation_p10 >= self.config.minimum_correlation_p10
                and np.isfinite(nrmse_p50)
                and nrmse_p50 <= self.config.maximum_nrmse_p50
            )
            rows.append(row)
        passing = [row for row in rows if row["passes"]]
        if not passing:
            raise RuntimeError("FRE resolution selector found no stable adequately sampled candidate.")
        selected = passing[-1]
        selected_count = int(selected["num_slices"])
        metrics = [copy.deepcopy(metric) for metric in context.configured_metrics]
        metrics[force_indices[0]] = SliceWiseForceReconstructionMetric(
            slice_config=SliceConfig(axis=axis, num_slices=selected_count)
        )
        diagnostics = {
            "mode": "finest_stable_noise_propagation",
            "truth_used_for_selection": False,
            "axis": axis,
            "selected_num_slices": selected_count,
            "selection_rule": "finest candidate passing sampling, coverage, correlation-p10 and NRMSE-p50 criteria",
            "thresholds": {
                "minimum_rows_per_slice": self.config.minimum_rows_per_slice,
                "minimum_points_per_slice": self.config.minimum_points_per_slice,
                "minimum_correlation_p10": self.config.minimum_correlation_p10,
                "maximum_nrmse_p50": self.config.maximum_nrmse_p50,
            },
            "noise": {
                "replicates": self.config.replicates,
                "random_seed": self.config.random_seed,
                "strain_noise_sigmas": list(self.config.strain_noise_sigmas),
                "strain_filter_sigmas_mm_yx": [list(value) for value in self.config.strain_filter_sigmas_mm],
                "force_noise_sigma": self.config.force_noise_sigma,
            },
            "candidates": rows,
        }
        if self.diagnostic_callback is not None:
            self.diagnostic_callback("fre_resolution_sweep", diagnostics)
        return PhasePreparationResult(metrics, diagnostics)

    def _replicates(
        self, context: PhasePreparationContext, axis: str,
    ) -> tuple[list[ExperimentData], list[npt.NDArray[np.float64]]]:
        experiment = context.experiment_data
        x = np.asarray(experiment.specimen_geometry.x, dtype=np.float64)
        y = np.asarray(experiment.specimen_geometry.y, dtype=np.float64)
        dx = float(np.nanmedian(np.abs(np.diff(x, axis=1))))
        dy = float(np.nanmedian(np.abs(np.diff(y, axis=0))))
        mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(x, y)
        output_experiments: list[ExperimentData] = []
        output_stresses: list[npt.NDArray[np.float64]] = []
        component_index = 0 if axis == "x" else 1
        for replicate_index in range(self.config.replicates):
            rng = np.random.default_rng(self.config.random_seed + replicate_index)
            replicate = copy.deepcopy(experiment)
            strain = np.asarray(experiment.strain, dtype=np.float64).copy()
            for component, (target_sigma, sigma_mm) in enumerate(zip(
                self.config.strain_noise_sigmas,
                self.config.strain_filter_sigmas_mm,
                strict=True,
            )):
                filter_sigma = (sigma_mm[0] / dy, sigma_mm[1] / dx)
                for state in range(strain.shape[0]):
                    sample = gaussian_filter(
                        rng.standard_normal(mask.shape), sigma=filter_sigma,
                        mode="reflect",
                    )
                    sample -= float(np.mean(sample[mask]))
                    scale = float(np.std(sample[mask]))
                    if scale <= np.finfo(np.float64).eps:
                        raise RuntimeError("Generated zero-variance FRE noise field.")
                    strain[state, component, mask] += target_sigma * sample[mask] / scale
            replicate.strain = strain
            force = np.asarray(experiment.boundary_conditions.force, dtype=np.float64).copy()
            force[:, component_index] += rng.normal(
                0.0, self.config.force_noise_sigma, force.shape[0]
            )
            replicate.boundary_conditions.force = force
            output_experiments.append(replicate)
            output_stresses.append(np.asarray(
                context.constitutive_law.calculate_stress(
                    replicate.strain, context.accepted_parameter_maps
                ),
                dtype=np.float64,
            ))
        return output_experiments, output_stresses


@dataclass(slots=True, frozen=True)
class FixedEgiSupportPreparation:
    """Install a previously selected fine/(middle)/broad EGI metric bank.

    This is the reproducibility path for a support decision made by the
    automatic probe sweep.  It deliberately validates role completeness and
    does not silently substitute a hand-selected window when selection fails.
    """

    supports: tuple[tuple[str, PhysicalEgiSupport], ...]

    def __post_init__(self) -> None:
        roles = tuple(role for role, _ in self.supports)
        if roles not in (("fine", "middle", "broad"), ("fine", "broad")):
            raise ValueError(
                "Fixed EGI supports must be ordered fine, middle, broad or fine, broad."
            )

    def prepare(self, context: PhasePreparationContext) -> PhasePreparationResult:
        templates = [
            metric for metric in context.configured_metrics
            if isinstance(metric, EquilibriumGapMetric)
        ]
        if not templates:
            raise ValueError(
                "Fixed EGI support preparation requires an EquilibriumGapMetric template."
            )
        template = templates[0]
        preserved = [
            metric for metric in context.configured_metrics
            if not isinstance(metric, EquilibriumGapMetric)
        ]
        metrics = [*preserved]
        support_diagnostics: dict[str, object] = {}
        for role, support in self.supports:
            metric = copy.deepcopy(template)
            metric.window_size = np.asarray(support.window_size, dtype=np.uint32)
            # Window-dependent caches cannot be shared with the template.
            metric._operator = None
            metric._kernel_fft_cache = {}
            metrics.append(metric)
            support_diagnostics[role] = support.diagnostics()
        return PhasePreparationResult(
            metrics=metrics,
            diagnostics={
                "mode": "fixed",
                "roles": support_diagnostics,
                "metric_order": [type(metric).__name__ for metric in metrics],
            },
        )


@dataclass(slots=True, frozen=True)
class UserFineEgiSupportPreparation:
    """Install user-declared fine plus geometry-derived middle/broad supports.

    The fine window is intentionally not inferred from residuals.  The broad
    support is the unchanged upper member of the odd-pixel geometry bank and
    the middle support is the valid odd bank member nearest the logarithmic
    midpoint.  All three metrics are installed once during phase preparation
    and therefore remain frozen for the complete BF trajectory.
    """

    fine_window: int
    bank_config: EgiSupportBankConfig = EgiSupportBankConfig()
    include_middle: bool = True

    def __post_init__(self) -> None:
        if self.fine_window < 3 or self.fine_window % 2 == 0:
            raise ValueError("fine_window must be an odd integer of at least 3.")

    def prepare(self, context: PhasePreparationContext) -> PhasePreparationResult:
        bank = generate_odd_pixel_egi_support_bank(
            context.experiment_data.specimen_geometry.x,
            context.experiment_data.specimen_geometry.y,
            self.bank_config,
        )
        by_pixels = {support.window_size[0]: support for support in bank}
        if self.fine_window not in by_pixels:
            raise ValueError(
                f"User fine EGI window {self.fine_window} is outside the "
                f"geometry bank [{bank[0].window_size[0]}, {bank[-1].window_size[0]}]."
            )
        fine = by_pixels[self.fine_window]
        broad = bank[-1]
        if fine.window_size == broad.window_size:
            raise ValueError("Fine EGI window must be smaller than the broad geometry cap.")
        eligible = [
            support for support in bank
            if fine.nominal_side_length < support.nominal_side_length
            < broad.nominal_side_length
        ]
        if not eligible:
            raise ValueError("No distinct middle EGI support exists between fine and broad.")
        target = 0.5 * (
            np.log(fine.nominal_side_length) + np.log(broad.nominal_side_length)
        )
        middle = min(
            eligible,
            key=lambda support: abs(np.log(support.nominal_side_length) - target),
        )
        selected = (("fine", fine), ("middle", middle), ("broad", broad))
        if not self.include_middle:
            selected = (("fine", fine), ("broad", broad))
        installed = FixedEgiSupportPreparation(selected).prepare(context)
        return PhasePreparationResult(
            installed.metrics,
            {
                "mode": "user_fine_geometry_middle_broad",
                "fine_source": "explicit_user_input",
                "middle_rule": "nearest_valid_logarithmic_midpoint",
                "broad_rule": "geometry_bank_maximum",
                "include_middle": self.include_middle,
                "roles": installed.diagnostics["roles"],
                "metric_order": installed.diagnostics["metric_order"],
            },
        )


@dataclass(slots=True, frozen=True)
class SimpleEgiSupportPreparation:
    """Select three scales from direct homogeneous EGI signal and coverage."""

    residual_noise_scale: float = 1.0
    bank_config: EgiSupportBankConfig = EgiSupportBankConfig()
    selection_config: EgiSignalSelectionConfig = EgiSignalSelectionConfig()
    diagnostic_callback: Callable[[str, dict[str, object]], None] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.residual_noise_scale) or self.residual_noise_scale <= 0.0:
            raise ValueError("residual_noise_scale must be positive.")

    def prepare(self, context: PhasePreparationContext) -> PhasePreparationResult:
        templates = [
            metric for metric in context.configured_metrics
            if isinstance(metric, EquilibriumGapMetric)
        ]
        if not templates:
            raise ValueError("Simple EGI preparation requires an EquilibriumGapMetric template.")
        x = context.experiment_data.specimen_geometry.x
        y = context.experiment_data.specimen_geometry.y
        supports = generate_odd_pixel_egi_support_bank(x, y, self.bank_config)
        candidate_metrics = [_copy_egi_template(templates[0], support) for support in supports]
        for metric in candidate_metrics:
            metric.initialise(context.experiment_data)
        base_results = evaluate_equilibrium_gap_batch(
            context.accepted_stress, candidate_metrics, include_diagnostics=True
        )
        sweep = analyse_egi_signal_sweep(
            supports,
            [
                np.asarray((result.additional_fields or {})["normalised_gap"])
                for result in base_results
            ],
            [self.residual_noise_scale for _ in supports],
            active_fraction=self.selection_config.active_fraction,
        )
        payload = {
            "phase_index": context.phase_index,
            "residual_field": "normalised_gap",
            "residual_noise_scale": self.residual_noise_scale,
            "sweep": sweep.diagnostics(),
        }
        if self.diagnostic_callback is not None:
            self.diagnostic_callback("simple_egi_support_sweep", payload)
        selection = select_log_spaced_egi_supports(sweep, self.selection_config)
        installed = FixedEgiSupportPreparation(
            tuple(selection.selected_supports(sweep).items())
        ).prepare(context)
        return PhasePreparationResult(
            installed.metrics,
            {
                "mode": "simple_signal_snr",
                "residual_noise": {
                    "kind": "scalar_diagonal_approximation",
                    "scale": self.residual_noise_scale,
                },
                "sweep": sweep.diagnostics(),
                "selection": selection.diagnostics(sweep),
                "installed": installed.diagnostics,
            },
        )


@dataclass(slots=True, frozen=True)
class AutomaticEgiSupportPreparation:
    """Select and install EGI supports from homogeneous-state yield probes.

    This version deliberately has a narrow, explicit contract: it screens
    yield-strength probes using a caller-declared physical parameter range and
    diagonal EGI residual standard deviation.  The latter is an approximation
    until a campaign supplies propagated arrays; it is persisted in the
    result rather than being hidden as a scalar objective weight.
    """

    yield_parameter_name: str = "yield_strength"
    yield_parameter_range: float = 1800.0
    perturbation_fraction: float = 0.01
    local_probe_count: int = 9
    local_probe_width: float | None = None
    residual_noise_scale: float = 1.0
    bank_config: EgiSupportBankConfig = EgiSupportBankConfig()
    selection_config: EgiSupportInformationSelectionConfig = (
        EgiSupportInformationSelectionConfig()
    )
    diagnostic_callback: Callable[[str, dict[str, object]], None] | None = None

    def __post_init__(self) -> None:
        if not self.yield_parameter_name:
            raise ValueError("yield_parameter_name cannot be empty.")
        if not np.isfinite(self.yield_parameter_range) or self.yield_parameter_range <= 0.0:
            raise ValueError("yield_parameter_range must be positive.")
        if not 0.0 < self.perturbation_fraction < 1.0:
            raise ValueError("perturbation_fraction must lie in (0, 1).")
        if self.local_probe_count < 1:
            raise ValueError("local_probe_count must be positive.")
        if self.local_probe_width is not None and (
            not np.isfinite(self.local_probe_width) or self.local_probe_width <= 0.0
        ):
            raise ValueError("local_probe_width must be positive when supplied.")
        if not np.isfinite(self.residual_noise_scale) or self.residual_noise_scale <= 0.0:
            raise ValueError("residual_noise_scale must be positive.")

    def prepare(self, context: PhasePreparationContext) -> PhasePreparationResult:
        if self.yield_parameter_name not in context.accepted_parameter_maps:
            raise ValueError(
                f"Accepted maps have no {self.yield_parameter_name!r} parameter."
            )
        templates = [
            metric for metric in context.configured_metrics
            if isinstance(metric, EquilibriumGapMetric)
        ]
        if not templates:
            raise ValueError(
                "Automatic EGI preparation requires an EquilibriumGapMetric template."
            )
        x = context.experiment_data.specimen_geometry.x
        y = context.experiment_data.specimen_geometry.y
        supports = generate_physical_egi_support_bank(x, y, self.bank_config)
        candidate_metrics = [
            _copy_egi_template(templates[0], support) for support in supports
        ]
        for metric in candidate_metrics:
            metric.initialise(context.experiment_data)
        base_results = evaluate_equilibrium_gap_batch(
            context.accepted_stress, candidate_metrics, include_diagnostics=True
        )
        probe_masks = _local_probe_masks(
            x, y, context.experiment_data.strain, self.local_probe_count,
            self.local_probe_width,
            eligible_mask=_yielded_or_near_yield_mask(
                context.accepted_stress,
                context.accepted_parameter_maps[self.yield_parameter_name],
            ),
        )
        amplitude = self.perturbation_fraction * self.yield_parameter_range
        probe_names = ("homogeneous_yield",) + tuple(
            f"local_yield_{index:02d}" for index in range(len(probe_masks))
        )
        responses: list[list[np.ndarray]] = [[] for _ in supports]
        for probe in (np.ones_like(x, dtype=np.float64), *probe_masks):
            plus_maps = _perturb_parameter_map(
                context.accepted_parameter_maps, self.yield_parameter_name,
                amplitude * probe,
            )
            minus_maps = _perturb_parameter_map(
                context.accepted_parameter_maps, self.yield_parameter_name,
                -amplitude * probe,
            )
            plus = context.constitutive_law.calculate_stress(
                context.experiment_data.strain, plus_maps
            )
            minus = context.constitutive_law.calculate_stress(
                context.experiment_data.strain, minus_maps
            )
            plus_results = evaluate_equilibrium_gap_batch(
                plus, candidate_metrics, include_diagnostics=True
            )
            minus_results = evaluate_equilibrium_gap_batch(
                minus, candidate_metrics, include_diagnostics=True
            )
            for index, (plus_result, minus_result) in enumerate(
                zip(plus_results, minus_results, strict=True)
            ):
                responses[index].append(
                    0.5 * (
                        np.asarray(plus_result.residual, dtype=np.float64)
                        - np.asarray(minus_result.residual, dtype=np.float64)
                    )
                )
        sweep = analyse_egi_support_information(
            supports,
            [np.asarray(result.residual, dtype=np.float64) for result in base_results],
            [self.residual_noise_scale for _ in supports],
            [np.stack(item) for item in responses],
            probe_names=probe_names,
            fisher_regularisation=self.selection_config.fisher_regularisation,
        )
        if self.diagnostic_callback is not None:
            self.diagnostic_callback(
                "egi_support_sweep",
                {
                    "phase_index": context.phase_index,
                    "yield_parameter_name": self.yield_parameter_name,
                    "yield_parameter_range": self.yield_parameter_range,
                    "perturbation_fraction": self.perturbation_fraction,
                    "probe_names": list(probe_names),
                    "residual_noise_scale": self.residual_noise_scale,
                    "sweep": sweep.diagnostics(),
                },
            )
        selection = select_information_egi_supports(sweep, self.selection_config)
        if selection.middle_index is None:
            raise RuntimeError(
                "Automatic EGI preparation found only two noise-resolved "
                "support directions; refusing to install an unverified bank."
            )
        installed = FixedEgiSupportPreparation(
            tuple(selection.selected_supports(sweep).items())
        ).prepare(context)
        diagnostics = {
            "mode": "automatic",
            "yield_parameter_name": self.yield_parameter_name,
            "yield_parameter_range": self.yield_parameter_range,
            "perturbation_fraction": self.perturbation_fraction,
            "probe_names": list(probe_names),
            "residual_noise": {
                "kind": "scalar_diagonal_approximation",
                "scale": self.residual_noise_scale,
            },
            "sweep": sweep.diagnostics(),
            "selection": selection.diagnostics(sweep),
            "installed": installed.diagnostics,
        }
        return PhasePreparationResult(installed.metrics, diagnostics)


def build_phase_preparation_context(
    *,
    phase_index: int,
    experiment_data: ExperimentData,
    constitutive_law: IConstitutiveLaw,
    parameter_map_size: npt.NDArray[np.uint32],
    accepted_parameter_maps: dict[str, npt.NDArray[np.float64]],
    configured_metrics: list[IMetric],
) -> PhasePreparationContext:
    """Copy predecessor maps and reconstruct their accepted stress state."""

    if phase_index <= 0:
        raise ValueError("Phase preparation requiring predecessor state starts at phase 1.")
    maps = {
        name: np.asarray(values, dtype=np.float64).copy()
        for name, values in accepted_parameter_maps.items()
    }
    stress = np.asarray(
        constitutive_law.calculate_stress(experiment_data.strain, maps),
        dtype=np.float64,
    ).copy()
    return PhasePreparationContext(
        phase_index=phase_index,
        experiment_data=experiment_data,
        constitutive_law=constitutive_law,
        parameter_map_size=np.asarray(parameter_map_size, dtype=np.uint32).copy(),
        accepted_parameter_maps=maps,
        accepted_stress=stress,
        configured_metrics=tuple(configured_metrics),
    )


def _copy_egi_template(
    template: EquilibriumGapMetric,
    support: PhysicalEgiSupport,
) -> EquilibriumGapMetric:
    metric = copy.deepcopy(template)
    metric.window_size = np.asarray(support.window_size, dtype=np.uint32)
    metric._operator = None
    metric._kernel_fft_cache = {}
    return metric


def _perturb_parameter_map(
    accepted_maps: dict[str, npt.NDArray[np.float64]],
    parameter_name: str,
    perturbation: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    maps = {
        name: np.asarray(values, dtype=np.float64).copy()
        for name, values in accepted_maps.items()
    }
    maps[parameter_name] = maps[parameter_name] + perturbation
    return maps


def _local_probe_masks(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    strain: npt.ArrayLike,
    count: int,
    width: float | None,
    *,
    eligible_mask: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Place reproducible smooth probes over finite specimen observations."""

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    valid = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & np.all(np.isfinite(np.asarray(strain, dtype=np.float64)), axis=(0, 1))
    )
    if eligible_mask is not None:
        eligible = np.asarray(eligible_mask, dtype=bool)
        if eligible.shape != valid.shape:
            raise ValueError("eligible_mask must match the specimen grid shape.")
        valid &= eligible
    if not np.any(valid):
        raise ValueError("No finite specimen points are available for local probes.")
    x_valid = x_values[valid]
    y_valid = y_values[valid]
    extent = min(float(np.ptp(x_valid)), float(np.ptp(y_valid)))
    if extent <= np.finfo(np.float64).eps:
        raise ValueError("Cannot place probes on a zero-extent specimen.")
    resolved_width = width if width is not None else max(extent / 12.0, np.finfo(float).eps)
    rows = int(np.ceil(np.sqrt(count)))
    cols = int(np.ceil(count / rows))
    x_centres = np.linspace(float(np.min(x_valid)), float(np.max(x_valid)), cols + 2)[1:-1]
    y_centres = np.linspace(float(np.min(y_valid)), float(np.max(y_valid)), rows + 2)[1:-1]
    masks: list[npt.NDArray[np.float64]] = []
    for y_centre in y_centres:
        for x_centre in x_centres:
            if len(masks) == count:
                break
            mask = np.exp(
                -0.5 * (
                    ((x_values - x_centre) / resolved_width) ** 2
                    + ((y_values - y_centre) / resolved_width) ** 2
                )
            )
            mask[~valid] = 0.0
            maximum = float(np.max(mask))
            if maximum <= np.finfo(np.float64).eps:
                continue
            masks.append(mask / maximum)
    if len(masks) != count:
        raise RuntimeError("Could not construct the requested local material probes.")
    return tuple(masks)


def _yielded_or_near_yield_mask(
    stress: npt.ArrayLike,
    yield_strength: npt.ArrayLike,
    *,
    threshold_fraction: float = 0.8,
) -> npt.NDArray[np.bool_]:
    """Return Phase-0 points that reached a yielded/near-yield stress state."""

    values = np.asarray(stress, dtype=np.float64)
    threshold = np.asarray(yield_strength, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("stress must have shape (frame, 3, y, x).")
    if threshold.shape != values.shape[2:]:
        raise ValueError("yield_strength map must match stress spatial shape.")
    equivalent = np.sqrt(
        values[:, 0] ** 2
        + values[:, 1] ** 2
        - values[:, 0] * values[:, 1]
        + 3.0 * values[:, 2] ** 2
    )
    return np.any(
        np.isfinite(equivalent)
        & np.isfinite(threshold)[np.newaxis]
        & (equivalent >= threshold_fraction * threshold[np.newaxis]),
        axis=0,
    )

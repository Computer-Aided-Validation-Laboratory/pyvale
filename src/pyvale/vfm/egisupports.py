"""Physical EGI support sweeps and sparse information-based selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from pyvale.vfm.residualfeatures import physical_length_to_odd_pixels


FloatArray = npt.NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class EgiSupportBankConfig:
    """Physical candidate-bank definition used before adaptive identification.

    Requested side lengths are logarithmically distributed in physical units;
    duplicate odd pixel windows are removed by
    :func:`resolve_physical_egi_supports`.
    """

    candidate_count: int = 10
    minimum_pixels: int = 3
    maximum_bbox_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.candidate_count < 2:
            raise ValueError("candidate_count must be at least two.")
        if self.minimum_pixels < 3 or self.minimum_pixels % 2 == 0:
            raise ValueError("minimum_pixels must be odd and at least three.")
        if not 0.0 < self.maximum_bbox_fraction <= 1.0:
            raise ValueError("maximum_bbox_fraction must lie in (0, 1].")


@dataclass(slots=True, frozen=True)
class EgiSignalSelectionConfig:
    """Simple signal/coverage gates for fine/log-middle/broad supports."""

    minimum_coverage_fraction: float = 0.5
    minimum_signal_to_noise: float = 1.0
    active_fraction: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_coverage_fraction <= 1.0:
            raise ValueError("minimum_coverage_fraction must lie in [0, 1].")
        if not np.isfinite(self.minimum_signal_to_noise) or self.minimum_signal_to_noise < 0.0:
            raise ValueError("minimum_signal_to_noise must be non-negative.")
        if not 0.0 < self.active_fraction <= 1.0:
            raise ValueError("active_fraction must lie in (0, 1].")


@dataclass(slots=True, frozen=True)
class EgiSignalEvidence:
    """Direct homogeneous-state EGI signal evidence for one support."""

    support: "PhysicalEgiSupport"
    valid_count: int
    total_count: int
    coverage_fraction: float
    characteristic_signal: float
    noise_rms: float
    signal_to_noise: float

    def diagnostics(self) -> dict[str, object]:
        return {
            "support": self.support.diagnostics(),
            "valid_count": self.valid_count,
            "total_count": self.total_count,
            "coverage_fraction": self.coverage_fraction,
            "characteristic_signal": self.characteristic_signal,
            "noise_rms": self.noise_rms,
            "signal_to_noise": self.signal_to_noise,
        }


@dataclass(slots=True, frozen=True)
class EgiSignalSweep:
    """Ordered direct-signal evidence for a candidate EGI support bank."""

    evidence: tuple[EgiSignalEvidence, ...]
    active_fraction: float

    def diagnostics(self) -> dict[str, object]:
        return {
            "active_fraction": self.active_fraction,
            "evidence": [item.diagnostics() for item in self.evidence],
        }


@dataclass(slots=True, frozen=True)
class EgiSignalSelection:
    """Fine, logarithmic-middle and broad indices from direct EGI SNR."""

    fine_index: int
    middle_index: int
    broad_index: int

    def roles(self) -> dict[str, int]:
        return {
            "fine": self.fine_index,
            "middle": self.middle_index,
            "broad": self.broad_index,
        }

    def selected_supports(self, sweep: "EgiSignalSweep") -> dict[str, "PhysicalEgiSupport"]:
        return {role: sweep.evidence[index].support for role, index in self.roles().items()}

    def diagnostics(self, sweep: "EgiSignalSweep") -> dict[str, object]:
        fine = sweep.evidence[self.fine_index].support.nominal_side_length
        broad = sweep.evidence[self.broad_index].support.nominal_side_length
        target = float(np.sqrt(fine * broad))
        return {
            "roles": self.roles(),
            "log_middle_target": target,
            "selected_supports": {
                role: sweep.evidence[index].support.diagnostics()
                for role, index in self.roles().items()
            },
        }


@dataclass(slots=True, frozen=True)
class EgiSupportInformationEvidence:
    """Whitened multi-probe information for one physical EGI support.

    ``sensitivity`` is intentionally retained in-memory for selection.  Its
    rows are the support's valid observations after diagonal noise whitening
    and within-support normalisation; columns are declared material probes.
    Campaign persistence should use :meth:`diagnostics`, not this array.
    """

    support: PhysicalEgiSupport
    valid_count: int
    total_count: int
    coverage_fraction: float
    probe_response_to_noise: tuple[float, ...]
    singular_values: tuple[float, ...]
    fisher_log_determinant: float
    sensitivity: FloatArray

    def diagnostics(self, probe_names: Sequence[str]) -> dict[str, object]:
        return {
            "support": self.support.diagnostics(),
            "valid_count": self.valid_count,
            "total_count": self.total_count,
            "coverage_fraction": self.coverage_fraction,
            "probe_response_to_noise": {
                name: value
                for name, value in zip(
                    probe_names, self.probe_response_to_noise, strict=True
                )
            },
            "singular_values": list(self.singular_values),
            "fisher_log_determinant": self.fisher_log_determinant,
        }


@dataclass(slots=True, frozen=True)
class EgiSupportInformationSweep:
    """Multi-probe evidence used by the fine/middle/broad selector."""

    evidence: tuple[EgiSupportInformationEvidence, ...]
    probe_names: tuple[str, ...]
    fisher_regularisation: float

    def diagnostics(self) -> dict[str, object]:
        return {
            "probe_names": list(self.probe_names),
            "fisher_regularisation": self.fisher_regularisation,
            "evidence": [item.diagnostics(self.probe_names) for item in self.evidence],
        }


@dataclass(slots=True, frozen=True)
class EgiSupportInformationSelectionConfig:
    """Declared gates for data-driven fine/middle/broad selection."""

    minimum_coverage_fraction: float = 0.5
    minimum_response_to_noise: float = 1.0
    minimum_local_probe_fraction: float = 0.5
    fisher_regularisation: float = 1.0e-12
    minimum_middle_information_gain: float = 0.0
    local_probe_prefix: str = "local_yield"
    homogeneous_probe_prefix: str = "homogeneous_yield"

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_coverage_fraction", self.minimum_coverage_fraction),
            ("minimum_local_probe_fraction", self.minimum_local_probe_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if not np.isfinite(self.minimum_response_to_noise) or self.minimum_response_to_noise < 0.0:
            raise ValueError("minimum_response_to_noise must be non-negative.")
        if not np.isfinite(self.fisher_regularisation) or self.fisher_regularisation <= 0.0:
            raise ValueError("fisher_regularisation must be positive.")
        if not np.isfinite(self.minimum_middle_information_gain):
            raise ValueError("minimum_middle_information_gain must be finite.")
        if not self.local_probe_prefix or not self.homogeneous_probe_prefix:
            raise ValueError("Probe prefixes cannot be empty.")


@dataclass(slots=True, frozen=True)
class EgiSupportInformationSelection:
    """Role-labelled support decision and auditable incremental evidence."""

    fine_index: int
    middle_index: int | None
    broad_index: int
    middle_information_gain: float | None
    middle_smallest_singular_gain: float | None
    status: str

    def roles(self) -> dict[str, int]:
        result = {"fine": self.fine_index}
        if self.middle_index is not None:
            result["middle"] = self.middle_index
        result["broad"] = self.broad_index
        return result

    def diagnostics(
        self, sweep: "EgiSupportInformationSweep"
    ) -> dict[str, object]:
        return {
            "roles": self.roles(),
            "status": self.status,
            "middle_information_gain": self.middle_information_gain,
            "middle_smallest_singular_gain": self.middle_smallest_singular_gain,
            "selected_supports": {
                role: sweep.evidence[index].support.diagnostics()
                for role, index in self.roles().items()
            },
        }

    def selected_supports(
        self, sweep: "EgiSupportInformationSweep"
    ) -> dict[str, PhysicalEgiSupport]:
        """Return role-labelled immutable supports for runtime metric creation."""

        return {
            role: sweep.evidence[index].support
            for role, index in self.roles().items()
        }


@dataclass(slots=True, frozen=True)
class PhysicalEgiSupport:
    """One unique pixel window resolved from physical side-length requests."""

    requested_side_lengths: tuple[float, ...]
    window_size: tuple[int, int]
    nominal_side_lengths: tuple[float, float]
    grid_spacing: tuple[float, float]

    @property
    def nominal_side_length(self) -> float:
        return float(np.mean(self.nominal_side_lengths))

    def diagnostics(self) -> dict[str, object]:
        result = asdict(self)
        result["nominal_side_length"] = self.nominal_side_length
        return result


@dataclass(slots=True, frozen=True)
class EgiSupportEvidence:
    """Coverage, residual, noise, and parameter-response evidence."""

    support: PhysicalEgiSupport
    valid_count: int
    total_count: int
    coverage_fraction: float
    residual_rms: float
    whitened_residual_rms: float
    noise_rms: float
    parameter_response_rms: tuple[tuple[str, float], ...]
    parameter_response_to_noise: tuple[tuple[str, float], ...]

    def response_to_noise(self, parameter_group: str) -> float:
        try:
            return dict(self.parameter_response_to_noise)[parameter_group]
        except KeyError as exc:
            raise KeyError(
                f"No response evidence for parameter group {parameter_group!r}."
            ) from exc

    def diagnostics(self) -> dict[str, object]:
        return {
            "support": self.support.diagnostics(),
            "valid_count": self.valid_count,
            "total_count": self.total_count,
            "coverage_fraction": self.coverage_fraction,
            "residual_rms": self.residual_rms,
            "whitened_residual_rms": self.whitened_residual_rms,
            "noise_rms": self.noise_rms,
            "parameter_response_rms": dict(self.parameter_response_rms),
            "parameter_response_to_noise": dict(
                self.parameter_response_to_noise
            ),
        }


@dataclass(slots=True, frozen=True)
class EgiSupportSweepResult:
    """Evidence and pairwise response redundancy for a support bank."""

    evidence: tuple[EgiSupportEvidence, ...]
    redundancy_parameter_group: str
    absolute_cosine_redundancy: FloatArray

    def diagnostics(self) -> dict[str, object]:
        return {
            "evidence": [item.diagnostics() for item in self.evidence],
            "redundancy_parameter_group": self.redundancy_parameter_group,
            "absolute_cosine_redundancy": (
                self.absolute_cosine_redundancy.tolist()
            ),
        }


@dataclass(slots=True, frozen=True)
class EgiSupportSelectionConfig:
    """Thresholds for role-based fine/middle/broad support selection."""

    parameter_group: str = "yield"
    maximum_supports: int = 3
    minimum_coverage_fraction: float = 0.5
    minimum_response_to_noise: float = 1.0
    maximum_middle_redundancy: float = 0.95

    def __post_init__(self) -> None:
        if not self.parameter_group:
            raise ValueError("parameter_group cannot be empty.")
        if not 1 <= self.maximum_supports <= 3:
            raise ValueError("maximum_supports must lie in [1, 3].")
        if not 0.0 <= self.minimum_coverage_fraction <= 1.0:
            raise ValueError("minimum_coverage_fraction must lie in [0, 1].")
        if not np.isfinite(self.minimum_response_to_noise) or (
            self.minimum_response_to_noise < 0.0
        ):
            raise ValueError("minimum_response_to_noise must be non-negative.")
        if not 0.0 <= self.maximum_middle_redundancy <= 1.0:
            raise ValueError("maximum_middle_redundancy must lie in [0, 1].")


@dataclass(slots=True, frozen=True)
class EgiSupportSelection:
    """Sparse support selection with explicit scientific roles."""

    selected_indices: tuple[int, ...]
    roles: tuple[tuple[str, int], ...]
    parameter_group: str

    def diagnostics(self, sweep: EgiSupportSweepResult) -> dict[str, object]:
        return {
            "selected_indices": list(self.selected_indices),
            "roles": dict(self.roles),
            "parameter_group": self.parameter_group,
            "selected_supports": [
                sweep.evidence[index].support.diagnostics()
                for index in self.selected_indices
            ],
        }


def resolve_physical_egi_supports(
    side_lengths: Sequence[float],
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    minimum_pixels: int = 3,
) -> tuple[PhysicalEgiSupport, ...]:
    """Resolve and deduplicate physical side lengths on a prepared grid.

    Coordinates and requested lengths must use the same physical unit. The
    returned window tuple follows array order ``(rows, columns)``.
    """

    requested = tuple(sorted(float(value) for value in side_lengths))
    if not requested or any(not np.isfinite(value) or value <= 0.0 for value in requested):
        raise ValueError("side_lengths must contain positive finite values.")
    if minimum_pixels < 3:
        raise ValueError("minimum_pixels must be at least three.")
    spacing_y, spacing_x = _grid_spacing(x, y)
    grouped: dict[tuple[int, int], list[float]] = {}
    for length in requested:
        window = (
            physical_length_to_odd_pixels(
                length,
                spacing_y,
                minimum=minimum_pixels,
            ),
            physical_length_to_odd_pixels(
                length,
                spacing_x,
                minimum=minimum_pixels,
            ),
        )
        grouped.setdefault(window, []).append(length)
    return tuple(
        PhysicalEgiSupport(
            requested_side_lengths=tuple(lengths),
            window_size=window,
            nominal_side_lengths=(
                window[0] * spacing_y,
                window[1] * spacing_x,
            ),
            grid_spacing=(spacing_y, spacing_x),
        )
        for window, lengths in sorted(
            grouped.items(),
            key=lambda item: np.mean(
                (item[0][0] * spacing_y, item[0][1] * spacing_x)
            ),
        )
    )


def generate_physical_egi_support_bank(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    config: EgiSupportBankConfig = EgiSupportBankConfig(),
) -> tuple[PhysicalEgiSupport, ...]:
    """Generate the version-1 physical EGI bank from a specimen grid.

    The lower endpoint is the larger physical side represented by the
    minimum odd pixel window in either grid direction.  The upper endpoint is
    a declared fraction of the smaller finite coordinate extent.  This avoids
    treating a raw pixel count as a material length on anisotropic grids.
    """

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    spacing_y, spacing_x = _grid_spacing(x_values, y_values)
    extent_x = _finite_extent(x_values, "x")
    extent_y = _finite_extent(y_values, "y")
    lower = config.minimum_pixels * max(spacing_y, spacing_x)
    upper = config.maximum_bbox_fraction * min(extent_x, extent_y)
    if upper < lower:
        raise ValueError(
            "The requested maximum EGI support is smaller than the minimum "
            "three-point physical support."
        )
    requested = np.geomspace(lower, upper, config.candidate_count)
    return resolve_physical_egi_supports(
        requested,
        x_values,
        y_values,
        minimum_pixels=config.minimum_pixels,
    )


def generate_odd_pixel_egi_support_bank(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    config: EgiSupportBankConfig = EgiSupportBankConfig(),
) -> tuple[PhysicalEgiSupport, ...]:
    """Generate every odd square pixel window up to the physical extent cap.

    This deliberately literal bank is used by the minimalist selector:
    ``3, 5, 7, ...`` datapoints. The upper pixel count is chosen so neither
    physical window dimension exceeds the configured fraction of the smaller
    specimen bounding-box dimension.
    """

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    spacing_y, spacing_x = _grid_spacing(x_values, y_values)
    upper_length = config.maximum_bbox_fraction * min(
        _finite_extent(x_values, "x"), _finite_extent(y_values, "y")
    )
    maximum_pixels = int(np.floor(
        upper_length / max(spacing_y, spacing_x) + 1.0e-9
    ))
    if maximum_pixels % 2 == 0:
        maximum_pixels -= 1
    if maximum_pixels < config.minimum_pixels:
        raise ValueError(
            "The requested maximum EGI support is smaller than the minimum odd window."
        )
    return tuple(
        PhysicalEgiSupport(
            requested_side_lengths=(float(size * max(spacing_y, spacing_x)),),
            window_size=(size, size),
            nominal_side_lengths=(size * spacing_y, size * spacing_x),
            grid_spacing=(spacing_y, spacing_x),
        )
        for size in range(config.minimum_pixels, maximum_pixels + 1, 2)
    )


def analyse_egi_signal_sweep(
    supports: Sequence[PhysicalEgiSupport],
    residual_fields: Sequence[npt.ArrayLike],
    noise_scales: Sequence[npt.ArrayLike],
    *,
    active_fraction: float = 0.2,
) -> EgiSignalSweep:
    """Measure characteristic homogeneous-state EGI signal above noise.

    The characteristic signal is the RMS of the largest ``active_fraction``
    of finite absolute, noise-normalised EGI observations. This deliberately
    answers only whether a scale contains a resolved mechanical discrepancy;
    it performs no material probing, redundancy calculation, SVD or Fisher
    optimisation.
    """

    if not supports or len(supports) != len(residual_fields) or len(supports) != len(noise_scales):
        raise ValueError("Supports, residual fields and noise scales must be non-empty and match.")
    if not 0.0 < active_fraction <= 1.0:
        raise ValueError("active_fraction must lie in (0, 1].")
    evidence: list[EgiSignalEvidence] = []
    for support, residual_raw, noise_raw in zip(
        supports, residual_fields, noise_scales, strict=True
    ):
        residual = np.asarray(residual_raw, dtype=np.float64)
        noise = np.broadcast_to(np.asarray(noise_raw, dtype=np.float64), residual.shape)
        valid = np.isfinite(residual) & np.isfinite(noise) & (noise > 0.0)
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            raise ValueError(f"Support {support.window_size} has no finite positive-noise observations.")
        whitened = np.abs(residual[valid] / noise[valid])
        active_count = max(1, int(np.ceil(active_fraction * whitened.size)))
        active = np.partition(whitened, whitened.size - active_count)[-active_count:]
        snr = _rms(active)
        noise_rms = _rms(noise[valid])
        evidence.append(EgiSignalEvidence(
            support=support,
            valid_count=valid_count,
            total_count=int(residual.size),
            coverage_fraction=float(valid_count / residual.size),
            characteristic_signal=float(snr * noise_rms),
            noise_rms=noise_rms,
            signal_to_noise=snr,
        ))
    return EgiSignalSweep(tuple(evidence), float(active_fraction))


def select_log_spaced_egi_supports(
    sweep: EgiSignalSweep,
    config: EgiSignalSelectionConfig = EgiSignalSelectionConfig(),
) -> EgiSignalSelection:
    """Choose the smallest resolved, logarithmic-middle and largest support."""

    eligible = [
        index for index, item in enumerate(sweep.evidence)
        if item.coverage_fraction >= config.minimum_coverage_fraction
        and item.signal_to_noise >= config.minimum_signal_to_noise
    ]
    if len(eligible) < 3:
        raise ValueError(
            "Fewer than three EGI supports pass the direct signal/coverage gate."
        )
    ordered = sorted(
        eligible,
        key=lambda index: sweep.evidence[index].support.nominal_side_length,
    )
    fine, broad = ordered[0], ordered[-1]
    fine_length = sweep.evidence[fine].support.nominal_side_length
    broad_length = sweep.evidence[broad].support.nominal_side_length
    log_target = 0.5 * (np.log(fine_length) + np.log(broad_length))
    middle = min(
        ordered[1:-1],
        key=lambda index: abs(
            np.log(sweep.evidence[index].support.nominal_side_length) - log_target
        ),
    )
    return EgiSignalSelection(fine, middle, broad)


def analyse_egi_support_information(
    supports: Sequence[PhysicalEgiSupport],
    residual_fields: Sequence[npt.ArrayLike],
    noise_scales: Sequence[npt.ArrayLike],
    probe_response_fields: Sequence[npt.ArrayLike],
    *,
    probe_names: Sequence[str],
    fisher_regularisation: float = 1.0e-12,
) -> EgiSupportInformationSweep:
    """Build normalised, whitened multi-probe EGI sensitivity matrices.

    Each response entry has shape ``(probe, *residual.shape)``.  A common
    finite mask across all probes is used for a support, giving a matrix whose
    columns can be compared and stacked without an implicit mask change.
    Rows are divided by ``sqrt(valid_count)`` so a large support is not
    selected solely because it has more valid centres.
    """

    count = len(supports)
    if count == 0:
        raise ValueError("At least one support is required.")
    if not (
        len(residual_fields) == len(noise_scales) == len(probe_response_fields) == count
    ):
        raise ValueError("Supports, residuals, noise, and responses must match.")
    names = tuple(str(name) for name in probe_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("probe_names must be non-empty and unique.")
    if not np.isfinite(fisher_regularisation) or fisher_regularisation <= 0.0:
        raise ValueError("fisher_regularisation must be positive.")

    evidence: list[EgiSupportInformationEvidence] = []
    for support, residual_raw, noise_raw, responses_raw in zip(
        supports, residual_fields, noise_scales, probe_response_fields, strict=True
    ):
        residual = np.asarray(residual_raw, dtype=np.float64)
        noise = np.broadcast_to(np.asarray(noise_raw, dtype=np.float64), residual.shape)
        responses = np.asarray(responses_raw, dtype=np.float64)
        if responses.shape != (len(names), *residual.shape):
            raise ValueError(
                "Probe responses must have shape (probe, *residual.shape); got "
                f"{responses.shape} for residual shape {residual.shape}."
            )
        valid = (
            np.isfinite(residual)
            & np.isfinite(noise)
            & (noise > 0.0)
            & np.all(np.isfinite(responses), axis=0)
        )
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            raise ValueError(
                f"Support {support.window_size} has no finite common probe observations."
            )
        # Shape (observation, probe), whitened then normalised by support size.
        sensitivity = (
            responses[:, valid].T / noise[valid, np.newaxis]
        ) / np.sqrt(valid_count)
        response_to_noise = tuple(
            float(np.sqrt(np.mean((responses[index, valid] / noise[valid]) ** 2)))
            for index in range(len(names))
        )
        singular_values = tuple(
            float(value) for value in np.linalg.svd(sensitivity, compute_uv=False)
        )
        fisher = sensitivity.T @ sensitivity
        fisher.flat[:: fisher.shape[0] + 1] += fisher_regularisation
        sign, logdet = np.linalg.slogdet(fisher)
        if sign <= 0.0:
            raise RuntimeError("Regularised Fisher matrix is not positive definite.")
        evidence.append(
            EgiSupportInformationEvidence(
                support=support,
                valid_count=valid_count,
                total_count=int(residual.size),
                coverage_fraction=float(valid_count / residual.size),
                probe_response_to_noise=response_to_noise,
                singular_values=singular_values,
                fisher_log_determinant=float(logdet),
                sensitivity=sensitivity,
            )
        )
    return EgiSupportInformationSweep(
        evidence=tuple(evidence),
        probe_names=names,
        fisher_regularisation=float(fisher_regularisation),
    )


def select_information_egi_supports(
    sweep: EgiSupportInformationSweep,
    config: EgiSupportInformationSelectionConfig = EgiSupportInformationSelectionConfig(),
) -> EgiSupportInformationSelection:
    """Select fine, broad, and Fisher-complementary middle EGI supports."""

    local = _probe_indices(sweep.probe_names, config.local_probe_prefix)
    homogeneous = _probe_indices(
        sweep.probe_names, config.homogeneous_probe_prefix
    )
    if not local:
        raise ValueError("At least one local yield probe is required for fine selection.")
    if not homogeneous:
        raise ValueError(
            "At least one homogeneous yield probe is required for broad selection."
        )
    coverage_eligible = [
        index
        for index, item in enumerate(sweep.evidence)
        if item.coverage_fraction >= config.minimum_coverage_fraction
    ]
    if len(coverage_eligible) < 2:
        raise ValueError("Fewer than two EGI supports pass the coverage gate.")

    def local_fraction(index: int) -> float:
        responses = sweep.evidence[index].probe_response_to_noise
        return float(np.mean([
            responses[probe] >= config.minimum_response_to_noise
            for probe in local
        ]))

    fine_eligible = [
        index for index in coverage_eligible
        if local_fraction(index) >= config.minimum_local_probe_fraction
    ]
    if not fine_eligible:
        raise ValueError("No EGI support passes the local-probe information gate.")
    fine = min(
        fine_eligible,
        key=lambda index: sweep.evidence[index].support.nominal_side_length,
    )
    broad_eligible = [
        index for index in coverage_eligible
        if max(
            sweep.evidence[index].probe_response_to_noise[probe]
            for probe in homogeneous
        ) >= config.minimum_response_to_noise
    ]
    if not broad_eligible:
        raise ValueError("No EGI support passes the homogeneous-probe information gate.")
    broad = max(
        broad_eligible,
        key=lambda index: sweep.evidence[index].support.nominal_side_length,
    )
    if broad == fine:
        raise ValueError("Fine and broad EGI selection resolved to the same support.")

    base = _stack_support_sensitivities(sweep, (fine, broad))
    base_logdet, base_smallest = _information_scores(
        base, sweep.fisher_regularisation
    )
    ranked: list[tuple[float, float, float, int]] = []
    for candidate in coverage_eligible:
        if candidate in {fine, broad}:
            continue
        logdet, smallest = _information_scores(
            _stack_support_sensitivities(sweep, (fine, candidate, broad)),
            sweep.fisher_regularisation,
        )
        gain = logdet - base_logdet
        smallest_gain = smallest - base_smallest
        # Prefer the largest Fisher gain, then its weakest resolved direction,
        # then the smaller physical support for deterministic ties.
        ranked.append((gain, smallest_gain, -sweep.evidence[candidate].support.nominal_side_length, candidate))
    if not ranked:
        return EgiSupportInformationSelection(
            fine, None, broad, None, None, "two_unique_directions"
        )
    gain, smallest_gain, _, middle = max(ranked)
    if gain <= config.minimum_middle_information_gain:
        return EgiSupportInformationSelection(
            fine, None, broad, float(gain), float(smallest_gain), "two_unique_directions"
        )
    return EgiSupportInformationSelection(
        fine, middle, broad, float(gain), float(smallest_gain), "three_resolved"
    )


def analyse_egi_support_sweep(
    supports: Sequence[PhysicalEgiSupport],
    residual_fields: Sequence[npt.ArrayLike],
    noise_scales: Sequence[npt.ArrayLike],
    parameter_responses: Mapping[str, Sequence[npt.ArrayLike]],
    *,
    redundancy_parameter_group: str = "yield",
) -> EgiSupportSweepResult:
    """Measure support coverage, response-to-noise, and redundancy.

    Each residual, noise scale, and parameter response must retain a common
    full-grid layout for pairwise redundancy. Invalid EGI centres should be
    represented by NaNs. ``noise_scales`` are observation standard deviations,
    not variances.
    """

    count = len(supports)
    if count == 0:
        raise ValueError("At least one support is required.")
    if len(residual_fields) != count or len(noise_scales) != count:
        raise ValueError("Residual and noise entries must match the supports.")
    if redundancy_parameter_group not in parameter_responses:
        raise ValueError(
            "The redundancy parameter group has no response arrays: "
            f"{redundancy_parameter_group!r}."
        )
    for name, responses in parameter_responses.items():
        if len(responses) != count:
            raise ValueError(
                f"Parameter response group {name!r} does not match supports."
            )

    evidence: list[EgiSupportEvidence] = []
    whitened_response_vectors: list[FloatArray] = []
    for index, support in enumerate(supports):
        residual = np.asarray(residual_fields[index], dtype=np.float64)
        noise = np.broadcast_to(
            np.asarray(noise_scales[index], dtype=np.float64),
            residual.shape,
        )
        valid = np.isfinite(residual) & np.isfinite(noise) & (noise > 0.0)
        if not np.any(valid):
            raise ValueError(
                f"Support {support.window_size} has no finite positive-noise observations."
            )
        response_rms: list[tuple[str, float]] = []
        response_snr: list[tuple[str, float]] = []
        group_vectors: dict[str, FloatArray] = {}
        for name, responses in parameter_responses.items():
            response = np.asarray(responses[index], dtype=np.float64)
            if response.shape != residual.shape:
                raise ValueError(
                    f"Response {name!r} shape {response.shape} does not match "
                    f"residual shape {residual.shape}."
                )
            group_valid = valid & np.isfinite(response)
            if not np.any(group_valid):
                raw_rms = 0.0
                snr = 0.0
            else:
                raw_rms = _rms(response[group_valid])
                snr = _rms(response[group_valid] / noise[group_valid])
            response_rms.append((name, raw_rms))
            response_snr.append((name, snr))
            vector = np.full(residual.shape, np.nan, dtype=np.float64)
            vector[group_valid] = response[group_valid] / noise[group_valid]
            group_vectors[name] = vector

        evidence.append(
            EgiSupportEvidence(
                support=support,
                valid_count=int(np.count_nonzero(valid)),
                total_count=int(residual.size),
                coverage_fraction=float(np.count_nonzero(valid) / residual.size),
                residual_rms=_rms(residual[valid]),
                whitened_residual_rms=_rms(residual[valid] / noise[valid]),
                noise_rms=_rms(noise[valid]),
                parameter_response_rms=tuple(response_rms),
                parameter_response_to_noise=tuple(response_snr),
            )
        )
        whitened_response_vectors.append(
            group_vectors[redundancy_parameter_group]
        )

    redundancy = np.eye(count, dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            value = _absolute_cosine(
                whitened_response_vectors[first],
                whitened_response_vectors[second],
            )
            redundancy[first, second] = value
            redundancy[second, first] = value
    return EgiSupportSweepResult(
        evidence=tuple(evidence),
        redundancy_parameter_group=redundancy_parameter_group,
        absolute_cosine_redundancy=redundancy,
    )


def select_sparse_egi_supports(
    sweep: EgiSupportSweepResult,
    config: EgiSupportSelectionConfig = EgiSupportSelectionConfig(),
) -> EgiSupportSelection:
    """Select fine/broad and optionally non-redundant middle supports."""

    if sweep.redundancy_parameter_group != config.parameter_group:
        raise ValueError(
            "Sweep redundancy group and selection parameter group differ."
        )
    eligible = [
        index
        for index, item in enumerate(sweep.evidence)
        if item.coverage_fraction >= config.minimum_coverage_fraction
        and item.response_to_noise(config.parameter_group)
        >= config.minimum_response_to_noise
    ]
    if not eligible:
        raise ValueError("No EGI support passes the coverage and response gates.")

    ordered = sorted(
        eligible,
        key=lambda index: sweep.evidence[index].support.nominal_side_length,
    )
    fine = ordered[0]
    selected = [fine]
    roles: list[tuple[str, int]] = [("fine", fine)]
    if config.maximum_supports >= 2 and ordered[-1] != fine:
        broad = ordered[-1]
        selected.append(broad)
        roles.append(("broad", broad))
    if config.maximum_supports >= 3 and len(ordered) > len(selected):
        candidates = [index for index in ordered if index not in selected]
        ranked: list[tuple[float, int]] = []
        for index in candidates:
            maximum_redundancy = max(
                float(sweep.absolute_cosine_redundancy[index, chosen])
                for chosen in selected
            )
            if maximum_redundancy > config.maximum_middle_redundancy:
                continue
            novelty = 1.0 - maximum_redundancy
            score = (
                sweep.evidence[index].response_to_noise(config.parameter_group)
                * novelty
            )
            ranked.append((score, index))
        if ranked:
            middle = max(ranked)[1]
            selected.append(middle)
            roles.append(("middle", middle))

    selected.sort(
        key=lambda index: sweep.evidence[index].support.nominal_side_length
    )
    return EgiSupportSelection(
        selected_indices=tuple(selected),
        roles=tuple(roles),
        parameter_group=config.parameter_group,
    )


def _grid_spacing(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
) -> tuple[float, float]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 2 or y_values.shape != x_values.shape:
        raise ValueError("x and y must be matching two-dimensional grids.")
    spacing_x = _positive_median_spacing(np.diff(x_values, axis=1), "x")
    spacing_y = _positive_median_spacing(np.diff(y_values, axis=0), "y")
    return spacing_y, spacing_x


def _finite_extent(values: FloatArray, axis_name: str) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Could not resolve finite {axis_name}-coordinate extent.")
    extent = float(np.max(finite) - np.min(finite))
    if extent <= np.finfo(np.float64).eps:
        raise ValueError(f"Could not resolve positive {axis_name}-coordinate extent.")
    return extent


def _probe_indices(names: Sequence[str], prefix: str) -> tuple[int, ...]:
    return tuple(index for index, name in enumerate(names) if name.startswith(prefix))


def _stack_support_sensitivities(
    sweep: EgiSupportInformationSweep,
    indices: Sequence[int],
) -> FloatArray:
    return np.concatenate(
        [sweep.evidence[index].sensitivity for index in indices], axis=0
    )


def _information_scores(
    sensitivity: FloatArray,
    regularisation: float,
) -> tuple[float, float]:
    fisher = sensitivity.T @ sensitivity
    fisher.flat[:: fisher.shape[0] + 1] += regularisation
    sign, logdet = np.linalg.slogdet(fisher)
    if sign <= 0.0:
        raise RuntimeError("Regularised Fisher matrix is not positive definite.")
    singular = np.linalg.svd(sensitivity, compute_uv=False)
    smallest = float(singular[-1]) if singular.size else 0.0
    return float(logdet), smallest


def _positive_median_spacing(values: FloatArray, axis_name: str) -> float:
    finite = np.abs(values[np.isfinite(values)])
    positive = finite[finite > np.finfo(np.float64).eps]
    if positive.size == 0:
        raise ValueError(f"Could not resolve positive {axis_name}-grid spacing.")
    return float(np.median(positive))


def _rms(values: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _absolute_cosine(first: FloatArray, second: FloatArray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if not np.any(valid):
        return 0.0
    first_valid = first[valid]
    second_valid = second[valid]
    denominator = float(
        np.linalg.norm(first_valid) * np.linalg.norm(second_valid)
    )
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(abs(np.dot(first_valid, second_valid)) / denominator)

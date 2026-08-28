from __future__ import annotations

import json
import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.hardening import (
    HardeningLinear,
    HardeningLudwik,
    HardeningSwift,
    HardeningVoce,
    IHardeningFunction,
)
from pyvale.vfm.identificationresult import (
    IdentificationResult,
    ObjectSnapshot,
    ParameterisationSnapshot,
    PhaseSnapshot,
)
from pyvale.vfm.metricequilibriumgap import EquilibriumGapMetric
from pyvale.vfm.metricsliceforce import (
    ForceReconstructionErrorResult,
    SliceWiseForceReconstructionMetric,
)
from pyvale.vfm.radialreturn import radial_return
from pyvale.vfm.slicewise_utils import SliceConfig


PARAMETER_NAME_ORDER = (
    "elastic_modulus",
    "poissons_ratio",
    "yield_strength",
    "hardening_modulus",
    "strength_coefficient",
    "strain_offset",
    "hardening_exponent",
    "saturation_stress",
    "rate_parameter",
)

PLOT_COMPONENT_INDEX = {
    "xx": 0,
    "yy": 1,
    "xy": 2,
}

PLOT_COMPONENT_LABEL = {
    "xx": "xx",
    "yy": "yy",
    "xy": "xy",
    "vm": "von Mises",
}


@dataclass(slots=True, frozen=True)
class StressCheckResult:
    """Comparison between recomputed stress and stress saved in a result bundle."""

    saved_stress_available: bool
    matches_saved_stress: bool | None
    max_abs_difference: float | None
    max_relative_difference: float | None
    rtol: float
    atol: float
    message: str

    def to_summary(self) -> dict[str, float | bool | str | None]:
        return {
            "saved_stress_available": self.saved_stress_available,
            "matches_saved_stress": self.matches_saved_stress,
            "stress_max_abs_difference": self.max_abs_difference,
            "stress_max_relative_difference": self.max_relative_difference,
            "stress_check_rtol": self.rtol,
            "stress_check_atol": self.atol,
            "stress_check_message": self.message,
        }


@dataclass(slots=True, frozen=True)
class PlasticityDiagnostics:
    """Yield and equivalent-plastic-strain diagnostics."""

    yielded_datapoints: npt.NDArray[np.bool_]
    equivalent_plastic_strain: npt.NDArray[np.float64]
    yielded_datapoint_count: int
    yielded_datapoint_fraction: float

    def to_summary(self) -> dict[str, float | int]:
        return {
            "yielded_datapoint_count": self.yielded_datapoint_count,
            "yielded_datapoint_fraction": self.yielded_datapoint_fraction,
        }


@dataclass(slots=True, frozen=True)
class ForceReconstructionDiagnostics:
    """Slice-wise force reconstruction diagnostics and plotting fields."""

    metric_result: ForceReconstructionErrorResult
    raw_residual_newtons: npt.NDArray[np.float64]
    normalised_residual_percent: npt.NDArray[np.float64]
    weighted_rms_newtons_by_slice: npt.NDArray[np.float64]
    weighted_rms_percent_by_slice: npt.NDArray[np.float64]
    weighted_rms_newtons_map: npt.NDArray[np.float64]
    weighted_rms_percent_map: npt.NDArray[np.float64]
    reconstructed_force: npt.NDArray[np.float64]
    applied_longitudinal_force: npt.NDArray[np.float64]
    slice_boundaries: npt.NDArray[np.float64]

    def to_summary(self) -> dict[str, float]:
        return {
            "weighted_force_reconstruction_error_percent": (
                100.0 * self.metric_result.weighted_spatiotemporal_rms
            ),
            "force_reconstruction_error_newtons_max": float(
                np.nanmax(self.weighted_rms_newtons_by_slice)
            ),
            "force_reconstruction_error_percent_max": float(
                np.nanmax(self.weighted_rms_percent_by_slice)
            ),
        }


@dataclass(slots=True, frozen=True)
class EquilibriumGapDiagnostics:
    """Equilibrium-gap diagnostics and plotting fields."""

    raw_gap: npt.NDArray[np.float64]
    normalised_gap: npt.NDArray[np.float64]
    weighted_temporal_rms_percent_map: npt.NDArray[np.float64] | None
    weighted_spatiotemporal_rms_percent: float | None
    window_size: tuple[int, int]

    def to_summary(self) -> dict[str, float | None]:
        return {
            "weighted_equilibrium_gap_percent": (
                self.weighted_spatiotemporal_rms_percent
            ),
            "equilibrium_gap_indicator_percent_max": (
                None
                if self.weighted_temporal_rms_percent_map is None
                else float(np.nanmax(self.weighted_temporal_rms_percent_map))
            ),
        }


@dataclass(slots=True, frozen=True)
class ParameterErrorDiagnostics:
    """Identified-minus-reference parameter-map errors."""

    error_maps: dict[str, npt.NDArray[np.float64]]
    percent_error_maps: dict[str, npt.NDArray[np.float64]]
    summary: dict[str, float]


def load_constitutive_law_from_result(
    result: IdentificationResult,
) -> IConstitutiveLaw:
    """Rebuild the saved constitutive law when the law is supported.

    The result bundle stores only class names and simple constructor options,
    not live Python objects. This helper deliberately supports the known VFM
    laws explicitly so old result bundles fail with a clear message if a newer
    project asks for a law this postprocessor does not know how to rebuild.
    """

    law_snapshot = result.metadata.config.constitutive_law
    if law_snapshot is None:
        raise ValueError("No constitutive-law snapshot was saved in the result.")

    if law_snapshot.type_name == "IsotropicVonMisesElastoplasticity":
        hardening_law = _load_hardening_law_from_snapshot(
            result.metadata.config.hardening_law
        )
        return IsotropicVonMisesElastoplasticity(
            hardening_law,
            **_constructor_options(
                IsotropicVonMisesElastoplasticity,
                law_snapshot.options,
                exclude={"hardening_function"},
            ),
        )

    if (
        law_snapshot.type_name == "CompiledLinearHardeningLaw"
        and law_snapshot.module == "cython_stress_recon.pyvale_adapter"
    ):
        try:
            from cython_stress_recon.pyvale_adapter import (
                CompiledLinearHardeningLaw,
            )
        except ImportError as exc:
            raise NotImplementedError(
                "This result bundle uses the optional Cython stress backend. "
                "Install cython-stress-recon in the active environment to "
                "postprocess it."
            ) from exc
        hardening_law = _load_hardening_law_from_snapshot(
            result.metadata.config.hardening_law
        )
        if not isinstance(hardening_law, HardeningLinear):
            raise NotImplementedError(
                "CompiledLinearHardeningLaw result bundles require "
                "HardeningLinear."
            )
        return CompiledLinearHardeningLaw(
            hardening_law,
            **_constructor_options(
                CompiledLinearHardeningLaw,
                law_snapshot.options,
                exclude={"hardening_function"},
            ),
        )

    raise NotImplementedError(
        "Unsupported constitutive law in result bundle: "
        f"{law_snapshot.type_name} ({law_snapshot.module})."
    )


def _load_hardening_law_from_snapshot(
    snapshot: ObjectSnapshot | None,
) -> IHardeningFunction:
    if snapshot is None:
        raise ValueError("No hardening-law snapshot was saved in the result.")

    hardening_types = {
        "HardeningLinear": HardeningLinear,
        "HardeningSwift": HardeningSwift,
        "HardeningVoce": HardeningVoce,
        "HardeningLudwik": HardeningLudwik,
    }
    hardening_type = hardening_types.get(snapshot.type_name)
    if hardening_type is None:
        raise NotImplementedError(
            "Unsupported hardening law in result bundle: "
            f"{snapshot.type_name} ({snapshot.module})."
        )
    return hardening_type(
        **_constructor_options(hardening_type, snapshot.options)
    )


def _constructor_options(
    constructor: object,
    options: dict[str, object],
    *,
    exclude: set[str] | None = None,
) -> dict[str, object]:
    supported_names = set(inspect.signature(constructor).parameters)
    excluded_names = set() if exclude is None else exclude
    return {
        name: value
        for name, value in options.items()
        if (
            name in supported_names
            and name not in excluded_names
            and value is not None
        )
    }


def compute_stress_from_result(
    experiment_data: ExperimentData,
    result: IdentificationResult,
    constitutive_law: IConstitutiveLaw,
) -> npt.NDArray[np.float64]:
    """Recompute final stress from a result's parameter maps."""

    return constitutive_law.calculate_stress(
        experiment_data.strain,
        result.parameter_maps,
    )


def check_stress_against_saved(
    computed_stress: npt.NDArray[np.float64],
    saved_stress: npt.NDArray[np.float64] | None,
    *,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-8,
) -> StressCheckResult:
    """Check that recomputed stress matches the optional bundle stress cache."""

    if saved_stress is None:
        return StressCheckResult(
            saved_stress_available=False,
            matches_saved_stress=None,
            max_abs_difference=None,
            max_relative_difference=None,
            rtol=rtol,
            atol=atol,
            message="No saved stress was available in the result bundle.",
        )

    if computed_stress.shape != saved_stress.shape:
        return StressCheckResult(
            saved_stress_available=True,
            matches_saved_stress=False,
            max_abs_difference=None,
            max_relative_difference=None,
            rtol=rtol,
            atol=atol,
            message=(
                "Computed stress shape does not match saved stress shape: "
                f"{computed_stress.shape} vs {saved_stress.shape}."
            ),
        )

    difference = np.asarray(computed_stress) - np.asarray(saved_stress)
    finite_difference = difference[np.isfinite(difference)]
    max_abs = (
        0.0
        if finite_difference.size == 0
        else float(np.max(np.abs(finite_difference)))
    )
    finite_saved = np.asarray(saved_stress)[np.isfinite(saved_stress)]
    stress_scale = (
        1.0
        if finite_saved.size == 0
        else float(np.max(np.abs(finite_saved)))
    )
    if stress_scale <= 0.0:
        stress_scale = 1.0
    max_relative = max_abs / stress_scale
    matches = bool(
        np.allclose(
            computed_stress,
            saved_stress,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    )
    return StressCheckResult(
        saved_stress_available=True,
        matches_saved_stress=matches,
        max_abs_difference=max_abs,
        max_relative_difference=max_relative,
        rtol=rtol,
        atol=atol,
        message=(
            "Computed stress matches saved stress."
            if matches
            else "Computed stress does not match saved stress."
        ),
    )


def compute_plasticity_diagnostics(
    experiment_data: ExperimentData,
    constitutive_law: IConstitutiveLaw,
    parameter_maps: dict[str, npt.NDArray[np.float64]],
) -> PlasticityDiagnostics | None:
    """Compute yielded datapoints and equivalent plastic strain when supported."""

    hardening_function = getattr(constitutive_law, "hardening_function", None)
    elastic_modulus_label = getattr(constitutive_law, "elastic_modulus_label", None)
    poissons_ratio_label = getattr(constitutive_law, "poissons_ratio_label", None)
    if (
        hardening_function is None
        or elastic_modulus_label is None
        or poissons_ratio_label is None
    ):
        return None

    _, _, yield_map, equivalent_plastic_strain = radial_return(
        experiment_data.strain,
        parameter_maps,
        parameter_maps[elastic_modulus_label],
        parameter_maps[poissons_ratio_label],
        hardening_function,
    )
    yielded_datapoints = np.any(yield_map, axis=0) & specimen_mask(
        experiment_data
    )
    specimen_count = int(np.count_nonzero(specimen_mask(experiment_data)))
    yielded_count = int(np.count_nonzero(yielded_datapoints))
    yielded_fraction = 0.0 if specimen_count == 0 else yielded_count / specimen_count
    return PlasticityDiagnostics(
        yielded_datapoints=yielded_datapoints,
        equivalent_plastic_strain=equivalent_plastic_strain,
        yielded_datapoint_count=yielded_count,
        yielded_datapoint_fraction=float(yielded_fraction),
    )


def compute_force_reconstruction_diagnostics(
    experiment_data: ExperimentData,
    stress: npt.NDArray[np.float64],
    *,
    axis: Literal["x", "y"],
    num_slices: int,
) -> ForceReconstructionDiagnostics:
    """Evaluate slice-wise force reconstruction error for final stress."""

    metric = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis=axis, num_slices=num_slices),
    )
    metric.initialise(experiment_data)
    if metric.slice_partition is None:
        raise RuntimeError("FRE metric partition was not initialised.")

    result = metric.evaluate_force_recon_error(stress, experiment_data)
    fields = result.metric_result.additional_fields or {}
    raw_residual = np.asarray(fields["raw_residual"], dtype=np.float64)
    normalised_residual = np.asarray(
        fields["normalised_residual"],
        dtype=np.float64,
    )
    temporal_weights = np.asarray(fields["temporal_weights"], dtype=np.float64)
    rms_newtons = np.sqrt(
        np.sum(temporal_weights[:, np.newaxis] * raw_residual**2, axis=0)
    )
    rms_percent = 100.0 * result.weighted_temporal_rms

    return ForceReconstructionDiagnostics(
        metric_result=result,
        raw_residual_newtons=raw_residual,
        normalised_residual_percent=100.0 * normalised_residual,
        weighted_rms_newtons_by_slice=rms_newtons,
        weighted_rms_percent_by_slice=rms_percent,
        weighted_rms_newtons_map=slice_values_to_grid(
            experiment_data,
            metric,
            rms_newtons,
        ),
        weighted_rms_percent_map=slice_values_to_grid(
            experiment_data,
            metric,
            rms_percent,
        ),
        reconstructed_force=np.asarray(
            fields["reconstructed_force"],
            dtype=np.float64,
        ),
        applied_longitudinal_force=np.asarray(
            fields["applied_longitudinal_force"],
            dtype=np.float64,
        ),
        slice_boundaries=np.asarray(
            metric.slice_partition.boundaries,
            dtype=np.float64,
        ),
    )


def compute_equilibrium_gap_diagnostics(
    experiment_data: ExperimentData,
    stress: npt.NDArray[np.float64],
    *,
    window_size: int | tuple[int, int],
) -> EquilibriumGapDiagnostics:
    """Evaluate equilibrium-gap diagnostics for final stress."""

    resolved_window = resolve_egi_window(
        experiment_data.specimen_geometry.x.shape,
        window_size,
    )
    metric = EquilibriumGapMetric(window_size=resolved_window)
    metric.initialise(experiment_data)
    result = metric.evaluate_equilibrium_gap(stress)
    weighted_temporal_percent_map = (
        None
        if result.weighted_temporal_rms is None
        else masked_map(experiment_data, 100.0 * result.weighted_temporal_rms)
    )
    weighted_spatiotemporal_percent = (
        None
        if result.weighted_spatiotemporal_rms is None
        else 100.0 * result.weighted_spatiotemporal_rms
    )
    return EquilibriumGapDiagnostics(
        raw_gap=result.raw_gap,
        normalised_gap=result.normalised_gap,
        weighted_temporal_rms_percent_map=weighted_temporal_percent_map,
        weighted_spatiotemporal_rms_percent=weighted_spatiotemporal_percent,
        window_size=resolved_window,
    )


def compute_parameter_error_diagnostics(
    identified_maps: dict[str, npt.NDArray[np.float64]],
    reference_maps: dict[str, npt.NDArray[np.float64]],
) -> ParameterErrorDiagnostics:
    """Compute absolute and percent errors against reference parameter maps."""

    shared_names = [
        name
        for name in ordered_parameter_names(identified_maps)
        if name in reference_maps
    ]
    error_maps = {
        name: (
            np.asarray(identified_maps[name], dtype=np.float64)
            - np.asarray(reference_maps[name], dtype=np.float64)
        )
        for name in shared_names
    }
    percent_error_maps = {
        name: np.divide(
            100.0 * error_maps[name],
            np.asarray(reference_maps[name], dtype=np.float64),
            out=np.full_like(error_maps[name], np.nan, dtype=np.float64),
            where=np.abs(
                np.asarray(reference_maps[name], dtype=np.float64)
            ) > 1.0e-12,
        )
        for name in shared_names
    }

    summary: dict[str, float] = {}
    for name in shared_names:
        summary[f"{name}_max_abs_error"] = float(
            np.nanmax(np.abs(error_maps[name]))
        )
        summary[f"{name}_max_abs_percent_error"] = float(
            np.nanmax(np.abs(percent_error_maps[name]))
        )
    return ParameterErrorDiagnostics(
        error_maps=error_maps,
        percent_error_maps=percent_error_maps,
        summary=summary,
    )


def cache_stress(
    cache_dir: Path,
    stress: npt.NDArray[np.float64],
) -> Path:
    """Cache recomputed stress in a disposable postprocessing folder."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "computed_stress.npz"
    np.savez(output_path, stress=stress)
    return output_path


def cache_plasticity_diagnostics(
    cache_dir: Path,
    plasticity: PlasticityDiagnostics | None,
) -> Path | None:
    """Cache optional yield and equivalent-plastic-strain diagnostics."""

    if plasticity is None:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "plasticity.npz"
    np.savez(
        output_path,
        yielded_datapoints=plasticity.yielded_datapoints,
        equivalent_plastic_strain=plasticity.equivalent_plastic_strain,
    )
    return output_path


def cache_force_reconstruction_diagnostics(
    cache_dir: Path,
    force_reconstruction: ForceReconstructionDiagnostics,
) -> Path:
    """Cache force-reconstruction diagnostics."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "force_reconstruction.npz"
    np.savez(
        output_path,
        raw_residual_newtons=force_reconstruction.raw_residual_newtons,
        normalised_residual_percent=(
            force_reconstruction.normalised_residual_percent
        ),
        weighted_rms_newtons_by_slice=(
            force_reconstruction.weighted_rms_newtons_by_slice
        ),
        weighted_rms_percent_by_slice=(
            force_reconstruction.weighted_rms_percent_by_slice
        ),
        weighted_rms_newtons_map=force_reconstruction.weighted_rms_newtons_map,
        weighted_rms_percent_map=force_reconstruction.weighted_rms_percent_map,
        reconstructed_force=force_reconstruction.reconstructed_force,
        applied_longitudinal_force=(
            force_reconstruction.applied_longitudinal_force
        ),
        slice_boundaries=force_reconstruction.slice_boundaries,
    )
    return output_path


def cache_equilibrium_gap_diagnostics(
    cache_dir: Path,
    equilibrium_gap: EquilibriumGapDiagnostics,
    *,
    cache_full_egi_history: bool = False,
) -> Path:
    """Cache equilibrium-gap diagnostics.

    The full raw and normalised gap histories can be large, so the default
    cache contains only compact plotting and summary fields.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "equilibrium_gap.npz"
    egi_cache = {
        "weighted_temporal_rms_percent_map": (
            np.array([], dtype=np.float64)
            if equilibrium_gap.weighted_temporal_rms_percent_map is None
            else equilibrium_gap.weighted_temporal_rms_percent_map
        ),
        "weighted_spatiotemporal_rms_percent": np.asarray(
            (
                np.nan
                if equilibrium_gap.weighted_spatiotemporal_rms_percent is None
                else equilibrium_gap.weighted_spatiotemporal_rms_percent
            ),
            dtype=np.float64,
        ),
        "window_size": np.asarray(equilibrium_gap.window_size, dtype=np.uint32),
    }
    if cache_full_egi_history:
        egi_cache["raw_gap"] = equilibrium_gap.raw_gap
        egi_cache["normalised_gap"] = equilibrium_gap.normalised_gap
    np.savez(output_path, **egi_cache)
    return output_path


def cache_parameter_error_diagnostics(
    cache_dir: Path,
    parameter_errors: ParameterErrorDiagnostics | None,
) -> tuple[Path, Path] | None:
    """Cache optional absolute and percent parameter-error maps."""

    if parameter_errors is None:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    error_path = cache_dir / "parameter_error_maps.npz"
    percent_error_path = cache_dir / "parameter_percent_error_maps.npz"
    np.savez(error_path, **parameter_errors.error_maps)
    np.savez(percent_error_path, **parameter_errors.percent_error_maps)
    return error_path, percent_error_path


def write_summary_json(
    output_path: Path,
    summary: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def load_known_parameter_maps(
    known_parameters: Path | None,
    input_dir: Path | None = None,
    *,
    parameter_names: set[str] | None = None,
) -> dict[str, npt.NDArray[np.float64]] | None:
    """Load optional reference parameter maps for error diagnostics."""

    if known_parameters is None:
        if input_dir is None:
            return None
        candidate = input_dir / "known_parameter_maps.npz"
        if not candidate.exists():
            return None
        known_parameters = candidate
    elif known_parameters.is_dir():
        known_parameters = known_parameters / "known_parameter_maps.npz"

    if not known_parameters.exists():
        raise FileNotFoundError(
            f"Known parameter maps file not found: {known_parameters}"
        )

    if parameter_names is None:
        parameter_names = set(PARAMETER_NAME_ORDER)
    with np.load(known_parameters) as loaded:
        return {
            name: np.asarray(loaded[name], dtype=np.float64)
            for name in loaded.files
            if name in parameter_names
        }


def parameter_map_summary(
    parameter_maps: dict[str, npt.NDArray[np.float64]],
) -> dict[str, float]:
    """Return min/mean/max values for each identified parameter map."""

    summary: dict[str, float] = {}
    for name in ordered_parameter_names(parameter_maps):
        parameter_map = np.asarray(parameter_maps[name], dtype=np.float64)
        summary[f"{name}_min"] = float(np.nanmin(parameter_map))
        summary[f"{name}_mean"] = float(np.nanmean(parameter_map))
        summary[f"{name}_max"] = float(np.nanmax(parameter_map))
    return summary


def ordered_parameter_names(
    parameter_maps: dict[str, npt.NDArray[np.float64]],
) -> list[str]:
    ordered = [name for name in PARAMETER_NAME_ORDER if name in parameter_maps]
    ordered.extend(name for name in parameter_maps if name not in ordered)
    return ordered


def parameter_label(name: str) -> str:
    labels = {
        "elastic_modulus": "Elastic Modulus [MPa]",
        "poissons_ratio": "Poisson Ratio [-]",
        "yield_strength": "Yield Strength [MPa]",
        "hardening_modulus": "Hardening Modulus [MPa]",
        "strength_coefficient": "Strength Coefficient [MPa]",
        "strain_offset": "Strain Offset [-]",
        "hardening_exponent": "Hardening Exponent [-]",
        "saturation_stress": "Saturation Stress [MPa]",
        "rate_parameter": "Rate Parameter [-]",
    }
    return labels.get(name, name.replace("_", " ").title())


def evaluate_snapshot_parameter_maps(
    snapshot: PhaseSnapshot,
    experiment_data: ExperimentData,
) -> dict[str, npt.NDArray[np.float64]]:
    """Evaluate identified parameter maps from a durable phase snapshot."""

    maps: dict[str, npt.NDArray[np.float64]] = {}
    for name, parameterisations in snapshot.spatial_parameterisations.items():
        identified = [
            item
            for item in parameterisations
            if item.summary.get("kind") != "known"
        ]
        if not identified:
            continue
        parameter_map = np.zeros(
            experiment_data.specimen_geometry.x.shape,
            dtype=np.float64,
        )
        for parameterisation in identified:
            parameter_map += _evaluate_parameterisation_snapshot(
                parameterisation,
                experiment_data,
            )
        maps[name] = parameter_map
    return maps


def _evaluate_parameterisation_snapshot(
    snapshot: ParameterisationSnapshot,
    experiment_data: ExperimentData,
) -> npt.NDArray[np.float64]:
    summary = snapshot.summary
    kind = summary.get("kind")
    shape = experiment_data.specimen_geometry.x.shape
    if kind == "homogeneous":
        return np.full(shape, float(summary["value"]), dtype=np.float64)
    if kind == "basis_functions":
        return _evaluate_basis_snapshot(summary, experiment_data)
    if kind == "slice_wise":
        return _evaluate_slice_snapshot(summary, experiment_data)
    raise ValueError(
        f"Cannot evaluate saved parameterisation type "
        f"'{snapshot.parameterisation_type}' with summary kind '{kind}'."
    )


def _evaluate_basis_snapshot(
    summary: dict,
    experiment_data: ExperimentData,
) -> npt.NDArray[np.float64]:
    x = experiment_data.specimen_geometry.x
    y = experiment_data.specimen_geometry.y
    parameter_map = np.zeros(x.shape, dtype=np.float64)
    for kernel in summary.get("kernels", []):
        centre_x, centre_y = (float(value) for value in kernel["centre"])
        dx = x - centre_x
        dy = y - centre_y
        kernel_type = kernel["kernel_type"]
        if kernel_type == "BasisFunctionKernelUnivariate":
            variance = float(kernel["variance"])
            exponent = -0.5 * (dx**2 + dy**2) / variance
        elif kernel_type == "BasisFunctionKernelBivariate":
            variance_x, variance_y = (
                float(value) for value in kernel["variance"]
            )
            angle = float(kernel["angle"])
            local_x = np.cos(angle) * dx + np.sin(angle) * dy
            local_y = -np.sin(angle) * dx + np.cos(angle) * dy
            exponent = -0.5 * (
                local_x**2 / variance_x + local_y**2 / variance_y
            )
        elif kernel_type == "BasisFunctionKernelBivariateSPD":
            inverse_covariance = np.linalg.inv(
                np.asarray(kernel["covariance"], dtype=np.float64)
            )
            exponent = -0.5 * (
                inverse_covariance[0, 0] * dx**2
                + 2.0 * inverse_covariance[0, 1] * dx * dy
                + inverse_covariance[1, 1] * dy**2
            )
        else:
            raise ValueError(f"Unsupported saved basis kernel '{kernel_type}'.")
        parameter_map += float(kernel["height"]) * np.exp(exponent)
    return parameter_map


def _evaluate_slice_snapshot(
    summary: dict,
    experiment_data: ExperimentData,
) -> npt.NDArray[np.float64]:
    boundaries = np.asarray(summary["boundaries"], dtype=np.float64)
    values = np.asarray(summary["values"], dtype=np.float64)
    coordinate = (
        experiment_data.specimen_geometry.x
        if summary["axis"] == "x"
        else experiment_data.specimen_geometry.y
    )
    indices = np.searchsorted(boundaries, coordinate, side="right") - 1
    indices = np.clip(indices, 0, values.size - 1)
    return values[indices]


def specimen_mask(
    experiment_data: ExperimentData,
) -> npt.NDArray[np.bool_]:
    return experiment_data.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment_data.specimen_geometry.x,
        experiment_data.specimen_geometry.y,
    )


def masked_map(
    experiment_data: ExperimentData,
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.where(specimen_mask(experiment_data), data, np.nan)


def slice_values_to_grid(
    experiment_data: ExperimentData,
    metric: SliceWiseForceReconstructionMetric,
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if metric.slice_partition is None:
        raise RuntimeError("Metric slice partition has not been initialised.")

    grid = np.full(
        experiment_data.specimen_geometry.x.shape,
        np.nan,
        dtype=np.float64,
    )
    for slice_index, value in enumerate(values):
        grid[metric.slice_partition.get_slice_mask(slice_index)] = value
    return grid


def plastic_parameter_names(
    constitutive_law: IConstitutiveLaw,
) -> set[str] | None:
    hardening_function = getattr(constitutive_law, "hardening_function", None)
    if hardening_function is None:
        return None
    return set(hardening_function.get_required_parameters())


def plot_map_collection(
    experiment_data: ExperimentData,
    parameter_maps: dict[str, npt.NDArray[np.float64]],
    output_path: Path,
    title: str,
    *,
    cmap: str,
    symmetric: bool = False,
    yielded_datapoints: npt.NDArray[np.bool_] | None = None,
    transparent_names: set[str] | None = None,
) -> None:
    names = ordered_parameter_names(parameter_maps)
    if not names:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = min(2, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5.4 * cols, 4.2 * rows),
        constrained_layout=True,
    )
    flat_axes = np.atleast_1d(axes).ravel()
    fig.suptitle(title)
    for ax, name in zip(flat_axes, names, strict=False):
        image = imshow_map(
            ax,
            experiment_data,
            parameter_maps[name],
            cmap=cmap,
            symmetric=symmetric,
            alpha=map_alpha(
                experiment_data,
                name,
                yielded_datapoints,
                transparent_names,
            ),
        )
        ax.set_title(parameter_label(name))
        fig.colorbar(image, ax=ax)
    for ax in flat_axes[len(names):]:
        ax.axis("off")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_solve_parameter_maps(
    experiment_data: ExperimentData,
    result: IdentificationResult,
    output_path: Path,
    *,
    cmap: str,
) -> None:
    """Plot every solve's identified parameter maps in chronological rows."""

    solve_maps = []
    for phase in result.history.phases:
        for solve in phase.solve_results:
            if solve.final_snapshot is None:
                raise ValueError(
                    "Solve-map plotting requires per-solve snapshots. "
                    "Rerun the identification with solve snapshot recording enabled."
                )
            solve_maps.append(
                (
                    phase.phase_index,
                    solve,
                    evaluate_snapshot_parameter_maps(
                        solve.final_snapshot,
                        experiment_data,
                    ),
                )
            )
    if not solve_maps:
        return

    names = ordered_parameter_names(
        {
            name: parameter_map
            for _, _, maps in solve_maps
            for name, parameter_map in maps.items()
        }
    )
    if not names:
        return

    limits = {
        name: (
            min(
                float(np.nanmin(maps[name]))
                for _, _, maps in solve_maps
                if name in maps
            ),
            max(
                float(np.nanmax(maps[name]))
                for _, _, maps in solve_maps
                if name in maps
            ),
        )
        for name in names
    }
    rows = len(solve_maps)
    cols = len(names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5.2 * cols, 3.5 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle("Identified Parameter Maps at Each Solve")
    column_images = []
    for row, (phase_index, solve, maps) in enumerate(solve_maps):
        for column, name in enumerate(names):
            ax = axes[row, column]
            if name not in maps:
                ax.axis("off")
                continue
            image = imshow_map(
                ax,
                experiment_data,
                maps[name],
                cmap=cmap,
                vmin=limits[name][0],
                vmax=limits[name][1],
            )
            if row == 0:
                ax.set_title(parameter_label(name))
            if column == 0:
                status = (
                    "accepted"
                    if solve.accepted
                    else "rejected"
                    if solve.accepted is False
                    else "unknown"
                )
                ax.set_ylabel(
                    f"Phase {phase_index + 1}, solve {solve.solve_iteration + 1}\n"
                    f"{status}"
                )
            column_images.append((column, image))
    for column, name in enumerate(names):
        image = next(
            image
            for image_column, image in column_images
            if image_column == column
        )
        fig.colorbar(image, ax=axes[:, column], label=parameter_label(name))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_phase_objective_histories(
    result: IdentificationResult,
    output_dir: Path,
) -> None:
    """Save one accepted/rejected objective scatter plot per phase."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for phase in result.history.phases:
        points = [
            (
                solve.solve_iteration + 1,
                solve.final_objective.get("cost"),
                solve.accepted,
            )
            for solve in phase.solve_results
            if solve.final_objective.get("cost") is not None
        ]
        if not points:
            continue
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        x_values = [point[0] for point in points]
        costs = [float(point[1]) for point in points]
        ax.plot(x_values, costs, color="0.7", linewidth=1.0, zorder=1)
        for accepted, marker, color, label in (
            (True, "o", "tab:green", "Accepted"),
            (False, "X", "tab:red", "Rejected"),
            (None, "s", "tab:gray", "Acceptance unknown"),
        ):
            selected = [point for point in points if point[2] is accepted]
            if selected:
                ax.scatter(
                    [point[0] for point in selected],
                    [float(point[1]) for point in selected],
                    marker=marker,
                    color=color,
                    s=60,
                    label=label,
                    zorder=2,
                )
        ax.set_title(f"Phase {phase.phase_index + 1} Final Objective by Solve")
        ax.set_xlabel("Solve iteration")
        ax.set_ylabel("Final objective value")
        ax.set_xticks(x_values)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(
            output_dir / f"objective_history_phase_{phase.phase_index + 1}.png",
            dpi=200,
        )
        plt.close(fig)


def component_history_map(
    field: npt.NDArray[np.float64],
    component: Literal["xx", "yy", "xy", "vm"],
) -> npt.NDArray[np.float64]:
    """Return a (timesteps, y, x) history for a requested component."""

    resolved_field = np.asarray(field, dtype=np.float64)
    if resolved_field.ndim != 4:
        raise ValueError(
            "Expected field with shape (timesteps, components, y, x), "
            f"got {resolved_field.shape}."
        )
    if resolved_field.shape[1] < 3:
        raise ValueError(
            "Expected at least 3 components [xx, yy, xy], "
            f"got {resolved_field.shape[1]}."
        )

    if component in PLOT_COMPONENT_INDEX:
        return resolved_field[:, PLOT_COMPONENT_INDEX[component], :, :]
    if component == "vm":
        comp_xx = resolved_field[:, 0, :, :]
        comp_yy = resolved_field[:, 1, :, :]
        comp_xy = resolved_field[:, 2, :, :]
        return np.sqrt(
            comp_xx**2 + comp_yy**2 - comp_xx * comp_yy + 3.0 * comp_xy**2
        )

    raise ValueError(
        f"Unknown component '{component}'. Supported values: xx, yy, xy, vm."
    )


def plot_stress_strain_tiled(
    strain: npt.NDArray[np.float64],
    stress: npt.NDArray[np.float64],
    component: Literal["xx", "yy", "xy", "vm"],
    point_rows: Sequence[int] | int | None = None,
    point_columns: Sequence[int] | int | None = None,
    *,
    timestep: int | None = None,
    output_path: Path,
    cmap: str = "viridis",
) -> None:
    """Plot a tiled figure with a strain map and local stress-strain curve."""

    strain_history = component_history_map(strain, component)
    stress_history = component_history_map(stress, component)
    if strain_history.shape != stress_history.shape:
        raise ValueError(
            "Strain and stress component histories must have the same shape, "
            f"got {strain_history.shape} and {stress_history.shape}."
        )

    def _to_list(values: Sequence[int] | int | None) -> list[int] | None:
        if values is None:
            return None
        if isinstance(values, int):
            return [int(values)]
        return [int(value) for value in values]

    row_list = _to_list(point_rows)
    col_list = _to_list(point_columns)
    if row_list is None and col_list is None:
        row_list = [strain_history.shape[1] // 2]
        col_list = [strain_history.shape[2] // 2]
    elif row_list is None or col_list is None:
        raise ValueError(
            "point_rows and point_columns must both be provided, "
            "or both omitted."
        )
    if not row_list or not col_list or len(row_list) != len(col_list):
        raise ValueError(
            "point_rows and point_columns must have equal non-zero length."
        )

    for row_index, col_index in zip(row_list, col_list, strict=False):
        if row_index < 0 or row_index >= strain_history.shape[1]:
            raise IndexError(
                f"point_row={row_index} is out of bounds for rows "
                f"[0, {strain_history.shape[1] - 1}]."
            )
        if col_index < 0 or col_index >= strain_history.shape[2]:
            raise IndexError(
                f"point_column={col_index} is out of bounds for columns "
                f"[0, {strain_history.shape[2] - 1}]."
            )

    map_index = -1 if timestep is None else int(timestep)
    map_index = map_index % strain_history.shape[0]
    map_data = strain_history[map_index, :, :]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    point_colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(row_list)))

    map_ax = axes[0]
    map_image = map_ax.imshow(
        np.ma.masked_invalid(map_data),
        origin="lower",
        cmap=cmap,
    )
    for row_index, col_index, color in zip(
        row_list,
        col_list,
        point_colors,
        strict=False,
    ):
        map_ax.scatter(
            col_index,
            row_index,
            color=color,
            marker="o",
            s=40,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
    map_ax.set_title(
        "Strain "
        f"{PLOT_COMPONENT_LABEL[component]} "
        f"(timestep {map_index})"
    )
    map_ax.set_xlabel("Column")
    map_ax.set_ylabel("Row")
    fig.colorbar(map_image, ax=map_ax)

    curve_ax = axes[1]
    for row_index, col_index, color in zip(
        row_list,
        col_list,
        point_colors,
        strict=False,
    ):
        strain_series = strain_history[:, row_index, col_index]
        stress_series = stress_history[:, row_index, col_index]
        valid_series = np.isfinite(strain_series) & np.isfinite(stress_series)
        if not np.any(valid_series):
            raise ValueError(
                "Selected point has no finite stress-strain data across "
                f"timesteps: (row={row_index}, col={col_index})."
            )
        curve_ax.plot(
            strain_series[valid_series],
            stress_series[valid_series],
            "-o",
            markersize=3,
            linewidth=1.4,
            color=color,
            label=f"r{row_index}, c{col_index}",
        )
    curve_ax.set_title(
        "Local stress-strain "
        f"({PLOT_COMPONENT_LABEL[component]})"
    )
    curve_ax.set_xlabel(f"Strain {PLOT_COMPONENT_LABEL[component]} [-]")
    curve_ax.set_ylabel(f"Stress {PLOT_COMPONENT_LABEL[component]} [MPa]")
    curve_ax.grid(True, alpha=0.3)
    curve_ax.legend(loc="best", fontsize=8)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_individual_maps(
    experiment_data: ExperimentData,
    maps: dict[str, npt.NDArray[np.float64]],
    output_dir: Path,
    prefix: str,
    *,
    cmap: str,
    symmetric: bool = False,
    yielded_datapoints: npt.NDArray[np.bool_] | None = None,
    transparent_names: set[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ordered_parameter_names(maps):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        image = imshow_map(
            ax,
            experiment_data,
            maps[name],
            cmap=cmap,
            symmetric=symmetric,
            alpha=map_alpha(
                experiment_data,
                name,
                yielded_datapoints,
                transparent_names,
            ),
        )
        ax.set_title(parameter_label(name))
        fig.colorbar(image, ax=ax)
        fig.savefig(output_dir / f"{prefix}_{name}.png", dpi=200)
        plt.close(fig)


def imshow_map(
    ax,
    experiment_data: ExperimentData,
    data: npt.NDArray[np.float64],
    *,
    cmap: str,
    symmetric: bool = False,
    alpha: npt.NDArray[np.float64] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    masked_data = masked_map(experiment_data, np.asarray(data, dtype=np.float64))
    if symmetric:
        max_abs = (
            float(np.nanmax(np.abs(masked_data)))
            if np.any(np.isfinite(masked_data))
            else 0.0
        )
        vmax = max_abs if max_abs > 0.0 else 1.0
        vmin = -vmax

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad((1.0, 1.0, 1.0, 0.0))
    return ax.imshow(
        np.ma.masked_invalid(masked_data),
        origin="lower",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
    )


def map_alpha(
    experiment_data: ExperimentData,
    name: str,
    yielded_datapoints: npt.NDArray[np.bool_] | None,
    transparent_names: set[str] | None,
    *,
    unyielded_alpha: float = 0.22,
) -> npt.NDArray[np.float64] | None:
    if (
        yielded_datapoints is None
        or transparent_names is None
        or name not in transparent_names
    ):
        return None

    mask = specimen_mask(experiment_data)
    alpha = np.zeros(mask.shape, dtype=np.float64)
    alpha[mask & ~yielded_datapoints] = unyielded_alpha
    alpha[mask & yielded_datapoints] = 1.0
    return alpha


def plot_yielded_datapoints(
    yielded_datapoints: npt.NDArray[np.bool_],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    image = ax.imshow(
        yielded_datapoints,
        origin="lower",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("Yielded Datapoints")
    fig.colorbar(image, ax=ax, ticks=[0, 1])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_grid_map(
    grid: npt.NDArray[np.float64],
    output_path: Path,
    title: str,
    *,
    cmap: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    image = ax.imshow(grid, origin="lower", cmap=cmap)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def resolve_egi_window(
    shape: tuple[int, int],
    requested_size: int | tuple[int, int],
) -> tuple[int, int]:
    if isinstance(requested_size, int):
        requested_rows = requested_size
        requested_cols = requested_size
    else:
        requested_rows, requested_cols = requested_size

    rows, cols = shape
    return (
        _resolve_odd_window_size(requested_rows, rows),
        _resolve_odd_window_size(requested_cols, cols),
    )


def _resolve_odd_window_size(
    requested_size: int,
    available_size: int,
) -> int:
    available = available_size if available_size % 2 == 1 else available_size - 1
    size = min(int(requested_size), available)
    size = max(3, size)
    if size % 2 == 0:
        size -= 1
    return size

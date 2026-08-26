from __future__ import annotations

"""
VFM identification result storage and loading helpers.

Postprocessing should depend only on the saved final parameter maps in
``final_parameter_maps.npz``. History and parameterisation snapshots are
auxiliary metadata used to explain or plot how a result was obtained, and
loading final maps must not require importing historical parameterisation
classes. There is no gaurentee that the parameterisation classes used in a
run will be available in the future so saved snapshots are deliberately lightweight
and do not include any executable code. The ``final_stress`` array is a derived audit
artifact saved when available. 
"""

import copy
import enum
import platform
import socket
import sys
import time
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import UnionType
from typing import Any, TypeAlias, get_args, get_origin, get_type_hints

import numpy as np
import numpy.typing as npt
import yaml

from pyvale.vfm.spatialparam import ISpatialParameterisation


JsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)
Summary: TypeAlias = dict[str, JsonValue]

RESULT_FILE_NAME = "identification_result.yaml"
FINAL_PARAMETER_MAPS_FILE_NAME = "final_parameter_maps.npz"
FINAL_IDENTIFIED_STRESS_FILE_NAME = "final_identified_stress.npz"


# ==================================================================================
# Dataclasses for identification result storage
# ==================================================================================

@dataclass(slots=True)
class ArraySummary:
    """Compact, YAML-safe description of a NumPy array."""

    shape: tuple[int, ...] = field(default_factory=tuple)
    dtype: str = ""
    finite_count: int = 0
    min: float | None = None
    max: float | None = None
    mean: float | None = None

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "ArraySummary":
        return _dataclass_from_dict(cls, data)


@dataclass(slots=True)
class ObjectSnapshot:
    """Class identity plus small YAML-safe options for an object used in a run."""

    type_name: str = "Unknown"
    module: str = ""
    options: Summary = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "ObjectSnapshot":
        data = data or {}
        return cls(
            type_name=str(data.get("type_name", data.get("name", "Unknown"))),
            module=str(data.get("module", "")),
            options=_ensure_summary(data.get("options", {})),
            notes=[str(note) for note in data.get("notes", [])],
        )


@dataclass(slots=True)
class ParameterSnapshot:
    """Initial value summary and bounds for one constitutive parameter."""

    lower_bound: float = np.nan
    upper_bound: float = np.nan
    initial_value: ArraySummary = field(default_factory=ArraySummary)

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "ParameterSnapshot":
        return _dataclass_from_dict(cls, data)


@dataclass(slots=True)
class PhaseConfigSnapshot:
    """Compact description of one configured identification phase."""

    phase_index: int = 0
    spatial_parameterisations: dict[str, list[ObjectSnapshot]] = field(
        default_factory=dict
    )
    metrics: list[ObjectSnapshot] = field(default_factory=list)
    objective_function: ObjectSnapshot | None = None
    optimiser: ObjectSnapshot | None = None
    refinement_policy: ObjectSnapshot | None = None
    optimisation_newton_tolerance: float = 1.0e-6
    cache_radial_return: bool = True

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "PhaseConfigSnapshot":
        return _dataclass_from_dict(cls, data)


@dataclass(slots=True)
class ConfigSnapshot:
    """Non-executable summary of the identification configuration."""

    constitutive_law: ObjectSnapshot | None = None
    hardening_law: ObjectSnapshot | None = None
    parameters: dict[str, ParameterSnapshot] = field(default_factory=dict)
    phases: list[PhaseConfigSnapshot] = field(default_factory=list)

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "ConfigSnapshot":
        return _dataclass_from_dict(cls, data)


@dataclass(slots=True)
class RunMetadata:
    """Execution provenance for an identification run."""

    started_at: str | None = None
    finished_at: str | None = None
    runtime_seconds: float | None = None
    pyvale_version: str | None = None
    python_version: str | None = None
    numpy_version: str | None = None
    scipy_version: str | None = None
    platform: str | None = None
    hostname: str | None = None
    perf_counter_started_at: float | None = field(
        default=None,
        repr=False,
        metadata={"serialize": False},
    )

    def finish(self) -> None:
        self.finished_at = _timestamp_now()
        if self.perf_counter_started_at is not None:
            self.runtime_seconds = (
                time.perf_counter() - self.perf_counter_started_at
            )

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "RunMetadata":
        return _dataclass_from_dict(cls, data)


@dataclass(slots=True)
class InputMetadata:
    """Compact description of the experiment data used by a run."""

    source_path: str | None = None
    strain_shape: tuple[int, ...] | None = None
    force_shape: tuple[int, ...] | None = None
    grid_shape: tuple[int, int] | None = None
    timestep_count: int | None = None
    thickness: float | None = None
    roi_type: str | None = None

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "InputMetadata":
        return _dataclass_from_dict(cls, data)


@dataclass(slots=True)
class IdentificationMetadata:
    """Grouped metadata describing the run, input data, and requested config."""

    run: RunMetadata = field(default_factory=RunMetadata)
    input: InputMetadata = field(default_factory=InputMetadata)
    config: ConfigSnapshot = field(default_factory=ConfigSnapshot)

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "IdentificationMetadata":
        return _dataclass_from_dict(cls, data)


@dataclass(slots=True)
class ParameterisationSnapshot:
    """
    Snapshot of a single spatial parameterisation at a saved history point.

    ``parameterisation`` is kept for in-memory compatibility with existing
    callers, but it is deliberately omitted from saved YAML. The durable part
    is the class identity, final degree-of-freedom values, and literal summary.
    """

    parameterisation: ISpatialParameterisation | None = field(
        default=None,
        metadata={"serialize": False},
    )
    dof_values: list[float] = field(default_factory=list)
    parameterisation_type: str = ""
    parameterisation_module: str = ""
    summary: Summary = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.parameterisation is None:
            return
        if not self.parameterisation_type:
            self.parameterisation_type = type(self.parameterisation).__name__
        if not self.parameterisation_module:
            self.parameterisation_module = type(self.parameterisation).__module__
        if not self.summary:
            self.summary = summarise_parameterisation(self.parameterisation)

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "ParameterisationSnapshot":
        data = data or {}
        parameterisation_type = str(
            data.get(
                "parameterisation_type",
                data.get("parameterisation", "Unknown"),
            )
        )
        return cls(
            parameterisation=None,
            dof_values=[
                float(value)
                for value in data.get("dof_values", [])
            ],
            parameterisation_type=parameterisation_type,
            parameterisation_module=str(data.get("parameterisation_module", "")),
            summary=_ensure_summary(data.get("summary", {})),
        )


@dataclass(slots=True)
class PhaseSnapshot:
    """
    Snapshot of an identification phase at a saved history point.
    """

    spatial_parameterisations: dict[str, list[ParameterisationSnapshot]] = field(
        default_factory=dict
    )
    """
    Mapping from constitutive parameter name to the snapshots of its spatial
    parameterisations, in definition order.
    """

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "PhaseSnapshot":
        return _dataclass_from_dict(cls, data)


@dataclass(slots=True)
class SolveResult:
    """Compact optimiser result for one solve attempt."""

    solve_iteration: int = 0
    optimiser: ObjectSnapshot = field(default_factory=ObjectSnapshot)
    runtime_seconds: float | None = None
    num_evaluations: int | None = None
    success: bool | None = None
    accepted: bool | None = None
    status: int | str | None = None
    message: str | None = None
    initial_dofs: list[float] = field(default_factory=list)
    final_dofs: list[float] = field(default_factory=list)
    initial_objective: Summary = field(default_factory=dict)
    final_objective: Summary = field(default_factory=dict)
    final_snapshot: PhaseSnapshot | None = None
    details: Summary = field(default_factory=dict)
    children: list["SolveResult"] = field(default_factory=list)

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "SolveResult":
        return _dataclass_from_dict(cls, data)


@dataclass(slots=True)
class RefinementEvent:
    """Compact record of one accepted refinement action."""

    event_index: int = 0
    policy: ObjectSnapshot = field(default_factory=ObjectSnapshot)
    action: ObjectSnapshot = field(default_factory=ObjectSnapshot)
    trigger_summary: Summary = field(default_factory=dict)
    before_summary: Summary = field(default_factory=dict)
    after_summary: Summary = field(default_factory=dict)

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "RefinementEvent":
        return _dataclass_from_dict(cls, data)


@dataclass(slots=True)
class PhaseResult:
    """Durable record of one executed identification phase."""

    phase_index: int = 0
    config: PhaseConfigSnapshot | None = None
    solve_results: list[SolveResult] = field(default_factory=list)
    refinement_events: list[RefinementEvent] = field(default_factory=list)
    final_snapshot: PhaseSnapshot | None = None

    @property
    def spatial_parameterisations(
        self,
    ) -> dict[str, list[ParameterisationSnapshot]]:
        """Compatibility access to the final phase snapshot."""

        if self.final_snapshot is None:
            return {}
        return self.final_snapshot.spatial_parameterisations

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "PhaseResult":
        data = data or {}
        # Backward compatibility with the original history format, where each
        # phase was directly a PhaseSnapshot-like dictionary.
        if "final_snapshot" not in data and "spatial_parameterisations" in data:
            return cls(
                phase_index=int(data.get("phase_index", 0)),
                final_snapshot=PhaseSnapshot.from_dict(data),
            )

        return _dataclass_from_dict(cls, data)


@dataclass(slots=True)
class IdentificationHistory:
    """Ordered, per-phase history of an identification run."""

    phases: list[PhaseResult] = field(default_factory=list)

    def to_dict(self) -> Summary:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "IdentificationHistory":
        return _dataclass_from_dict(cls, data)


@dataclass(slots=True)
class OptimisationOutcome:
    """Optimiser output plus optional result metadata."""

    spatial_parameterisations: dict[str, list[ISpatialParameterisation]]
    solve_result: SolveResult | None = None


@dataclass(slots=True)
class IdentificationResult:
    """
    Result of a VFM identification run.

    ``parameter_maps`` are the canonical durable result. ``final_stress`` is a
    derived audit artifact saved when available so postprocessing code can be
    checked against the stress history calculated during identification.
    """

    parameter_maps: dict[str, npt.NDArray[np.float64]]
    history: IdentificationHistory = field(default_factory=IdentificationHistory)
    final_stress: npt.NDArray[np.float64] | None = None
    metadata: IdentificationMetadata = field(default_factory=IdentificationMetadata)

    def save_to_yaml(self, output_dir: str | Path | None = None) -> Path:
        """
        Save a minimal durable run bundle.

        The bundle contains this YAML manifest plus ``final_parameter_maps.npz``.
        When ``final_stress`` is available, ``final_identified_stress.npz`` is
        also written as a derived check artifact for later postprocessing.
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            output_dir = f"vfm-identification-result_{timestamp}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        parameter_maps_file = output_dir / FINAL_PARAMETER_MAPS_FILE_NAME
        np.savez(
            parameter_maps_file,
            **{
                name: np.asarray(parameter_map, dtype=np.float64)
                for name, parameter_map in self.parameter_maps.items()
            },
        )

        final_stress_file: str | None = None
        if self.final_stress is not None:
            final_stress_file = FINAL_IDENTIFIED_STRESS_FILE_NAME
            np.savez(
                output_dir / final_stress_file,
                stress=np.asarray(self.final_stress, dtype=np.float64),
            )

        content = {
            "result_type": "pyvale.vfm.IdentificationResult",
            "files": {
                "final_parameter_maps": FINAL_PARAMETER_MAPS_FILE_NAME,
                "final_identified_stress": final_stress_file,
            },
            "metadata": self.metadata.to_dict(),
            "history": self.history.to_dict(),
        }

        result_file = output_dir / RESULT_FILE_NAME
        result_file.write_text(
            yaml.safe_dump(content, sort_keys=False),
            encoding="utf-8",
        )

        return result_file

    save_run_bundle = save_to_yaml

# ==================================================================================
# Public load / save entry points
# ==================================================================================

def load_identification_result(
    result_path: str | Path,
) -> IdentificationResult:
    """
    Load a saved result bundle without importing historical VFM classes.

    ``result_path`` may be either the bundle directory or the
    ``identification_result.yaml`` file inside it.
    """

    result_file = Path(result_path)
    if result_file.is_dir():
        result_file = result_file / RESULT_FILE_NAME

    base_dir = result_file.parent
    content = yaml.safe_load(
        result_file.read_text(encoding="utf-8")
    ) or {}

    files = content.get("files", {})
    parameter_maps = _load_parameter_maps(base_dir, content, files)
    final_stress = _load_final_stress(base_dir, files)

    return IdentificationResult(
        parameter_maps=parameter_maps,
        history=IdentificationHistory.from_dict(content.get("history")),
        final_stress=final_stress,
        metadata=IdentificationMetadata.from_dict(content.get("metadata")),
    )


# ==================================================================================
# Snapshot helpers
# ==================================================================================

def summarise_array(
    array: npt.ArrayLike,
) -> ArraySummary:
    """Create a compact, YAML-safe summary of a NumPy-compatible array."""

    resolved = np.asarray(array)
    finite_values = resolved[np.isfinite(resolved)]
    if finite_values.size == 0:
        return ArraySummary(
            shape=tuple(int(value) for value in resolved.shape),
            dtype=str(resolved.dtype),
            finite_count=0,
        )

    return ArraySummary(
        shape=tuple(int(value) for value in resolved.shape),
        dtype=str(resolved.dtype),
        finite_count=int(finite_values.size),
        min=float(np.min(finite_values)),
        max=float(np.max(finite_values)),
        mean=float(np.mean(finite_values)),
    )


def snapshot_object(
    obj: object | None,
    *,
    options: Summary | None = None,
    notes: list[str] | None = None,
) -> ObjectSnapshot:
    """Create a small serialisable description of a Python object."""

    if obj is None:
        return ObjectSnapshot("None", "", {}, ["Object was None."])

    resolved_options = options
    if resolved_options is None:
        resolved_options = _collect_object_options(obj)

    return ObjectSnapshot(
        type_name=type(obj).__name__,
        module=type(obj).__module__,
        options=resolved_options,
        notes=[] if notes is None else list(notes),
    )


def snapshot_parameter(
    parameter: object,
) -> ParameterSnapshot:
    """Create a saved summary of one constitutive parameter."""

    return ParameterSnapshot(
        lower_bound=float(getattr(parameter, "lower_bound")),
        upper_bound=float(getattr(parameter, "upper_bound")),
        initial_value=summarise_array(getattr(parameter, "map")),
    )


def snapshot_phase_config(
    phase_index: int,
    phase: object,
) -> PhaseConfigSnapshot:
    """Create a saved summary of one configured identification phase."""

    refinement_policy = getattr(phase, "refinement_policy")
    return PhaseConfigSnapshot(
        phase_index=phase_index,
        spatial_parameterisations={
            name: [
                snapshot_object(parameterisation)
                for parameterisation in parameterisations
            ]
            for name, parameterisations
            in getattr(phase, "spatial_parameterisations").items()
        },
        metrics=[
            snapshot_object(metric)
            for metric in getattr(phase, "metrics")
        ],
        objective_function=snapshot_object(getattr(phase, "objective_function")),
        optimiser=snapshot_object(getattr(phase, "optimiser")),
        refinement_policy=(
            None
            if refinement_policy is None
            else snapshot_object(refinement_policy)
        ),
        optimisation_newton_tolerance=float(
            getattr(phase, "optimisation_newton_tolerance", 1.0e-6)
        ),
        cache_radial_return=bool(getattr(phase, "cache_radial_return", True)),
    )


def snapshot_identification_config(
    identification_config: object,
) -> ConfigSnapshot:
    """Create a saved summary of an identification configuration."""

    constitutive_law = getattr(identification_config, "constitutive_law")
    hardening_law = getattr(constitutive_law, "hardening_function", None)
    return ConfigSnapshot(
        constitutive_law=snapshot_object(constitutive_law),
        hardening_law=(
            None
            if hardening_law is None
            else snapshot_object(hardening_law)
        ),
        # Create dictionary of parameter snapshots for each constitutive parameter, keyed by name.
        parameters={
            name: snapshot_parameter(parameter)
            for name, parameter
            in getattr(identification_config, "parameters").items()
        },
        # Create list of phase snapshots for each configured identification phase, preserving order.
        phases=[
            snapshot_phase_config(phase_index, phase)
            for phase_index, phase
            in enumerate(getattr(identification_config, "phases"))
        ],
    )


def start_run_metadata() -> RunMetadata:
    """Capture metadata known at the start of an identification run."""

    return RunMetadata(
        started_at=_timestamp_now(),
        pyvale_version=_package_version("pyvale"),
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        scipy_version=_package_version("scipy"),
        platform=platform.platform(),
        hostname=socket.gethostname(),
        perf_counter_started_at=time.perf_counter(),
    )


def input_metadata_from_experiment_data(
    experiment_data: object,
    *,
    source_path: str | Path | None = None,
) -> InputMetadata:
    """Create compact metadata for the experiment data used by a run."""

    specimen_geometry = getattr(experiment_data, "specimen_geometry")
    boundary_conditions = getattr(experiment_data, "boundary_conditions")
    timesteps = np.asarray(getattr(experiment_data, "timesteps"))
    return InputMetadata(
        source_path=None if source_path is None else str(source_path),
        strain_shape=tuple(
            int(value)
            for value in np.asarray(getattr(experiment_data, "strain")).shape
        ),
        force_shape=tuple(
            int(value)
            for value in np.asarray(getattr(boundary_conditions, "force")).shape
        ),
        grid_shape=tuple(
            int(value)
            for value in np.asarray(getattr(specimen_geometry, "x")).shape
        ),
        timestep_count=int(timesteps.size),
        thickness=float(getattr(specimen_geometry, "thickness")),
        roi_type=type(getattr(specimen_geometry, "region_of_interest")).__name__,
    )


def snapshot_parameterisation(
    parameterisation: ISpatialParameterisation,
) -> ParameterisationSnapshot:
    """Create a saved snapshot of one spatial parameterisation."""

    return ParameterisationSnapshot(
        parameterisation=copy.deepcopy(parameterisation),
        dof_values=[
            float(dof.value)
            for dof in parameterisation.collect_degrees_of_freedom()
        ],
        parameterisation_type=type(parameterisation).__name__,
        parameterisation_module=type(parameterisation).__module__,
        summary=summarise_parameterisation(parameterisation),
    )


def snapshot_phase(
    spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
) -> PhaseSnapshot:
    """Create a saved snapshot of one completed identification phase."""

    snapshot: dict[str, list[ParameterisationSnapshot]] = {}
    for name, parameterisation_list in spatial_parameterisations.items():
        snapshot[name] = [
            snapshot_parameterisation(parameterisation)
            for parameterisation in parameterisation_list
        ]
    return PhaseSnapshot(snapshot)


def generic_completed_solve_result(
    *,
    solve_iteration: int,
    optimiser: object,
    runtime_seconds: float,
    initial_dofs: list[float],
    final_dofs: list[float],
) -> SolveResult:
    """Create generic solve metadata for optimisers without detailed output."""

    return SolveResult(
        solve_iteration=solve_iteration,
        optimiser=snapshot_object(optimiser),
        runtime_seconds=float(runtime_seconds),
        success=True,
        status="completed",
        message=(
            "Optimiser did not return detailed result metadata; "
            "recorded generic solve completion."
        ),
        initial_dofs=initial_dofs,
        final_dofs=final_dofs,
    )


def summarise_parameterisation(
    parameterisation: ISpatialParameterisation,
) -> Summary:
    """Return a YAML-safe plotting/explanation summary for known parameterisations."""

    try:
        from pyvale.vfm.spatialparambasisfuncs import (
            BasisFunctionKernelBivariate,
            BasisFunctionKernelUnivariate,
            SpatialParameterisationBasisFunction,
        )
        from pyvale.vfm.spatialparamhomogeneous import (
            SpatialParameterisationHomogeneous,
        )
        from pyvale.vfm.spatialparamknown import SpatialParameterisationKnown
        from pyvale.vfm.spatialparamslicewise import (
            SliceWiseSpatialParameterisation,
        )
    except ImportError as exc:
        return {
            "kind": "unknown",
            "note": f"Could not import parameterisation summary helpers: {exc}",
        }

    if isinstance(parameterisation, SpatialParameterisationHomogeneous):
        return {
            "kind": "homogeneous",
            "value": _jsonify_value(_resolve_value(parameterisation.value)),
        }

    if isinstance(parameterisation, SpatialParameterisationKnown):
        return {
            "kind": "known",
            "value_summary": (
                None
                if parameterisation.value is None
                else summarise_array(parameterisation.value).to_dict()
            ),
        }

    if isinstance(parameterisation, SliceWiseSpatialParameterisation):
        partition = parameterisation.slice_partition
        config = parameterisation.slice_config
        values = parameterisation.values or []
        summary: Summary = {
            "kind": "slice_wise",
            "values": [
                _jsonify_value(_resolve_value(value))
                for value in values
            ],
        }
        if partition is not None:
            summary.update(
                {
                    "axis": partition.axis,
                    "num_slices": int(partition.num_slices),
                    "boundaries": _jsonify_value(partition.boundaries),
                    "centres": _jsonify_value(partition.centres),
                    "widths": _jsonify_value(partition.widths),
                    "coordinate_system": "physical",
                }
            )
        elif config is not None:
            summary.update(
                {
                    "axis": config.axis,
                    "num_slices": config.num_slices,
                    "boundaries": _jsonify_value(config.boundaries),
                    "coordinate_system": "physical",
                }
            )
        return summary

    if isinstance(parameterisation, SpatialParameterisationBasisFunction):
        kernels: list[JsonValue] = []
        for kernel, height in zip(
            parameterisation.kernels,
            parameterisation.heights,
            strict=True,
        ):
            kernel_summary: Summary = {
                "kernel_type": type(kernel).__name__,
                "centre": [
                    _jsonify_value(_resolve_value(kernel.x)),
                    _jsonify_value(_resolve_value(kernel.y)),
                ],
                "height": _jsonify_value(_resolve_value(height)),
                "coordinate_system": "physical",
            }
            if isinstance(kernel, BasisFunctionKernelUnivariate):
                variance = _resolve_value(kernel.variance)
                kernel_summary.update(
                    {
                        "variance": _jsonify_value(variance),
                        "width": _jsonify_value(np.sqrt(float(variance))),
                        "angle": 0.0,
                    }
                )
            elif isinstance(kernel, BasisFunctionKernelBivariate):
                variance_x = _resolve_value(kernel.variance_x)
                variance_y = _resolve_value(kernel.variance_y)
                kernel_summary.update(
                    {
                        "variance": [
                            _jsonify_value(variance_x),
                            _jsonify_value(variance_y),
                        ],
                        "width": [
                            _jsonify_value(np.sqrt(float(variance_x))),
                            _jsonify_value(np.sqrt(float(variance_y))),
                        ],
                        "angle": _jsonify_value(_resolve_value(kernel.angle)),
                    }
                )
            kernels.append(kernel_summary)

        return {
            "kind": "basis_functions",
            "num_kernels": len(kernels),
            "kernels": kernels,
        }

    return {
        "kind": "unknown",
        "note": "No detailed parameterisation summary available.",
    }


def summarise_refinement_target(
    target: object,
) -> Summary:
    """Return a literal support summary suitable for a refinement event."""

    try:
        from pyvale.vfm.spatialparambasisfuncs import SupportBasis
        from pyvale.vfm.spatialparamslicewise import SupportSlice
    except ImportError as exc:
        return {
            "kind": "unknown",
            "note": f"Could not import support summary helpers: {exc}",
        }

    if isinstance(target, SupportSlice):
        partition = target.slice_partition
        config = target.slice_config
        if partition is not None:
            return {
                "kind": "slice_support",
                "axis": partition.axis,
                "num_slices": int(partition.num_slices),
                "boundaries": _jsonify_value(partition.boundaries),
                "centres": _jsonify_value(partition.centres),
                "widths": _jsonify_value(partition.widths),
                "coordinate_system": "physical",
            }
        if config is not None:
            return {
                "kind": "slice_support",
                "axis": config.axis,
                "num_slices": config.num_slices,
                "boundaries": _jsonify_value(config.boundaries),
                "coordinate_system": "physical",
            }

    if isinstance(target, SupportBasis):
        return {
            "kind": "basis_support",
            "grid_shape": _jsonify_value(np.asarray(target.x).shape),
            "num_kernels": 0 if target.kernels is None else len(target.kernels),
            "x_summary": summarise_array(target.x).to_dict(),
            "y_summary": summarise_array(target.y).to_dict(),
        }

    return {
        "kind": "unknown",
        "type_name": type(target).__name__,
        "module": type(target).__module__,
    }


def summarise_refinement_action(
    action: object,
    *,
    before_summary: Summary,
    after_summary: Summary,
) -> Summary:
    """Return compact action-specific details without storing diagnostic arrays."""

    summary: Summary = {
        "action_type": type(action).__name__,
        "num_slices_before": before_summary.get("num_slices"),
        "num_slices_after": after_summary.get("num_slices"),
        "num_kernels_before": before_summary.get("num_kernels"),
        "num_kernels_after": after_summary.get("num_kernels"),
    }
    refined_boundaries = getattr(action, "refined_boundaries", None)
    if refined_boundaries is not None:
        summary["refined_boundary_count"] = int(np.asarray(refined_boundaries).size)
    return summary


def snapshot_refinement_policy(
    policy: object,
) -> ObjectSnapshot:
    """Return a lightweight policy snapshot for refinement history."""

    return snapshot_object(
        policy,
        options=_collect_named_options(
            policy,
            (
                "max_refinements",
                "merge_parameter_tolerance",
                "split_error_threshold",
                "max_basis_functions",
                "relative_improvement_threshold",
                "refinement_height_fraction",
                "smoothing_points",
                "minimum_separation_points",
            ),
        ),
    )


def snapshot_refinement_action(
    action: object,
) -> ObjectSnapshot:
    """Return a lightweight action snapshot for refinement history."""

    return snapshot_object(action, options={})

# ==================================================================================
# File loading helpers
# ==================================================================================

def _load_parameter_maps(
    base_dir: Path,
    content: dict[str, Any],
    files: dict[str, Any],
) -> dict[str, npt.NDArray[np.float64]]:
    parameter_maps_file = files.get("final_parameter_maps")
    if parameter_maps_file is not None:
        with np.load(base_dir / str(parameter_maps_file)) as loaded:
            return {
                name: np.asarray(loaded[name], dtype=np.float64)
                for name in loaded.files
            }

    # Backward compatibility with the previous writer.
    parameter_map_files = content.get("parameter_maps", {})
    return {
        name: np.asarray(np.load(base_dir / filename), dtype=np.float64)
        for name, filename in parameter_map_files.items()
    }


def _load_final_stress(
    base_dir: Path,
    files: dict[str, Any],
) -> npt.NDArray[np.float64] | None:
    stress_file = files.get("final_identified_stress")
    if stress_file is None:
        return None
    stress_path = base_dir / str(stress_file)
    if not stress_path.exists():
        return None
    with np.load(stress_path) as loaded:
        if "stress" in loaded.files:
            return np.asarray(loaded["stress"], dtype=np.float64)
        if loaded.files:
            return np.asarray(loaded[loaded.files[0]], dtype=np.float64)
    return None

# ==================================================================================
# Serialisation helpers for dataclasses and other objects that may be stored in YAML.
# ==================================================================================

def _dataclass_to_dict(
    value: object,
) -> Summary:
    """Convert result dataclasses to YAML-safe dictionaries."""

    json_value = _jsonify_value(value)
    if isinstance(json_value, dict):
        return json_value
    return {}


def _dataclass_from_dict(
    cls,
    data: dict[str, Any] | None,
):
    """Create a result dataclass from a dictionary using field annotations."""

    data = data or {}
    type_hints = get_type_hints(cls)
    init_kwargs = {}
    for data_field in fields(cls):
        if not data_field.init:
            continue
        if data_field.metadata.get("serialize") is False:
            continue
        if data_field.name not in data:
            continue
        init_kwargs[data_field.name] = _coerce_loaded_value(
            type_hints.get(data_field.name, Any),
            data[data_field.name],
            field_name=data_field.name,
        )
    return cls(**init_kwargs)


def _coerce_loaded_value(
    type_hint: object,
    value: Any,
    *,
    field_name: str = "",
) -> Any:
    if value is None:
        return None
    if _field_is_summary(field_name):
        return _ensure_summary(value)
    if type_hint is Any:
        return value

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin in (UnionType, getattr(__import__("typing"), "Union")):
        non_none_args = [arg for arg in args if arg is not type(None)]
        for arg in non_none_args:
            try:
                return _coerce_loaded_value(arg, value, field_name=field_name)
            except (TypeError, ValueError):
                continue
        return value

    if origin is list:
        item_type = args[0] if args else Any
        return [
            _coerce_loaded_value(item_type, item, field_name=field_name)
            for item in value
        ]

    if origin is dict:
        key_type = args[0] if args else str
        value_type = args[1] if len(args) > 1 else Any
        return {
            _coerce_loaded_value(key_type, key, field_name=field_name):
            _coerce_loaded_value(value_type, item, field_name=field_name)
            for key, item in value.items()
        }

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _coerce_loaded_value(args[0], item, field_name=field_name)
                for item in value
            )
        return tuple(
            _coerce_loaded_value(arg, item, field_name=field_name)
            for arg, item in zip(args, value, strict=False)
        )

    if isinstance(type_hint, type) and is_dataclass(type_hint):
        from_dict = getattr(type_hint, "from_dict", None)
        if from_dict is not None:
            return from_dict(value)
        return _dataclass_from_dict(type_hint, value)

    if type_hint is str:
        return str(value)
    if type_hint is int:
        return int(value)
    if type_hint is float:
        return float(value)
    if type_hint is bool:
        return bool(value)
    return value


def _field_is_summary(
    field_name: str,
) -> bool:
    return (
        field_name == "options"
        or field_name == "details"
        or field_name == "summary"
        or field_name.endswith("_summary")
        or field_name.endswith("_objective")
    )


def _collect_object_options(
    obj: object,
) -> Summary:
    options: Summary = {}
    candidate_names = _iter_public_data_attributes(obj)
    for name in candidate_names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        json_value = _jsonify_value(value)
        if json_value is not _UNSERIALISABLE:
            options[name] = json_value
    return options


def _collect_named_options(
    obj: object,
    names: tuple[str, ...],
) -> Summary:
    options: Summary = {}
    for name in names:
        if not hasattr(obj, name):
            continue
        value = _jsonify_value(getattr(obj, name))
        if value is not _UNSERIALISABLE:
            options[name] = value
    return options


def _iter_public_data_attributes(
    obj: object,
) -> list[str]:
    if is_dataclass(obj):
        return [
            data_field.name
            for data_field in fields(obj)
            if not data_field.name.startswith("_")
        ]

    names: list[str] = []
    slots = getattr(type(obj), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    names.extend(name for name in slots if not str(name).startswith("_"))

    instance_dict = getattr(obj, "__dict__", None)
    if isinstance(instance_dict, dict):
        names.extend(name for name in instance_dict if not name.startswith("_"))

    for name, value in vars(type(obj)).items():
        if name.startswith("_"):
            continue
        if callable(value) or isinstance(value, property):
            continue
        names.append(name)

    return sorted(set(names))


_UNSERIALISABLE = object()


def _jsonify_value(
    value: Any,
) -> JsonValue | object:
    # StrEnum is also an instance of str, so Enum must be handled first or
    # PyYAML receives the enum object rather than its plain scalar value.
    if isinstance(value, enum.Enum):
        return value.value if isinstance(value.value, (str, int, float, bool)) else value.name
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        resolved = float(value)
        return resolved if np.isfinite(resolved) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return _jsonify_value(value.tolist())
        return summarise_array(value).to_dict()
    if isinstance(value, tuple):
        return _jsonify_sequence(value)
    if isinstance(value, list):
        return _jsonify_sequence(value)
    if isinstance(value, dict):
        resolved_dict: Summary = {}
        for key, item in value.items():
            json_item = _jsonify_value(item)
            if json_item is _UNSERIALISABLE:
                return _UNSERIALISABLE
            resolved_dict[str(key)] = json_item
        return resolved_dict
    if is_dataclass(value):
        resolved_dict = {}
        for data_field in fields(value):
            if data_field.name.startswith("_"):
                continue
            if data_field.metadata.get("serialize") is False:
                continue
            json_item = _jsonify_value(getattr(value, data_field.name))
            if json_item is _UNSERIALISABLE:
                continue
            resolved_dict[data_field.name] = json_item
        return resolved_dict
    return _UNSERIALISABLE


def _jsonify_sequence(
    values: list[Any] | tuple[Any, ...],
) -> JsonValue | object:
    resolved_values: list[JsonValue] = []
    for value in values:
        json_value = _jsonify_value(value)
        if json_value is _UNSERIALISABLE:
            return _UNSERIALISABLE
        resolved_values.append(json_value)
    return resolved_values


def _resolve_value(
    value: Any,
) -> Any:
    return getattr(value, "value", value)


def _ensure_summary(
    data: Any,
) -> Summary:
    json_value = _jsonify_value(data)
    if isinstance(json_value, dict):
        return json_value
    return {}


def _timestamp_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _package_version(
    package_name: str,
) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None

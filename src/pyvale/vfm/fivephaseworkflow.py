"""State handoff helpers for the fixed-geometry five-phase VFM workflow.

The helpers in this module only translate durable Phase-2 snapshots into the
parameterisations required by Phases 3--5.  They do not choose basis geometry,
objectives, metrics, or optimisation algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.identificationconfig import IdentificationPhase
from pyvale.vfm.identificationresult import PhaseSnapshot
from pyvale.vfm.spatialparambasisfuncs import (
    BasisFunctionKernel,
    BasisFunctionKernelBivariate,
    BasisFunctionKernelBivariateSPD,
    BasisFunctionKernelUnivariate,
    SpatialParameterisationBasisFunction,
    SupportBasis,
)
from pyvale.vfm.spatialparamhomogeneous import SpatialParameterisationHomogeneous
from pyvale.vfm.spatialparamknown import SpatialParameterisationKnown


@dataclass(slots=True, frozen=True)
class FixedGeometryState:
    """Selected additive-basis state with literal, inactive kernel geometry."""

    kernel_type: str
    kernel_summaries: tuple[dict[str, Any], ...]
    yield_homogeneous: float
    yield_amplitudes: tuple[float, ...]

    @property
    def basis_count(self) -> int:
        return len(self.kernel_summaries)

    def to_summary(self) -> dict[str, Any]:
        return {
            "basis_count": self.basis_count,
            "kernel_type": self.kernel_type,
            "yield_homogeneous_mpa": self.yield_homogeneous,
            "yield_amplitudes_mpa": list(self.yield_amplitudes),
            "kernels": [dict(kernel) for kernel in self.kernel_summaries],
        }


def fixed_geometry_state_from_snapshot(
    snapshot: PhaseSnapshot,
    *,
    parameter_name: str = "yield_strength",
) -> FixedGeometryState:
    """Extract the selected homogeneous value, amplitudes, and BF geometry."""

    homogeneous: float | None = None
    basis_summary: dict[str, Any] | None = None
    for item in snapshot.spatial_parameterisations.get(parameter_name, []):
        kind = item.summary.get("kind")
        if kind == "homogeneous":
            homogeneous = float(item.summary["value"])
        elif kind == "basis_functions":
            basis_summary = item.summary

    if homogeneous is None or basis_summary is None:
        raise ValueError(
            f"Snapshot for {parameter_name!r} must contain homogeneous and "
            "basis-function components."
        )

    kernels = tuple(dict(value) for value in basis_summary.get("kernels", []))
    if not kernels:
        raise ValueError("The selected Phase-2 state contains no basis functions.")
    amplitudes = tuple(float(kernel["height"]) for kernel in kernels)
    kernel_types = {str(kernel["kernel_type"]) for kernel in kernels}
    if len(kernel_types) != 1:
        raise ValueError("All selected Phase-2 kernels must have the same type.")

    return FixedGeometryState(
        kernel_type=_parameterisation_kernel_type(kernel_types.pop()),
        kernel_summaries=kernels,
        yield_homogeneous=homogeneous,
        yield_amplitudes=amplitudes,
    )


def selected_phase_2_snapshot(
    phase_result,
    basis_count: int,
) -> PhaseSnapshot:
    """Return the persisted Phase-2 solve snapshot with the selected BF count."""

    matches: list[PhaseSnapshot] = []
    for solve in phase_result.solve_results:
        snapshot = solve.final_snapshot
        if snapshot is None:
            continue
        try:
            state = fixed_geometry_state_from_snapshot(snapshot)
        except ValueError:
            continue
        if state.basis_count == basis_count:
            matches.append(snapshot)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one persisted BF{basis_count} state, found "
            f"{len(matches)}."
        )
    return matches[0]


def make_phase_3_parameterisations(
    hardening_homogeneous: float,
    hardening_bounds: tuple[float, float],
) -> dict[str, list[object]]:
    """Freeze the complete selected yield field and release homogeneous H."""

    return {
        "elastic_modulus": [SpatialParameterisationKnown()],
        "poissons_ratio": [SpatialParameterisationKnown()],
        "yield_strength": [SpatialParameterisationKnown()],
        "hardening_modulus": [
            SpatialParameterisationHomogeneous(DegreeOfFreedom(
                hardening_homogeneous, *hardening_bounds
            ))
        ],
    }


def make_phase_4_parameterisations(
    geometry: FixedGeometryState,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    *,
    hardening_amplitude_bound: float,
) -> dict[str, list[object]]:
    """Freeze yield/H0 and release only H amplitudes on Phase-2 geometry."""

    support = _fixed_support(geometry, x, y)
    hardening_basis = _basis_parameterisation(
        geometry,
        support,
        amplitudes=(0.0,) * geometry.basis_count,
        amplitude_bound=hardening_amplitude_bound,
    )
    return {
        "elastic_modulus": [SpatialParameterisationKnown()],
        "poissons_ratio": [SpatialParameterisationKnown()],
        "yield_strength": [SpatialParameterisationKnown()],
        "hardening_modulus": [SpatialParameterisationKnown(), hardening_basis],
    }


def make_phase_5_parameterisations(
    geometry: FixedGeometryState,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    *,
    hardening_homogeneous: float,
    hardening_amplitudes: tuple[float, ...],
    yield_bounds: tuple[float, float],
    hardening_bounds: tuple[float, float],
) -> dict[str, list[object]]:
    """Release yield/H homogeneous terms and amplitudes; keep geometry fixed."""

    if len(hardening_amplitudes) != geometry.basis_count:
        raise ValueError("Hardening amplitudes must match the selected BF count.")
    support = _fixed_support(geometry, x, y)
    yield_basis = _basis_parameterisation(
        geometry,
        support,
        amplitudes=geometry.yield_amplitudes,
        amplitude_bound=yield_bounds[1] - yield_bounds[0],
    )
    hardening_basis = _basis_parameterisation(
        geometry,
        support,
        amplitudes=hardening_amplitudes,
        amplitude_bound=hardening_bounds[1] - hardening_bounds[0],
    )
    return {
        "elastic_modulus": [SpatialParameterisationKnown()],
        "poissons_ratio": [SpatialParameterisationKnown()],
        "yield_strength": [
            SpatialParameterisationHomogeneous(DegreeOfFreedom(
                geometry.yield_homogeneous, *yield_bounds
            )),
            yield_basis,
        ],
        "hardening_modulus": [
            SpatialParameterisationHomogeneous(DegreeOfFreedom(
                hardening_homogeneous, *hardening_bounds
            )),
            hardening_basis,
        ],
    }


def basis_amplitudes_from_snapshot(
    snapshot: PhaseSnapshot,
    parameter_name: str,
) -> tuple[float, ...]:
    """Read additive BF amplitudes from a durable phase snapshot."""

    for item in snapshot.spatial_parameterisations.get(parameter_name, []):
        if item.summary.get("kind") == "basis_functions":
            return tuple(
                float(kernel["height"])
                for kernel in item.summary.get("kernels", [])
            )
    raise ValueError(f"No basis-function state found for {parameter_name!r}.")


def active_dof_summary(phase: IdentificationPhase) -> list[dict[str, Any]]:
    """Describe configured active DOFs, including fixed-geometry status."""

    rows: list[dict[str, Any]] = []
    for parameter_name, parameterisations in phase.spatial_parameterisations.items():
        for component_index, parameterisation in enumerate(parameterisations):
            role = _role(parameterisation)
            for local_index, dof in enumerate(
                parameterisation.collect_degrees_of_freedom()
            ):
                rows.append({
                    "parameter": parameter_name,
                    "component_index": component_index,
                    "role": role,
                    "local_index": local_index,
                    "value": float(dof.value),
                    "lower_bound": float(dof.lower_bound),
                    "upper_bound": float(dof.upper_bound),
                })
    supports: set[int] = set()
    for parameterisations in phase.spatial_parameterisations.values():
        for parameterisation in parameterisations:
            support = getattr(parameterisation, "support", None)
            if support is None or id(support) in supports:
                continue
            supports.add(id(support))
            for local_index, dof in enumerate(support.collect_degrees_of_freedom()):
                rows.append({
                    "parameter": "shared_geometry",
                    "component_index": 0,
                    "role": "geometry",
                    "local_index": local_index,
                    "value": float(dof.value),
                    "lower_bound": float(dof.lower_bound),
                    "upper_bound": float(dof.upper_bound),
                })
    return rows


def snapshot_active_dof_summary(
    snapshot: PhaseSnapshot,
    *,
    include_geometry: bool,
) -> list[dict[str, Any]]:
    """Label active homogeneous/amplitude and optional Phase-2 geometry DOFs."""

    rows: list[dict[str, Any]] = []
    if include_geometry:
        for items in snapshot.spatial_parameterisations.values():
            basis_item = next(
                (
                    item for item in items
                    if item.summary.get("kind") == "basis_functions"
                ),
                None,
            )
            if basis_item is None:
                continue
            for kernel_index, kernel in enumerate(
                basis_item.summary.get("kernels", [])
            ):
                for name, value in _kernel_geometry_values(kernel):
                    rows.append({
                        "parameter": "shared_geometry",
                        "component_index": kernel_index,
                        "role": "geometry",
                        "name": name,
                        "value": value,
                    })
            break
    for parameter_name, items in snapshot.spatial_parameterisations.items():
        for component_index, item in enumerate(items):
            kind = item.summary.get("kind")
            if kind == "homogeneous":
                for local_index, value in enumerate(item.dof_values):
                    rows.append({
                        "parameter": parameter_name,
                        "component_index": component_index,
                        "role": "homogeneous",
                        "local_index": local_index,
                        "value": float(value),
                    })
            elif kind == "basis_functions":
                for local_index, value in enumerate(item.dof_values):
                    rows.append({
                        "parameter": parameter_name,
                        "component_index": component_index,
                        "role": "amplitude",
                        "local_index": local_index,
                        "value": float(value),
                    })
    return rows


def _fixed_support(
    geometry: FixedGeometryState,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> SupportBasis:
    kernels = [_fixed_kernel(summary) for summary in geometry.kernel_summaries]
    support = SupportBasis(np.asarray(x), np.asarray(y), kernels)
    if support.get_num_degrees_of_freedom() != 0:
        raise RuntimeError("Reconstructed Phase-2 geometry is not fixed.")
    return support


def _fixed_kernel(summary: dict[str, Any]) -> BasisFunctionKernel:
    centre_x, centre_y = (float(value) for value in summary["centre"])
    type_name = str(summary["kernel_type"])
    if type_name == "BasisFunctionKernelUnivariate":
        return BasisFunctionKernelUnivariate(
            centre_x, centre_y, float(summary["variance"])
        )
    if type_name == "BasisFunctionKernelBivariate":
        variance_x, variance_y = (float(value) for value in summary["variance"])
        return BasisFunctionKernelBivariate(
            centre_x,
            centre_y,
            variance_x,
            variance_y,
            float(summary["angle"]),
        )
    if type_name == "BasisFunctionKernelBivariateSPD":
        log_covariance = np.asarray(summary["log_covariance"], dtype=np.float64)
        return BasisFunctionKernelBivariateSPD(
            centre_x,
            centre_y,
            float(log_covariance[0, 0]),
            float(log_covariance[0, 1]),
            float(log_covariance[1, 1]),
            float(summary["reference_variance"]),
        )
    raise ValueError(f"Unsupported Phase-2 kernel type: {type_name}.")


def _basis_parameterisation(
    geometry: FixedGeometryState,
    support: SupportBasis,
    *,
    amplitudes: tuple[float, ...],
    amplitude_bound: float,
) -> SpatialParameterisationBasisFunction:
    if amplitude_bound <= 0.0:
        raise ValueError("Amplitude bound must be positive.")
    heights = [
        DegreeOfFreedom(float(value), -amplitude_bound, amplitude_bound)
        for value in amplitudes
    ]
    return SpatialParameterisationBasisFunction(
        support=support,
        heights=heights,
        kernel_type=geometry.kernel_type,
    )


def _parameterisation_kernel_type(type_name: str) -> str:
    values = {
        "BasisFunctionKernelUnivariate": "univariate",
        "BasisFunctionKernelBivariate": "bivariate",
        "BasisFunctionKernelBivariateSPD": "bivariate_spd",
    }
    try:
        return values[type_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Phase-2 kernel type: {type_name}.") from exc


def _kernel_geometry_values(
    kernel: dict[str, Any],
) -> list[tuple[str, float]]:
    centre = kernel["centre"]
    rows = [("centre_x", float(centre[0])), ("centre_y", float(centre[1]))]
    type_name = kernel["kernel_type"]
    if type_name == "BasisFunctionKernelUnivariate":
        rows.append(("variance", float(kernel["variance"])))
    elif type_name == "BasisFunctionKernelBivariate":
        rows.extend((
            ("variance_x", float(kernel["variance"][0])),
            ("variance_y", float(kernel["variance"][1])),
            ("angle", float(kernel["angle"])),
        ))
    elif type_name == "BasisFunctionKernelBivariateSPD":
        log_covariance = kernel["log_covariance"]
        rows.extend((
            ("log_covariance_11", float(log_covariance[0][0])),
            ("log_covariance_12", float(log_covariance[0][1])),
            ("log_covariance_22", float(log_covariance[1][1])),
        ))
    else:
        raise ValueError(f"Unsupported Phase-2 kernel type: {type_name}.")
    return rows


def _role(parameterisation: object) -> str:
    if isinstance(parameterisation, SpatialParameterisationHomogeneous):
        return "homogeneous"
    if isinstance(parameterisation, SpatialParameterisationBasisFunction):
        return "amplitude"
    return "fixed"

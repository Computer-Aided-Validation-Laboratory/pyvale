"""Rank-revealing material sensitivity subspaces for residual projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import TYPE_CHECKING, Mapping

import numpy as np
import numpy.typing as npt

from pyvale.vfm.loadregimes import ResolvedLoadRegimes
from pyvale.vfm.residualblocks import CanonicalResidualLayout, ResidualBlockSpec

if TYPE_CHECKING:
    from pyvale.vfm.solvepreparation import SolvePreparationContext


@dataclass(slots=True, frozen=True)
class ProjectionBasis:
    basis: npt.NDArray[np.float64]
    rank: int
    singular_values: tuple[float, ...]
    condition_estimate: float
    relative_tolerance: float

    def diagnostics(self) -> dict[str, object]:
        result = asdict(self)
        result.pop("basis")
        result["observation_count"] = int(self.basis.shape[0])
        return result


@dataclass(slots=True, frozen=True)
class MaterialProjectionBases:
    full: ProjectionBasis
    yield_basis: ProjectionBasis | None
    hardening_basis: ProjectionBasis | None
    yield_unique: ProjectionBasis | None
    yield_hardening_max_correlation: float | None

    def diagnostics(self) -> dict[str, object]:
        return {
            "full": self.full.diagnostics(),
            "yield": None if self.yield_basis is None else self.yield_basis.diagnostics(),
            "hardening": None if self.hardening_basis is None else self.hardening_basis.diagnostics(),
            "yield_unique": None if self.yield_unique is None else self.yield_unique.diagnostics(),
            "yield_hardening_max_correlation": self.yield_hardening_max_correlation,
        }


@dataclass(slots=True, frozen=True)
class FiniteDifferenceSensitivity:
    """Bound-aware native-DOF residual Jacobian and audit metadata."""

    matrix: npt.NDArray[np.float64]
    reference_residual: npt.NDArray[np.float64]
    schemes: tuple[str, ...]
    step_sizes: tuple[float, ...]

    def diagnostics(self) -> dict[str, object]:
        return {
            "observation_count": int(self.matrix.shape[0]),
            "degree_of_freedom_count": int(self.matrix.shape[1]),
            "schemes": list(self.schemes),
            "step_sizes": list(self.step_sizes),
        }


@dataclass(slots=True, frozen=True)
class NativeDofSensitivityAudit:
    """Serializable audit of the canonical residual at one fixed-BF state."""

    sensitivity: FiniteDifferenceSensitivity
    projection_bases: MaterialProjectionBases
    parameter_groups: tuple[str, ...]
    column_norms: tuple[float, ...]
    runtime_seconds: float
    residual_layout_diagnostics: dict[str, object]

    def diagnostics(self) -> dict[str, object]:
        return {
            "finite_difference": self.sensitivity.diagnostics(),
            "projection_bases": self.projection_bases.diagnostics(),
            "parameter_groups": list(self.parameter_groups),
            "column_norms": list(self.column_norms),
            "runtime_seconds": self.runtime_seconds,
            "residual_layout": self.residual_layout_diagnostics,
        }


@dataclass(slots=True, frozen=True)
class NativeDofSensitivityAuditConfig:
    """Opt-in configuration for an audit prepared before a fixed-BF solve."""

    load_regimes: ResolvedLoadRegimes
    residual_blocks: tuple[ResidualBlockSpec, ...]
    step: float = 1.0e-3
    relative_tolerance: float = 1.0e-8
    parameter_group_by_name: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.residual_blocks:
            raise ValueError("Sensitivity audit requires residual blocks.")
        if not np.isfinite(self.step) or self.step <= 0.0:
            raise ValueError("Sensitivity audit step must be positive.")
        if (
            not np.isfinite(self.relative_tolerance)
            or self.relative_tolerance <= 0.0
        ):
            raise ValueError("Sensitivity audit tolerance must be positive.")
        names = [name for name, _ in self.parameter_group_by_name]
        if len(names) != len(set(names)):
            raise ValueError("Parameter group mapping names must be unique.")

    def prepare(
        self,
        context: "SolvePreparationContext",
        *,
        progress_callback=None,
    ) -> NativeDofSensitivityAudit:
        layout = context.prepare_residual_layout(
            self.load_regimes,
            self.residual_blocks,
        )
        return prepare_native_dof_sensitivity_audit(
            context,
            layout,
            step=self.step,
            relative_tolerance=self.relative_tolerance,
            parameter_group_by_name=dict(self.parameter_group_by_name),
            progress_callback=progress_callback,
        )


def build_material_projection_bases(
    sensitivity: npt.ArrayLike,
    *,
    parameter_groups: tuple[str, ...],
    observation_scale: npt.ArrayLike | None = None,
    relative_tolerance: float = 1.0e-8,
) -> MaterialProjectionBases:
    """Build full, grouped, and yield-unique bases from residual derivatives.

    ``sensitivity`` has shape ``(observations, active_dofs)``.  Each column is
    the derivative with respect to an actual normalised active DOF.  Optional
    observation scales are standard deviations and are used for diagonal
    whitening before the rank-revealing SVD.
    """

    matrix = np.asarray(sensitivity, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(parameter_groups):
        raise ValueError("parameter_groups must label every sensitivity column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("sensitivity must contain only finite values.")
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive.")
    if observation_scale is not None:
        scale = np.asarray(observation_scale, dtype=np.float64)
        if scale.shape != (matrix.shape[0],) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("observation_scale must be positive with one value per row.")
        matrix = matrix / scale[:, np.newaxis]

    groups = np.asarray(parameter_groups, dtype=object)
    yield_columns = np.flatnonzero(groups == "yield")
    hardening_columns = np.flatnonzero(groups == "hardening")
    full = _svd_basis(matrix, relative_tolerance)
    yield_basis = _optional_basis(matrix[:, yield_columns], relative_tolerance)
    hardening_basis = _optional_basis(matrix[:, hardening_columns], relative_tolerance)
    yield_unique = None
    if yield_columns.size:
        unique_matrix = matrix[:, yield_columns]
        if hardening_basis is not None and hardening_basis.rank:
            unique_matrix = unique_matrix - hardening_basis.basis @ (
                hardening_basis.basis.T @ unique_matrix
            )
        yield_unique = _svd_basis(unique_matrix, relative_tolerance)

    max_correlation = None
    if yield_basis is not None and hardening_basis is not None and yield_basis.rank and hardening_basis.rank:
        max_correlation = float(np.max(np.linalg.svd(
            yield_basis.basis.T @ hardening_basis.basis,
            compute_uv=False,
        )))
    return MaterialProjectionBases(
        full=full,
        yield_basis=yield_basis,
        hardening_basis=hardening_basis,
        yield_unique=yield_unique,
        yield_hardening_max_correlation=max_correlation,
    )


def central_difference_sensitivity(
    reference_dofs: npt.ArrayLike,
    evaluate_residual,
    *,
    step: float = 1.0e-3,
    progress_callback=None,
) -> npt.NDArray[np.float64]:
    """Differentiate a residual vector without mutating the accepted DOFs."""

    return bound_aware_sensitivity(
        reference_dofs,
        evaluate_residual,
        step=step,
        lower_bounds=-np.inf,
        upper_bounds=np.inf,
        progress_callback=progress_callback,
    ).matrix


def prepare_native_dof_sensitivity_audit(
    context: "SolvePreparationContext",
    residual_layout: CanonicalResidualLayout,
    *,
    step: float = 1.0e-3,
    relative_tolerance: float = 1.0e-8,
    parameter_group_by_name: Mapping[str, str] | None = None,
    progress_callback=None,
) -> NativeDofSensitivityAudit:
    """Differentiate the frozen canonical vector with respect to active DOFs.

    This service is diagnostic only: candidates are evaluated through the
    copied solve-preparation state and neither the runtime model nor objective
    is changed. The layout's masks and diagonal whitening remain frozen for
    every perturbation.
    """

    started_at = time.perf_counter()

    def evaluate(dofs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        metric_results = context.evaluate_metric_results(dofs)
        return residual_layout.evaluate(metric_results).weighted

    sensitivity = bound_aware_sensitivity(
        context.normalised_degrees_of_freedom,
        evaluate,
        step=step,
        lower_bounds=0.0,
        upper_bounds=1.0,
        progress_callback=progress_callback,
    )
    groups = tuple(
        _resolve_parameter_group(
            descriptor.parameter_names,
            parameter_group_by_name,
        )
        for descriptor in context.degrees_of_freedom
    )
    bases = build_material_projection_bases(
        sensitivity.matrix,
        parameter_groups=groups,
        relative_tolerance=relative_tolerance,
    )
    return NativeDofSensitivityAudit(
        sensitivity=sensitivity,
        projection_bases=bases,
        parameter_groups=groups,
        column_norms=tuple(
            float(value) for value in np.linalg.norm(sensitivity.matrix, axis=0)
        ),
        runtime_seconds=time.perf_counter() - started_at,
        residual_layout_diagnostics=residual_layout.diagnostics(),
    )


def _resolve_parameter_group(
    parameter_names: tuple[str, ...],
    parameter_group_by_name: Mapping[str, str] | None,
) -> str:
    mapping = parameter_group_by_name or {}
    groups: set[str] = set()
    for name in parameter_names:
        if name in mapping:
            groups.add(mapping[name])
        elif "yield" in name.lower():
            groups.add("yield")
        elif "harden" in name.lower():
            groups.add("hardening")
        else:
            groups.add(name)
    if not groups:
        return "unassigned"
    if len(groups) == 1:
        return next(iter(groups))
    return "shared:" + "+".join(sorted(groups))


def bound_aware_sensitivity(
    reference_dofs: npt.ArrayLike,
    evaluate_residual,
    *,
    step: float = 1.0e-3,
    lower_bounds: npt.ArrayLike | float = 0.0,
    upper_bounds: npt.ArrayLike | float = 1.0,
    progress_callback=None,
) -> FiniteDifferenceSensitivity:
    """Differentiate a residual vector using feasible normalised-DOF steps.

    Central differences are used in the interior and first-order one-sided
    differences at active bounds. The accepted point is evaluated once more
    in a ``finally`` block, giving stateful residual evaluators an exact and
    exception-safe restoration call.
    """

    reference = np.asarray(reference_dofs, dtype=np.float64).copy()
    if reference.ndim != 1 or not np.all(np.isfinite(reference)):
        raise ValueError("reference_dofs must be a finite vector.")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive.")
    lower = _broadcast_bounds(lower_bounds, reference, "lower_bounds")
    upper = _broadcast_bounds(upper_bounds, reference, "upper_bounds")
    if np.any(lower >= upper):
        raise ValueError("Every lower bound must be below its upper bound.")
    if np.any(reference < lower) or np.any(reference > upper):
        raise ValueError("reference_dofs must lie within the supplied bounds.")

    columns: list[npt.NDArray[np.float64]] = []
    schemes: list[str] = []
    step_sizes: list[float] = []
    total = reference.size
    reference_residual = _finite_residual(evaluate_residual(reference))
    try:
        for index in range(total):
            forward_room = float(upper[index] - reference[index])
            backward_room = float(reference[index] - lower[index])
            if forward_room >= step and backward_room >= step:
                plus = reference.copy()
                minus = reference.copy()
                plus[index] += step
                minus[index] -= step
                forward = _finite_residual(evaluate_residual(plus))
                backward = _finite_residual(evaluate_residual(minus))
                _require_matching_residual(forward, backward, reference_residual)
                column = (forward - backward) / (2.0 * step)
                scheme = "central"
                used_step = step
            elif (
                forward_room >= backward_room
                and forward_room > np.finfo(np.float64).eps
            ):
                used_step = min(step, forward_room)
                plus = reference.copy()
                plus[index] += used_step
                forward = _finite_residual(evaluate_residual(plus))
                _require_matching_residual(forward, reference_residual)
                column = (forward - reference_residual) / used_step
                scheme = "forward"
            elif backward_room > np.finfo(np.float64).eps:
                used_step = min(step, backward_room)
                minus = reference.copy()
                minus[index] -= used_step
                backward = _finite_residual(evaluate_residual(minus))
                _require_matching_residual(backward, reference_residual)
                column = (reference_residual - backward) / used_step
                scheme = "backward"
            else:
                raise ValueError(
                    f"DOF {index} has no numerically usable perturbation room."
                )
            columns.append(column)
            schemes.append(scheme)
            step_sizes.append(float(used_step))
            if progress_callback is not None:
                progress_callback(index + 1, total)
    finally:
        restored = _finite_residual(evaluate_residual(reference))
        _require_matching_residual(restored, reference_residual)

    matrix = (
        np.column_stack(columns)
        if columns
        else np.empty((reference_residual.size, 0), dtype=np.float64)
    )
    return FiniteDifferenceSensitivity(
        matrix=matrix,
        reference_residual=reference_residual,
        schemes=tuple(schemes),
        step_sizes=tuple(step_sizes),
    )


def _broadcast_bounds(
    values: npt.ArrayLike | float,
    reference: npt.NDArray[np.float64],
    name: str,
) -> npt.NDArray[np.float64]:
    try:
        result = np.broadcast_to(
            np.asarray(values, dtype=np.float64),
            reference.shape,
        ).copy()
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to the reference shape.") from exc
    if np.any(np.isnan(result)):
        raise ValueError(f"{name} cannot contain NaNs.")
    return result


def _finite_residual(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    residual = np.asarray(values, dtype=np.float64).ravel()
    if np.any(~np.isfinite(residual)):
        raise ValueError("Residual evaluations must return finite vectors.")
    return residual


def _require_matching_residual(
    *residuals: npt.NDArray[np.float64],
) -> None:
    if len({residual.shape for residual in residuals}) != 1:
        raise ValueError("Residual perturbations must return matching vectors.")


def _optional_basis(matrix: npt.NDArray[np.float64], tolerance: float) -> ProjectionBasis | None:
    return None if matrix.shape[1] == 0 else _svd_basis(matrix, tolerance)


def _svd_basis(matrix: npt.NDArray[np.float64], tolerance: float) -> ProjectionBasis:
    if matrix.ndim != 2:
        raise ValueError("Sensitivity matrix must be two-dimensional.")
    if matrix.shape[1] == 0:
        return ProjectionBasis(np.empty((matrix.shape[0], 0)), 0, (), float("inf"), tolerance)
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    threshold = tolerance * singular[0] if singular.size and singular[0] > 0.0 else 0.0
    rank = int(np.sum(singular > threshold))
    retained = singular[:rank]
    condition = float(retained[0] / retained[-1]) if rank else float("inf")
    return ProjectionBasis(
        basis=left[:, :rank],
        rank=rank,
        singular_values=tuple(float(value) for value in singular),
        condition_estimate=condition,
        relative_tolerance=tolerance,
    )

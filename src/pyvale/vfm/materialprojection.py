"""Rank-revealing material sensitivity subspaces for residual projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt


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

    reference = np.asarray(reference_dofs, dtype=np.float64).copy()
    if reference.ndim != 1 or not np.all(np.isfinite(reference)):
        raise ValueError("reference_dofs must be a finite vector.")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive.")
    columns: list[npt.NDArray[np.float64]] = []
    total = reference.size
    for index in range(total):
        plus = reference.copy()
        minus = reference.copy()
        plus[index] += step
        minus[index] -= step
        forward = np.asarray(evaluate_residual(plus), dtype=np.float64).ravel()
        backward = np.asarray(evaluate_residual(minus), dtype=np.float64).ravel()
        if forward.shape != backward.shape or np.any(~np.isfinite(forward)) or np.any(~np.isfinite(backward)):
            raise ValueError("Residual perturbations must return matching finite vectors.")
        columns.append((forward - backward) / (2.0 * step))
        if progress_callback is not None:
            progress_callback(index + 1, total)
    # A final evaluation gives stateful callers an explicit restoration hook.
    np.asarray(evaluate_residual(reference), dtype=np.float64)
    return np.column_stack(columns) if columns else np.empty((0, 0), dtype=np.float64)


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

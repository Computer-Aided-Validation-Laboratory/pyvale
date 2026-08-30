"""Frozen sensitivity-information objective with FRE and broad-EGI guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

from pyvale.vfm.loadregimes import ResolvedLoadRegimes
from pyvale.vfm.materialprojection import bound_aware_sensitivity
from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefunc import IScalarObjectiveFunction
from pyvale.vfm.residualblocks import CanonicalResidualLayout, ResidualBlockSpec
from pyvale.vfm.solvepreparation import SolvePreparationContext


FloatArray = npt.NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class SensitivityInformationObjectiveConfig:
    """Fixed numerical rules for one prepared sensitivity objective."""

    load_regimes: ResolvedLoadRegimes
    residual_blocks: tuple[ResidualBlockSpec, ...]
    finite_difference_step: float = 1.0e-3
    meaningful_dof_movement: float = 1.0e-2
    minimum_noise_response: float = 1.0
    projection_covariance_floor: float = 1.0e-12
    robust_transition: float = 1.5

    def __post_init__(self) -> None:
        if not self.residual_blocks:
            raise ValueError("Sensitivity objective requires residual blocks.")
        names = {block.name for block in self.residual_blocks}
        if len(names) != len(self.residual_blocks):
            raise ValueError("Residual block names must be unique.")
        roles = {block.role for block in self.residual_blocks}
        if "fre_guard" not in roles or "broad_egi_guard" not in roles:
            raise ValueError(
                "Residual blocks must contain fre_guard and broad_egi_guard roles."
            )
        for name, value in (
            ("finite_difference_step", self.finite_difference_step),
            ("meaningful_dof_movement", self.meaningful_dof_movement),
            ("minimum_noise_response", self.minimum_noise_response),
            ("projection_covariance_floor", self.projection_covariance_floor),
            ("robust_transition", self.robust_transition),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass(slots=True, frozen=True)
class PreparedSensitivityInformation:
    """Read-only transforms calculated at a fixed-basis solve start."""

    layout: CanonicalResidualLayout
    basis: FloatArray
    projection_inverse_sqrt_covariance: FloatArray
    singular_values: tuple[float, ...]
    retained_rank: int
    absolute_singular_threshold: float
    reference_dofs: FloatArray
    fre_slices: tuple[slice, ...]
    broad_egi_slices: tuple[slice, ...]
    diagnostics_summary: dict[str, object]


@dataclass(slots=True, frozen=True)
class SensitivityInformationObjectiveResult:
    total_cost: float
    material_cost: float
    fre_guard_cost: float
    broad_egi_guard_cost: float
    retained_rank: int


class SensitivityInformationObjective(IScalarObjectiveFunction):
    """Equal-role projected information, FRE closure, and broad-EGI closure.

    ``prepare_solve`` is required exactly once for each fixed-BF solve.  The
    resulting layout, Jacobian, projection, and noise transform are immutable
    during optimiser evaluations.
    """

    def __init__(
        self,
        config: SensitivityInformationObjectiveConfig,
        *,
        diagnostic_callback: Callable[[str, dict[str, object]], None] | None = None,
        basis_growth_objective: IScalarObjectiveFunction | None = None,
    ) -> None:
        self.config = config
        self.diagnostic_callback = diagnostic_callback
        # Used only by basis-placement policies which require the established
        # mechanical-closure cotangent. It does not enter this objective's cost.
        self.global_objective = basis_growth_objective
        self._prepared: PreparedSensitivityInformation | None = None
        self.preparation_count = 0
        self.last_result: SensitivityInformationObjectiveResult | None = None

    @property
    def baseline(self):
        """Expose the placement-only closure baseline to phase lifecycle code."""

        return getattr(self.global_objective, "baseline", None)

    @property
    def spatial_weighting(self):
        return getattr(self.global_objective, "spatial_weighting", None)

    def resolve_from_prior_phase(self, metric_results: list[MetricResult]) -> None:
        resolver = getattr(self.global_objective, "resolve_from_prior_phase", None)
        if resolver is not None:
            resolver(metric_results)

    def resolve_spatial_weights(self, **kwargs) -> None:
        resolver = getattr(self.global_objective, "resolve_spatial_weights", None)
        if resolver is not None:
            resolver(**kwargs)

    def baseline_diagnostics(self) -> dict[str, object]:
        diagnostics = getattr(self.global_objective, "baseline_diagnostics", None)
        return {} if diagnostics is None else diagnostics()

    def spatial_weighting_diagnostics(self) -> dict[str, object]:
        diagnostics = getattr(self.global_objective, "spatial_weighting_diagnostics", None)
        return {} if diagnostics is None else diagnostics()

    def prepare_solve(self, context: SolvePreparationContext) -> dict[str, object]:
        layout = context.prepare_residual_layout(
            self.config.load_regimes, self.config.residual_blocks
        )

        def evaluate(dofs: npt.ArrayLike) -> FloatArray:
            return layout.evaluate(context.evaluate_metric_results(dofs)).weighted

        finite_difference = bound_aware_sensitivity(
            context.normalised_degrees_of_freedom,
            evaluate,
            step=self.config.finite_difference_step,
            lower_bounds=0.0,
            upper_bounds=1.0,
        )
        sensitivity = finite_difference.matrix
        left, singular, _ = np.linalg.svd(sensitivity, full_matrices=False)
        absolute_threshold = (
            self.config.minimum_noise_response
            / self.config.meaningful_dof_movement
        )
        rank = int(np.count_nonzero(singular >= absolute_threshold))
        basis = left[:, :rank].copy()
        basis.setflags(write=False)
        inverse_sqrt, covariance_eigenvalues = _projection_inverse_sqrt_covariance(
            layout, basis, self.config.projection_covariance_floor
        )
        inverse_sqrt.setflags(write=False)
        reference_dofs = context.normalised_degrees_of_freedom.copy()
        reference_dofs.setflags(write=False)
        slices = dict(
            (name, slice(start, stop))
            for name, start, stop in layout.evaluate(context.metric_results).block_slices
        )
        fre_slices = tuple(
            slices[block.spec.name]
            for block in layout.blocks
            if block.spec.role == "fre_guard"
        )
        broad_slices = tuple(
            slices[block.spec.name]
            for block in layout.blocks
            if block.spec.role == "broad_egi_guard"
        )
        diagnostics = {
            "preparation_count": self.preparation_count + 1,
            "phase_index": context.phase_index,
            "solve_iteration": context.solve_iteration,
            "reference_dof_count": int(reference_dofs.size),
            "finite_difference": finite_difference.diagnostics(),
            "singular_values": [float(value) for value in singular],
            "retained_rank": rank,
            "absolute_singular_threshold": absolute_threshold,
            "projection_covariance_eigenvalues": covariance_eigenvalues.tolist(),
            "residual_layout": layout.diagnostics(),
            "roles": {
                "fre_guard_blocks": [
                    block.spec.name for block in layout.blocks
                    if block.spec.role == "fre_guard"
                ],
                "broad_egi_guard_blocks": [
                    block.spec.name for block in layout.blocks
                    if block.spec.role == "broad_egi_guard"
                ],
            },
        }
        self._prepared = PreparedSensitivityInformation(
            layout=layout,
            basis=basis,
            projection_inverse_sqrt_covariance=inverse_sqrt,
            singular_values=tuple(float(value) for value in singular),
            retained_rank=rank,
            absolute_singular_threshold=float(absolute_threshold),
            reference_dofs=reference_dofs,
            fre_slices=fre_slices,
            broad_egi_slices=broad_slices,
            diagnostics_summary=diagnostics,
        )
        self.preparation_count += 1
        if self.diagnostic_callback is not None:
            self.diagnostic_callback(
                "solve_sensitivity",
                {
                    "phase_index": context.phase_index,
                    "solve_iteration": context.solve_iteration,
                    "reference_dofs": reference_dofs,
                    "sensitivity": sensitivity.copy(),
                    "basis": basis.copy(),
                    "projection_inverse_sqrt_covariance": inverse_sqrt.copy(),
                    "singular_values": singular.copy(),
                    "retained_rank": rank,
                    "absolute_singular_threshold": absolute_threshold,
                    "blocks": [
                        {
                            "name": block.spec.name,
                            "role": block.spec.role,
                            "metric_kind": block.spec.metric_kind,
                            "metric_index": block.spec.metric_index,
                            "physical_support": block.spec.physical_support,
                            "pixel_support": block.spec.pixel_support,
                            "source_shape": block.source_shape,
                            "frame_indices": block.frame_indices,
                            "valid_indices": block.valid_indices.copy(),
                            "square_root_weights": block.square_root_weights.copy(),
                            "noise_scale": block.noise_scale.copy(),
                        }
                        for block in layout.blocks
                    ],
                },
            )
        return self.diagnostics()

    def evaluate(self, metric_results: list[MetricResult]) -> float:
        prepared = self._require_prepared()
        vector = prepared.layout.evaluate(metric_results)
        material_cost = 0.0
        if prepared.retained_rank:
            coordinates = prepared.projection_inverse_sqrt_covariance @ (
                prepared.basis.T @ vector.weighted
            )
            material_cost = _mean_huber(coordinates, self.config.robust_transition)
        fre_cost = _mean_huber(
            _join_slices(vector.whitened, prepared.fre_slices),
            self.config.robust_transition,
        )
        broad_cost = _mean_huber(
            _join_slices(vector.whitened, prepared.broad_egi_slices),
            self.config.robust_transition,
        )
        total = float((material_cost + fre_cost + broad_cost) / 3.0)
        self.last_result = SensitivityInformationObjectiveResult(
            total_cost=total,
            material_cost=material_cost,
            fre_guard_cost=fre_cost,
            broad_egi_guard_cost=broad_cost,
            retained_rank=prepared.retained_rank,
        )
        return total

    def diagnostics(self) -> dict[str, object]:
        prepared = self._require_prepared()
        result = dict(prepared.diagnostics_summary)
        if self.last_result is not None:
            result["last_costs"] = {
                "total": self.last_result.total_cost,
                "material": self.last_result.material_cost,
                "fre_guard": self.last_result.fre_guard_cost,
                "broad_egi_guard": self.last_result.broad_egi_guard_cost,
            }
        return result

    def _require_prepared(self) -> PreparedSensitivityInformation:
        if self._prepared is None:
            raise RuntimeError(
                "SensitivityInformationObjective must be prepared before evaluation."
            )
        return self._prepared


def _projection_inverse_sqrt_covariance(
    layout: CanonicalResidualLayout,
    basis: FloatArray,
    floor: float,
) -> tuple[FloatArray, FloatArray]:
    if basis.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64)
    square_weights = np.concatenate([
        block.square_root_weights ** 2 for block in layout.blocks
    ])
    covariance = basis.T @ (square_weights[:, np.newaxis] * basis)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped = np.maximum(eigenvalues, floor)
    inverse_sqrt = (eigenvectors / np.sqrt(clipped)) @ eigenvectors.T
    return np.asarray(inverse_sqrt, dtype=np.float64), np.asarray(eigenvalues, dtype=np.float64)


def _join_slices(values: FloatArray, slices: Sequence[slice]) -> FloatArray:
    parts = [values[item] for item in slices]
    if not parts:
        raise RuntimeError("Prepared objective has no required guard observations.")
    return np.concatenate(parts)


def _mean_huber(values: FloatArray, transition: float) -> float:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    loss = np.where(
        absolute <= transition,
        0.5 * absolute ** 2,
        transition * (absolute - 0.5 * transition),
    )
    return float(np.mean(loss))

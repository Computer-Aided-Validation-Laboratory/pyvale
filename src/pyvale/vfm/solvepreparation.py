"""Immutable state supplied to optional fixed-BF solve preparation hooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.metric import IMetric, MetricResult
from pyvale.vfm.optimiser import evaluate_metrics
from pyvale.vfm.residualblocks import (
    CanonicalResidualLayout,
    ResidualBlockSpec,
    prepare_canonical_residual_layout,
)
from pyvale.vfm.loadregimes import ResolvedLoadRegimes
from pyvale.vfm.spatialparam import PhaseSpatialState


@dataclass(slots=True, frozen=True)
class SolveDegreeOfFreedom:
    """Serializable description of one active optimiser coordinate."""

    index: int
    local_index: int
    parameter_names: tuple[str, ...]
    owner_type: str
    role: str
    value: float
    lower_bound: float
    upper_bound: float
    scaling: str
    normalised_value: float


@dataclass(slots=True)
class SolvePreparationContext:
    """Frozen fixed-BF state available immediately before optimisation.

    The stored spatial state is a deep copy of the runtime state. Candidate
    evaluations therefore cannot mutate the accepted stage-start model. This
    context is intentionally richer than the old ``prepare_solve`` metric-list
    argument so objectives and audit services can build native-DOF
    sensitivities from exactly the state the optimiser will receive.
    """

    phase_index: int
    solve_iteration: int
    constitutive_law: IConstitutiveLaw
    parameter_map_size: npt.NDArray[np.uint32]
    spatial_state: PhaseSpatialState
    metrics: tuple[IMetric, ...]
    experiment_data: ExperimentData
    metric_results: tuple[MetricResult, ...]
    parameter_maps: dict[str, npt.NDArray[np.float64]]
    stress: npt.NDArray[np.float64]
    normalised_degrees_of_freedom: npt.NDArray[np.float64]
    degrees_of_freedom: tuple[SolveDegreeOfFreedom, ...]

    def evaluate_metric_results(
        self,
        normalised_degrees_of_freedom: npt.ArrayLike | None = None,
        *,
        include_egi_diagnostics: bool = True,
    ) -> list[MetricResult]:
        """Evaluate metrics on a copied candidate state without mutation."""

        candidate = self.normalised_degrees_of_freedom.copy()
        if normalised_degrees_of_freedom is not None:
            candidate = np.asarray(
                normalised_degrees_of_freedom,
                dtype=np.float64,
            )
            if candidate.shape != self.normalised_degrees_of_freedom.shape:
                raise ValueError(
                    "Candidate normalised DOFs do not match the prepared "
                    f"shape: {candidate.shape} vs "
                    f"{self.normalised_degrees_of_freedom.shape}."
                )
            if np.any(~np.isfinite(candidate)):
                raise ValueError("Candidate normalised DOFs must be finite.")
            if np.any((candidate < 0.0) | (candidate > 1.0)):
                raise ValueError("Candidate normalised DOFs must lie in [0, 1].")

        state = self.spatial_state.copy()
        state.update_from_normalised_degrees_of_freedom(candidate)
        parameter_maps = state.evaluate_parameter_maps(self.parameter_map_size)
        stress = self.constitutive_law.calculate_stress(
            self.experiment_data.strain,
            parameter_maps,
        )
        return evaluate_metrics(
            stress,
            self.constitutive_law,
            self.parameter_map_size,
            state.spatial_parameterisations,
            list(self.metrics),
            self.experiment_data,
            include_egi_diagnostics=include_egi_diagnostics,
        )

    def prepare_residual_layout(
        self,
        load_regimes: ResolvedLoadRegimes,
        specs: tuple[ResidualBlockSpec, ...] | list[ResidualBlockSpec],
    ) -> CanonicalResidualLayout:
        """Freeze a canonical residual layout at this accepted solve state."""

        return prepare_canonical_residual_layout(
            self.metric_results,
            load_regimes,
            specs,
        )


def build_solve_preparation_context(
    *,
    phase_index: int,
    solve_iteration: int,
    constitutive_law: IConstitutiveLaw,
    parameter_map_size: npt.NDArray[np.uint32],
    spatial_state: PhaseSpatialState,
    metrics: list[IMetric],
    experiment_data: ExperimentData,
) -> SolvePreparationContext:
    """Snapshot and evaluate the current state for a preparation hook."""

    if phase_index < 0 or solve_iteration < 0:
        raise ValueError("Phase and solve indices must be non-negative.")
    prepared_state = spatial_state.copy()
    normalised = prepared_state.collect_normalised_degrees_of_freedom()
    parameter_maps = {
        name: np.asarray(values, dtype=np.float64).copy()
        for name, values in prepared_state.evaluate_parameter_maps(
            parameter_map_size
        ).items()
    }
    stress = np.asarray(
        constitutive_law.calculate_stress(
            experiment_data.strain,
            parameter_maps,
        ),
        dtype=np.float64,
    )
    metric_results = evaluate_metrics(
        stress,
        constitutive_law,
        parameter_map_size,
        prepared_state.spatial_parameterisations,
        metrics,
        experiment_data,
        include_egi_diagnostics=True,
    )
    return SolvePreparationContext(
        phase_index=phase_index,
        solve_iteration=solve_iteration,
        constitutive_law=constitutive_law,
        parameter_map_size=np.asarray(parameter_map_size, dtype=np.uint32).copy(),
        spatial_state=prepared_state,
        metrics=tuple(metrics),
        experiment_data=experiment_data,
        metric_results=tuple(metric_results),
        parameter_maps=parameter_maps,
        stress=stress.copy(),
        normalised_degrees_of_freedom=normalised.copy(),
        degrees_of_freedom=_snapshot_degrees_of_freedom(
            prepared_state,
            normalised,
        ),
    )


def _snapshot_degrees_of_freedom(
    state: PhaseSpatialState,
    normalised: npt.NDArray[np.float64],
) -> tuple[SolveDegreeOfFreedom, ...]:
    support_parameters: dict[int, set[str]] = {}
    for parameter_name, parameterisations in (
        state.spatial_parameterisations.items()
    ):
        for parameterisation in parameterisations:
            support = getattr(parameterisation, "support", None)
            if support is not None:
                support_parameters.setdefault(id(support), set()).add(
                    parameter_name
                )

    labelled_owners: list[tuple[object, tuple[str, ...], str]] = [
        (
            support,
            tuple(sorted(support_parameters.get(id(support), set()))),
            "geometry",
        )
        for support in state.supports
    ]
    for parameter_name, parameterisations in (
        state.spatial_parameterisations.items()
    ):
        for parameterisation in parameterisations:
            labelled_owners.append(
                (
                    parameterisation,
                    (parameter_name,),
                    _parameterisation_role(parameterisation),
                )
            )

    snapshots: list[SolveDegreeOfFreedom] = []
    global_index = 0
    for owner, parameter_names, role in labelled_owners:
        owner_dofs: list[DegreeOfFreedom] = owner.collect_degrees_of_freedom()
        for local_index, dof in enumerate(owner_dofs):
            snapshots.append(
                SolveDegreeOfFreedom(
                    index=global_index,
                    local_index=local_index,
                    parameter_names=parameter_names,
                    owner_type=type(owner).__name__,
                    role=role,
                    value=float(dof.value),
                    lower_bound=float(dof.lower_bound),
                    upper_bound=float(dof.upper_bound),
                    scaling=str(dof.scaling),
                    normalised_value=float(normalised[global_index]),
                )
            )
            global_index += 1
    if global_index != normalised.size:
        raise RuntimeError(
            "Prepared DOF descriptors do not match the packed DOF vector."
        )
    return tuple(snapshots)


def _parameterisation_role(parameterisation: object) -> str:
    type_name = type(parameterisation).__name__.lower()
    if "homogeneous" in type_name:
        return "homogeneous"
    if "basisfunction" in type_name:
        return "amplitude"
    if "known" in type_name:
        return "fixed"
    return "parameterisation"

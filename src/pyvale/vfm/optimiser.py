from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.identificationresult import OptimisationOutcome
from pyvale.vfm.metric import IMetric, MetricResult
from pyvale.vfm.objectivefunc import IObjectiveFunction
from pyvale.vfm.spatialparam import (
    ISpatialParameterisation,
    PhaseSpatialState,
)


class IOptimiser(ABC):
    """
    Interface (abstract base class) for a VFM optimisation algorithm.

    Drives the parameter search by repeatedly evaluating candidate stress
    fields, computing metric/objective values, and updating the spatial
    parameterisations until convergence
    """

    @abstractmethod
    def get_required_objective_function_type(self) -> type:
        """
        Return the required objective function type for this optimiser.

        Returns
        -------
        type
            ``IScalarObjectiveFunction`` or ``IVectorObjectiveFunction``
        """
        pass

    @abstractmethod
    def optimise(
        self,
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        metrics: list[IMetric],
        objective_function: IObjectiveFunction,
        experiment_data: ExperimentData,
        progress_callback=None,
    ) -> OptimisationOutcome | dict[str, list[ISpatialParameterisation]]:
        """
        Run the optimisation loop for one identification phase.

        Parameters
        ----------
        constitutive_law : IConstitutiveLaw
            Constitutive model whose parameters are being identified
        parameter_map_size : npt.NDArray[np.uint32]
            Spatial dimensions ``(y, x)`` of the parameter maps
        spatial_parameterisations : dict[str, list[ISpatialParameterisation]]
            Initial parameter distributions keyed by parameter name
        metrics : list[IMetric]
            Virtual-work metrics to evaluate candidate stress fields
        objective_function : IObjectiveFunction
            Aggregates metric results into the quantity to be minimised
        experiment_data : ExperimentData
            Measured DIC data
        progress_callback
            Optional callable receiving lightweight progress events

        Returns
        -------
        OptimisationOutcome | dict[str, list[ISpatialParameterisation]]
            Optimised spatial parameterisations after convergence, optionally
            with solver metadata for the identification result history.
        """
        pass


def evaluate_candidate(
    degrees_of_freedom: npt.NDArray[np.float64],
    constitutive_law: IConstitutiveLaw,
    parameter_map_size: npt.NDArray[np.uint32],
    phase_spatial_state: PhaseSpatialState,
    metrics: list[IMetric],
    objective_function: IObjectiveFunction,
    experiment_data: ExperimentData,
) -> float | npt.NDArray[np.float64]:
    """
    Evaluate one candidate point in the design space.

    Unpacks the normalised degrees of freedom into spatial parameterisations,
    computes the resulting stress via the constitutive law, evaluates all
    metrics, and aggregates them with the objective function.

    Parameters
    ----------
    degrees_of_freedom : npt.NDArray[np.float64]
        Normalised spatial parameterisation variables
    constitutive_law : IConstitutiveLaw
        Constitutive model
    parameter_map_size : npt.NDArray[np.uint32]
        Spatial dimensions ``(y, x)`` of the parameter maps
    phase_spatial_state : PhaseSpatialState
        Reference phase spatial state (cloned internally)
    metrics : list[IMetric]
        Virtual-work metrics
    objective_function : IObjectiveFunction
        Scalar or vector objective
    experiment_data : ExperimentData
        Measured DIC data

    Returns
    -------
    float | npt.NDArray[np.float64]
        Scalar or vector objective value for the candidate
    """
    updated_phase_spatial_state = phase_spatial_state.copy()
    updated_phase_spatial_state.update_from_normalised_degrees_of_freedom(
        degrees_of_freedom,
    )

    updated_spatial_parameterisations = (
        updated_phase_spatial_state.spatial_parameterisations
    )
    updated_constitutive_parameter_maps = (
        updated_phase_spatial_state.evaluate_parameter_maps(parameter_map_size)
    )

    updated_stress = constitutive_law.calculate_stress(
        experiment_data.strain, updated_constitutive_parameter_maps,
    )

    metric_results: list[MetricResult] = []
    for metric in metrics:
        metric_results.append(
            metric.evaluate(
                updated_stress,
                constitutive_law,
                parameter_map_size,
                updated_spatial_parameterisations,
                experiment_data,
            ),
        )

    return objective_function.evaluate(metric_results)

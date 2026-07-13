from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.metric import IMetric
from pyvale.vfm.objectivefunc import IObjectiveFunction
from pyvale.vfm.spatialparam import (
    ISpatialParameterisation,
    evaluate_parameterisations_to_map,
    unpack_spatial_parameterisations,
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
    ) -> dict[str, list[ISpatialParameterisation]]:
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

        Returns
        -------
        dict[str, list[ISpatialParameterisation]]
            Optimised spatial parameterisations after convergence
        """
        pass


def evaluate_candidate(
    degrees_of_freedom: npt.NDArray[np.float64],
    constitutive_law: IConstitutiveLaw,
    parameter_map_size: npt.NDArray[np.uint32],
    spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
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
    spatial_parameterisations : dict[str, list[ISpatialParameterisation]]
        Reference spatial parameterisations (cloned internally)
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

    print("Evaluating candidate with degrees of freedom:", degrees_of_freedom)
    updated_spatial_parameterisations = unpack_spatial_parameterisations(
        spatial_parameterisations,
        degrees_of_freedom,
    )

    updated_constitutive_parameter_maps = {
        param_name: evaluate_parameterisations_to_map(sps, parameter_map_size)
        for (param_name, sps) in updated_spatial_parameterisations.items()
    }

    updated_stress = constitutive_law.calculate_stress(
        experiment_data.strain, updated_constitutive_parameter_maps,
    )

    metric_results = []
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

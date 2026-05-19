from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_law import IConstitutiveLaw
from pyvale.vfm.experiment_data import ExperimentData
from pyvale.vfm.metrics.metric import IMetric
from pyvale.vfm.objective_functions.objective_function import IObjectiveFunction
from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    ISpatialParameterisation,
    unpack_spatial_parameterisations,
)


class IOptimiser(ABC):
    # Run a set of optimisation passes until a best guess is found
    # TODO: figure out what to return for different optimisers
    @abstractmethod
    def optimise(
        self,
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, ISpatialParameterisation],
        metrics: list[IMetric],
        objective_function: IObjectiveFunction,
        experiment_data: ExperimentData,
    ) -> dict[str, ISpatialParameterisation]:
        pass


# TODO: in some optimisers will the input be a scalar not a vector?
# TODO: should we constrain return type? or have 2 functions for scalar/vector?
#   or have the scalar representation just be a 1 elem vector?
def evaluate_candidate(
    degrees_of_freedom: npt.NDArray[np.float64],
    constitutive_law: IConstitutiveLaw,
    parameter_map_size: npt.NDArray[np.uint32],
    spatial_parameterisations: dict[str, ISpatialParameterisation],
    metrics: list[IMetric],
    objective_function: IObjectiveFunction,
    experiment_data: ExperimentData,
) -> float | npt.NDArray[np.float64]:
    updated_spatial_parameterisations = unpack_spatial_parameterisations(
        spatial_parameterisations,
        degrees_of_freedom
    )

    updated_constitutive_parameter_maps = {
        param_name: sp.to_map(parameter_map_size)
        for (param_name, sp) in updated_spatial_parameterisations.items()
    }

    updated_stress = constitutive_law.calculate_stress(
        experiment_data.strain, updated_constitutive_parameter_maps
    )

    metric_results = []
    for metric in metrics:
        metric_results.append(
            metric.evaluate(
                updated_stress,
                constitutive_law,
                parameter_map_size,
                updated_spatial_parameterisations,
                experiment_data
            )
        )

    return objective_function.evaluate(metric_results)


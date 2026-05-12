import copy
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_law import ConstitutiveLaw
from pyvale.vfm.experiment_data import ExperimentData
from pyvale.vfm.metrics.metric import Metric
from pyvale.vfm.objective_functions.objective_function import ObjectiveFunction
from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    SpatialParameterisation,
)


class Optimiser(ABC):
    # Run a set of optimisation passes until a best guess is found
    # TODO: figure out what to return for different optimisers
    @abstractmethod
    def optimise(
        self,
        constitutive_law: ConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, SpatialParameterisation],
        metrics: list[Metric],
        objective_function: ObjectiveFunction,
        experiment_data: ExperimentData,
    ) -> float | npt.NDArray[np.float64]:
        pass


# TODO: in some optimisers will the input be a scalar not a vector?
# TODO: should we constrain return type? or have 2 functions for scalar/vector?
#   or have the scalar representation just be a 1 elem vector?
def evaluate_candidate(
    vector: npt.NDArray[np.float64],
    constitutive_law: ConstitutiveLaw,
    parameter_map_size: npt.NDArray[np.uint32],
    spatial_parameterisations: dict[str, SpatialParameterisation],
    metrics: list[Metric],
    objective_function: ObjectiveFunction,
    experiment_data: ExperimentData,
) -> float | npt.NDArray[np.float64]:
    updated_spatial_parameterisations: dict[str, SpatialParameterisation] = {}

    index = 0
    for param_name, sp in spatial_parameterisations.items():
        num_dofs = sp.num_degrees_of_freedom

        if num_dofs == 0:
            updated_spatial_parameterisations[param_name] = sp
            continue

        updated_sp = copy.deepcopy(sp)

        sp_dofs = vector[index:index + num_dofs]

        updated_sp.update_from_packed_degrees_of_freedom(sp_dofs)
        updated_spatial_parameterisations[param_name] = updated_sp

        index += num_dofs


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
                updated_spatial_parameterisations,
                experiment_data
            )
        )

    return objective_function.evaluate(metric_results)


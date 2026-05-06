import copy
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_law import ConstitutiveLaw
from pyvale.vfm.metrics.metric import Metric
from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    SpatialParameterisation,
)


class Optimiser(ABC):
    # Run a set of optimisation passes until a best guess is found
    @abstractmethod
    def optimise(
        self,
        constitutive_law: ConstitutiveLaw,
        strain: npt.NDArray[np.float64],
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, SpatialParameterisation],
        weighted_metrics: list[tuple[float, Metric]],
    ) -> None:
        pass


def evaluate_candidate(
    vector: npt.NDArray[np.float64],
    constitutive_law: ConstitutiveLaw,
    strain: npt.NDArray[np.float64],
    parameter_map_size: npt.NDArray[np.uint32],
    spatial_parameterisations: dict[str, SpatialParameterisation],
    weighted_metrics: list[tuple[float, Metric]],
) -> float:
    updated_parameterisations: dict[str, SpatialParameterisation] = {}

    index = 0
    for param_name, sp in spatial_parameterisations.items():
        num_dofs = sp.num_degrees_of_freedom

        if num_dofs == 0:
            updated_parameterisations[param_name] = sp
            continue

        updated_sp = copy.deepcopy(sp)

        sp_dofs = vector[index:index + num_dofs]

        updated_sp.update_from_packed_degrees_of_freedom(sp_dofs)
        updated_parameterisations[param_name] = updated_sp

        index += num_dofs

    updated_constitutive_parameter_maps = {
        param_name: sp.to_map(parameter_map_size)
        for (param_name, sp) in updated_parameterisations.items()
    }

    updated_stress = constitutive_law.calculate_stress(
        strain, updated_constitutive_parameter_maps
    )

    cost = 0
    for weight, metric in weighted_metrics:
        metric_cost = metric.evaluate(updated_stress)
        cost += metric_cost * weight

    return cost

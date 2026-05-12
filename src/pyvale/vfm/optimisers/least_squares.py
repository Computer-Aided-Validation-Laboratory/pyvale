import copy

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from pyvale.vfm.constitutive_laws.constitutive_law import ConstitutiveLaw
from pyvale.vfm.experiment_data import ExperimentData
from pyvale.vfm.metrics.metric import Metric
from pyvale.vfm.optimisers.optimiser import Optimiser
from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    SpatialParameterisation,
)


# TODO: do we need to have customisation for things like:
#   - ftol
#   - xtol
#   - gtol
#   - max_nfev
#   if we need these, should treat the below as a dataclass and
#   take these options as inputs in construction
# TODO: how should I calculate residuals?
#   For sbvfs, to calc residual we need:
#     - stress
#     - sbvfs
#     - force
#     - area
#     - thickness
class LeastSquares(Optimiser):
    def optimise(
        self,
        constitutive_law: ConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, SpatialParameterisation],
        weighted_metrics: list[tuple[float, Metric]],
        experiment_data: ExperimentData,
    ) -> None:
        dofs = []
        dofs_lower_bounds = []
        dofs_upper_bounds = []

        for sp in spatial_parameterisations.values():
            (sp_dofs, sp_dofs_lower_bounds, sp_dofs_upper_bounds) = (
                sp.pack_degrees_of_freedom()
            )

            dofs.append(sp_dofs)
            dofs_lower_bounds.append(sp_dofs_lower_bounds)
            dofs_upper_bounds.append(sp_dofs_upper_bounds)

        result = least_squares(
            # TODO: replace with a function which calculates residuals
            self.evaluate_candidate,
            dofs,
            bounds=(dofs_lower_bounds, dofs_upper_bounds),
            # TODO: should we change class name to align with least squares method?
            # I suspect we might want to do that for LM so maybe for trf/dogbox too?
            method="trf",
            args=(
                constitutive_law,
                parameter_map_size,
                spatial_parameterisations,
                weighted_metrics,
                experiment_data,
            ),
        )

        return result.x

    def evaluate_candidate(
        self,
        vector: npt.NDArray[np.float64],
        constitutive_law: ConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, SpatialParameterisation],
        weighted_metrics: list[tuple[float, Metric]],
        experiment_data: ExperimentData,
    ) -> npt.NDArray[np.float64]:
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
            experiment_data.strain, updated_constitutive_parameter_maps
        )

        cost = 0
        for weight, metric in weighted_metrics:
            # TODO: Metric evaluation returns an array of different kinds for differnt metrics
            #   (though always a numpy array I think, with different shapes)
            #   This metric result then needs to be passed into a function which
            #   changes it into the form needed for the optimiser
            #   (and you can do whatever data manipulation if you create your own one)
            metric_result = metric.evaluate(updated_stress, experiment_data)
            # cost += metric_cost * weight

        # return cost

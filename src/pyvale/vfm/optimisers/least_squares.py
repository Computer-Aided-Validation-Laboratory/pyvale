import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from pyvale.vfm.constitutive_laws.constitutive_law import ConstitutiveLaw
from pyvale.vfm.experiment_data import ExperimentData
from pyvale.vfm.metrics.metric import Metric
from pyvale.vfm.objective_functions.objective_function import ObjectiveFunction
from pyvale.vfm.optimisers.optimiser import (
    Optimiser,
    evaluate_candidate,
)
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
        metrics: list[Metric],
        objective_function: ObjectiveFunction,
        experiment_data: ExperimentData,
    ) -> float | npt.NDArray[np.float64]:
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

        dofs = np.concatenate(dofs)
        dofs_lower_bounds = np.concatenate(dofs_lower_bounds)
        dofs_upper_bounds = np.concatenate(dofs_upper_bounds)

        result = least_squares(
            evaluate_candidate,
            dofs,
            bounds=(dofs_lower_bounds, dofs_upper_bounds),
            # TODO: should we change class name to align with least squares method?
            # I suspect we might want to do that for LM so maybe for trf/dogbox too?
            method="trf",
            args=(
                constitutive_law,
                parameter_map_size,
                spatial_parameterisations,
                metrics,
                objective_function,
                experiment_data,
            ),
        )

        return result.x

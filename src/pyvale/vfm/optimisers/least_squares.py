import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from pyvale.vfm.constitutive_laws.constitutive_law import IConstitutiveLaw
from pyvale.vfm.experiment_data import ExperimentData
from pyvale.vfm.metrics.metric import IMetric
from pyvale.vfm.normalisation import normalise_degrees_of_freedom
from pyvale.vfm.objective_functions.objective_function import IObjectiveFunction
from pyvale.vfm.optimisers.optimiser import (
    IOptimiser,
    evaluate_candidate,
)
from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    ISpatialParameterisation,
    unpack_spatial_parameterisations,
)


# TODO: do we need to have customisation for things like:
#   - ftol
#   - xtol
#   - gtol
#   - max_nfev
#   if we need these, should treat the below as a dataclass and
#   take these options as inputs in construction
class LeastSquares(IOptimiser):
    def optimise(
        self,
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, ISpatialParameterisation],
        metrics: list[IMetric],
        objective_function: IObjectiveFunction,
        experiment_data: ExperimentData,
    ) -> dict[str, ISpatialParameterisation]:
        normalised_degrees_of_freedom = []

        for sp in spatial_parameterisations.values():
            if sp.num_degrees_of_freedom == 0:
                continue

            degrees_of_freedom = sp.collect_degrees_of_freedom()

            normalised_degrees_of_freedom.append(
                normalise_degrees_of_freedom(degrees_of_freedom)
            )

        dofs = np.concatenate(normalised_degrees_of_freedom)

        result = least_squares(
            evaluate_candidate,
            dofs,
            # TODO: add bounds
            # TODO: should we change class name to align with least squares method?
            # I suspect we might want to do that for LM so maybe for trf/dogbox too?
            method="lm",
            args=(
                constitutive_law,
                parameter_map_size,
                spatial_parameterisations,
                metrics,
                objective_function,
                experiment_data,
            )
        )

        optimised_spatial_parameterisations = unpack_spatial_parameterisations(
            spatial_parameterisations,
            result.x
        )

        return optimised_spatial_parameterisations

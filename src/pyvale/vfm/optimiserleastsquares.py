import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.metric import IMetric
from pyvale.vfm.normalisation import normalise_degrees_of_freedom
from pyvale.vfm.objectivefunc import (
    IObjectiveFunction,
    IVectorObjectiveFunction,
)
from pyvale.vfm.optimiser import (
    IOptimiser,
    evaluate_candidate,
)
from pyvale.vfm.spatialparam import (
    PhaseSpatialState,
    ISpatialParameterisation,
)


# TODO: do we need to have customisation for things like:
#   - ftol
#   - xtol
#   - gtol
#   - max_nfev
#   if we need these, should treat the below as a dataclass and
#   take these options as inputs in construction
class OptimiserLeastSquares(IOptimiser):
    """
    Least-squares optimiser driving the parameter search.

    Wraps ``scipy.optimize.least_squares`` (Levenberg-Marquardt) to
    minimise a vector objective over the active, normalised degrees of
    freedom. Requires an ``IVectorObjectiveFunction``
    """

    def get_required_objective_function_type(self) -> type:
        return IVectorObjectiveFunction

    def optimise(
        self,
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        metrics: list[IMetric],
        objective_function: IObjectiveFunction,
        experiment_data: ExperimentData,
    ) -> dict[str, list[ISpatialParameterisation]]:
        phase_spatial_state = PhaseSpatialState(spatial_parameterisations)
        dofs = phase_spatial_state.collect_normalised_degrees_of_freedom()
        if dofs.size == 0:
            return spatial_parameterisations

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
                phase_spatial_state,
                metrics,
                objective_function,
                experiment_data,
            )
        )

        optimised_phase_spatial_state = phase_spatial_state.copy()
        optimised_phase_spatial_state.update_from_normalised_degrees_of_freedom(
            result.x
        )

        return optimised_phase_spatial_state.spatial_parameterisations

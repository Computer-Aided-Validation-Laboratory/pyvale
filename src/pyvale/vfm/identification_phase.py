from dataclasses import dataclass

from pyvale.vfm.metrics.metric import Metric
from pyvale.vfm.optimisers.optimiser import Optimiser
from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import SpatialParameterisation


@dataclass(slots=True)
class IdentificationPhase:
    # Param name str must be same as in the input params
    spatial_parameteristaions: dict[str, SpatialParameterisation]
    weighted_metrics: list[tuple[float, Metric]]
    optimiser: Optimiser

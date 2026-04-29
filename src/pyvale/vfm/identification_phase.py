from dataclasses import dataclass

from pyvale.vfm.metrics.metric import Metric
from pyvale.vfm.optimisers.optimiser import Optimiser
from pyvale.vfm.parameterisations.parameterisation import Parameterisation


@dataclass(slots=True)
class IdentificationPhase:
    # Param name str must be same as in the input params
    parameteristaions: dict[str, Parameterisation]
    # Metric with a weighting
    metrics: list[tuple[Metric, float]]
    optimiser: Optimiser

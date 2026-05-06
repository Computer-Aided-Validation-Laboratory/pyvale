import enum
from dataclasses import dataclass

from pyvale.vfm.constitutive_laws.constitutive_law import ConstitutiveLaw
from pyvale.vfm.constitutive_parameter import ConstitutiveParameter
from pyvale.vfm.metrics.metric import Metric
from pyvale.vfm.optimisers.optimiser import Optimiser
from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    SpatialParameterisation,
)


class EIdentificationType(enum.Enum):
    Linear = enum.auto()
    Nonlinear = enum.auto()


@dataclass(slots=True)
class IdentificationPhase:
    # Param name str must be same as in the input params
    spatial_parameteristaions: dict[str, SpatialParameterisation]
    weighted_metrics: list[tuple[float, Metric]]
    optimiser: Optimiser


@dataclass(slots=True)
class Identification():
    constitutive_law: ConstitutiveLaw
    parameters: dict[str, ConstitutiveParameter]
    phases: list[IdentificationPhase]


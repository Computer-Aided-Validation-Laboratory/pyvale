from dataclasses import dataclass

from pyvale.vfm.constitutive_laws.constitutive_law import IConstitutiveLaw
from pyvale.vfm.constitutive_laws.constitutive_parameter import (
    ConstitutiveParameter,
)
from pyvale.vfm.metrics.metric import IMetric
from pyvale.vfm.objective_functions.objective_function import IObjectiveFunction
from pyvale.vfm.optimisers.optimiser import IOptimiser
from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    ISpatialParameterisation,
)


@dataclass(slots=True)
class IdentificationPhase:
    # Param name str must be same as in the input params
    spatial_parameterisations: dict[str, ISpatialParameterisation]
    metrics: list[IMetric]
    objective_function: IObjectiveFunction
    optimiser: IOptimiser


@dataclass(slots=True)
class Identification():
    constitutive_law: IConstitutiveLaw
    parameters: dict[str, ConstitutiveParameter]
    phases: list[IdentificationPhase]


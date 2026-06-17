from dataclasses import dataclass

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.metric import IMetric
from pyvale.vfm.objectivefunc import IObjectiveFunction
from pyvale.vfm.optimiser import IOptimiser
from pyvale.vfm.spatialparam import ISpatialParameterisation


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


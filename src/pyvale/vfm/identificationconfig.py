from dataclasses import dataclass

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.metric import IMetric
from pyvale.vfm.objectivefunc import IObjectiveFunction
from pyvale.vfm.optimiser import IOptimiser
from pyvale.vfm.spatialparam import ISpatialParameterisation


@dataclass(slots=True)
class IdentificationPhase:
    """
    A single identification phase.

    Each phase defines its own spatial parameterisations, metrics, objective
    function, and optimiser
    """

    spatial_parameterisations: dict[str, list[ISpatialParameterisation]]
    """
    Mapping from constitutive parameter name to its list of spatial
    parameterisations. The parameter map is the sum of each parameterisation's
    map, evaluated in list (definition) order
    """

    metrics: list[IMetric]
    """Virtual-work metrics used to evaluate candidate stress fields"""

    objective_function: IObjectiveFunction
    """Scalar or vector objective that aggregates metric results"""

    optimiser: IOptimiser
    """Optimisation algorithm that drives the parameter search"""


@dataclass(slots=True)
class IdentificationConfig:
    """
    Complete configuration for a VFM identification run.

    Combines the constitutive law to identify, initial guessed for its
    parameters, and one or more identification phases that
    are executed sequentially, where the output of one phase
    becomes the initial guess for the next
    """

    constitutive_law: IConstitutiveLaw
    """Constitutive model whose parameters are being identified"""

    parameters: dict[str, ConstitutiveParameter]
    """Initial guess, lower bound, and upper bound for each parameter"""

    phases: list[IdentificationPhase]
    """Identification phases that are executed in order"""

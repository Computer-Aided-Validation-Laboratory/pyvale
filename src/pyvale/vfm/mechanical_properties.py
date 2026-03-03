import enum
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class IdentificationType(enum.Enum):
    Known = enum.auto()
    Unknown = enum.auto()


class ParameterName(enum.Enum):
    ElasticModulus = enum.auto()
    PoissonsRatio = enum.auto()
    HardeningModulus = enum.auto()
    YieldStrength = enum.auto()


# Bound of the physical mechanical property (e.g. yield strength)
@dataclass(slots=True)
class ParameterBounds:
    lower_bound: int | float
    upper_bound: int | float


ParameterValue = (
    int |
    float |
    npt.NDArray[np.int64] |
    npt.NDArray[np.float64]
)


@dataclass(slots=True)
class HomogeneousParameter:
    identification_type: IdentificationType
    parameter_type: ParameterName
    bounds: ParameterBounds
    value: ParameterValue


@dataclass(slots=True)
class MeshParameter:
    identification_type: IdentificationType
    parameter_type: ParameterName
    value: ParameterValue


@dataclass(slots=True)
class BasisFunctionParameter:
    identification_type: IdentificationType
    parameter_type: ParameterName
    value: ParameterValue


Parameter = (
    HomogeneousParameter |
    MeshParameter |
    BasisFunctionParameter
)


class ConstituitiveLaw(enum.Enum):
    LinearHardening = enum.auto()


@dataclass(slots=True)
class MechanicalProperties:
    constituitive_law: ConstituitiveLaw
    parameters: list[Parameter]


def check_validity(mechanical_properties: MechanicalProperties) -> bool:
    match mechanical_properties.constituitive_law:
        case ConstituitiveLaw.LinearHardening:
            required_parameters = {
                ParameterName.ElasticModulus,
                ParameterName.PoissonsRatio,
                ParameterName.HardeningModulus,
                ParameterName.YieldStrength,
            }

            parameters = {param.parameter_type for param in mechanical_properties.parameters}

            return required_parameters.issubset(parameters)


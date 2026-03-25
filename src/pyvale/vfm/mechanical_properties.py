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
    StrengthCoefficient = enum.auto()
    HardeningExponent = enum.auto()
    StrainOffset = enum.auto()
    SaturationStress = enum.auto()
    RateParameter = enum.auto()


# Bound of the physical mechanical property (e.g. yield strength)
@dataclass(slots=True)
class ParameterBounds:
    lower_bound: int | float
    upper_bound: int | float


@dataclass(slots=True)
class ScalarValue:
    value: int | float


# TODO: could consider this being an int32 (maybe for perf but idk)?
@dataclass(slots=True)
class MapValue:
    value: npt.NDArray[np.int64] | npt.NDArray[np.float64]


ParameterValue = ScalarValue | MapValue

@dataclass(slots=True)
class HomogeneousParameter:
    identification_type: IdentificationType
    bounds: ParameterBounds
    value: ParameterValue


@dataclass(slots=True)
class MeshParameter:
    identification_type: IdentificationType
    value: ParameterValue


@dataclass(slots=True)
class BasisFunctionParameter:
    identification_type: IdentificationType
    value: ParameterValue


Parameter = (
    HomogeneousParameter |
    MeshParameter |
    BasisFunctionParameter
)


class ConstituitiveLaw(enum.Enum):
    LinearHardening = enum.auto()
    SwiftHardening = enum.auto()
    VoceHardening = enum.auto()
    LudwikHardening = enum.auto()


@dataclass(slots=True)
class MechanicalProperties:
    constituitive_law: ConstituitiveLaw
    parameters: dict[ParameterName, Parameter]


def check_validity(mechanical_properties: MechanicalProperties) -> bool:
    match mechanical_properties.constituitive_law:
        case ConstituitiveLaw.LinearHardening:
            required_parameters = {
                ParameterName.ElasticModulus,
                ParameterName.PoissonsRatio,
                ParameterName.HardeningModulus,
                ParameterName.YieldStrength,
            }

            return required_parameters.issubset(mechanical_properties.parameters.keys())
        case ConstituitiveLaw.SwiftHardening:
            required_parameters = {
                ParameterName.ElasticModulus,
                ParameterName.PoissonsRatio,
                ParameterName.StrengthCoefficient,
                ParameterName.StrainOffset,
                ParameterName.HardeningExponent,
            }

            return required_parameters.issubset(mechanical_properties.parameters.keys())
        case ConstituitiveLaw.VoceHardening:
            required_parameters = {
                ParameterName.ElasticModulus,
                ParameterName.PoissonsRatio,
                ParameterName.YieldStrength,
                ParameterName.HardeningModulus,
                ParameterName.SaturationStress,
                ParameterName.RateParameter,
            }

            return required_parameters.issubset(mechanical_properties.parameters.keys())
        case ConstituitiveLaw.LudwikHardening:
            required_parameters = {
                ParameterName.ElasticModulus,
                ParameterName.PoissonsRatio,
                ParameterName.YieldStrength,
                ParameterName.StrengthCoefficient,
                ParameterName.HardeningExponent,
            }

            return required_parameters.issubset(mechanical_properties.parameters.keys())


# TODO: support the other kinds of Parameter
def parameter_to_scalar(param: Parameter) -> int | float:
    match param:
        case HomogeneousParameter(_, _, ScalarValue(value)):
            return value
        case HomogeneousParameter(_, _, MapValue(value)):
            return value[0, 0]

    raise TypeError("Unsupported parameter")


# TODO: support the other kinds of Parameter
def parameter_to_map(
    param: Parameter,
    size_x: int,
    size_y: int
) -> npt.NDArray[np.int64] | npt.NDArray[np.float64]:
    match param:
        case HomogeneousParameter(_, _, ScalarValue(value)):
            return np.full((size_y, size_x), value)
        case HomogeneousParameter(_, _, MapValue(value)):
            return value

    raise TypeError("Unsupported parameter")

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


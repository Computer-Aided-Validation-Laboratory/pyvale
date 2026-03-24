import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class EParameterLabel(enum.Enum):
    ElasticModulus = enum.auto()
    PoissonsRatio = enum.auto()
    HardeningModulus = enum.auto()
    YieldStrength = enum.auto()
    StrengthCoefficient = enum.auto()
    HardeningExponent = enum.auto()
    StrainOffset = enum.auto()
    SaturationStress = enum.auto()
    RateParameter = enum.auto()


# TODO: update with mesh dofs
class EDOFLabel(enum.Enum):
    Value = enum.auto()
    X = enum.auto()
    Y = enum.auto()
    Height = enum.auto()
    Variance = enum.auto()
    Variance1 = enum.auto()
    Variance2 = enum.auto()
    Angle = enum.auto()


@dataclass(slots=True)
class BoundedValue:
    value: float
    lower_bound: float
    upper_bound: float


class IParameterisation(ABC):
    @abstractmethod
    def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def get_degrees_of_freedom(self) -> dict[EDOFLabel, BoundedValue]:
        pass

    @abstractmethod
    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: dict[EDOFLabel, BoundedValue]
    ) -> None:
        pass


@dataclass
class Homogeneous(IParameterisation):
    value: BoundedValue

    def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
        return np.full((size_y, size_x), self.value.value)

    def get_degrees_of_freedom(self) -> dict[EDOFLabel, BoundedValue]:
        return { EDOFLabel.Value: self.value }

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: dict[EDOFLabel, BoundedValue]
    ) -> None:
        self.value = degrees_of_freedom[EDOFLabel.Value]


@dataclass(slots=True)
class UnivariateBasisFunction(IParameterisation):
    x: BoundedValue
    y: BoundedValue
    height: BoundedValue
    variance: BoundedValue

    # TODO: implement
    # def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
    #     return ...

    def get_degrees_of_freedom(self) -> dict[EDOFLabel, BoundedValue]:
        return {
            EDOFLabel.X: self.x,
            EDOFLabel.Y: self.y,
            EDOFLabel.Height: self.height,
            EDOFLabel.Variance: self.variance
        }

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: dict[EDOFLabel, BoundedValue]
    ) -> None:
        self.x = degrees_of_freedom[EDOFLabel.X]
        self.y = degrees_of_freedom[EDOFLabel.Y]
        self.height = degrees_of_freedom[EDOFLabel.Height]
        self.variance = degrees_of_freedom[EDOFLabel.Variance]


@dataclass(slots=True)
class BivariateBasisFunction(IParameterisation):
    x: BoundedValue
    y: BoundedValue
    height: BoundedValue
    variance_1: BoundedValue
    variance_2: BoundedValue
    angle: BoundedValue

    # TODO: implement
    # def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
    #     return ...

    def get_degrees_of_freedom(self) -> dict[EDOFLabel, BoundedValue]:
        return {
            EDOFLabel.X: self.x,
            EDOFLabel.Y: self.y,
            EDOFLabel.Height: self.height,
            EDOFLabel.Variance1: self.variance_1,
            EDOFLabel.Variance2: self.variance_2,
            EDOFLabel.Angle: self.angle
        }

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: dict[EDOFLabel, BoundedValue]
    ) -> None:
        self.x = degrees_of_freedom[EDOFLabel.X]
        self.y = degrees_of_freedom[EDOFLabel.Y]
        self.height = degrees_of_freedom[EDOFLabel.Height]
        self.variance_1 = degrees_of_freedom[EDOFLabel.Variance1]
        self.variance_2 = degrees_of_freedom[EDOFLabel.Variance2]
        self.angle = degrees_of_freedom[EDOFLabel.Angle]


@dataclass(slots=True)
class Mesh(IParameterisation):
    # TODO: should this just be size and take a numpy array?
    size_x: int
    size_y: int
    # TODO: add other fields like element ordering/shape functions etc

    # TODO: implement
    # def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
    #     return ...

    # TODO: implement
    # def get_degrees_of_freedom(self) -> dict[EDOFLabel, BoundedValue]:
    #     return ...

    # TODO: implement
    # def update_from_degrees_of_freedom(self, degrees_of_freedom: dict[EDOFLabel, BoundedValue]) -> None:
    #     return ...

Parameterisation = (
    Homogeneous |
    UnivariateBasisFunction |
    BivariateBasisFunction |
    Mesh
)


class IParameter(ABC):
    @abstractmethod
    def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
        pass


@dataclass(slots=True)
class UnknownParameter(IParameter):
    lower_bound: float
    upper_bound: float
    parameterisation: list[Parameterisation]

    def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
        maps = [p.to_map(size_x, size_y) for p in self.parameterisation]
        return np.sum(maps, axis=0)

    def get_degrees_of_freedom(self) -> list[dict[EDOFLabel, BoundedValue]]:
        return [
            p.get_degrees_of_freedom() for p in self.parameterisation
        ]

    def update_degrees_of_freedom(
        self,
        degrees_of_freedom: list[dict[EDOFLabel, BoundedValue]]
    ) -> None:
        for i, p in enumerate(self.parameterisation):
            p.update_from_degrees_of_freedom(degrees_of_freedom[i])


@dataclass(slots=True)
class KnownParameter(IParameter):
    value: npt.NDArray[np.float64]

    def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
        return self.value


Parameter = KnownParameter | UnknownParameter


class EConstituitiveLaw(enum.Enum):
    LinearHardening = enum.auto()
    SwiftHardening = enum.auto()
    VoceHardening = enum.auto()
    LudwikHardening = enum.auto()


@dataclass(slots=True)
class MechanicalProperties:
    constituitive_law: EConstituitiveLaw
    parameters: dict[EParameterLabel, Parameter]


def check_validity(mechanical_properties: MechanicalProperties) -> bool:
    is_valid = True
    match mechanical_properties.constituitive_law:
        case EConstituitiveLaw.LinearHardening:
            required_parameters = {
                EParameterLabel.ElasticModulus,
                EParameterLabel.PoissonsRatio,
                EParameterLabel.HardeningModulus,
                EParameterLabel.YieldStrength,
            }

            if not required_parameters.issubset(
                mechanical_properties.parameters.keys()
            ):
                is_valid = False

        case EConstituitiveLaw.SwiftHardening:
            required_parameters = {
                EParameterName.ElasticModulus,
                EParameterName.PoissonsRatio,
                EParameterName.StrengthCoefficient,
                EParameterName.StrainOffset,
                EParameterName.HardeningExponent,
            }

            if not required_parameters.issubset(
                mechanical_properties.parameters.keys()
            ):
                is_valid = False

        case EConstituitiveLaw.VoceHardening:
            required_parameters = {
                EParameterName.ElasticModulus,
                EParameterName.PoissonsRatio,
                EParameterName.YieldStrength,
                EParameterName.HardeningModulus,
                EParameterName.SaturationStress,
                EParameterName.RateParameter,
            }

            if not required_parameters.issubset(
                mechanical_properties.parameters.keys()
            ):
                is_valid = False

        case EConstituitiveLaw.LudwikHardening:
            required_parameters = {
                EParameterName.ElasticModulus,
                EParameterName.PoissonsRatio,
                EParameterName.YieldStrength,
                EParameterName.StrengthCoefficient,
                EParameterName.HardeningExponent,
            }

            if not required_parameters.issubset(
                mechanical_properties.parameters.keys()
            ):
                is_valid = False

    return is_valid


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

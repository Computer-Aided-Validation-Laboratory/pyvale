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
class DegreeOfFreedom:
    value: float
    lower_bound: float
    upper_bound: float


class IParameterisation(ABC):
    @abstractmethod
    def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def get_degrees_of_freedom(self) -> dict[EDOFLabel, DegreeOfFreedom]:
        pass

    @abstractmethod
    def update_degree_of_freedom_value(
        self,
        degree_of_freedom: EDOFLabel,
        value: float
    ) -> None:
        pass

    # TODO: do we actually need this?
    @abstractmethod
    def update_degrees_of_freedom(
        self,
        degrees_of_freedom: dict[EDOFLabel, DegreeOfFreedom]
    ) -> None:
        pass


@dataclass
class Homogeneous(IParameterisation):
    value: DegreeOfFreedom

    def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
        return np.full((size_y, size_x), self.value.value)

    def get_degrees_of_freedom(self) -> dict[EDOFLabel, DegreeOfFreedom]:
        return { EDOFLabel.Value: self.value }

    def update_degree_of_freedom_value(
        self,
        degree_of_freedom: EDOFLabel,
        value: float
    ) -> None:
        match degree_of_freedom:
            case EDOFLabel.Value:
                self.value.value = value
            case _:
                raise ValueError(
                    "Invalid Degree of Freedom for Homogeneous "
                    f"parameterisation: {degree_of_freedom.name}"
                )

    def update_degrees_of_freedom(
        self,
        degrees_of_freedom: dict[EDOFLabel, DegreeOfFreedom]
    ) -> None:
        self.value = degrees_of_freedom[EDOFLabel.Value]


@dataclass(slots=True)
class UnivariateBasisFunction(IParameterisation):
    x: DegreeOfFreedom
    y: DegreeOfFreedom
    height: DegreeOfFreedom
    variance: DegreeOfFreedom

    # TODO: implement
    # def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
    #     return ...

    def get_degrees_of_freedom(self) -> dict[EDOFLabel, DegreeOfFreedom]:
        return {
            EDOFLabel.X: self.x,
            EDOFLabel.Y: self.y,
            EDOFLabel.Height: self.height,
            EDOFLabel.Variance: self.variance
        }

    def update_degree_of_freedom_value(
        self,
        degree_of_freedom: EDOFLabel,
        value: float
    ) -> None:
        match degree_of_freedom:
            case EDOFLabel.X:
                self.x.value = value
            case EDOFLabel.Y:
                self.y.value = value
            case EDOFLabel.Height:
                self.height.value = value
            case EDOFLabel.Variance:
                self.variance.value = value
            case _:
                raise ValueError(
                    "Invalid Degree of Freedom for Univariate Basis Function "
                    f"parameterisation: {degree_of_freedom.name}"
                )

    def update_degrees_of_freedom(
        self,
        degrees_of_freedom: dict[EDOFLabel, DegreeOfFreedom]
    ) -> None:
        self.x = degrees_of_freedom[EDOFLabel.X]
        self.y = degrees_of_freedom[EDOFLabel.Y]
        self.height = degrees_of_freedom[EDOFLabel.Height]
        self.variance = degrees_of_freedom[EDOFLabel.Variance]


@dataclass(slots=True)
class BivariateBasisFunction(IParameterisation):
    x: DegreeOfFreedom
    y: DegreeOfFreedom
    height: DegreeOfFreedom
    variance_1: DegreeOfFreedom
    variance_2: DegreeOfFreedom
    angle: DegreeOfFreedom

    # TODO: implement
    # def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
    #     return ...

    def get_degrees_of_freedom(self) -> dict[EDOFLabel, DegreeOfFreedom]:
        return {
            EDOFLabel.X: self.x,
            EDOFLabel.Y: self.y,
            EDOFLabel.Height: self.height,
            EDOFLabel.Variance1: self.variance_1,
            EDOFLabel.Variance2: self.variance_2,
            EDOFLabel.Angle: self.angle
        }

    def update_degree_of_freedom_value(
        self,
        degree_of_freedom: EDOFLabel,
        value: float
    ) -> None:
        match degree_of_freedom:
            case EDOFLabel.X:
                self.x.value = value
            case EDOFLabel.Y:
                self.y.value = value
            case EDOFLabel.Height:
                self.height.value = value
            case EDOFLabel.Variance1:
                self.variance_1.value = value
            case EDOFLabel.Variance2:
                self.variance_2.value = value
            case EDOFLabel.Angle:
                self.angle.value = value
            case _:
                raise ValueError(
                    "Invalid Degree of Freedom for Bivariate Basis Function "
                    f"parameterisation: {degree_of_freedom.name}"
                )

    def update_degrees_of_freedom(
        self,
        degrees_of_freedom: dict[EDOFLabel, DegreeOfFreedom]
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
    # def get_degrees_of_freedom(self) -> dict[EDOFLabel, DegreeOfFreedom]:
    #     return ...

    # TODO: implement
    # def update_degree_of_freedom_value(
    #     self,
    #     degree_of_freedom: EDOFLabel,
    #     value: float
    # ) -> None:
    #     return ...

    # TODO: implement
    # def update_degrees_of_freedom(self, degrees_of_freedom: dict[EDOFLabel, DegreeOfFreedom]) -> None:
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

    def get_degrees_of_freedom(self) -> list[dict[EDOFLabel, DegreeOfFreedom]]:
        return [
            p.get_degrees_of_freedom() for p in self.parameterisation
        ]

    # TODO: should this be removed fully?
    # def update_degrees_of_freedom(
    #     self,
    #     degrees_of_freedom: list[dict[EDOFLabel, DegreeOfFreedom]]
    # ) -> None:
    #     for i, p in enumerate(self.parameterisation):
    #         p.update_degrees_of_freedom(degrees_of_freedom[i])


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

    def get_unknown_parameters(self) -> dict[EParameterLabel, UnknownParameter]:
        return {
            label: param
            for label, param in self.parameters.items()
            if isinstance(param, UnknownParameter)
        }

    # TODO: should this be removed fully?
    # def update_degrees_of_freedom(
    #     self,
    #     degrees_of_freedom: dict[
    #         EParameterLabel,
    #         list[dict[EDOFLabel, DegreeOfFreedom]]
    #     ]
    # ) -> None:
    #     for label, dofs in degrees_of_freedom.items():
    #         param = self.parameters[label]

    #         match param:
    #             case UnknownParameter():
    #                 param.update_degrees_of_freedom(dofs)
    #             case KnownParameter():
    #                 raise(TypeError(
    #                     f"Expected UnknownParameter, got {type(param).__name__}"
    #                 ))


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

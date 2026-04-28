from __future__ import annotations
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


class EParameterName(enum.Enum):
    """Supported constitutive parameter labels. The actual parameters required depend on the selected constitutive law."""

    ElasticModulus = enum.auto()
    PoissonsRatio = enum.auto()
    HardeningModulus = enum.auto()
    YieldStrength = enum.auto()
    StrengthCoefficient = enum.auto()
    HardeningExponent = enum.auto()
    StrainOffset = enum.auto()
    SaturationStress = enum.auto()
    RateParameter = enum.auto()


class EConstituitiveLaw(enum.Enum):
    """Supported constitutive laws used by the return-mapping code."""

    Elastic = enum.auto()
    LinearHardening = enum.auto()
    SwiftHardening = enum.auto()
    VoceHardening = enum.auto()
    LudwikHardening = enum.auto()


# Dictionary mapping each constitutive law to required parameters
REQUIRED_PARAMETERS = {
    EConstituitiveLaw.LinearHardening: (
        EParameterName.ElasticModulus,
        EParameterName.PoissonsRatio,
        EParameterName.YieldStrength,
        EParameterName.HardeningModulus,
    ),
    EConstituitiveLaw.Elastic: (
        EParameterName.ElasticModulus,
        EParameterName.PoissonsRatio,
    ),
    EConstituitiveLaw.SwiftHardening: (
        EParameterName.ElasticModulus,
        EParameterName.PoissonsRatio,
        EParameterName.StrengthCoefficient,
        EParameterName.StrainOffset,
        EParameterName.HardeningExponent,
    ),
    EConstituitiveLaw.VoceHardening: (
        EParameterName.ElasticModulus,
        EParameterName.PoissonsRatio,
        EParameterName.YieldStrength,
        EParameterName.HardeningModulus,
        EParameterName.SaturationStress,
        EParameterName.RateParameter,
    ),
    EConstituitiveLaw.LudwikHardening: (
        EParameterName.ElasticModulus,
        EParameterName.PoissonsRatio,
        EParameterName.YieldStrength,
        EParameterName.StrengthCoefficient,
        EParameterName.HardeningExponent,
    ),
}


@dataclass(slots=True)
class ParameterBounds:
    """Simple lower/upper bounds container."""

    lower: float
    upper: float


class ConstitutiveParameter(ABC):
    """ABC for constitutive parameters."""

    @abstractmethod
    def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
        """Return a 2d array of constitutive parameter values for all datapoints."""


@dataclass(slots=True)
class KnownParameter(ConstitutiveParameter):
    """Constitutive parameter fully defined by user-provided scalar or a 2D array."""

    value: npt.NDArray[np.float64] | float

    def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
        value_array = np.asarray(self.value, dtype=np.float64)

        if value_array.ndim == 0:
            return np.full((size_y, size_x), float(value_array), dtype=np.float64)

        if value_array.shape != (size_y, size_x):
            raise ValueError(
                "KnownParameter map shape does not match the requested grid. "
                f"Expected {(size_y, size_x)}, got {value_array.shape}."
            )

        return value_array


@dataclass(slots=True)
class HomogeneousParameter(ConstitutiveParameter):
    """Constitutive parameter for which all datapoints have the same value."""

    bounds: ParameterBounds  
    value: float      

    def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
        return np.full((size_y, size_x), self.value, dtype=np.float64)


# Define a type alias for parameter types
Parameter = KnownParameter | HomogeneousParameter


@dataclass(slots=True)
class MechanicalProperties:
    """Constitutive law plus resolved parameter fields."""

    constituitive_law: EConstituitiveLaw
    parameters: dict[EParameterName, Parameter]

    def validate(self) -> None:
        required = set(REQUIRED_PARAMETERS[self.constituitive_law])
        missing = sorted(
            required.difference(self.parameters.keys()),
            key=lambda name: name.name,
        )

        if missing:
            missing_names = ", ".join(name.name for name in missing)
            raise ValueError(
                "MechanicalProperties is missing parameters for "
                f"{self.constituitive_law.name}: {missing_names}"
            )
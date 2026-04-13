from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class ParameterName(enum.Enum):
    """Supported constitutive-parameter labels."""

    ElasticModulus = enum.auto()
    PoissonsRatio = enum.auto()
    HardeningModulus = enum.auto()
    YieldStrength = enum.auto()
    StrengthCoefficient = enum.auto()
    HardeningExponent = enum.auto()
    StrainOffset = enum.auto()
    SaturationStress = enum.auto()
    RateParameter = enum.auto()


EParameterLabel = ParameterName
EParameterName = ParameterName


class ConstituitiveLaw(enum.Enum):
    """Supported constitutive laws used by the return-mapping code."""

    LinearHardening = enum.auto()
    Elastic = enum.auto()
    SwiftHardening = enum.auto()
    VoceHardening = enum.auto()
    LudwikHardening = enum.auto()


ConstitutiveLaw = ConstituitiveLaw
EConstituitiveLaw = ConstituitiveLaw


class IdentificationType(enum.Enum):
    """Compatibility enum kept for the older homogeneous-parameter tests.

    The new toolkit parameterisation workflow does not use this enum to
    decide whether a parameter is fixed or optimised. That now lives in the
    spatial-parameterisation layer.
    """

    Known = enum.auto()
    Unknown = enum.auto()


EIdentificationType = IdentificationType


@dataclass(slots=True)
class ParameterBounds:
    """Simple lower/upper bounds container."""

    lower: float
    upper: float

    @property
    def lower_bound(self) -> float:
        return self.lower

    @property
    def upper_bound(self) -> float:
        return self.upper


@dataclass(slots=True)
class ScalarValue:
    """Small wrapper kept for compatibility with the existing tests."""

    value: float


class ConstitutiveParameter(ABC):
    """Resolved constitutive parameter available to the material model.

    By the time the constitutive law sees a parameter it should already be a
    concrete scalar field over the specimen. How that field was created
    (homogeneous, mesh, basis functions, linked phase, and so on) is handled
    elsewhere.
    """

    @abstractmethod
    def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
        """Return the parameter as a dense 2D map on the test-data grid."""


@dataclass(slots=True)
class KnownParameter(ConstitutiveParameter):
    """Resolved parameter defined directly by a scalar or a 2D map."""

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
    """Resolved parameter represented by one scalar everywhere.

    This class is intentionally narrow: it exists mainly because the
    radial-return tests already use it, and because a uniform parameter is a
    useful resolved material representation. It is not the place where the
    identification logic decides whether a parameter is fixed or unknown.
    """

    identification_type: IdentificationType
    bounds: ParameterBounds
    value: ScalarValue

    def to_map(self, size_x: int, size_y: int) -> npt.NDArray[np.float64]:
        return np.full((size_y, size_x), self.value.value, dtype=np.float64)

    @property
    def lower_bound(self) -> float:
        return self.bounds.lower

    @property
    def upper_bound(self) -> float:
        return self.bounds.upper


Parameter = KnownParameter | HomogeneousParameter


REQUIRED_PARAMETERS = {
    ConstituitiveLaw.LinearHardening: (
        ParameterName.ElasticModulus,
        ParameterName.PoissonsRatio,
        ParameterName.YieldStrength,
        ParameterName.HardeningModulus,
    ),
    ConstituitiveLaw.Elastic: (
        ParameterName.ElasticModulus,
        ParameterName.PoissonsRatio,
    ),
    ConstituitiveLaw.SwiftHardening: (
        ParameterName.ElasticModulus,
        ParameterName.PoissonsRatio,
        ParameterName.StrengthCoefficient,
        ParameterName.StrainOffset,
        ParameterName.HardeningExponent,
    ),
    ConstituitiveLaw.VoceHardening: (
        ParameterName.ElasticModulus,
        ParameterName.PoissonsRatio,
        ParameterName.YieldStrength,
        ParameterName.HardeningModulus,
        ParameterName.SaturationStress,
        ParameterName.RateParameter,
    ),
    ConstituitiveLaw.LudwikHardening: (
        ParameterName.ElasticModulus,
        ParameterName.PoissonsRatio,
        ParameterName.YieldStrength,
        ParameterName.StrengthCoefficient,
        ParameterName.HardeningExponent,
    ),
}


def coerce_parameter_name(value: ParameterName | str) -> ParameterName:
    """Accept either an enum member or its name."""

    if isinstance(value, ParameterName):
        return value
    return ParameterName[value]


def coerce_constituitive_law(
    value: ConstituitiveLaw | str,
) -> ConstituitiveLaw:
    """Accept either a constitutive-law enum member or its name."""

    if isinstance(value, ConstituitiveLaw):
        return value
    return ConstituitiveLaw[value]


def required_parameters_for_law(
    constituitive_law: ConstituitiveLaw | str,
) -> tuple[ParameterName, ...]:
    """Return the required parameters for the selected law."""

    law = coerce_constituitive_law(constituitive_law)
    return REQUIRED_PARAMETERS[law]


@dataclass(slots=True)
class MechanicalProperties:
    """Constitutive law plus resolved parameter fields."""

    constituitive_law: ConstituitiveLaw
    parameters: dict[ParameterName, Parameter]

    def __post_init__(self) -> None:
        self.constituitive_law = coerce_constituitive_law(self.constituitive_law)
        self.parameters = {
            coerce_parameter_name(label): parameter
            for label, parameter in self.parameters.items()
        }

    @property
    def constitutive_law(self) -> ConstituitiveLaw:
        return self.constituitive_law

    def validate(self) -> None:
        required = set(required_parameters_for_law(self.constituitive_law))
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

    def get_unknown_parameters(self) -> dict[ParameterName, Parameter]:
        """Legacy compatibility hook for older callers.

        The new toolkit stores unknowns in the spatial parameterisation
        state, so this always returns an empty mapping.
        """

        return {}


def check_validity(mechanical_properties: MechanicalProperties) -> bool:
    """Return True when the material definition is valid for its law."""

    try:
        mechanical_properties.validate()
    except ValueError:
        return False
    return True

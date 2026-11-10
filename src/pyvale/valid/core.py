"""Named physical and uncertainty axes for validation data."""

from dataclasses import dataclass
from enum import Enum, StrEnum, auto

import numpy as np


class EValAxis(Enum):
    POINT = auto()
    COMPONENT = auto()
    TIME = auto()
    EPISTEMIC_TRANS = auto()
    EPISTEMIC_NUIS = auto()
    ALEATORY = auto()


AXIS_ORDER = tuple(EValAxis)
PHYSICAL_AXES = AXIS_ORDER[:3]
SAMPLE_AXES = AXIS_ORDER[3:]


class EValGeometry(Enum):
    POINT = auto()
    LINE = auto()
    SURFACE = auto()
    VOLUME = auto()
    GLOBAL = auto()


class EValOperator(Enum):
    SAMPLE = auto()
    MEAN = auto()
    INTEGRAL = auto()


class EValSource(StrEnum):
    SIM = "sim"
    EXP = "exp"


@dataclass(slots=True)
class ValGeometry:
    kind: EValGeometry
    coords: np.ndarray | None
    connect: dict[str, np.ndarray] | None = None
    coordinate_units: str = ""
    frame: str = ""

    def __post_init__(self) -> None:
        if self.kind is EValGeometry.GLOBAL:
            if self.coords is not None:
                raise ValueError("GLOBAL geometry cannot have point coordinates.")
        elif self.coords is None or self.coords.ndim != 2:
            raise ValueError("Spatial geometry requires coordinates (point, xyz).")
        if self.coords is not None and self.coords.shape[1] != 3:
            raise ValueError("Coordinates must have x, y, z columns.")


@dataclass(slots=True)
class ValObservable:
    geometry: ValGeometry
    operator: EValOperator
    components: tuple[str, ...]
    times: np.ndarray | None
    units: str
    time_units: str = ""

    def __post_init__(self) -> None:
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("Observable components must be nonempty and unique.")
        if self.times is not None:
            if self.times.ndim != 1 or len(self.times) == 0:
                raise ValueError("Times must be a nonempty 1D array.")
            if not np.all(np.isfinite(self.times)):
                raise ValueError("Times must be finite.")
            if np.any(np.diff(self.times) <= 0):
                raise ValueError("Times must increase strictly.")


@dataclass(slots=True)
class ValDomain:
    observables: dict[str, ValObservable]

    def __post_init__(self) -> None:
        if not self.observables:
            raise ValueError("A validation domain requires an observable.")


@dataclass(slots=True)
class EpistemicOffsets:
    """Nonnegative magnitudes for coherent lower/upper value shifts."""

    lower: np.ndarray | float
    upper: np.ndarray | float
    axes: tuple[EValAxis, ...] = ()

    def __post_init__(self) -> None:
        if any(axis not in PHYSICAL_AXES for axis in self.axes):
            raise ValueError("Offset axes must be physical axes.")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("Offset axes must be unique.")
        if tuple(sorted(self.axes, key=AXIS_ORDER.index)) != self.axes:
            raise ValueError("Offset axes must be in canonical order.")
        for name, value in (("lower", self.lower), ("upper", self.upper)):
            array = np.asarray(value)
            if array.ndim != len(self.axes):
                raise ValueError(f"{name} offset rank does not match axes.")
            if not np.all(np.isfinite(array)) or np.any(array < 0):
                raise ValueError(f"{name} offset must be finite and nonnegative.")


@dataclass(slots=True)
class ValArray:
    """Data with explicit axes; the final axis, when present, is aleatory."""

    data: np.ndarray
    axes: tuple[EValAxis, ...]
    offsets: EpistemicOffsets | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != len(self.axes):
            raise ValueError("Array rank does not match named axes.")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("Array axes must be unique.")
        if tuple(sorted(self.axes, key=AXIS_ORDER.index)) != self.axes:
            raise ValueError("Array axes must be in canonical order.")
        if any(size == 0 for size in self.data.shape):
            raise ValueError("Validation axes cannot be empty.")


@dataclass(slots=True)
class ValSampleInputs:
    parameters: dict[str, ValArray] | None = None
    run_ids: ValArray | None = None


@dataclass(slots=True)
class ValData:
    quantities: dict[str, ValArray]
    sample_inputs: ValSampleInputs | None = None


@dataclass(slots=True)
class ValCase:
    domain: ValDomain
    sim: ValData
    exp: ValData

    def __post_init__(self) -> None:
        expected = set(self.domain.observables)
        if set(self.sim.quantities) != expected or set(self.exp.quantities) != expected:
            raise ValueError("Simulation and experiment must match domain keys.")
        for name, observable in self.domain.observables.items():
            for source in (self.sim, self.exp):
                quantity = source.quantities[name]
                if not np.issubdtype(quantity.data.dtype, np.number):
                    raise TypeError(f"{name}: validation quantity must be numeric.")
                lengths = {
                    EValAxis.POINT: (
                        1
                        if observable.geometry.coords is None
                        else len(observable.geometry.coords)
                    ),
                    EValAxis.COMPONENT: len(observable.components),
                    EValAxis.TIME: 1
                    if observable.times is None
                    else len(observable.times),
                }
                for axis, length in lengths.items():
                    if axis in quantity.axes:
                        if quantity.data.shape[quantity.axes.index(axis)] != length:
                            raise ValueError(f"{name}: {axis.name} length mismatch.")
                    elif length != 1:
                        raise ValueError(f"{name}: missing non-singleton {axis.name}.")
                if quantity.offsets is not None:
                    offsets = quantity.offsets
                    for axis in offsets.axes:
                        size = lengths[axis]
                        for value in (offsets.lower, offsets.upper):
                            if (
                                np.asarray(value).shape[offsets.axes.index(axis)]
                                != size
                            ):
                                raise ValueError(
                                    f"{name}: offset {axis.name} mismatch."
                                )
                provenance = source.sample_inputs
                if provenance is not None:
                    trace_arrays = list((provenance.parameters or {}).values())
                    if provenance.run_ids is not None:
                        trace_arrays.append(provenance.run_ids)
                    for trace in trace_arrays:
                        if any(axis not in SAMPLE_AXES for axis in trace.axes):
                            raise ValueError("Sample inputs need uncertainty axes.")
                        for axis, length in zip(
                            trace.axes, trace.data.shape, strict=True
                        ):
                            if axis not in quantity.axes:
                                raise ValueError(
                                    f"{name}: sample input axis {axis.name} absent."
                                )
                            source_length = quantity.data.shape[
                                quantity.axes.index(axis)
                            ]
                            if length != source_length:
                                raise ValueError(
                                    f"{name}: sample input {axis.name} mismatch."
                                )


def val_from_arrays(
    quantities: dict[str, ValArray],
    *,
    sample_inputs: ValSampleInputs | None = None,
) -> ValData:
    """Wrap prepared arrays without copying them."""
    return ValData(quantities=quantities, sample_inputs=sample_inputs)


def assemble_val_case(domain: ValDomain, sim: ValData, exp: ValData) -> ValCase:
    """Validate common observable metadata and assemble a comparison."""
    return ValCase(domain=domain, sim=sim, exp=exp)

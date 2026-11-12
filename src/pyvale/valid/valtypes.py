# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Named physical and uncertainty axes for validation data."""

from dataclasses import dataclass, field
from enum import Enum, IntEnum, StrEnum, auto

import numpy as np

from pyvale.valid.inputvalidation import validate_real_array


class EValMetricPart(Enum):
    TOTAL = auto()
    D_PLUS = auto()
    D_MINUS = auto()
    BALANCE = auto()


class EEpisOffset(IntEnum):
    NOMINAL = 0
    LOWER = 1
    UPPER = 2


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
    coord_units: str = ""
    coord_frame: str = ""
    """Spatial coordinate reference frame for mesh and observation locations."""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EValGeometry):
            raise TypeError("Geometry kind must be EValGeometry.")
        if self.coords is not None:
            validate_real_array(self.coords, "coords", allow_nan=False)
        if self.kind is EValGeometry.GLOBAL:
            if self.coords is not None:
                raise ValueError("GLOBAL geometry cannot have point coordinates.")
        elif self.coords is None or self.coords.ndim != 2:
            raise ValueError("Spatial geometry requires coordinates (point, xyz).")

        if self.coords is not None and (
            self.coords.shape[1] != 3 or len(self.coords) == 0
        ):
            raise ValueError("Coordinates must have x, y, z columns.")


@dataclass(slots=True)
class ValObservable:
    geometry: ValGeometry
    operator: EValOperator
    components: tuple[str, ...]
    times: np.ndarray | None = None
    phys_units: str = ""
    time_units: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.operator, EValOperator):
            raise TypeError("Observable operator must be EValOperator.")
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("Observable components must be nonempty and unique.")
        if self.times is not None:
            validate_real_array(self.times, "times", allow_nan=False)
            if self.times.ndim != 1 or len(self.times) == 0:
                raise ValueError("Times must be a nonempty 1D array.")
            if np.any(self.times[1:] <= self.times[:-1]):
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

        if any(not isinstance(axis, EValAxis) for axis in self.axes):
            raise TypeError("Axes must be EValAxis.")
        ordered_axes = tuple(sorted(self.axes, key=AXIS_ORDER.index))
        if ordered_axes != self.axes:
            raise ValueError("Offset axes must be in canonical order.")

        for name, value in (("lower", self.lower), ("upper", self.upper)):
            array = validate_real_array(value, name, allow_nan=False)
            if array.ndim != len(self.axes):
                raise ValueError(f"{name} offset rank does not match axes.")
            if np.any(array < 0):
                raise ValueError(f"{name} offset must be finite and nonnegative.")


@dataclass(slots=True)
class ValArray:
    """Borrow data with explicit axes; nominal selections share its memory.

    Observation dtype and finiteness are checked when assembling a ValCase.
    Other arrays may carry nonnumeric run identifiers. Borrowed observations
    and metadata must remain unchanged for the lifetime of a case.
    """

    data: np.ndarray
    axes: tuple[EValAxis, ...]
    offsets: EpistemicOffsets | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != len(self.axes):
            raise ValueError("Array rank does not match named axes.")

        if len(set(self.axes)) != len(self.axes):
            raise ValueError("Array axes must be unique.")

        if any(not isinstance(axis, EValAxis) for axis in self.axes):
            raise TypeError("Axes must be EValAxis.")
        ordered_axes = tuple(sorted(self.axes, key=AXIS_ORDER.index))
        if ordered_axes != self.axes:
            raise ValueError("Array axes must be in canonical order.")

        if any(size == 0 for size in self.data.shape):
            raise ValueError("Validation axes cannot be empty.")

    def select_aleatory(
        self,
        point_idx: int,
        component_idx: int,
        time_idx: int,
        *,
        epis_trans_idx: int = 0,
        epis_nuis_idx: int = 0,
        offset: EEpisOffset = EEpisOffset.NOMINAL,
    ) -> np.ndarray:
        """Select one conditional 1D sample, applying a coherent value shift."""
        selection_idxs = {
            EValAxis.POINT: point_idx,
            EValAxis.COMPONENT: component_idx,
            EValAxis.TIME: time_idx,
            EValAxis.EPISTEMIC_TRANS: epis_trans_idx,
            EValAxis.EPISTEMIC_NUIS: epis_nuis_idx,
            EValAxis.ALEATORY: slice(None),
        }
        for axis, selection_idx in selection_idxs.items():
            if axis is EValAxis.ALEATORY:
                continue
            size = self.data.shape[self.axes.index(axis)] if axis in self.axes else 1
            if isinstance(selection_idx, (bool, np.bool_)) or not isinstance(
                selection_idx, (int, np.integer)
            ):
                raise TypeError(f"{axis.name} index must be an integer.")
            if not 0 <= selection_idx < size:
                raise IndexError(
                    f"{axis.name} index {selection_idx} is outside [0, {size})."
                )
        if not isinstance(offset, EEpisOffset):
            raise TypeError("Offset scenario must be EEpisOffset.")
        if offset != EEpisOffset.NOMINAL and self.offsets is None:
            raise ValueError("Quantity has no offset scenarios.")
        return self._select_aleatory(
            point_idx, component_idx, time_idx, epis_trans_idx, epis_nuis_idx, offset
        )

    def _select_aleatory(
        self,
        point_idx: int,
        component_idx: int,
        time_idx: int,
        epis_trans_idx: int,
        epis_nuis_idx: int,
        offset: EEpisOffset,
    ) -> np.ndarray:
        selection_idxs = {
            EValAxis.POINT: point_idx,
            EValAxis.COMPONENT: component_idx,
            EValAxis.TIME: time_idx,
            EValAxis.EPISTEMIC_TRANS: epis_trans_idx,
            EValAxis.EPISTEMIC_NUIS: epis_nuis_idx,
            EValAxis.ALEATORY: slice(None),
        }
        values = np.atleast_1d(
            self.data[tuple(selection_idxs[axis] for axis in self.axes)]
        )
        if offset == EEpisOffset.NOMINAL:
            return values
        offsets = self.offsets
        assert offsets is not None
        physical = {
            EValAxis.POINT: point_idx,
            EValAxis.COMPONENT: component_idx,
            EValAxis.TIME: time_idx,
        }
        bound = offsets.lower if offset == EEpisOffset.LOWER else offsets.upper
        magnitude = np.asarray(bound)[tuple(physical[axis] for axis in offsets.axes)]
        sign = -1.0 if offset == EEpisOffset.LOWER else 1.0
        return values + sign * magnitude


@dataclass(slots=True)
class ValSampleInputs:
    parameters: dict[str, ValArray] | None = None
    run_ids: ValArray | None = None


@dataclass(slots=True)
class ValCase:
    """Validate and borrow observations and metadata without copying large arrays."""

    domain: ValDomain
    sim: dict[str, ValArray]
    exp: dict[str, ValArray]
    sim_inputs: ValSampleInputs | None = field(default=None, kw_only=True)
    exp_inputs: ValSampleInputs | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        expected = set(self.domain.observables)
        if set(self.sim) != expected or set(self.exp) != expected:
            raise ValueError("Simulation and experiment must match domain keys.")
        for name, observable in self.domain.observables.items():
            for source, provenance in (
                (self.sim, self.sim_inputs),
                (self.exp, self.exp_inputs),
            ):
                quantity = source[name]
                validate_real_array(quantity.data, name)
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


def assemble_val_case(
    domain: ValDomain, sim: dict[str, ValArray], exp: dict[str, ValArray]
) -> ValCase:
    """Validate common observable metadata and assemble a comparison."""
    return ValCase(domain=domain, sim=sim, exp=exp)


class EMAVMDuplicates(Enum):
    """Selects the MAVM empirical-CDF integration algorithm."""

    REJECT = "reject"
    """Full field compatible loop for unique floating point observations."""

    INCLUDE = "include"
    """Probability weighted loop that supports repeated observations."""


@dataclass(slots=True)
class MAVMResult:
    """Container for Modified Area Validation Metric (MAVM) calculation
    results.
    """

    d_plus: float
    """Positive support-shift mismatch area, matching fullfieldvalmetrics."""

    d_minus: float
    """Negative support-shift mismatch area, matching fullfieldvalmetrics."""

    d_total: float
    """Total mismatch distance (d_plus + d_minus)."""

    model_quantiles: np.ndarray
    """Quantiles (support values) of the model empirical CDF."""

    model_probs: np.ndarray
    """Cumulative probabilities corresponding to model quantiles."""

    exp_quantiles: np.ndarray
    """Quantiles (support values) of the experimental empirical CDF."""

    exp_probs: np.ndarray
    """Cumulative probabilities corresponding to experimental quantiles."""

    exp_conf_lower: np.ndarray
    """Lower confidence bound quantiles for the experimental CDF."""

    exp_conf_upper: np.ndarray
    """Upper confidence bound quantiles for the experimental CDF."""

    alpha: float
    """Significance level used for the confidence interval (e.g. 0.05)."""


@dataclass(slots=True)
class ValECDF:
    """Borrow validated support/probabilities, each shaped (n_support,).

    Support and cumulative probabilities increase strictly; probabilities end
    at one. Arrays must remain unchanged while the ECDF is in use.
    """

    quantiles: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        validate_real_array(self.quantiles, "quantiles", allow_nan=False)
        validate_real_array(self.probabilities, "probabilities", allow_nan=False)
        if (
            self.quantiles.ndim != 1
            or not len(self.quantiles)
            or self.quantiles.shape != self.probabilities.shape
        ):
            raise ValueError(
                "ECDF support and probabilities need matching nonempty 1D shapes."
            )
        if np.any(self.quantiles[1:] <= self.quantiles[:-1]):
            raise ValueError("ECDF support must increase strictly.")
        if (
            np.any(self.probabilities <= 0)
            or np.any(self.probabilities > 1)
            or np.any(self.probabilities[1:] <= self.probabilities[:-1])
            or self.probabilities[-1] != 1
        ):
            raise ValueError(
                "ECDF probabilities must increase strictly in (0, 1] and end at one."
            )


class EValECDFBounds(Enum):
    POINTWISE = auto()
    SUPPORT_SUM = auto()


class EValMAVMDiagnostic(Enum):
    DISTRIBUTIONS = auto()
    COMPARISONS = auto()
    CORRECTION = auto()
    MODEL_CORRECTION = auto()


@dataclass(slots=True)
class ValECDFSummary:
    """Conditional quantile curves on a shared probability grid, each shaped (n,).

    nominal is the arithmetic mean of conditional quantiles, not a pooled CDF.
    SUPPORT_SUM selects whole scenarios and does not guarantee pointwise bounds.
    Indices refer to input scenario order; pointwise bounds have no single index.
    """

    probabilities: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    nominal: np.ndarray
    bounds: EValECDFBounds
    lower_idx: int | None
    upper_idx: int | None


@dataclass(slots=True)
class MAVMCorrection:
    """Results defining the lower/upper support shifts and four endpoint comparisons.

    Comparison order is model-lower/exp-lower, model-upper/exp-lower,
    model-lower/exp-upper, model-upper/exp-upper. lower/upper retain the selected
    MAVMResult objects. Lower is anchored to model-lower; upper to model-upper.
    """

    lower: MAVMResult
    upper: MAVMResult
    comparisons: tuple[MAVMResult, MAVMResult, MAVMResult, MAVMResult]
    lower_exp_idx: int
    upper_exp_idx: int

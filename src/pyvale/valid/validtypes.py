# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Named physical and uncertainty axes for validation data."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, StrEnum, auto
from typing import TYPE_CHECKING

import numpy as np

from pyvale.valid.validchecks import validate_real_array

if TYPE_CHECKING:
    from pyvale.valid.metricstrategy import EMetricType, IValidMetric


class EMetricOutput(Enum):
    D_PLUS = auto()
    D_MINUS = auto()
    DISTANCE = auto()
    STATISTIC = auto()
    ERROR = auto()


class EEpisOffset(IntEnum):
    NOMINAL = 0
    LOWER = 1
    UPPER = 2


class EValidAxis(Enum):
    POINT = auto()
    COMPONENT = auto()
    TIME = auto()
    EPISTEMIC_TRANS = auto()
    EPISTEMIC_NUIS = auto()
    ALEATORY = auto()


AXIS_ORDER = tuple(EValidAxis)
PHYSICAL_AXES = AXIS_ORDER[:3]
SAMPLE_AXES = AXIS_ORDER[3:]


class EValidGeometry(Enum):
    POINT = auto()
    LINE = auto()
    SURFACE = auto()
    VOLUME = auto()
    GLOBAL = auto()


class EValidSource(StrEnum):
    SIM = "sim"
    EXP = "exp"


@dataclass(slots=True)
class ValidObservable:
    components: tuple[str, ...]
    geometry: EValidGeometry
    coords: np.ndarray | None = None
    connect: dict[str, np.ndarray] | None = None
    times: np.ndarray | None = None
    phys_units: str = ""
    coord_units: str = ""
    coord_frame: str = ""
    time_units: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, EValidGeometry):
            raise TypeError("Geometry kind must be EValidGeometry.")
        if self.coords is not None:
            validate_real_array(self.coords, "coords", allow_nan=False)
        if self.geometry is EValidGeometry.GLOBAL:
            if self.coords is not None:
                raise ValueError("GLOBAL geometry cannot have point coordinates.")
        elif self.coords is None or self.coords.ndim != 2:
            raise ValueError("Spatial geometry requires coordinates (point, xyz).")

        if self.coords is not None and (
            self.coords.shape[1] != 3 or self.coords.shape[0] == 0
        ):
            raise ValueError("Coordinates must have x, y, z columns.")
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("Observable components must be nonempty and unique.")
        if self.times is not None:
            validate_real_array(self.times, "times", allow_nan=False)
            if self.times.ndim != 1 or self.times.shape[0] == 0:
                raise ValueError("Times must be a nonempty 1D array.")
            if np.any(self.times[1:] <= self.times[:-1]):
                raise ValueError("Times must increase strictly.")


@dataclass(slots=True)
class EpistemicOffsets:
    """Nonnegative magnitudes for coherent lower/upper value shifts."""

    lower: np.ndarray | float
    upper: np.ndarray | float
    axes: tuple[EValidAxis, ...] = ()

    def __post_init__(self) -> None:
        if any(axis not in PHYSICAL_AXES for axis in self.axes):
            raise ValueError("Offset axes must be physical axes.")

        if len(set(self.axes)) != len(self.axes):
            raise ValueError("Offset axes must be unique.")

        if any(not isinstance(axis, EValidAxis) for axis in self.axes):
            raise TypeError("Axes must be EValidAxis.")
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
class ValidArray:
    """Borrow data with explicit axes; nominal selections share its memory.

    Observation dtype and finiteness are checked when assembling a ValidCase.
    Other arrays may carry nonnumeric run identifiers. Borrowed observations
    and metadata must remain unchanged for the lifetime of a case.
    """

    data: np.ndarray
    axes: tuple[EValidAxis, ...]
    offsets: EpistemicOffsets | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != len(self.axes):
            raise ValueError("Array rank does not match named axes.")

        if len(set(self.axes)) != len(self.axes):
            raise ValueError("Array axes must be unique.")

        if any(not isinstance(axis, EValidAxis) for axis in self.axes):
            raise TypeError("Axes must be EValidAxis.")

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
            EValidAxis.POINT: point_idx,
            EValidAxis.COMPONENT: component_idx,
            EValidAxis.TIME: time_idx,
            EValidAxis.EPISTEMIC_TRANS: epis_trans_idx,
            EValidAxis.EPISTEMIC_NUIS: epis_nuis_idx,
            EValidAxis.ALEATORY: slice(None),
        }

        for axis, selection_idx in selection_idxs.items():
            if axis is EValidAxis.ALEATORY:
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
            EValidAxis.POINT: point_idx,
            EValidAxis.COMPONENT: component_idx,
            EValidAxis.TIME: time_idx,
            EValidAxis.EPISTEMIC_TRANS: epis_trans_idx,
            EValidAxis.EPISTEMIC_NUIS: epis_nuis_idx,
            EValidAxis.ALEATORY: slice(None),
        }

        values = np.atleast_1d(
            self.data[tuple(selection_idxs[axis] for axis in self.axes)]
        )

        if offset == EEpisOffset.NOMINAL:
            return values

        offsets = self.offsets
        assert offsets is not None

        physical = {
            EValidAxis.POINT: point_idx,
            EValidAxis.COMPONENT: component_idx,
            EValidAxis.TIME: time_idx,
        }
        bound = offsets.lower if offset == EEpisOffset.LOWER else offsets.upper

        magnitude = np.asarray(bound)[tuple(physical[axis] for axis in offsets.axes)]

        sign = -1.0 if offset == EEpisOffset.LOWER else 1.0

        return values + sign * magnitude


@dataclass(slots=True)
class ValidSampleInputs:
    parameters: dict[str, ValidArray] | None = None
    run_ids: ValidArray | None = None


@dataclass(slots=True)
class ValidCase:
    """Validate and borrow observations and metadata without copying large arrays."""

    observables: dict[str, ValidObservable]
    sim: dict[str, ValidArray]
    exp: dict[str, ValidArray]
    sim_inputs: ValidSampleInputs | None = field(default=None, kw_only=True)
    exp_inputs: ValidSampleInputs | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not self.observables:
            raise ValueError("A validation case requires an observable.")
        expected = set(self.observables)

        if set(self.sim) != expected or set(self.exp) != expected:
            raise ValueError("Simulation and experiment must match observables keys.")

        for name, observable in self.observables.items():
            leng_pt = 1
            if observable.coords is not None:
                leng_pt = observable.coords.shape[0]
            leng_comp = len(observable.components)
            leng_time = 1
            if observable.times is not None:
                leng_time = observable.times.shape[0]
            lengths = {
                EValidAxis.POINT: leng_pt,
                EValidAxis.COMPONENT: leng_comp,
                EValidAxis.TIME: leng_time,
            }
            for source, provenance in (
                (self.sim, self.sim_inputs),
                (self.exp, self.exp_inputs),
            ):
                quantity = source[name]
                validate_real_array(quantity.data, name)

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
class ValidECDF:
    """Borrow support/probabilities, each shaped (n_support,).

    Support and cumulative probabilities increase strictly; probabilities end
    at one. Arrays must remain unchanged while the ECDF is in use.
    """

    quantiles: np.ndarray
    probabilities: np.ndarray


class EValidECDFBounds(Enum):
    POINTWISE = auto()
    SUPPORT_SUM = auto()


class EValidMAVMDiagnostic(Enum):
    DISTRIBUTIONS = auto()
    COMPARISONS = auto()
    CORRECTION = auto()
    MODEL_CORRECTION = auto()


@dataclass(slots=True)
class ValidECDFSummary:
    """Conditional quantile curves on a shared probability grid, each shaped (n,).

    nominal is the arithmetic mean of conditional quantiles, not a pooled CDF.
    SUPPORT_SUM selects whole scenarios and does not guarantee pointwise bounds.
    Indices refer to input scenario order; pointwise bounds have no single index.
    """

    probabilities: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    nominal: np.ndarray
    bounds: EValidECDFBounds
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


def validate_mavm_config(alpha: float, duplicates: EMAVMDuplicates, tol: float) -> None:
    if not isinstance(duplicates, EMAVMDuplicates):
        raise TypeError("MAVM duplicates must be EMAVMDuplicates.")

    if (
        not isinstance(alpha, (int, float, np.integer, np.floating))
        or isinstance(alpha, (bool, np.bool_))
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("MAVM alpha must be between zero and one.")

    if (
        not isinstance(tol, (int, float, np.integer, np.floating))
        or isinstance(tol, (bool, np.bool_))
        or not np.isfinite(tol)
        or tol < 0
    ):
        raise ValueError("MAVM tolerance must be finite and nonnegative.")


def validate_ecdf(cdf: ValidECDF) -> None:
    if not isinstance(cdf, ValidECDF):
        raise TypeError("Model CDFs must be ValidECDF records.")

    validate_real_array(cdf.quantiles, "quantiles", allow_nan=False)
    validate_real_array(cdf.probabilities, "probabilities", allow_nan=False)

    if (
        cdf.quantiles.ndim != 1
        or not len(cdf.quantiles)
        or cdf.quantiles.shape != cdf.probabilities.shape
    ):
        raise ValueError(
            "ECDF support and probabilities need matching nonempty 1D shapes."
        )

    if np.any(cdf.quantiles[1:] <= cdf.quantiles[:-1]):
        raise ValueError("ECDF support must increase strictly.")

    if (
        np.any(cdf.probabilities <= 0)
        or np.any(cdf.probabilities > 1)
        or np.any(cdf.probabilities[1:] <= cdf.probabilities[:-1])
        or cdf.probabilities[-1] != 1
    ):
        raise ValueError(
            "ECDF probabilities must increase strictly in (0, 1] and end at one."
        )


class EValidResAxis(StrEnum):
    POINT = "point"
    COMPONENT = "component"
    TIME = "time"
    SIM_TRANS = "sim_trans"
    SIM_NUIS = "sim_nuis"
    EXP_TRANS = "exp_trans"
    EXP_NUIS = "exp_nuis"
    SIM_OFFSET = "sim_offset"
    EXP_OFFSET = "exp_offset"


VALID_RES_AXES = tuple(EValidResAxis)


@dataclass(slots=True)
class ValidMetricField:
    """Metric arrays in canonical result-axis order.

    Axes are point, component, time, simulation transferable/nuisance epistemic,
    experiment transferable/nuisance epistemic, simulation offset, experiment
    offset. Physical axes are always present; singleton uncertainty axes are
    omitted. All output arrays share the same shape.
    A steady scalar observable commonly has shape (n_points, 1, 1); a transient
    vector has shape (n_points, n_components, n_times) before scenario axes.
    """

    axes: tuple[EValidResAxis, ...]
    outputs: dict[EMetricOutput, np.ndarray]
    scenarios: dict[EValidResAxis, tuple[EEpisOffset, ...]] | None = None


def resolve_metric_output(
    field: ValidMetricField, output: EMetricOutput | None
) -> EMetricOutput:
    """Select the sole output or require an explicit choice for multiple outputs."""
    if output is None:
        if len(field.outputs) != 1:
            raise ValueError("Select an output explicitly for a multi-output metric.")
        return next(iter(field.outputs))
    if not isinstance(output, EMetricOutput):
        raise TypeError("Metric output must be EMetricOutput.")
    if output not in field.outputs:
        raise ValueError(f"Metric field has no {output.name} output.")
    return output


@dataclass(slots=True)
class ValidResult:
    """Borrow observables/field arrays and snapshot metric configurations."""

    observables: dict[str, ValidObservable]
    metrics: dict[str, dict[EMetricType, ValidMetricField]]
    metric_configs: dict[EMetricType, IValidMetric]

    def __post_init__(self) -> None:
        self.metric_configs = deepcopy(self.metric_configs)

    def find_extreme(
        self,
        observable: str,
        metric: EMetricType,
        *,
        output: EMetricOutput | None = None,
        maximum: bool = True,
    ) -> tuple[float, dict[EValidResAxis, int]]:
        """Locate a finite extreme without reducing the stored result field."""
        field = self.metrics[observable][metric]
        values = field.outputs[resolve_metric_output(field, output)]
        if not np.any(np.isfinite(values)):
            raise ValueError("Metric field contains no finite results.")
        selectable = np.where(
            np.isfinite(values), values, -np.inf if maximum else np.inf
        )
        flat_idx = np.argmax(selectable) if maximum else np.argmin(selectable)
        selection_idx = tuple(
            int(index) for index in np.unravel_index(flat_idx, values.shape)
        )
        return float(values[selection_idx]), dict(
            zip(field.axes, selection_idx, strict=True)
        )

# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Named physical and uncertainty axes for validation data."""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, StrEnum, auto
from itertools import product

import numpy as np

from pyvale.validation.arraychecks import _validate_real_array
from pyvale.validation.imetric import (
    EMetricOut,
    EMetricType,
    IValidMetric,
    MAVMResult,
)


class EEpisOffset(IntEnum):
    """Select nominal observations or a coherent lower or upper shift."""

    NOMINAL = 0
    LOWER = 1
    UPPER = 2


class EValidAxis(Enum):
    """Name physical and uncertainty dimensions of observation arrays."""

    POINT = auto()
    COMPONENT = auto()
    TIME = auto()
    EPISTEMIC_TRANS = auto()
    EPISTEMIC = EPISTEMIC_TRANS
    EPISTEMIC_NUIS = auto()
    ALEATORY = auto()


AXIS_ORDER = tuple(EValidAxis)
PHYSICAL_AXES = AXIS_ORDER[:3]
SAMPLE_AXES = AXIS_ORDER[3:]


class EValidGeometry(Enum):
    """Identify how an observable is placed on physical support."""

    POINT = auto()
    LINE = auto()
    SURFACE = auto()
    VOLUME = auto()
    GLOBAL = auto()


@dataclass(slots=True)
class ValidObservable:
    """Describe the physical support, components and units of an observable.

    Attributes
    ----------
    components : tuple[str, ...]
        Unique names in component-axis order.
    geometry : EValidGeometry
        Global or spatial geometry of the observable.
    coords : ndarray or None
        Finite sampling coordinates of shape (n_points, 3), with xyz columns in
        coord_units and coord_frame. Omit for global data.
    connect : dict[str, ndarray] or None
        Optional mesh connectivity keyed by element type. Each array has shape
        (n_elements, n_nodes_per_element) and indexes the coordinate rows.
    times : ndarray or None
        Strictly increasing evaluation times of shape (n_times,), in time_units.
        None denotes steady data.
    phys_units : str
        Units of observations and physical metric outputs.
    coord_units : str
        Units of xyz coordinates.
    coord_frame : str
        Coordinate frame of the sampling positions.
    time_units : str
        Units of evaluation times.
    """

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
        """Validate observable geometry, component names and evaluation times."""
        if not isinstance(self.geometry, EValidGeometry):
            raise TypeError("Geometry kind must be EValidGeometry.")

        if self.coords is not None:
            _validate_real_array(self.coords, "coords", allow_nan=False)

        if self.geometry is EValidGeometry.GLOBAL:
            if self.coords is not None:
                raise ValueError("GLOBAL geometry cannot have point coordinates.")
        elif self.coords is None and self.geometry is not EValidGeometry.POINT:
            raise ValueError("Spatial geometry requires coordinates (point, xyz).")

        if self.coords is not None and (
            self.coords.ndim != 2
            or self.coords.shape[1] != 3
            or self.coords.shape[0] == 0
        ):
            raise ValueError("Coordinates must have x, y, z columns.")

        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("Observable components must be nonempty and unique.")

        if self.times is not None:
            _validate_real_array(self.times, "times", allow_nan=False)

            if self.times.ndim != 1 or self.times.shape[0] == 0:
                raise ValueError("Times must be a nonempty 1D array.")

            if np.any(self.times[1:] <= self.times[:-1]):
                raise ValueError("Times must increase strictly.")


@dataclass(slots=True)
class EpistemicOffsets:
    """Nonnegative magnitudes for coherent lower/upper value shifts.

    Attributes
    ----------
    lower : ndarray or float
        Nonnegative magnitude subtracted from the samples. Shape follows axes, or
        scalar if axes is empty.
    upper : ndarray or float
        Nonnegative magnitude added to the samples. Shape follows axes, or scalar if
        axes is empty.
    axes : tuple[EValidAxis, ...]
        Subset of (point, component, time) in canonical order. Magnitudes use
        observation units and do not vary across aleatory or sampled epistemic
        dimensions.
    """

    lower: np.ndarray | float
    upper: np.ndarray | float
    axes: tuple[EValidAxis, ...] = ()

    def __post_init__(self) -> None:
        """Check that offset magnitudes match the named physical dimensions."""
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
            array = _validate_real_array(value, name, allow_nan=False)

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

    Attributes
    ----------
    data : ndarray
        Array with one dimension per entry in axes. Observation arrays store real
        values and may use NaN for missing observations.
    axes : tuple[EValidAxis, ...]
        Present dimensions in canonical order (point, component, time, transferable,
        nuisance, aleatory). Absent dimensions are treated as singletons.
    offsets : EpistemicOffsets or None
        Optional coherent shifts in the same units as the observations.
    """

    data: np.ndarray
    axes: tuple[EValidAxis, ...]
    offsets: EpistemicOffsets | None = None

    def __post_init__(self) -> None:
        """Check that the data rank and axis labels agree."""
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
        """Select one conditional 1D sample, applying a coherent value shift.

        Parameters
        ----------
        point_idx : int
            Zero-based physical sampling point index.
        component_idx : int
            Zero-based observable component index.
        time_idx : int
            Zero-based evaluation time index.
        epis_trans_idx : int, optional
            Zero-based transferable epistemic case index.
        epis_nuis_idx : int, optional
            Zero-based nuisance epistemic case index.
        offset : EEpisOffset, optional
            Coherent offset to apply to the selected aleatory observations.

        Returns
        -------
        ndarray
            A single conditional sample of shape (n_aleatory,). If the aleatory axis is
            absent, the shape is (1,). Nominal values can share the input memory.
        """
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
            raise TypeError("Offset must be EEpisOffset.")

        if offset != EEpisOffset.NOMINAL and self.offsets is None:
            raise ValueError("Quantity has no coherent offsets.")

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
        """Select a conditional sample using prevalidated indices.

        Parameters
        ----------
        point_idx : int
            Zero-based physical sampling point index.
        component_idx : int
            Zero-based observable component index.
        time_idx : int
            Zero-based evaluation time index.
        epis_trans_idx : int
            Zero-based transferable epistemic case index.
        epis_nuis_idx : int
            Zero-based nuisance epistemic case index.
        offset : EEpisOffset
            Coherent offset to apply to the selected aleatory observations.

        Returns
        -------
        ndarray
            Aleatory observations of shape (n_aleatory,), with the selected coherent
            offset applied.
        """
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
    """Record provenance aligned with the uncertainty axes of observations.

    Attributes
    ----------
    input_params : dict[str, ValidArray] or None
        Input parameter arrays keyed by name, indexed by named uncertainty
        dimensions.
    run_ids : ValidArray or None
        Optional run identifiers on a matching subset of uncertainty dimensions.
    """

    input_params: dict[str, ValidArray] | None = None
    run_ids: ValidArray | None = None


def _get_axis_size(quantity: ValidArray, axis: EValidAxis) -> int:
    """Get the length of a named axis, treating an absent axis as length one.

    Parameters
    ----------
    quantity : ValidArray
        Observations paired with their canonical named axes.
    axis : EValidAxis
        Named axis whose length is requested.

    Returns
    -------
    int
        Axis length, or one when the named dimension is absent.
    """
    if axis not in quantity.axes:
        return 1

    return quantity.data.shape[quantity.axes.index(axis)]


def _select_aleatory_cases(
    quantity: ValidArray,
    *,
    point_idx: int,
    component_idx: int,
    time_idx: int,
    offset: EEpisOffset | None,
    trans_idxs: tuple[int, ...] | None = None,
    nuis_idxs: tuple[int, ...] | None = None,
) -> tuple[tuple[tuple[int, int, EEpisOffset], ...], tuple[np.ndarray, ...]]:
    """Return case labels and samples in transferable, nuisance, offset order.

    Parameters
    ----------
    quantity : ValidArray
        Observations paired with their canonical named axes.
    point_idx : int
        Zero-based physical sampling point index.
    component_idx : int
        Zero-based observable component index.
    time_idx : int
        Zero-based evaluation time index.
    offset : EEpisOffset or None
        Offset to select. None spans all available offsets, including nominal.
    trans_idxs : tuple[int, ...] or None, optional
        Transferable case indices in selection order. Use None for all cases.
    nuis_idxs : tuple[int, ...] or None, optional
        Nuisance case indices in selection order. Use None for all cases.

    Returns
    -------
    tuple[tuple[tuple[int, int, EEpisOffset], ...], tuple[ndarray, ...]]
        Case labels and aligned one-dimensional sample arrays in (transferable,
        nuisance, offset) product order.
    """

    if offset is not None and not isinstance(offset, EEpisOffset):
        raise TypeError("Offset must be EEpisOffset or None.")

    if trans_idxs is None:
        trans_idxs = tuple(range(_get_axis_size(quantity, EValidAxis.EPISTEMIC_TRANS)))

    if nuis_idxs is None:
        nuis_idxs = tuple(range(_get_axis_size(quantity, EValidAxis.EPISTEMIC_NUIS)))

    offsets = (offset,) if offset is not None else (EEpisOffset.NOMINAL,)

    if offset is None and quantity.offsets is not None:
        offsets = tuple(EEpisOffset)

    cases = tuple(product(trans_idxs, nuis_idxs, offsets))
    samples = tuple(
        quantity.select_aleatory(
            point_idx,
            component_idx,
            time_idx,
            epis_trans_idx=trans_idx,
            epis_nuis_idx=nuis_idx,
            offset=case_offset,
        )
        for trans_idx, nuis_idx, case_offset in cases
    )
    return cases, samples


@dataclass(slots=True)
class ValidCase:
    """Validate and borrow observations and metadata without copying large arrays.

    Attributes
    ----------
    observables : dict[str, ValidObservable]
        Observable metadata keyed by the shared observable tags.
    sim : dict[str, ValidArray]
        Simulation arrays keyed by observable tag. Physical dimensions match the
        metadata; uncertainty dimensions can differ from exp.
    exp : dict[str, ValidArray]
        Experimental arrays keyed by observable tag in the corresponding physical
        coordinates and units.
    sim_inputs : ValidSampleInputs or None
        Optional input parameter arrays and run IDs aligned with simulation
        uncertainty axes.
    exp_inputs : ValidSampleInputs or None
        Optional input parameter arrays and run IDs aligned with experiment
        uncertainty axes.
    """

    observables: dict[str, ValidObservable]
    sim: dict[str, ValidArray]
    exp: dict[str, ValidArray]
    sim_inputs: ValidSampleInputs | None = field(default=None, kw_only=True)
    exp_inputs: ValidSampleInputs | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Check observation shapes and provenance against the observable metadata."""
        if not self.observables:
            raise ValueError("A validation case requires an observable.")

        expected = set(self.observables)

        if set(self.sim) != expected or set(self.exp) != expected:
            raise ValueError("Simulation and experiment must match observables keys.")

        for name, observable in self.observables.items():
            leng_pt = 1

            if observable.coords is not None:
                leng_pt = observable.coords.shape[0]
            elif observable.geometry is EValidGeometry.POINT:
                leng_pt = _get_axis_size(self.sim[name], EValidAxis.POINT)

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
                _validate_real_array(quantity.data, name)

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
                    trace_arrays = list((provenance.input_params or {}).values())

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


class EMAVMDupes(Enum):
    """Selects the MAVM empirical-CDF integration algorithm."""

    REJECT = "reject"
    """Full field compatible loop for unique floating point observations."""

    INCLUDE = "include"
    """Probability weighted loop that supports repeated observations."""


@dataclass(slots=True)
class MAVMBoundPairs:
    """Results defining the lower/upper support shifts and four endpoint comparisons.

    Comparison order is model-lower/exp-lower, model-upper/exp-lower,
    model-lower/exp-upper, model-upper/exp-upper. lower/upper retain the selected
    MAVMResult objects. Lower is anchored to model-lower; upper to model-upper.

    Attributes
    ----------
    lower : MAVMResult
        Comparison supplying the lower shift of the lower simulation curve.
    upper : MAVMResult
        Comparison supplying the upper shift of the upper simulation curve.
    comparisons : tuple[MAVMResult, MAVMResult, MAVMResult, MAVMResult]
        Four ordered endpoint comparisons, with simulation bound varying first
        inside each experimental bound.
    lower_exp_idx : int
        Experimental endpoint index (0 for lower, 1 for upper) controlling the lower
        correction.
    upper_exp_idx : int
        Experimental endpoint index controlling the upper correction.
    """

    lower: MAVMResult
    upper: MAVMResult
    comparisons: tuple[MAVMResult, MAVMResult, MAVMResult, MAVMResult]
    lower_exp_idx: int
    upper_exp_idx: int


class EValidResAxis(StrEnum):
    """Name dimensions of metric results, distinguishing simulation and experiment cases."""

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
class MetricField:
    """Metric arrays in canonical result-axis order.

    Axes are point, component, time, simulation transferable/nuisance epistemic,
    experiment transferable/nuisance epistemic, simulation offset, experiment
    offset. Physical axes are always present; singleton uncertainty axes are
    omitted. All output arrays share the same shape.
    A steady scalar observable commonly has shape (n_points, 1, 1); a transient
    vector has shape (n_points, n_components, n_times) before epistemic and offset axes.

    Attributes
    ----------
    axes : tuple[EValidResAxis, ...]
        Canonical names of the stored result dimensions.
    metric_vals : dict[EMetricOut, ndarray]
        One array per metric output, all with the same shape in axes order. A cell
        represents one conditional simulation–experiment comparison.
    epis_offsets : dict[EValidResAxis, tuple[EEpisOffset, ...]] or None
        Ordered offset labels for any retained simulation and experiment offset
        axes.
    """

    axes: tuple[EValidResAxis, ...]
    metric_vals: dict[EMetricOut, np.ndarray]
    epis_offsets: dict[EValidResAxis, tuple[EEpisOffset, ...]] | None = None

    def _resolve_output(self, output: EMetricOut | None) -> EMetricOut:
        """Select the sole output or require an explicit choice for multiple outputs.

        Parameters
        ----------
        output : EMetricOut or None
            Metric output to select. Use None to infer a single available output.

        Returns
        -------
        EMetricOut
            Declared output tag to use.
        """

        if output is None:
            if len(self.metric_vals) != 1:
                raise ValueError("Select an output explicitly for a multi-output metric.")

            return next(iter(self.metric_vals))

        if not isinstance(output, EMetricOut):
            raise TypeError("Metric output must be EMetricOut.")

        if output not in self.metric_vals:
            raise ValueError(f"Metric field has no {output.name} output.")

        return output


@dataclass(slots=True)
class ValidMetricExtrema:
    """Extrema over reduced axes; index arrays append one column per reduced axis.

    All-missing cells contain NaN and index -1. Ties choose the first original
    index in canonical result-axis order. Selected singleton axes are omitted.

    Attributes
    ----------
    axes : tuple[EValidResAxis, ...]
        Retained result axes, in array dimension order.
    reduced_axes : tuple[EValidResAxis, ...]
        Reduced dimensions represented by the final index-array axis.
    minimum : dict[EMetricOut, ndarray]
        Minimum metric values by output, each of shape retained_shape.
    maximum : dict[EMetricOut, ndarray]
        Maximum metric values by output, each of shape retained_shape.
    minimum_indices : dict[EMetricOut, ndarray]
        Controlling indices by output, of shape ``(*retained_shape, n_reduced_axes)``.
    maximum_indices : dict[EMetricOut, ndarray]
        Controlling indices for the maxima, with the same shape as minimum_indices.
    selectors : dict[EValidResAxis, int]
        Fixed axis indices applied before reducing the fields.
    """

    axes: tuple[EValidResAxis, ...]
    reduced_axes: tuple[EValidResAxis, ...]
    minimum: dict[EMetricOut, np.ndarray]
    maximum: dict[EMetricOut, np.ndarray]
    minimum_indices: dict[EMetricOut, np.ndarray]
    maximum_indices: dict[EMetricOut, np.ndarray]
    selectors: dict[EValidResAxis, int]


@dataclass(slots=True)
class ValidResult:
    """Borrow observables/field arrays and snapshot metric configurations.

    Attributes
    ----------
    observables : dict[str, ValidObservable]
        Observable metadata keyed by tag.
    metrics : dict[str, dict[EMetricType, MetricField]]
        Metric fields keyed first by observable tag, then by metric type.
    metric_configs : dict[EMetricType, IValidMetric]
        Snapshot of the metric strategies and settings used to calculate the fields.
    """

    observables: dict[str, ValidObservable]
    metrics: dict[str, dict[EMetricType, MetricField]]
    metric_configs: dict[EMetricType, IValidMetric]

    def __post_init__(self) -> None:
        """Snapshot metric settings for stable interpretation of calculated results."""
        self.metric_configs = deepcopy(self.metric_configs)

    def find_extreme(
        self,
        observable: str,
        metric: EMetricType,
        *,
        output: EMetricOut | None = None,
        maximum: bool = True,
    ) -> tuple[float, dict[EValidResAxis, int]]:
        """Locate a finite extreme without reducing the stored result field.

        Parameters
        ----------
        observable : str
            Observable tag identifying matching metadata and observations.
        metric : EMetricType
            Metric to select or evaluate.
        output : EMetricOut or None, optional
            Metric output to select. Use None to infer a single available output.
        maximum : bool, optional
            If True select the maximum; otherwise select the minimum. Nonfinite values
            are omitted.

        Returns
        -------
        tuple[float, dict[EValidResAxis, int]]
            Extreme value and its zero-based indices keyed by every stored result axis.
        """
        field = self.metrics[observable][metric]
        values = field.metric_vals[field._resolve_output(output)]

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

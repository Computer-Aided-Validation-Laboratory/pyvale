# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Sample sim fields and prepare exp data with named valid axes."""

from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass
from typing import cast

import numpy as np

from pyvale.dataio.expdata import ExpData
from pyvale.dataio.simdata import SimData
from pyvale.sensorsim.field import IField, build_sim_data_sampler
from pyvale.sensorsim.fieldsampler import sample_field_with_sensor_data
from pyvale.sensorsim.sensordata import SensorData
from pyvale.validation.arraychecks import _validate_real_array
from pyvale.validation.validtypes import (
    EValidAxis,
    EValidGeometry,
    ValidArray,
    ValidCase,
    ValidObservable,
    ValidSampleInputs,
)


def sample_field_for_valid(
    field: IField,
    observable: ValidObservable,
    *,
    sensor_data: SensorData | None = None,
) -> ValidArray:
    """Sample a bound field into point/component/time axes without rebinding it.

    Observable coordinates and times specify the sampling locations. Optional
    SensorData supplies orientations and spatial averaging, with matching
    positions and sample times. Component order follows the observable.

    Parameters
    ----------
    field : IField
        Spatial field bound to the simulation to be sampled.
    observable : ValidObservable
        Spatial observable with xyz coordinates and evaluation times.
    sensor_data : SensorData or None, optional
        Optional sensor orientation and footprint settings at the same coordinates
        and times.

    Returns
    -------
    ValidArray
        Sampled values of shape (n_points, n_components, n_times), in observable
        component order and units.
    """

    if not isinstance(field, IField):
        raise TypeError("Sampling requires an IField instance.")

    if observable.geometry is EValidGeometry.GLOBAL:
        raise ValueError("Use sample_sim_globals for global variables.")

    if observable.times is None:
        raise ValueError("Field sampling needs explicit evaluation times.")

    sim_data = field.get_sim_data()

    if sim_data.time is None or sim_data.time.size == 0:
        raise ValueError("Sim data has no time coordinates.")

    if (
        observable.times[0] < sim_data.time[0]
        or observable.times[-1] > sim_data.time[-1]
    ):
        raise ValueError("Evaluation times leave the sim range.")

    if observable.coords is None:
        raise ValueError("Field sampling requires observable coordinates.")

    if sim_data.coords is None:
        raise ValueError("Sim data has no coordinates.")

    mesh_lower = np.min(sim_data.coords, axis=0)
    mesh_upper = np.max(sim_data.coords, axis=0)

    if np.any(observable.coords < mesh_lower) or np.any(observable.coords > mesh_upper):
        raise ValueError("Sample points leave the sim bounds.")

    all_components = field.get_all_components()
    component_idxs = [all_components.index(key) for key in observable.components]

    if sensor_data is None:
        sampled = field.sample_field(observable.coords, observable.times)
    else:
        if sensor_data.positions is None or not np.array_equal(
            sensor_data.positions, observable.coords
        ):
            raise ValueError("SensorData positions differ from the observable.")

        if sensor_data.sample_times is None or not np.array_equal(
            sensor_data.sample_times, observable.times
        ):
            raise ValueError("SensorData times differ from the observable.")

        sampled = sample_field_with_sensor_data(field, sensor_data)

    return ValidArray(
        sampled[:, component_idxs, :],
        (EValidAxis.POINT, EValidAxis.COMPONENT, EValidAxis.TIME),
    )


def sample_sim_globals(sim_data: SimData, observable: ValidObservable) -> ValidArray:
    """Interpolate stored global variables at the observable times.

    Return one point with components in observable order and an explicit time
    axis. This reads existing global variables; it does not reduce nodal fields.

    Parameters
    ----------
    sim_data : SimData
        Simulation with named global traces and a one-dimensional time array.
    observable : ValidObservable
        Global observable naming stored variables and target times.

    Returns
    -------
    ValidArray
        Interpolated values of shape (1, n_components, n_times). The point axis is a
        singleton.
    """

    if observable.geometry is not EValidGeometry.GLOBAL:
        raise ValueError("Global sampling requires GLOBAL geometry.")

    if observable.times is None:
        raise ValueError("Global sampling needs explicit evaluation times.")

    if sim_data.glob_vars is None or sim_data.time is None or sim_data.time.size == 0:
        raise ValueError("Global sampling needs variables and time coordinates.")

    if (
        observable.times[0] < sim_data.time[0]
        or observable.times[-1] > sim_data.time[-1]
    ):
        raise ValueError("Evaluation times leave the sim range.")

    sampled = np.empty((1, len(observable.components), observable.times.size))

    for component_idx, key in enumerate(observable.components):
        sampled[0, component_idx] = np.interp(
            observable.times, sim_data.time, sim_data.glob_vars[key]
        )

    return ValidArray(
        sampled, (EValidAxis.POINT, EValidAxis.COMPONENT, EValidAxis.TIME)
    )


def sample_sim_sweep_for_valid(
    fields: dict[str, IField],
    observables: dict[str, ValidObservable],
    runs: list[SimData] | Iterable[tuple[tuple[int, ...], SimData]],
    *,
    sample_axes: tuple[EValidAxis, ...] = (),
    sample_shape: tuple[int, ...] = (),
    sensor_data: dict[str, SensorData] | None = None,
) -> dict[str, ValidArray]:
    """Sample simulation runs and assemble their values along named axes.

    Parameters
    ----------
    fields : dict[str, IField]
        Sampling fields keyed by spatial observable name. Supply exactly one
        entry per spatial observable. Global observables are sampled from the
        simulation's stored global variables.
    observables : dict[str, ValidObservable]
        Observable definitions keyed by name. Every observable must specify
        evaluation times. Spatial observables also specify sampling coordinates.
    runs : list[SimData] or iterable of tuple[tuple[int, ...], SimData]
        For one sample axis, pass a flat list of simulations in axis order.
        For multiple axes, pass pairs of zero-based index tuples and simulations.
        For example, `((1, 2), sim_data)` selects transferable case 1 and nuisance
        case 2 when the axes are `(EPISTEMIC_TRANS, EPISTEMIC_NUIS)`. Indexed
        pairs can arrive in any order and must supply each cell exactly once.
        Indexed pairs also work for a single axis. With no sample axes, pass
        one pair with an empty index tuple, `((), sim_data)`.
    sample_axes : tuple of EValidAxis, optional
        Axes addressed by the run indices, in canonical order. Use any subset
        of transferable epistemic, nuisance epistemic, and aleatory, in that
        order. The default is no sample axes.
    sample_shape : tuple of int, optional
        Number of runs along each sample axis, in the same order as
        `sample_axes`. Supply this for indexed sweeps with sample axes. For a
        flat list, it defaults to the list length as a one-element tuple.
    sensor_data : dict[str, SensorData] or None, optional
        Sensor settings keyed by spatial observable name. Entries can supply
        sensor orientations and spatial averaging. Their positions and sample
        times must match the corresponding observable. For an omitted entry,
        the field is sampled directly at the observable's coordinates and times.

    Returns
    -------
    dict[str, ValidArray]
        Sampled values keyed by observable name. Each array has point, component,
        and time axes followed by `sample_axes`. Global observables have a
        singleton point axis.

    Notes
    -----
    Each distinct field is rebound in place through `set_sim_data` once per run
    and remains bound to the last run processed. Use separate field instances
    for concurrent sampling.
    """
    spatial_names = {
        name
        for name, observable in observables.items()
        if observable.geometry is not EValidGeometry.GLOBAL
    }

    if set(fields) != spatial_names:
        raise ValueError("Field keys must match spatial observable keys.")

    if any(not isinstance(field, IField) for field in fields.values()):
        raise TypeError("Fields must be IField instances.")

    indexed_runs: Iterable[tuple[tuple[int, ...], SimData]]

    if isinstance(runs, list) and runs and isinstance(runs[0], SimData):
        if len(sample_axes) != 1:
            raise ValueError("A flat simulation list requires exactly one sample axis.")

        inferred_shape = (len(runs),)

        if sample_shape and sample_shape != inferred_shape:
            raise ValueError("Sample shape must match the flat simulation list length.")

        if any(not isinstance(run, SimData) for run in runs):
            raise TypeError("A flat simulation list must contain only SimData.")

        sample_shape = inferred_shape
        flat_runs = cast(list[SimData], runs)
        indexed_runs = (((idx,), sim) for idx, sim in enumerate(flat_runs))
    else:
        indexed_runs = cast(Iterable[tuple[tuple[int, ...], SimData]], runs)

    if len(sample_axes) != len(sample_shape):
        raise ValueError("Sample axes and shape must have the same length.")

    if any(
        axis
        not in (
            EValidAxis.EPISTEMIC_TRANS,
            EValidAxis.EPISTEMIC_NUIS,
            EValidAxis.ALEATORY,
        )
        for axis in sample_axes
    ):
        raise ValueError("Run axes must be uncertainty axes.")

    if tuple(sorted(sample_axes, key=tuple(EValidAxis).index)) != sample_axes:
        raise ValueError("Run axes must be in canonical order.")

    if len(set(sample_axes)) != len(sample_axes) or any(
        size <= 0 for size in sample_shape
    ):
        raise ValueError("Run axes must be unique and nonempty.")

    arrays: dict[str, ValidArray] = {}

    for name, observable in observables.items():
        if observable.times is None:
            raise ValueError("Sim sampling needs explicit evaluation times.")

        point_count = 1

        if observable.coords is not None:
            point_count = observable.coords.shape[0]

        shape = (
            point_count,
            len(observable.components),
            observable.times.size,
            *sample_shape,
        )
        arrays[name] = ValidArray(
            np.full(shape, np.nan),
            (EValidAxis.POINT, EValidAxis.COMPONENT, EValidAxis.TIME, *sample_axes),
        )

    # Multiple observable tags may use the same tensor field.
    distinct_fields = {id(field): field for field in fields.values()}
    seen: set[tuple[int, ...]] = set()

    for selection_idx, sim_data in indexed_runs:
        if len(selection_idx) != len(sample_axes):
            raise ValueError("Run index rank does not match sample axes.")

        if any(
            index < 0 or index >= size
            for index, size in zip(selection_idx, sample_shape, strict=True)
        ):
            raise ValueError("Run index is outside sample shape.")

        if selection_idx in seen:
            raise ValueError(f"Duplicate sim run index: {selection_idx}.")

        seen.add(selection_idx)

        bound_fields: dict[int, IField] = {}
        for field_id, field in distinct_fields.items():
            sampler = build_sim_data_sampler(
                sim_data,
                comp_keys=field.get_all_components(),
                spatial_dims=field.get_spatial_dims(),
            )
            bound_fields[field_id] = field.with_sampler(sampler)

        for name, observable in observables.items():
            if observable.geometry is EValidGeometry.GLOBAL:
                sampled = sample_sim_globals(sim_data, observable)
            else:
                sensor = None if sensor_data is None else sensor_data.get(name)
                sampled = sample_field_for_valid(
                    bound_fields[id(fields[name])], observable, sensor_data=sensor
                )

            arrays[name].data[
                (slice(None), slice(None), slice(None), *selection_idx)
            ] = sampled.data

    if len(seen) != (int(np.prod(sample_shape)) if sample_shape else 1):
        raise ValueError("Sim run tensor is incomplete.")

    return arrays


def assemble_exp_arrays(
    sources: dict[str, ExpData],
    observables: dict[str, ValidObservable],
    *,
    steady_window: slice | None = None,
    channel_map: dict[str, tuple[str, str]] | None = None,
) -> dict[str, ValidArray]:
    """Assemble single-component validation arrays from experimental traces.

    By default, keys in `sources` must match the observables, and acquisition
    rows are used as points in their existing order. Supply `channel_map` to
    select one row per observable by (source key, sensor label). Coordinates
    are compared when supplied by both the observable and acquisition.

    A steady window treats selected trace values as aleatory observations.
    Without a window, acquisition times must match the observable's times.

    Parameters
    ----------
    sources : dict[str, ExpData]
        Experimental acquisitions. Each fields array has shape (n_channels,
        n_samples).
    observables : dict[str, ValidObservable]
        Target single-component observable definitions keyed by tag.
    steady_window : slice or None, optional
        Slice of trace samples used as aleatory observations for a steady
        observable.
    channel_map : dict[str, tuple[str, str]] or None, optional
        Optional mapping from observable tag to (acquisition key, sensor label).

    Returns
    -------
    dict[str, ValidArray]
        Arrays keyed by observable tag. Steady data have shape (n_points, 1,
        n_aleatory); transient data have shape (n_points, 1, n_times).
    """

    if channel_map is None and set(sources) != set(observables):
        raise ValueError("Experimental source keys must match the observables.")

    if channel_map is not None and set(channel_map) != set(observables):
        raise ValueError("Experimental channel map keys must match the observables.")

    arrays = {}

    for name in sources if channel_map is None else observables:
        observable = observables[name]

        if channel_map is None:
            exp_data = sources[name]
            rows = slice(None)
        else:
            source_key, sensor_label = channel_map[name]
            exp_data = sources[source_key]
            row_idx = exp_data.sens_label_to_ind[sensor_label]
            rows = slice(row_idx, row_idx + 1)

        if exp_data.fields.ndim != 2 or len(observable.components) != 1:
            raise ValueError(f"{name}: ExpData source must have one component.")

        fields = exp_data.fields[rows]
        coords = observable.coords

        if coords is not None:
            point_count = len(coords)

            if exp_data.coords is not None and not np.array_equal(
                coords, exp_data.coords[rows]
            ):
                raise ValueError(
                    f"{name}: experimental coordinates differ from observables."
                )
        elif observable.geometry is EValidGeometry.GLOBAL:
            point_count = 1
        else:
            point_count = len(fields)

        if not len(fields) or len(fields) != point_count:
            raise ValueError(f"{name}: experimental point count mismatch.")

        if steady_window is not None:
            if observable.times is not None:
                raise ValueError("Steady window requires observables without times.")

            values = fields[:, steady_window]
            arrays[name] = ValidArray(
                values[:, None, :],
                (EValidAxis.POINT, EValidAxis.COMPONENT, EValidAxis.ALEATORY),
            )
        else:
            if observable.times is None or exp_data.times is None:
                raise ValueError("Transient acquisition requires time coordinates.")

            if not np.array_equal(observable.times, exp_data.times):
                raise ValueError(f"{name}: acquisition times differ from observables.")

            arrays[name] = ValidArray(
                fields[:, None, :],
                (EValidAxis.POINT, EValidAxis.COMPONENT, EValidAxis.TIME),
            )

    return arrays


@dataclass(slots=True)
class SensorOutputSelection:
    """Select a component and supply nominal sensor coordinates for SensorSim import.

    Attributes
    ----------
    sensor : str
        Sensor array key in the SensorSim export.
    component : int
        Zero-based component index in the exported measurements.
    coords : ndarray
        Nominal xyz coordinates of shape (n_sensors, 3), in the observable
        coordinate units and frame.
    """

    sensor: str
    component: int
    coords: np.ndarray

    def __post_init__(self) -> None:
        """Validate the component index and nominal XYZ coordinates."""
        if isinstance(self.component, (bool, np.bool_)) or not isinstance(
            self.component, (int, np.integer)
        ):
            raise TypeError("Component index must be an integer.")

        if self.component < 0:
            raise ValueError("Component index must be nonnegative.")

        _validate_real_array(self.coords, "coords", allow_nan=False)

        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError("Sensor coordinates must have shape (point, xyz).")


def extract_exp_sim_data(
    output: dict[tuple[str, ...], np.ndarray],
    *,
    sim_key: str,
    selections: dict[str, SensorOutputSelection],
    measurement_key: str = "meas",
    time_key: str = "sens_times",
) -> list[dict[str, ExpData]]:
    """Split SensorSim experiments into per-run, per-component ExpData.

    ``selections`` maps observable names to SensorOutputSelection records.
    The caller supplies coordinates because SensorSim output
    contains measurements but does not guarantee a nominal position array.

    Parameters
    ----------
    output : dict[tuple[str, ...], ndarray]
        SensorSim export keyed by (simulation, sensor, data key). Measurements have
        shape (n_runs, n_sensors, n_components, n_times); times have shape
        (n_times,).
    sim_key : str
        Simulation key in the SensorSim export.
    selections : dict[str, SensorOutputSelection]
        Sensor array, component and nominal coordinates for each output observable.
    measurement_key : str, optional
        Export key for the measurement arrays.
    time_key : str, optional
        Export key for the evaluation times.

    Returns
    -------
    list[dict[str, ExpData]]
        One dictionary per run, keyed by observable tag. Each ExpData fields array
        has shape (n_sensors, n_times).
    """
    counts = {
        output[(sim_key, selection.sensor, measurement_key)].shape[0]
        for selection in selections.values()
    }

    if len(counts) != 1:
        raise ValueError("Selected SensorSim arrays have unequal run counts.")

    count = counts.pop()

    converted: list[dict[str, ExpData]] = []

    for run_idx in range(count):
        data = {}

        for name, selection in selections.items():
            sensor, component_idx, coords = (
                selection.sensor,
                selection.component,
                selection.coords,
            )
            values = output[(sim_key, sensor, measurement_key)]

            if (
                values.ndim != 4
                or component_idx < 0
                or component_idx >= values.shape[2]
            ):
                raise ValueError(f"{name}: invalid SensorSim measurement shape.")

            times = np.asarray(output[(sim_key, sensor, time_key)])

            if times.ndim != 1 or len(times) != values.shape[3]:
                raise ValueError(f"{name}: invalid SensorSim time coordinates.")

            if coords.shape != (values.shape[1], 3):
                raise ValueError(f"{name}: invalid sensor coordinates.")

            labels = {
                f"S{point_idx}": point_idx for point_idx in range(values.shape[1])
            }
            reverse = {point_idx: label for label, point_idx in labels.items()}
            data[name] = ExpData(
                fields=values[run_idx, :, component_idx, :],
                sens_label_to_ind=labels,
                ind_to_sens_label=reverse,
                coords=coords,
                times=times,
            )

        converted.append(data)

    return converted


def _compare_offsets(left: ValidArray, right: ValidArray) -> bool:
    """Check that two arrays have equivalent coherent offsets.

    Parameters
    ----------
    left : ValidArray
        First observation array and offset metadata.
    right : ValidArray
        Second observation array and offset metadata.

    Returns
    -------
    bool
        Whether both offset selections and magnitudes match.
    """
    if left.offsets is None or right.offsets is None:
        return left.offsets is right.offsets

    return (
        left.offsets.axes == right.offsets.axes
        and np.array_equal(left.offsets.lower, right.offsets.lower)
        and np.array_equal(left.offsets.upper, right.offsets.upper)
    )


def _compare_sample_inputs(
    left: ValidSampleInputs | None,
    right: ValidSampleInputs | None,
) -> bool:
    """Compare optional sample provenance for case concatenation.

    Parameters
    ----------
    left : ValidSampleInputs or None
        First set of input parameters and run IDs, if supplied.
    right : ValidSampleInputs or None
        Second set of input parameters and run IDs, if supplied.

    Returns
    -------
    bool
        Whether the provenance records have matching keys, axes and entries.
    """
    if left is None or right is None:
        return left is right

    if (left.run_ids is None) != (right.run_ids is None):
        return False

    left_arrays = dict(left.input_params or {})
    right_arrays = dict(right.input_params or {})

    if set(left_arrays) != set(right_arrays):
        return False

    for key, value in left_arrays.items():
        other = right_arrays[key]
        equal_nan = np.issubdtype(value.data.dtype, np.inexact) and np.issubdtype(
            other.data.dtype, np.inexact
        )

        if value.axes != other.axes or not np.array_equal(
            value.data, other.data, equal_nan=equal_nan
        ):
            return False

    if left.run_ids is None:
        return True

    assert right.run_ids is not None
    return left.run_ids.axes == right.run_ids.axes and np.array_equal(
        left.run_ids.data, right.run_ids.data
    )


def concatenate_valid_cases(*cases: ValidCase) -> ValidCase:
    """Join compatible acquisitions on the experimental aleatory axis.

    Simulations and observables metadata must be identical. For transient cases,
    each complete trace becomes a joint aleatory realization at every time.

    Parameters
    ----------
    *cases : ValidCase
        Cases with matching simulation observations and physical metadata.

    Returns
    -------
    ValidCase
        Case with experimental observations concatenated along the aleatory
        dimension. Transient samples remain aligned across time.
    """

    if not cases:
        raise ValueError("At least one valid_case is required.")

    first = cases[0]
    combined = {}
    run_ids = []

    for case_idx, valid_case in enumerate(cases):
        if set(valid_case.observables) != set(first.observables):
            raise ValueError("Cases have different observable keys.")

        if not _compare_sample_inputs(valid_case.sim_inputs, first.sim_inputs):
            raise ValueError("Simulation sample provenance differs across cases.")

        first_quantity = next(iter(valid_case.exp.values()))
        sample_count = (
            first_quantity.data.shape[first_quantity.axes.index(EValidAxis.ALEATORY)]
            if EValidAxis.ALEATORY in first_quantity.axes
            else 1
        )
        provenance = valid_case.exp_inputs

        if provenance is not None and provenance.run_ids is not None:
            identifiers = provenance.run_ids

            if identifiers.axes != (EValidAxis.ALEATORY,):
                raise ValueError("Combined run IDs must use the ALEATORY axis.")

            if len(identifiers.data) != sample_count:
                raise ValueError("Run ID count does not match observations.")

            run_ids.append(identifiers.data.astype(str))
        else:
            run_ids.append(np.full(sample_count, f"case_{case_idx}"))

    for name, observable in first.observables.items():
        reference = first.exp[name]
        base_axes = tuple(
            axis for axis in reference.axes if axis is not EValidAxis.ALEATORY
        )
        pieces = []

        for case_idx, valid_case in enumerate(cases):
            other = valid_case.observables.get(name)

            if other is None:
                raise ValueError(f"{name}: incompatible observable.")

            if (
                other.components != observable.components
                or other.phys_units != observable.phys_units
                or other.time_units != observable.time_units
            ):
                raise ValueError(f"{name}: incompatible components or units.")

            if (
                other.geometry is not observable.geometry
                or other.coord_units != observable.coord_units
                or other.coord_frame != observable.coord_frame
            ):
                raise ValueError(f"{name}: incompatible geometry or coord_frame.")

            left_connect = observable.connect or {}
            right_connect = other.connect or {}

            if set(left_connect) != set(right_connect) or any(
                not np.array_equal(left_connect[key], right_connect[key])
                for key in left_connect
            ):
                raise ValueError(f"{name}: incompatible geometry connectivity.")

            for other_coordinates, reference_coordinates in (
                (other.times, observable.times),
                (other.coords, observable.coords),
            ):
                if (other_coordinates is None) != (reference_coordinates is None):
                    raise ValueError(f"{name}: incompatible observables coordinates.")

                if (
                    other_coordinates is not None
                    and reference_coordinates is not None
                    and not np.array_equal(other_coordinates, reference_coordinates)
                ):
                    raise ValueError(f"{name}: incompatible observables coordinates.")

            reference_sim = first.sim[name]
            other_sim = valid_case.sim[name]

            if reference_sim.axes != other_sim.axes or not np.array_equal(
                reference_sim.data, other_sim.data, equal_nan=True
            ):
                raise ValueError(f"{name}: simulation data differs across cases.")

            if not _compare_offsets(reference_sim, other_sim):
                raise ValueError(f"{name}: simulation offsets differ across cases.")

            quantity = valid_case.exp[name]
            quantity_count = (
                quantity.data.shape[quantity.axes.index(EValidAxis.ALEATORY)]
                if EValidAxis.ALEATORY in quantity.axes
                else 1
            )

            if quantity_count != len(run_ids[case_idx]):
                raise ValueError("Observables have inconsistent run counts.")

            if not _compare_offsets(quantity, reference):
                raise ValueError(f"{name}: offsets differ across cases.")

            if EValidAxis.ALEATORY in quantity.axes:
                if quantity.axes != (*base_axes, EValidAxis.ALEATORY):
                    raise ValueError(f"{name}: experimental axes differ.")

                pieces.append(quantity.data)
            else:
                if quantity.axes != base_axes:
                    raise ValueError(f"{name}: experimental axes differ.")

                pieces.append(quantity.data[..., None])

        axes = (*base_axes, EValidAxis.ALEATORY)
        combined[name] = copy(reference)
        combined[name].data = np.concatenate(pieces, axis=-1)
        combined[name].axes = axes

    parameter_keys = set(
        (first.exp_inputs.input_params or {}) if first.exp_inputs is not None else {}
    )
    parameters = {}

    for valid_case in cases:
        inputs = valid_case.exp_inputs

        if (
            set((inputs.input_params or {}) if inputs is not None else {})
            != parameter_keys
        ):
            raise ValueError("Cases have different experimental parameter keys.")

    for key in parameter_keys:
        assert first.exp_inputs is not None and first.exp_inputs.input_params is not None
        reference_parameter = first.exp_inputs.input_params[key]
        base_axes = tuple(
            axis for axis in reference_parameter.axes if axis is not EValidAxis.ALEATORY
        )
        pieces = []

        for case_idx, valid_case in enumerate(cases):
            assert (
                valid_case.exp_inputs is not None
                and valid_case.exp_inputs.input_params is not None
            )
            parameter = valid_case.exp_inputs.input_params[key]

            if EValidAxis.ALEATORY in parameter.axes:
                if parameter.axes != (*base_axes, EValidAxis.ALEATORY):
                    raise ValueError(f"{key}: parameter axes differ.")

                pieces.append(parameter.data)
            else:
                if parameter.axes != base_axes:
                    raise ValueError(f"{key}: parameter axes differ.")

                pieces.append(
                    np.repeat(
                        parameter.data[..., None], len(run_ids[case_idx]), axis=-1
                    )
                )

        parameters[key] = copy(reference_parameter)
        parameters[key].data = np.concatenate(pieces, axis=-1)
        parameters[key].axes = (*base_axes, EValidAxis.ALEATORY)
        parameters[key].offsets = None

    inputs = ValidSampleInputs(
        input_params=parameters or None,
        run_ids=ValidArray(np.concatenate(run_ids), (EValidAxis.ALEATORY,)),
    )

    combined_case = copy(first)
    combined_case.exp = combined
    combined_case.exp_inputs = inputs

    return combined_case

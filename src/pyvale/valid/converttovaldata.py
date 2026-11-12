# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Convert simulation and acquisition data to tagged validation arrays."""

from collections.abc import Iterable

import numpy as np

from pyvale.dataio.expdata import ExpData
from pyvale.dataio.simdata import SimData
from pyvale.sensorsim.field import IField
from pyvale.sensorsim.fieldsampler import sample_field_with_sensor_data
from pyvale.sensorsim.sensordata import SensorData
from pyvale.valid.valtypes import (
    EValAxis,
    EValGeometry,
    EValOperator,
    ValArray,
    ValCase,
    ValDomain,
    ValSampleInputs,
)


def val_from_sim(
    fields: dict[str, IField | None],
    domain: ValDomain,
    runs: Iterable[tuple[tuple[int, ...], SimData]],
    *,
    sample_axes: tuple[EValAxis, ...] = (),
    sample_shape: tuple[int, ...] = (),
    sensor_data: dict[str, SensorData] | None = None,
) -> dict[str, ValArray]:
    """Sample a stream of SimData runs into explicit rectangular sample axes.

    Each run key gives its index on ``sample_axes``. A run is processed and
    discarded before the next one; only extracted sensor values are retained.
    The domain must supply explicit evaluation times, even for steady data.
    """
    if set(fields) != set(domain.observables):
        raise ValueError("Field keys must match domain observables.")
    if len(sample_axes) != len(sample_shape):
        raise ValueError("Sample axes and shape must have the same length.")
    if any(
        axis
        not in (EValAxis.EPISTEMIC_TRANS, EValAxis.EPISTEMIC_NUIS, EValAxis.ALEATORY)
        for axis in sample_axes
    ):
        raise ValueError("Run axes must be uncertainty axes.")
    if tuple(sorted(sample_axes, key=tuple(EValAxis).index)) != sample_axes:
        raise ValueError("Run axes must be in canonical order.")
    if len(set(sample_axes)) != len(sample_axes) or any(n <= 0 for n in sample_shape):
        raise ValueError("Run axes must be unique and nonempty.")

    arrays: dict[str, ValArray] = {}
    for name, observable in domain.observables.items():
        if observable.times is None:
            raise ValueError("Simulation extraction needs explicit evaluation times.")
        if observable.geometry.kind is EValGeometry.GLOBAL:
            if observable.operator is EValOperator.SAMPLE:
                raise ValueError("GLOBAL observables need MEAN or INTEGRAL.")
            point_count = 1
        else:
            assert observable.geometry.coords is not None
            point_count = len(observable.geometry.coords)
            if observable.operator is EValOperator.MEAN:
                if sensor_data is None or name not in sensor_data:
                    raise ValueError(f"{name}: MEAN requires SensorData.")
                sensor = sensor_data[name]
                if sensor.positions is None or not np.array_equal(
                    sensor.positions, observable.geometry.coords
                ):
                    raise ValueError(f"{name}: SensorData positions differ.")
                if sensor.sample_times is None or not np.array_equal(
                    sensor.sample_times, observable.times
                ):
                    raise ValueError(f"{name}: SensorData times differ.")
                if sensor.spatial_averager is None:
                    raise ValueError(f"{name}: MEAN needs a spatial averager.")
            elif observable.operator is not EValOperator.SAMPLE:
                raise ValueError(f"{name}: unsupported spatial operator.")
        shape = (
            point_count,
            len(observable.components),
            len(observable.times),
            *sample_shape,
        )
        arrays[name] = ValArray(
            data=np.empty(shape, dtype=np.float64),
            axes=(EValAxis.POINT, EValAxis.COMPONENT, EValAxis.TIME, *sample_axes),
        )

    seen: set[tuple[int, ...]] = set()
    for selection_idx, sim_data in runs:
        if len(selection_idx) != len(sample_axes):
            raise ValueError("Run index rank does not match sample axes.")
        if any(
            sample_idx < 0 or sample_idx >= size
            for sample_idx, size in zip(selection_idx, sample_shape, strict=True)
        ):
            raise ValueError("Run index is outside sample shape.")
        if selection_idx in seen:
            raise ValueError(f"Duplicate simulation run index: {selection_idx}.")
        seen.add(selection_idx)
        mesh_lower = (
            np.min(sim_data.coords, axis=0) if sim_data.coords is not None else None
        )
        mesh_upper = (
            np.max(sim_data.coords, axis=0) if sim_data.coords is not None else None
        )
        for name, field in fields.items():
            observable = domain.observables[name]
            assert observable.times is not None
            if sim_data.time is None or len(sim_data.time) == 0:
                raise ValueError(f"{name}: simulation has no time coordinates.")
            if (
                observable.times[0] < sim_data.time[0]
                or observable.times[-1] > sim_data.time[-1]
            ):
                raise ValueError(f"{name}: evaluation times leave simulation range.")
            if observable.geometry.kind is EValGeometry.GLOBAL:
                if sim_data.glob_vars is None or sim_data.time is None:
                    raise ValueError("Global extraction needs variables and time.")
                sampled = np.empty(
                    (1, len(observable.components), len(observable.times)),
                    dtype=np.float64,
                )
                for component_idx, key in enumerate(observable.components):
                    sampled[0, component_idx, :] = np.interp(
                        observable.times, sim_data.time, sim_data.glob_vars[key]
                    )
                arrays[name].data[
                    (slice(None), slice(None), slice(None), *selection_idx)
                ] = sampled
                continue
            if field is None:
                raise ValueError(f"{name}: spatial observable needs IField.")
            assert observable.geometry.coords is not None
            if sim_data.coords is None:
                raise ValueError(f"{name}: simulation has no coordinates.")
            points = observable.geometry.coords
            assert mesh_lower is not None and mesh_upper is not None
            if np.any(points < mesh_lower) or np.any(points > mesh_upper):
                raise ValueError(f"{name}: sample points leave simulation bounds.")
            field.set_sim_data(sim_data)
            all_components = field.get_all_components()
            component_idxs = [all_components.index(c) for c in observable.components]
            if observable.operator is EValOperator.MEAN:
                assert sensor_data is not None
                sampled = sample_field_with_sensor_data(field, sensor_data[name])
            else:
                sampled = field.sample_field(
                    observable.geometry.coords,
                    observable.times,
                )
            arrays[name].data[
                (slice(None), slice(None), slice(None), *selection_idx)
            ] = sampled[:, component_idxs, :]
    if len(seen) != (int(np.prod(sample_shape)) if sample_shape else 1):
        raise ValueError("Simulation run tensor is incomplete.")
    return arrays


def val_from_exp(
    sources: dict[str, ExpData],
    domain: ValDomain,
    *,
    steady_window: slice | None = None,
) -> dict[str, ValArray]:
    """Select one acquisition per observable on a registered domain.

    A steady window treats trace values as aleatory observations. Transient
    acquisition times must exactly match the requested domain times.
    """
    if set(sources) != set(domain.observables):
        raise ValueError("Experimental source keys must match the domain.")
    arrays = {}
    for name, exp_data in sources.items():
        observable = domain.observables[name]
        coords = observable.geometry.coords
        if coords is None or exp_data.coords is None:
            raise ValueError(f"{name}: explicit registered coordinates required.")
        if not np.array_equal(coords, exp_data.coords):
            raise ValueError(f"{name}: experimental coordinates differ from domain.")
        if exp_data.fields.ndim != 2 or len(observable.components) != 1:
            raise ValueError(f"{name}: ExpData source must have one component.")
        if exp_data.fields.shape[0] != len(coords):
            raise ValueError(f"{name}: experimental point count mismatch.")
        if steady_window is not None:
            if observable.times is not None:
                raise ValueError("Steady window requires a domain without times.")
            values = exp_data.fields[:, steady_window]
            arrays[name] = ValArray(
                values[:, None, :],
                (EValAxis.POINT, EValAxis.COMPONENT, EValAxis.ALEATORY),
            )
        else:
            if observable.times is None or exp_data.times is None:
                raise ValueError("Transient acquisition requires time coordinates.")
            if not np.array_equal(observable.times, exp_data.times):
                raise ValueError(f"{name}: acquisition times differ from domain.")
            arrays[name] = ValArray(
                exp_data.fields[:, None, :],
                (EValAxis.POINT, EValAxis.COMPONENT, EValAxis.TIME),
            )
    return arrays


def conv_exp_sim_to_data(
    output: dict[tuple[str, ...], np.ndarray],
    *,
    sim_key: str,
    selections: dict[str, tuple[str, int, np.ndarray]],
    measurement_key: str = "meas",
    time_key: str = "sens_times",
) -> list[dict[str, ExpData]]:
    """Split SensorSim experiments into per-run, per-component ExpData.

    ``selections`` maps an observable to (sensor key, component index, xyz
    coordinates). The caller supplies coordinates because SensorSim output
    contains measurements but does not guarantee a nominal position array.
    """
    counts = {
        output[(sim_key, sensor, measurement_key)].shape[0]
        for sensor, _, _ in selections.values()
    }
    if len(counts) != 1:
        raise ValueError("Selected SensorSim arrays have unequal run counts.")
    count = counts.pop()
    converted: list[dict[str, ExpData]] = []
    for run_idx in range(count):
        data = {}
        for name, (sensor, component_idx, coords) in selections.items():
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


def _same_offsets(left: ValArray, right: ValArray) -> bool:
    if left.offsets is None or right.offsets is None:
        return left.offsets is right.offsets
    return (
        left.offsets.axes == right.offsets.axes
        and np.array_equal(left.offsets.lower, right.offsets.lower)
        and np.array_equal(left.offsets.upper, right.offsets.upper)
    )


def _same_sample_inputs(
    left: ValSampleInputs | None,
    right: ValSampleInputs | None,
) -> bool:
    if left is None or right is None:
        return left is right
    if (left.run_ids is None) != (right.run_ids is None):
        return False
    left_arrays = dict(left.parameters or {})
    right_arrays = dict(right.parameters or {})
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
    return left.run_ids.axes == right.run_ids.axes and np.array_equal(
        left.run_ids.data, right.run_ids.data
    )


def combine_val_cases(*cases: ValCase) -> ValCase:
    """Join compatible acquisitions on the experimental aleatory axis.

    Simulations and domain metadata must be identical. For transient cases,
    each complete trace becomes a joint aleatory realization at every time.
    """
    if not cases:
        raise ValueError("At least one val_case is required.")
    first = cases[0]
    combined = {}
    run_ids = []
    for case_idx, val_case in enumerate(cases):
        if set(val_case.domain.observables) != set(first.domain.observables):
            raise ValueError("Cases have different observable keys.")
        if not _same_sample_inputs(val_case.sim_inputs, first.sim_inputs):
            raise ValueError("Simulation sample provenance differs across cases.")
        first_quantity = next(iter(val_case.exp.values()))
        sample_count = (
            first_quantity.data.shape[first_quantity.axes.index(EValAxis.ALEATORY)]
            if EValAxis.ALEATORY in first_quantity.axes
            else 1
        )
        provenance = val_case.exp_inputs
        if provenance is not None and provenance.run_ids is not None:
            identifiers = provenance.run_ids
            if identifiers.axes != (EValAxis.ALEATORY,):
                raise ValueError("Combined run IDs must use the ALEATORY axis.")
            if len(identifiers.data) != sample_count:
                raise ValueError("Run ID count does not match observations.")
            run_ids.append(identifiers.data.astype(str))
        else:
            run_ids.append(np.full(sample_count, f"case_{case_idx}"))
    for name, observable in first.domain.observables.items():
        reference = first.exp[name]
        base_axes = tuple(
            axis for axis in reference.axes if axis is not EValAxis.ALEATORY
        )
        pieces = []
        for case_idx, val_case in enumerate(cases):
            other = val_case.domain.observables.get(name)
            if other is None or other.operator is not observable.operator:
                raise ValueError(f"{name}: incompatible observable.")
            if (
                other.components != observable.components
                or other.phys_units != observable.phys_units
                or other.time_units != observable.time_units
            ):
                raise ValueError(f"{name}: incompatible components or units.")
            if (
                other.geometry.kind is not observable.geometry.kind
                or other.geometry.coord_units != observable.geometry.coord_units
                or other.geometry.coord_frame != observable.geometry.coord_frame
            ):
                raise ValueError(f"{name}: incompatible geometry or coord_frame.")
            left_connect = observable.geometry.connect or {}
            right_connect = other.geometry.connect or {}
            if set(left_connect) != set(right_connect) or any(
                not np.array_equal(left_connect[key], right_connect[key])
                for key in left_connect
            ):
                raise ValueError(f"{name}: incompatible geometry connectivity.")
            for a, b in (
                (other.times, observable.times),
                (other.geometry.coords, observable.geometry.coords),
            ):
                if (a is None) != (b is None):
                    raise ValueError(f"{name}: incompatible domain coordinates.")
                if a is not None and not np.array_equal(a, b):
                    raise ValueError(f"{name}: incompatible domain coordinates.")
            sim_a = first.sim[name]
            sim_b = val_case.sim[name]
            if sim_a.axes != sim_b.axes or not np.array_equal(
                sim_a.data, sim_b.data, equal_nan=True
            ):
                raise ValueError(f"{name}: simulation data differs across cases.")
            if not _same_offsets(sim_a, sim_b):
                raise ValueError(f"{name}: simulation offsets differ across cases.")
            quantity = val_case.exp[name]
            quantity_count = (
                quantity.data.shape[quantity.axes.index(EValAxis.ALEATORY)]
                if EValAxis.ALEATORY in quantity.axes
                else 1
            )
            if quantity_count != len(run_ids[case_idx]):
                raise ValueError("Observables have inconsistent run counts.")
            if not _same_offsets(quantity, reference):
                raise ValueError(f"{name}: offsets differ across cases.")
            if EValAxis.ALEATORY in quantity.axes:
                if quantity.axes != (*base_axes, EValAxis.ALEATORY):
                    raise ValueError(f"{name}: experimental axes differ.")
                pieces.append(quantity.data)
            else:
                if quantity.axes != base_axes:
                    raise ValueError(f"{name}: experimental axes differ.")
                pieces.append(quantity.data[..., None])
        axes = (*base_axes, EValAxis.ALEATORY)
        combined[name] = ValArray(
            np.concatenate(pieces, axis=-1), axes, reference.offsets
        )
    parameter_keys = set(
        (first.exp_inputs.parameters or {}) if first.exp_inputs is not None else {}
    )
    parameters = {}
    for val_case in cases:
        inputs = val_case.exp_inputs
        if (
            set((inputs.parameters or {}) if inputs is not None else {})
            != parameter_keys
        ):
            raise ValueError("Cases have different experimental parameter keys.")
    for key in parameter_keys:
        reference_parameter = first.exp_inputs.parameters[key]
        base_axes = tuple(
            axis for axis in reference_parameter.axes if axis is not EValAxis.ALEATORY
        )
        pieces = []
        for case_idx, val_case in enumerate(cases):
            parameter = val_case.exp_inputs.parameters[key]
            if EValAxis.ALEATORY in parameter.axes:
                if parameter.axes != (*base_axes, EValAxis.ALEATORY):
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
        parameters[key] = ValArray(
            np.concatenate(pieces, axis=-1), (*base_axes, EValAxis.ALEATORY)
        )
    inputs = ValSampleInputs(
        parameters=parameters or None,
        run_ids=ValArray(np.concatenate(run_ids), (EValAxis.ALEATORY,)),
    )
    return ValCase(
        first.domain,
        first.sim,
        combined,
        sim_inputs=first.sim_inputs,
        exp_inputs=inputs,
    )

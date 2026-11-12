# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Run standalone metrics over shared physical and uncertainty axes."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from math import ceil, prod
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from os import cpu_count

import numpy as np

from pyvale.valid.inputvalidation import validate_mavm_config, validate_unique_sample
from pyvale.valid.metrics import validate_relative_error_config
from pyvale.valid.metricstrategy import (
    EMetricType,
    IValMetric,
    MetricMAVM,
    MetricRelErr,
    MetricRMSE,
    calculate_trusted_metric,
)
from pyvale.valid.valtypes import (
    EEpisOffset,
    EMAVMDuplicates,
    EpistemicOffsets,
    EValAxis,
    MAVMResult,
    ValArray,
    ValCase,
    ValDomain,
)


class EValResAxis(StrEnum):
    POINT = "point"
    COMPONENT = "component"
    TIME = "time"
    SIM_TRANS = "sim_trans"
    SIM_NUIS = "sim_nuis"
    EXP_TRANS = "exp_trans"
    EXP_NUIS = "exp_nuis"
    SIM_OFFSET = "sim_offset"
    EXP_OFFSET = "exp_offset"


VAL_RES_AXES = tuple(EValResAxis)


@dataclass(slots=True)
class ValMetricField:
    """Metric arrays in canonical result-axis order.

    Axes are point, component, time, simulation transferable/nuisance epistemic,
    experiment transferable/nuisance epistemic, simulation offset, experiment
    offset. Physical axes are always present; singleton uncertainty axes are
    omitted. ``metric``, ``d_plus`` and ``d_minus`` share the same shape.
    A steady scalar observable commonly has shape (n_points, 1, 1); a transient
    vector has shape (n_points, n_components, n_times) before scenario axes.
    """

    axes: tuple[EValResAxis, ...]
    metric: np.ndarray
    d_plus: np.ndarray | None = None
    d_minus: np.ndarray | None = None
    scenarios: dict[EValResAxis, tuple[EEpisOffset, ...]] | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(axis, EValResAxis) for axis in self.axes):
            raise TypeError("Result axes must be EValResAxis.")
        if (
            len(set(self.axes)) != len(self.axes)
            or tuple(sorted(self.axes, key=VAL_RES_AXES.index)) != self.axes
        ):
            raise ValueError("Result axes must be unique and canonical.")
        if (
            self.axes[:3] != VAL_RES_AXES[:3]
            or self.metric.ndim != len(self.axes)
            or any(size == 0 for size in self.metric.shape)
        ):
            raise ValueError(
                "Metric shape must match nonempty axes including physical axes."
            )
        if (self.d_plus is None) != (self.d_minus is None):
            raise ValueError("Directional arrays must be supplied together.")
        if self.metric.dtype.kind not in "iuf":
            raise TypeError("Metric arrays must contain real numbers.")
        for directional in (self.d_plus, self.d_minus):
            if directional is not None and directional.dtype.kind not in "iuf":
                raise TypeError("Directional metric arrays must contain real numbers.")
            if directional is not None and directional.shape != self.metric.shape:
                raise ValueError("Directional arrays must match metric shape.")
        for axis, scenarios in (self.scenarios or {}).items():
            if (
                not isinstance(axis, EValResAxis)
                or axis not in (EValResAxis.SIM_OFFSET, EValResAxis.EXP_OFFSET)
                or axis not in self.axes
            ):
                raise ValueError("Scenarios must describe stored offset axes.")
            if (
                len(scenarios) != self.metric.shape[self.axes.index(axis)]
                or len(set(scenarios)) != len(scenarios)
                or any(not isinstance(scenario, EEpisOffset) for scenario in scenarios)
            ):
                raise ValueError(
                    "Offset scenarios must match the axis and use unique EEpisOffset values."
                )


@dataclass(slots=True)
class ValResult:
    """Borrow domain/field arrays and snapshot metric configurations."""

    domain: ValDomain
    fields: dict[str, dict[EMetricType, ValMetricField]]
    metric_configs: dict[EMetricType, IValMetric]

    def __post_init__(self) -> None:
        self.metric_configs = deepcopy(self.metric_configs)

    def get(self, observable: str, metric: EMetricType) -> ValMetricField:
        """Select the labelled field for one observable and metric."""
        return self.fields[observable][metric]

    def find_extreme(
        self,
        observable: str,
        metric: EMetricType,
        *,
        maximum: bool = True,
    ) -> tuple[float, dict[EValResAxis, int]]:
        """Locate a finite extreme without reducing the stored result field."""
        field = self.get(observable, metric)
        if not np.any(np.isfinite(field.metric)):
            raise ValueError("Metric field contains no finite results.")
        selectable = np.where(
            np.isfinite(field.metric), field.metric, -np.inf if maximum else np.inf
        )
        flat_idx = np.argmax(selectable) if maximum else np.argmin(selectable)
        selection_idx = tuple(
            int(index) for index in np.unravel_index(flat_idx, field.metric.shape)
        )
        return float(field.metric[selection_idx]), dict(
            zip(field.axes, selection_idx, strict=True)
        )


@dataclass(slots=True)
class ValEngineOpts:
    workers: int | None = 1
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("workers", self.workers), ("chunk_size", self.chunk_size)):
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name} must be an integer or None.")
                if value < 1:
                    raise ValueError(f"{name} must be positive.")


@dataclass(slots=True)
class _SharedArray:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(slots=True)
class _SharedQuantity:
    data: _SharedArray
    axes: tuple[EValAxis, ...]
    lower: _SharedArray | None
    upper: _SharedArray | None
    offset_axes: tuple[EValAxis, ...]


@dataclass(slots=True)
class _Workload:
    name: str
    sim: ValArray
    exp: ValArray
    shape: tuple[int, ...]


@dataclass(slots=True)
class _SharedWorkload:
    name: str
    sim: _SharedQuantity
    exp: _SharedQuantity
    shape: tuple[int, ...]


@dataclass(slots=True)
class _SharedTask:
    workload: _SharedWorkload
    metrics: tuple[IValMetric, ...]
    start: int
    stop: int


class ValidationEngine:
    """Snapshot metric/options configuration and retain epistemic scenarios."""

    __slots__ = ("metrics", "opts")
    metrics: tuple[IValMetric, ...]
    opts: ValEngineOpts

    def __init__(
        self, metrics: list[IValMetric], *, opts: ValEngineOpts | None = None
    ) -> None:
        if not metrics:
            raise ValueError("At least one metric is required.")
        if any(not isinstance(metric, IValMetric) for metric in metrics):
            raise TypeError("Metrics must implement IValMetric.")
        kinds = [metric.get_kind() for metric in metrics]
        if any(not isinstance(kind, EMetricType) for kind in kinds):
            raise TypeError("Metric kinds must be EMetricType.")
        if len(set(kinds)) != len(kinds):
            raise ValueError("Metric types must be unique in one engine.")
        self.metrics = tuple(deepcopy(metrics))
        self.opts = (
            ValEngineOpts()
            if opts is None
            else ValEngineOpts(workers=opts.workers, chunk_size=opts.chunk_size)
        )
        _validate_metric_configs(self.metrics)

    def run(self, val_case: ValCase) -> ValResult:
        """Return all physical, sampled epistemic, and offset scenarios."""
        _validate_metric_configs(self.metrics)
        # Observations are borrowed from validated containers; callers must not
        # mutate them during evaluation. Preflight scenario invariants before output.
        from pyvale.valid.valtools import extract_aleatory_scenarios

        for name in val_case.domain.observables:
            sim, exp = val_case.sim[name], val_case.exp[name]
            requires_pairs = any(
                metric.get_kind() in (EMetricType.RMSE, EMetricType.RELATIVE_ERROR)
                for metric in self.metrics
            )
            if requires_pairs and _get_axis_size(
                sim, EValAxis.ALEATORY
            ) != _get_axis_size(exp, EValAxis.ALEATORY):
                raise ValueError("Paired metrics require equal aleatory counts.")
            if any(
                isinstance(metric, MetricMAVM)
                and metric.duplicates is EMAVMDuplicates.REJECT
                for metric in self.metrics
            ):
                physical_shape = tuple(
                    _get_axis_size(sim, axis)
                    for axis in (EValAxis.POINT, EValAxis.COMPONENT, EValAxis.TIME)
                )
                for quantity in (sim, exp):
                    for point, component, time in np.ndindex(physical_shape):
                        for samples in extract_aleatory_scenarios(
                            quantity,
                            point_idx=point,
                            component_idx=component,
                            time_idx=time,
                            offset=None,
                        ):
                            validate_unique_sample(
                                samples[np.isfinite(samples)], "engine"
                            )
        results: dict[str, dict[EMetricType, ValMetricField]] = {}
        workloads = []
        for name, observable in val_case.domain.observables.items():
            sim = val_case.sim[name]
            exp = val_case.exp[name]
            axis_sizes = {
                EValResAxis.POINT: (
                    1
                    if observable.geometry.coords is None
                    else len(observable.geometry.coords)
                ),
                EValResAxis.COMPONENT: len(observable.components),
                EValResAxis.TIME: 1
                if observable.times is None
                else len(observable.times),
                EValResAxis.SIM_TRANS: _get_axis_size(sim, EValAxis.EPISTEMIC_TRANS),
                EValResAxis.SIM_NUIS: _get_axis_size(sim, EValAxis.EPISTEMIC_NUIS),
                EValResAxis.EXP_TRANS: _get_axis_size(exp, EValAxis.EPISTEMIC_TRANS),
                EValResAxis.EXP_NUIS: _get_axis_size(exp, EValAxis.EPISTEMIC_NUIS),
                EValResAxis.SIM_OFFSET: 3 if sim.offsets is not None else 1,
                EValResAxis.EXP_OFFSET: 3 if exp.offsets is not None else 1,
            }
            retained_axes = []
            for axis in VAL_RES_AXES:
                if axis in VAL_RES_AXES[:3] or axis_sizes[axis] > 1:
                    retained_axes.append(axis)
            axes = tuple(retained_axes)
            shape = tuple(axis_sizes[axis] for axis in axes)
            scenarios = {}
            for axis in (EValResAxis.SIM_OFFSET, EValResAxis.EXP_OFFSET):
                if axis in axes:
                    scenarios[axis] = tuple(EEpisOffset)
            fields = {}
            for metric in self.metrics:
                directional = metric.get_kind() is EMetricType.MAVM
                fields[metric.get_kind()] = ValMetricField(
                    axes=axes,
                    metric=np.full(shape, np.nan),
                    d_plus=np.full(shape, np.nan) if directional else None,
                    d_minus=np.full(shape, np.nan) if directional else None,
                    scenarios=scenarios or None,
                )
            full_shape = tuple(axis_sizes[axis] for axis in VAL_RES_AXES)
            workloads.append(_Workload(name, sim, exp, full_shape))
            results[name] = fields
        workers = self.opts.workers
        if workers is None:
            workers = cpu_count() or 1
        if workers == 1:
            for workload in workloads:
                name, sim, exp, full_shape = (
                    workload.name,
                    workload.sim,
                    workload.exp,
                    workload.shape,
                )
                chunk_size = self.opts.chunk_size or 1024
                for start in range(0, prod(full_shape), chunk_size):
                    calculated = _calculate_chunk(
                        sim,
                        exp,
                        full_shape,
                        self.metrics,
                        start,
                        min(start + chunk_size, prod(full_shape)),
                    )
                    _store_chunk(results[name], self.metrics, start, calculated)
        else:
            total = sum(prod(workload.shape) for workload in workloads)
            chunk_size = self.opts.chunk_size or max(1, ceil(total / (4 * workers)))
            with ExitStack() as stack:
                shared_workloads = []
                for workload in workloads:
                    shared_workloads.append(
                        _SharedWorkload(
                            workload.name,
                            _share_quantity(workload.sim, stack),
                            _share_quantity(workload.exp, stack),
                            workload.shape,
                        )
                    )
                with get_context("spawn").Pool(workers) as pool:
                    tasks = (
                        _SharedTask(
                            workload,
                            self.metrics,
                            start,
                            min(start + chunk_size, prod(workload.shape)),
                        )
                        for workload in shared_workloads
                        for start in range(0, prod(workload.shape), chunk_size)
                    )
                    for name, start, calculated in pool.imap_unordered(
                        _calculate_shared_chunk, tasks, chunksize=1
                    ):
                        _store_chunk(results[name], self.metrics, start, calculated)
        configs = {metric.get_kind(): metric for metric in self.metrics}
        return ValResult(domain=val_case.domain, fields=results, metric_configs=configs)


def _get_axis_size(quantity: ValArray, axis: EValAxis) -> int:
    if axis not in quantity.axes:
        return 1
    return quantity.data.shape[quantity.axes.index(axis)]


def _share_array(array: np.ndarray, stack: ExitStack) -> _SharedArray:
    contiguous = np.asarray(array)
    memory = SharedMemory(create=True, size=contiguous.nbytes)
    stack.callback(memory.unlink)
    stack.callback(memory.close)
    shared = np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=memory.buf)
    shared[...] = contiguous
    return _SharedArray(memory.name, contiguous.shape, contiguous.dtype)


def _share_quantity(quantity: ValArray, stack: ExitStack) -> _SharedQuantity:
    offsets = quantity.offsets
    return _SharedQuantity(
        _share_array(quantity.data, stack),
        quantity.axes,
        None if offsets is None else _share_array(np.asarray(offsets.lower), stack),
        None if offsets is None else _share_array(np.asarray(offsets.upper), stack),
        () if offsets is None else offsets.axes,
    )


def _attach_array(description: _SharedArray, stack: ExitStack) -> np.ndarray:
    memory = SharedMemory(name=description.name)
    stack.callback(memory.close)
    array = np.ndarray(description.shape, dtype=description.dtype, buffer=memory.buf)
    array.flags.writeable = False
    return array


def _attach_quantity(description: _SharedQuantity, stack: ExitStack) -> ValArray:
    offsets = None
    if description.lower is not None:
        assert description.upper is not None
        offsets = EpistemicOffsets(
            _attach_array(description.lower, stack),
            _attach_array(description.upper, stack),
            description.offset_axes,
        )
    return ValArray(_attach_array(description.data, stack), description.axes, offsets)


def _calculate_shared_chunk(
    task: _SharedTask,
) -> tuple[str, int, np.ndarray]:
    workload = task.workload
    with ExitStack() as stack:
        calculated = _calculate_chunk(
            _attach_quantity(workload.sim, stack),
            _attach_quantity(workload.exp, stack),
            workload.shape,
            task.metrics,
            task.start,
            task.stop,
        )
    return workload.name, task.start, calculated


def _calculate_chunk(
    sim: ValArray,
    exp: ValArray,
    shape: tuple[int, ...],
    metrics: tuple[IValMetric, ...],
    start: int,
    stop: int,
) -> np.ndarray:
    requires_pairs = any(
        type(metric) in (MetricRMSE, MetricRelErr) for metric in metrics
    )
    calculated = np.full((len(metrics), 3, stop - start), np.nan)
    for local_idx, flat_idx in enumerate(range(start, stop)):
        indices = tuple(int(index) for index in np.unravel_index(flat_idx, shape))
        (
            point,
            component,
            time,
            sim_trans,
            sim_nuis,
            exp_trans,
            exp_nuis,
            sim_off,
            exp_off,
        ) = indices
        sim_samples = sim._select_aleatory(
            point, component, time, sim_trans, sim_nuis, EEpisOffset(sim_off)
        )
        exp_samples = exp._select_aleatory(
            point, component, time, exp_trans, exp_nuis, EEpisOffset(exp_off)
        )
        sim_present = ~np.isnan(sim_samples)
        exp_present = ~np.isnan(exp_samples)
        simulation = np.asarray(sim_samples[sim_present], dtype=np.float64)
        experiment = np.asarray(exp_samples[exp_present], dtype=np.float64)
        if not len(simulation) or not len(experiment):
            continue
        paired_simulation, paired_experiment = simulation, experiment
        if requires_pairs:
            paired = sim_present & exp_present
            paired_simulation = np.asarray(sim_samples[paired], dtype=np.float64)
            paired_experiment = np.asarray(exp_samples[paired], dtype=np.float64)
        for metric_idx, metric in enumerate(metrics):
            if type(metric) in (MetricRMSE, MetricRelErr):
                metric_result = calculate_trusted_metric(
                    metric, paired_simulation, paired_experiment
                )
            else:
                metric_result = calculate_trusted_metric(metric, simulation, experiment)
            if isinstance(metric_result, MAVMResult):
                calculated[metric_idx, :, local_idx] = (
                    metric_result.d_total,
                    metric_result.d_plus,
                    metric_result.d_minus,
                )
            else:
                calculated[metric_idx, 0, local_idx] = metric_result.value
    return calculated


def _store_chunk(
    fields: dict[EMetricType, ValMetricField],
    metrics: tuple[IValMetric, ...],
    start: int,
    calculated: np.ndarray,
) -> None:
    stop = start + calculated.shape[-1]
    for metric_idx, metric in enumerate(metrics):
        field = fields[metric.get_kind()]
        field.metric.reshape(-1)[start:stop] = calculated[metric_idx, 0]
        if field.d_plus is not None:
            assert field.d_minus is not None
            field.d_plus.reshape(-1)[start:stop] = calculated[metric_idx, 1]
            field.d_minus.reshape(-1)[start:stop] = calculated[metric_idx, 2]


def _validate_metric_configs(metrics: tuple[IValMetric, ...]) -> None:
    for metric in metrics:
        if isinstance(metric, MetricMAVM):
            validate_mavm_config(metric.alpha, metric.duplicates, metric.tol)
        elif isinstance(metric, MetricRelErr):
            validate_relative_error_config(metric.normalization, metric.constant)

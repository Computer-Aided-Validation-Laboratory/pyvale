# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Run standalone metrics over shared physical and uncertainty axes."""

from dataclasses import dataclass
from enum import StrEnum
from contextlib import ExitStack
from math import ceil, prod
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from os import cpu_count

import numpy as np

from pyvale.valid.metrics import MAVMResult
from pyvale.valid.metricstrategy import EMetricType, IValMetric
from pyvale.valid.valtypes import (
    EEpisOffset, EValAxis, EpistemicOffsets, ValArray, ValCase, ValDomain,
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


@dataclass(slots=True)
class ValResult:
    domain: ValDomain
    fields: dict[str, dict[EMetricType, ValMetricField]]
    metric_configs: dict[EMetricType, IValMetric]

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
        selection_idx = np.unravel_index(flat_idx, field.metric.shape)
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


class ValidationEngine:
    """Evaluate configured metrics without reducing epistemic scenarios."""

    __slots__ = ("metrics", "opts")

    def __init__(
        self, metrics: list[IValMetric], *, opts: ValEngineOpts | None = None
    ) -> None:
        if not metrics:
            raise ValueError("At least one metric is required.")
        kinds = [metric.get_kind() for metric in metrics]
        if len(set(kinds)) != len(kinds):
            raise ValueError("Metric types must be unique in one engine.")
        self.metrics = metrics
        self.opts = ValEngineOpts() if opts is None else opts

    def run(self, val_case: ValCase) -> ValResult:
        """Return all physical, sampled epistemic, and offset scenarios."""
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
                EValResAxis.SIM_TRANS: _axis_size(sim, EValAxis.EPISTEMIC_TRANS),
                EValResAxis.SIM_NUIS: _axis_size(sim, EValAxis.EPISTEMIC_NUIS),
                EValResAxis.EXP_TRANS: _axis_size(exp, EValAxis.EPISTEMIC_TRANS),
                EValResAxis.EXP_NUIS: _axis_size(exp, EValAxis.EPISTEMIC_NUIS),
                EValResAxis.SIM_OFFSET: 3 if sim.offsets is not None else 1,
                EValResAxis.EXP_OFFSET: 3 if exp.offsets is not None else 1,
            }
            axes = tuple(
                axis
                for axis in VAL_RES_AXES
                if axis in VAL_RES_AXES[:3] or axis_sizes[axis] > 1
            )
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
            workloads.append((name, sim, exp, full_shape))
            results[name] = fields
        workers = self.opts.workers
        if workers is None:
            workers = cpu_count() or 1
        if workers == 1:
            for name, sim, exp, full_shape in workloads:
                chunk_size = self.opts.chunk_size or 1024
                for start in range(0, prod(full_shape), chunk_size):
                    calculated = _calculate_chunk(
                        sim, exp, full_shape, self.metrics, start,
                        min(start + chunk_size, prod(full_shape)),
                    )
                    _store_chunk(results[name], self.metrics, start, calculated)
        else:
            total = sum(prod(workload[3]) for workload in workloads)
            chunk_size = self.opts.chunk_size or max(1, ceil(total / (4 * workers)))
            with ExitStack() as stack:
                shared_workloads = [
                    (
                        name, _share_quantity(sim, stack),
                        _share_quantity(exp, stack), shape,
                    )
                    for name, sim, exp, shape in workloads
                ]
                with get_context("spawn").Pool(workers) as pool:
                    tasks = (
                        (name, sim, exp, shape, self.metrics, start,
                         min(start + chunk_size, prod(shape)))
                        for name, sim, exp, shape in shared_workloads
                        for start in range(0, prod(shape), chunk_size)
                    )
                    for name, start, calculated in pool.imap_unordered(
                        _calculate_shared_chunk, tasks, chunksize=1
                    ):
                        _store_chunk(results[name], self.metrics, start, calculated)
        configs = {metric.get_kind(): metric for metric in self.metrics}
        return ValResult(domain=val_case.domain, fields=results, metric_configs=configs)


def _axis_size(quantity: ValArray, axis: EValAxis) -> int:
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
        _share_array(quantity.data, stack), quantity.axes,
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
            _attach_array(description.upper, stack), description.offset_axes,
        )
    return ValArray(_attach_array(description.data, stack), description.axes, offsets)


def _calculate_shared_chunk(
    task: tuple[str, _SharedQuantity, _SharedQuantity, tuple[int, ...],
                list[IValMetric], int, int],
) -> tuple[str, int, np.ndarray]:
    name, sim, exp, shape, metrics, start, stop = task
    with ExitStack() as stack:
        calculated = _calculate_chunk(
            _attach_quantity(sim, stack), _attach_quantity(exp, stack),
            shape, metrics, start, stop,
        )
    return name, start, calculated


def _calculate_chunk(
    sim: ValArray, exp: ValArray, shape: tuple[int, ...],
    metrics: list[IValMetric], start: int, stop: int,
) -> np.ndarray:
    calculated = np.full((len(metrics), 3, stop - start), np.nan)
    for local_idx, flat_idx in enumerate(range(start, stop)):
        indices = tuple(int(index) for index in np.unravel_index(flat_idx, shape))
        (
            point, component, time, sim_trans, sim_nuis,
            exp_trans, exp_nuis, sim_off, exp_off,
        ) = indices
        sim_samples = sim.select_aleatory(
            point, component, time, sim_trans, sim_nuis, sim_off
        )
        exp_samples = exp.select_aleatory(
            point, component, time, exp_trans, exp_nuis, exp_off
        )
        if not np.any(np.isfinite(sim_samples)) or not np.any(np.isfinite(exp_samples)):
            continue
        for metric_idx, metric in enumerate(metrics):
            metric_result = metric.calc(sim_samples, exp_samples)
            if isinstance(metric_result, MAVMResult):
                calculated[metric_idx, :, local_idx] = (
                    metric_result.d_total, metric_result.d_plus, metric_result.d_minus
                )
            else:
                calculated[metric_idx, 0, local_idx] = metric_result.value
    return calculated


def _store_chunk(
    fields: dict[EMetricType, ValMetricField], metrics: list[IValMetric],
    start: int, calculated: np.ndarray,
) -> None:
    stop = start + calculated.shape[-1]
    for metric_idx, metric in enumerate(metrics):
        field = fields[metric.get_kind()]
        field.metric.reshape(-1)[start:stop] = calculated[metric_idx, 0]
        if field.d_plus is not None:
            assert field.d_minus is not None
            field.d_plus.reshape(-1)[start:stop] = calculated[metric_idx, 1]
            field.d_minus.reshape(-1)[start:stop] = calculated[metric_idx, 2]

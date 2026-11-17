# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Run standalone metrics over shared physical and uncertainty axes."""

from copy import copy, deepcopy
from dataclasses import dataclass
from math import prod
from multiprocessing.pool import AsyncResult, Pool
from os import cpu_count

import numpy as np

from pyvale.valid.cython.mavm import (
    _calculate_prepared_mavm,
    _prepare_mavm_distribution,
    _prepare_mavm_exp,
    _PreparedECDF,
    _PreparedMAVMExp,
)
from pyvale.valid.metrics import validate_relative_error_config
from pyvale.valid.metricstrategy import (
    EMetricComparison,
    EMetricType,
    IValidMetric,
    MetricAVM,
    MetricCVM,
    MetricKS,
    MetricMAVM,
    MetricRelErr,
    MetricRMSE,
    calculate_trusted_metric,
)
from pyvale.valid.validchecks import validate_unique_sample
from pyvale.valid.validecdf import extract_aleatory_scenarios
from pyvale.valid.validtypes import (
    VALID_RES_AXES,
    EEpisOffset,
    EMAVMDuplicates,
    EMetricOutput,
    EValidAxis,
    EValidResAxis,
    ValidArray,
    ValidCase,
    ValidMetricField,
    ValidResult,
    validate_mavm_config,
)


@dataclass(slots=True)
class ValidEngineOpts:
    workers: int | None = 1

    def __post_init__(self) -> None:
        if self.workers is not None:
            if isinstance(self.workers, bool) or not isinstance(self.workers, int):
                raise TypeError("workers must be an integer or None.")
            if self.workers < 1:
                raise ValueError("workers must be positive.")


@dataclass(slots=True)
class _Workload:
    name: str
    sim: ValidArray
    exp: ValidArray
    shape: tuple[int, ...]
    start: int
    stop: int


class ValidEngine:
    """Snapshot metric/options configuration and retain epistemic scenarios."""

    __slots__ = ("metrics", "opts")
    metrics: tuple[IValidMetric, ...]
    opts: ValidEngineOpts

    def __init__(
        self, metrics: list[IValidMetric], *, opts: ValidEngineOpts | None = None
    ) -> None:
        if not metrics:
            raise ValueError("At least one metric is required.")
        if any(not isinstance(metric, IValidMetric) for metric in metrics):
            raise TypeError("Metrics must implement IValidMetric.")
        kinds = [metric.get_kind() for metric in metrics]
        if any(not isinstance(kind, EMetricType) for kind in kinds):
            raise TypeError("Metric kinds must be EMetricType.")
        if len(set(kinds)) != len(kinds):
            raise ValueError("Metric types must be unique in one engine.")
        for metric in metrics:
            outputs = metric.get_outputs()
            if not outputs or len(set(outputs)) != len(outputs):
                raise ValueError("Metric outputs must be nonempty and unique.")
            if any(not isinstance(output, EMetricOutput) for output in outputs):
                raise TypeError("Metric outputs must be EMetricOutput.")
        self.metrics = tuple(deepcopy(metrics))
        self.opts = (
            ValidEngineOpts() if opts is None else ValidEngineOpts(workers=opts.workers)
        )

    def run(self, valid_case: ValidCase) -> ValidResult:
        """Return all physical, sampled epistemic, and offset scenarios."""
        _validate_metric_configs(self.metrics)
        # Observations are borrowed from validated containers; callers must not
        # mutate them during evaluation. Preflight scenario invariants before output.

        for name in valid_case.observables:
            sim, exp = valid_case.sim[name], valid_case.exp[name]
            requires_pairs = any(
                metric.get_comparison() is EMetricComparison.PAIRED
                for metric in self.metrics
            )
            if requires_pairs and _get_axis_size(
                sim, EValidAxis.ALEATORY
            ) != _get_axis_size(exp, EValidAxis.ALEATORY):
                raise ValueError("Paired metrics require equal aleatory counts.")
            if any(
                isinstance(metric, MetricMAVM)
                and metric.duplicates is EMAVMDuplicates.REJECT
                for metric in self.metrics
            ):
                physical_shape = tuple(
                    _get_axis_size(sim, axis)
                    for axis in (
                        EValidAxis.POINT,
                        EValidAxis.COMPONENT,
                        EValidAxis.TIME,
                    )
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

        results: dict[str, dict[EMetricType, ValidMetricField]] = {}
        workloads: list[_Workload] = []

        for name, observable in valid_case.observables.items():
            sim = valid_case.sim[name]
            exp = valid_case.exp[name]
            leng_pt = 1
            if observable.coords is not None:
                leng_pt = observable.coords.shape[0]
            leng_comp = len(observable.components)
            leng_time = 1
            if observable.times is not None:
                leng_time = observable.times.shape[0]
            axis_sizes = {
                EValidResAxis.POINT: leng_pt,
                EValidResAxis.COMPONENT: leng_comp,
                EValidResAxis.TIME: leng_time,
                EValidResAxis.SIM_TRANS: _get_axis_size(
                    sim, EValidAxis.EPISTEMIC_TRANS
                ),
                EValidResAxis.SIM_NUIS: _get_axis_size(sim, EValidAxis.EPISTEMIC_NUIS),
                EValidResAxis.EXP_TRANS: _get_axis_size(
                    exp, EValidAxis.EPISTEMIC_TRANS
                ),
                EValidResAxis.EXP_NUIS: _get_axis_size(exp, EValidAxis.EPISTEMIC_NUIS),
                EValidResAxis.SIM_OFFSET: 3 if sim.offsets is not None else 1,
                EValidResAxis.EXP_OFFSET: 3 if exp.offsets is not None else 1,
            }
            retained_axes = []
            for axis in VALID_RES_AXES:
                if axis in VALID_RES_AXES[:3] or axis_sizes[axis] > 1:
                    retained_axes.append(axis)
            axes = tuple(retained_axes)
            shape = tuple(axis_sizes[axis] for axis in axes)
            scenarios = {}
            for axis in (EValidResAxis.SIM_OFFSET, EValidResAxis.EXP_OFFSET):
                if axis in axes:
                    scenarios[axis] = tuple(EEpisOffset)
            metric_fields = {}
            for metric in self.metrics:
                metric_fields[metric.get_kind()] = ValidMetricField(
                    axes=axes,
                    outputs={
                        output: np.full(shape, np.nan)
                        for output in metric.get_outputs()
                    },
                    scenarios=scenarios or None,
                )
            full_shape = tuple(axis_sizes[axis] for axis in VALID_RES_AXES)
            workloads.append(_Workload(name, sim, exp, full_shape, 0, prod(full_shape)))
            results[name] = metric_fields
        workers = self.opts.workers
        if workers is None:
            workers = cpu_count() or 1
        total = sum(workload.stop for workload in workloads)
        workers = min(workers, total)
        if workers == 1:
            for name, start, calculated in _calculate_partition(
                workloads, self.metrics
            ):
                _store_chunk(results[name], self.metrics, start, calculated)
        else:
            with Pool(workers) as pool:
                pending: list[AsyncResult] = []
                for worker_idx in range(workers):
                    partition_start = total * worker_idx // workers
                    partition_stop = total * (worker_idx + 1) // workers
                    partition: list[_Workload] = []
                    observable_start = 0
                    for workload in workloads:
                        start = max(0, partition_start - observable_start)
                        stop = min(workload.stop, partition_stop - observable_start)
                        observable_start += workload.stop
                        if start >= stop:
                            continue
                        cells_per_point = prod(workload.shape[1:])
                        point_start = start // cells_per_point
                        point_stop = (stop - 1) // cells_per_point + 1
                        partition.append(
                            _Workload(
                                workload.name,
                                _slice_points(workload.sim, point_start, point_stop),
                                _slice_points(workload.exp, point_start, point_stop),
                                (point_stop - point_start, *workload.shape[1:]),
                                start,
                                stop,
                            )
                        )
                    pending.append(
                        pool.apply_async(
                            _calculate_partition, args=(partition, self.metrics)
                        )
                    )
                for task in pending:
                    for name, start, calculated in task.get():
                        _store_chunk(results[name], self.metrics, start, calculated)
        configs = {metric.get_kind(): metric for metric in self.metrics}
        return ValidResult(
            observables=valid_case.observables, metrics=results, metric_configs=configs
        )


def _get_axis_size(quantity: ValidArray, axis: EValidAxis) -> int:
    if axis not in quantity.axes:
        return 1
    return quantity.data.shape[quantity.axes.index(axis)]


def _slice_points(quantity: ValidArray, start: int, stop: int) -> ValidArray:
    if start == 0 and stop == _get_axis_size(quantity, EValidAxis.POINT):
        return quantity
    selection = tuple(
        slice(start, stop) if axis is EValidAxis.POINT else slice(None)
        for axis in quantity.axes
    )
    offsets = quantity.offsets
    if offsets is not None and EValidAxis.POINT in offsets.axes:
        offset_selection = tuple(
            slice(start, stop) if axis is EValidAxis.POINT else slice(None)
            for axis in offsets.axes
        )
        offsets = copy(offsets)
        offsets.lower = np.asarray(offsets.lower)[offset_selection]
        offsets.upper = np.asarray(offsets.upper)[offset_selection]
    sliced = copy(quantity)
    sliced.data = quantity.data[selection]
    sliced.offsets = offsets
    return sliced


def _calculate_partition(
    workloads: list[_Workload], metrics: tuple[IValidMetric, ...]
) -> list[tuple[str, int, np.ndarray]]:
    calculated_workloads: list[tuple[str, int, np.ndarray]] = []
    check_output_count = tuple(
        type(metric)
        not in (MetricMAVM, MetricAVM, MetricKS, MetricCVM, MetricRMSE, MetricRelErr)
        for metric in metrics
    )
    paired_metrics = tuple(
        metric.get_comparison() is EMetricComparison.PAIRED for metric in metrics
    )
    requires_pairs = any(paired_metrics)
    for workload in workloads:
        sim_array, exp_array, shape = workload.sim, workload.exp, workload.shape
        start = workload.start % prod(shape[1:])
        stop = start + workload.stop - workload.start
        model_distributions: dict[tuple[int, ...], _PreparedECDF] = {}
        exp_distributions: dict[tuple[int, ...], _PreparedMAVMExp] = {}
        metric_outputs = tuple(metric.get_outputs() for metric in metrics)
        output_count = sum(len(outputs) for outputs in metric_outputs)
        calculated = np.full((output_count, stop - start), np.nan)
        physical_cell: tuple[int, int, int] | None = None
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
            current_cell = (point, component, time)
            if current_cell != physical_cell:
                model_distributions.clear()
                exp_distributions.clear()
                physical_cell = current_cell
            sim_samples = sim_array._select_aleatory(
                point, component, time, sim_trans, sim_nuis, EEpisOffset(sim_off)
            )
            exp_samples = exp_array._select_aleatory(
                point, component, time, exp_trans, exp_nuis, EEpisOffset(exp_off)
            )
            sim_present = ~np.isnan(sim_samples)
            exp_present = ~np.isnan(exp_samples)
            sim = np.asarray(sim_samples[sim_present], dtype=np.float64)
            exp = np.asarray(exp_samples[exp_present], dtype=np.float64)
            if not len(sim) or not len(exp):
                continue
            paired_sim, paired_exp = sim, exp
            if requires_pairs:
                paired = sim_present & exp_present
                paired_sim = np.asarray(sim_samples[paired], dtype=np.float64)
                paired_exp = np.asarray(exp_samples[paired], dtype=np.float64)
            output_start = 0
            for metric, outputs, check_count, paired_metric in zip(
                metrics, metric_outputs, check_output_count, paired_metrics, strict=True
            ):
                if type(metric) is MetricMAVM:
                    model_key = (point, component, time, sim_trans, sim_nuis, sim_off)
                    exp_key = (
                        point,
                        component,
                        time,
                        exp_trans,
                        exp_nuis,
                        exp_off,
                    )
                    if model_key not in model_distributions:
                        model_distributions[model_key] = _prepare_mavm_distribution(sim)
                    if exp_key not in exp_distributions:
                        exp_distributions[exp_key] = _prepare_mavm_exp(
                            exp, metric.alpha
                        )
                    metric_result = _calculate_prepared_mavm(
                        model_distributions[model_key],
                        exp_distributions[exp_key],
                        metric.alpha,
                        metric.duplicates,
                        metric.tol,
                    )
                elif paired_metric:
                    metric_result = calculate_trusted_metric(
                        metric, paired_sim, paired_exp
                    )
                else:
                    metric_result = calculate_trusted_metric(metric, sim, exp)
                output_stop = output_start + len(outputs)
                output_values = metric.extract_outputs(metric_result)
                if check_count and len(output_values) != len(outputs):
                    raise ValueError(
                        "Calculated outputs must match the metric declaration."
                    )
                calculated[output_start:output_stop, local_idx] = output_values
                output_start = output_stop
        calculated_workloads.append((workload.name, workload.start, calculated))
    return calculated_workloads


def _store_chunk(
    metric_fields: dict[EMetricType, ValidMetricField],
    metrics: tuple[IValidMetric, ...],
    start: int,
    calculated: np.ndarray,
) -> None:
    stop = start + calculated.shape[-1]
    output_idx = 0
    for metric in metrics:
        field = metric_fields[metric.get_kind()]
        for output in metric.get_outputs():
            field.outputs[output].reshape(-1)[start:stop] = calculated[output_idx]
            output_idx += 1


def _validate_metric_configs(metrics: tuple[IValidMetric, ...]) -> None:
    for metric in metrics:
        if isinstance(metric, MetricMAVM):
            validate_mavm_config(metric.alpha, metric.duplicates, metric.tol)
        elif isinstance(metric, MetricRelErr):
            validate_relative_error_config(metric.normalisation, metric.constant)

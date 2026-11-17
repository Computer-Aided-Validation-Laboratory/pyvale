# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Run standalone metrics over shared physical and uncertainty axes."""

from copy import copy, deepcopy
from dataclasses import dataclass
from enum import StrEnum
from math import prod
from multiprocessing.pool import AsyncResult, Pool
from os import cpu_count

import numpy as np

from pyvale.valid.cython.mavm import (
    _calculate_prepared_mavm,
    _prepare_mavm_distribution,
    _prepare_mavm_experiment,
    _PreparedECDF,
    _PreparedMAVMExperiment,
)
from pyvale.valid.inputvalidation import validate_mavm_config, validate_unique_sample
from pyvale.valid.metrics import validate_relative_error_config
from pyvale.valid.metricstrategy import (
    EMetricType,
    IValMetric,
    MetricAVM,
    MetricCVM,
    MetricKS,
    MetricMAVM,
    MetricRelErr,
    MetricRMSE,
    calculate_trusted_metric,
)
from pyvale.valid.valtypes import (
    EEpisOffset,
    EMAVMDuplicates,
    EMetricOutput,
    EValAxis,
    ValArray,
    ValCase,
    ValObservable,
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
    omitted. All output arrays share the same shape.
    A steady scalar observable commonly has shape (n_points, 1, 1); a transient
    vector has shape (n_points, n_components, n_times) before scenario axes.
    """

    axes: tuple[EValResAxis, ...]
    outputs: dict[EMetricOutput, np.ndarray]
    scenarios: dict[EValResAxis, tuple[EEpisOffset, ...]] | None = None


def resolve_metric_output(
    field: ValMetricField, output: EMetricOutput | None
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
class ValResult:
    """Borrow observables/field arrays and snapshot metric configurations."""

    observables: dict[str, ValObservable]
    metrics: dict[str, dict[EMetricType, ValMetricField]]
    metric_configs: dict[EMetricType, IValMetric]

    def __post_init__(self) -> None:
        self.metric_configs = deepcopy(self.metric_configs)

    def find_extreme(
        self,
        observable: str,
        metric: EMetricType,
        *,
        output: EMetricOutput | None = None,
        maximum: bool = True,
    ) -> tuple[float, dict[EValResAxis, int]]:
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


@dataclass(slots=True)
class ValEngineOpts:
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
    sim: ValArray
    exp: ValArray
    shape: tuple[int, ...]
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
        for metric in metrics:
            outputs = metric.get_outputs()
            if not outputs or len(set(outputs)) != len(outputs):
                raise ValueError("Metric outputs must be nonempty and unique.")
            if any(not isinstance(output, EMetricOutput) for output in outputs):
                raise TypeError("Metric outputs must be EMetricOutput.")
        self.metrics = tuple(deepcopy(metrics))
        self.opts = (
            ValEngineOpts() if opts is None else ValEngineOpts(workers=opts.workers)
        )

    def run(self, val_case: ValCase) -> ValResult:
        """Return all physical, sampled epistemic, and offset scenarios."""
        _validate_metric_configs(self.metrics)
        # Observations are borrowed from validated containers; callers must not
        # mutate them during evaluation. Preflight scenario invariants before output.
        from pyvale.valid.valtools import extract_aleatory_scenarios

        for name in val_case.observables:
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
        workloads: list[_Workload] = []

        for name, observable in val_case.observables.items():
            sim = val_case.sim[name]
            exp = val_case.exp[name]
            leng_pt = 1
            if observable.coords is not None:
                leng_pt = observable.coords.shape[0]
            leng_comp = len(observable.components)
            leng_time = 1
            if observable.times is not None:
                leng_time = observable.times.shape[0]
            axis_sizes = {
                EValResAxis.POINT: leng_pt,
                EValResAxis.COMPONENT: leng_comp,
                EValResAxis.TIME: leng_time,
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
            metric_fields = {}
            for metric in self.metrics:
                metric_fields[metric.get_kind()] = ValMetricField(
                    axes=axes,
                    outputs={
                        output: np.full(shape, np.nan)
                        for output in metric.get_outputs()
                    },
                    scenarios=scenarios or None,
                )
            full_shape = tuple(axis_sizes[axis] for axis in VAL_RES_AXES)
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
        return ValResult(
            observables=val_case.observables, metrics=results, metric_configs=configs
        )


def _get_axis_size(quantity: ValArray, axis: EValAxis) -> int:
    if axis not in quantity.axes:
        return 1
    return quantity.data.shape[quantity.axes.index(axis)]


def _slice_points(quantity: ValArray, start: int, stop: int) -> ValArray:
    if start == 0 and stop == _get_axis_size(quantity, EValAxis.POINT):
        return quantity
    selection = tuple(
        slice(start, stop) if axis is EValAxis.POINT else slice(None)
        for axis in quantity.axes
    )
    offsets = quantity.offsets
    if offsets is not None and EValAxis.POINT in offsets.axes:
        offset_selection = tuple(
            slice(start, stop) if axis is EValAxis.POINT else slice(None)
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
    workloads: list[_Workload], metrics: tuple[IValMetric, ...]
) -> list[tuple[str, int, np.ndarray]]:
    calculated_workloads: list[tuple[str, int, np.ndarray]] = []
    check_output_count = tuple(
        type(metric)
        not in (MetricMAVM, MetricAVM, MetricKS, MetricCVM, MetricRMSE, MetricRelErr)
        for metric in metrics
    )
    for workload in workloads:
        sim, exp, shape = workload.sim, workload.exp, workload.shape
        start = workload.start % prod(shape[1:])
        stop = start + workload.stop - workload.start
        requires_pairs = any(
            type(metric) in (MetricRMSE, MetricRelErr) for metric in metrics
        )
        model_distributions: dict[tuple[int, ...], _PreparedECDF] = {}
        experimental_distributions: dict[tuple[int, ...], _PreparedMAVMExperiment] = {}
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
                experimental_distributions.clear()
                physical_cell = current_cell
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
            output_start = 0
            for metric, outputs, check_count in zip(
                metrics, metric_outputs, check_output_count, strict=True
            ):
                if type(metric) is MetricMAVM:
                    model_key = (point, component, time, sim_trans, sim_nuis, sim_off)
                    experimental_key = (
                        point,
                        component,
                        time,
                        exp_trans,
                        exp_nuis,
                        exp_off,
                    )
                    if model_key not in model_distributions:
                        model_distributions[model_key] = _prepare_mavm_distribution(
                            simulation
                        )
                    if experimental_key not in experimental_distributions:
                        experimental_distributions[experimental_key] = (
                            _prepare_mavm_experiment(experiment, metric.alpha)
                        )
                    metric_result = _calculate_prepared_mavm(
                        model_distributions[model_key],
                        experimental_distributions[experimental_key],
                        metric.alpha,
                        metric.duplicates,
                        metric.tol,
                    )
                elif type(metric) in (MetricRMSE, MetricRelErr):
                    metric_result = calculate_trusted_metric(
                        metric, paired_simulation, paired_experiment
                    )
                else:
                    metric_result = calculate_trusted_metric(
                        metric, simulation, experiment
                    )
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
    metric_fields: dict[EMetricType, ValMetricField],
    metrics: tuple[IValMetric, ...],
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


def _validate_metric_configs(metrics: tuple[IValMetric, ...]) -> None:
    for metric in metrics:
        if isinstance(metric, MetricMAVM):
            validate_mavm_config(metric.alpha, metric.duplicates, metric.tol)
        elif isinstance(metric, MetricRelErr):
            validate_relative_error_config(metric.normalisation, metric.constant)

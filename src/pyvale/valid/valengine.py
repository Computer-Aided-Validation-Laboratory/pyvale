# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Run standalone metrics over shared physical and uncertainty axes."""

from dataclasses import dataclass
from enum import StrEnum
from itertools import product

import numpy as np

from pyvale.valid.metrics import MAVMResult
from pyvale.valid.metricstrategy import EMetricType, IValMetric
from pyvale.valid.valtypes import EEpisOffset, EValAxis, ValArray, ValCase, ValDomain


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
        flat = np.argmax(selectable) if maximum else np.argmin(selectable)
        index = np.unravel_index(flat, field.metric.shape)
        return float(field.metric[index]), dict(zip(field.axes, index, strict=True))


class ValidationEngine:
    """Evaluate configured metrics without reducing epistemic scenarios."""

    __slots__ = ("metrics",)

    def __init__(self, metrics: list[IValMetric]) -> None:
        if not metrics:
            raise ValueError("At least one metric is required.")
        kinds = [metric.get_kind() for metric in metrics]
        if len(set(kinds)) != len(kinds):
            raise ValueError("Metric types must be unique in one engine.")
        self.metrics = metrics

    def run(self, val_case: ValCase) -> ValResult:
        """Return all physical, sampled epistemic, and offset scenarios."""
        results: dict[str, dict[EMetricType, ValMetricField]] = {}
        for name, observable in val_case.domain.observables.items():
            sim = val_case.sim.quantities[name]
            exp = val_case.exp.quantities[name]
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
            for full_index in product(*(range(axis_sizes[a]) for a in VAL_RES_AXES)):
                selected = dict(zip(VAL_RES_AXES, full_index, strict=True))
                sim_samples = sim.select_aleatory(
                    selected["point"],
                    selected["component"],
                    selected["time"],
                    selected["sim_trans"],
                    selected["sim_nuis"],
                    selected["sim_offset"],
                )
                exp_samples = exp.select_aleatory(
                    selected["point"],
                    selected["component"],
                    selected["time"],
                    selected["exp_trans"],
                    selected["exp_nuis"],
                    selected["exp_offset"],
                )
                if not np.any(np.isfinite(sim_samples)):
                    continue
                if not np.any(np.isfinite(exp_samples)):
                    continue
                out_index = tuple(selected[axis] for axis in axes)
                for metric in self.metrics:
                    field = fields[metric.get_kind()]
                    metric_result = metric.calc(sim_samples, exp_samples)
                    if isinstance(metric_result, MAVMResult):
                        field.metric[out_index] = metric_result.d_total
                        assert field.d_plus is not None
                        assert field.d_minus is not None
                        field.d_plus[out_index] = metric_result.d_plus
                        field.d_minus[out_index] = metric_result.d_minus
                    else:
                        field.metric[out_index] = metric_result.value
            results[name] = fields
        configs = {metric.get_kind(): metric for metric in self.metrics}
        return ValResult(domain=val_case.domain, fields=results, metric_configs=configs)


def _axis_size(quantity: ValArray, axis: EValAxis) -> int:
    if axis not in quantity.axes:
        return 1
    return quantity.data.shape[quantity.axes.index(axis)]

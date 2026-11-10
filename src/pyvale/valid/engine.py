"""Run standalone metrics over shared physical and uncertainty axes."""

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from itertools import product

import numpy as np

from pyvale.valid.core import EValAxis, ValArray, ValCase, ValDomain
from pyvale.valid.strategy import EMetricType, IValMetric, MAVMMetricResult


class EResultAxis(StrEnum):
    POINT = "point"
    COMPONENT = "component"
    TIME = "time"
    SIM_TRANS = "sim_trans"
    SIM_NUIS = "sim_nuis"
    EXP_TRANS = "exp_trans"
    EXP_NUIS = "exp_nuis"
    SIM_OFFSET = "sim_offset"
    EXP_OFFSET = "exp_offset"


class EOffsetScenario(IntEnum):
    NOMINAL = 0
    LOWER = 1
    UPPER = 2


RESULT_AXES = tuple(EResultAxis)


@dataclass(slots=True)
class ValMetricField:
    axes: tuple[EResultAxis, ...]
    value: np.ndarray
    d_plus: np.ndarray | None = None
    d_minus: np.ndarray | None = None
    scenarios: dict[EResultAxis, tuple[EOffsetScenario, ...]] | None = None


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
    ) -> tuple[float, dict[EResultAxis, int]]:
        """Locate a finite extreme without reducing the stored result field."""
        field = self.get(observable, metric)
        if not np.any(np.isfinite(field.value)):
            raise ValueError("Metric field contains no finite results.")
        selectable = np.where(
            np.isfinite(field.value), field.value, -np.inf if maximum else np.inf
        )
        flat = np.argmax(selectable) if maximum else np.argmin(selectable)
        index = np.unravel_index(flat, field.value.shape)
        return float(field.value[index]), dict(zip(field.axes, index, strict=True))


def select_aleatory(
    quantity: ValArray,
    point: int,
    component: int,
    time: int,
    trans: int = 0,
    nuis: int = 0,
    offset: EOffsetScenario | int = EOffsetScenario.NOMINAL,
) -> np.ndarray:
    """Select one conditional 1D sample, applying a coherent value shift."""
    indices = {
        EValAxis.POINT: point,
        EValAxis.COMPONENT: component,
        EValAxis.TIME: time,
        EValAxis.EPISTEMIC_TRANS: trans,
        EValAxis.EPISTEMIC_NUIS: nuis,
        EValAxis.ALEATORY: slice(None),
    }
    selected = quantity.data[tuple(indices[axis] for axis in quantity.axes)]
    values = np.atleast_1d(selected)
    if np.any(np.isinf(values)):
        raise ValueError("Validation observations cannot contain infinity.")
    if isinstance(offset, bool) or offset not in (0, 1, 2):
        raise ValueError("Offset scenario must be nominal, lower, or upper.")
    if offset != EOffsetScenario.NOMINAL and quantity.offsets is None:
        raise ValueError("Quantity has no offset scenarios.")
    if offset == EOffsetScenario.NOMINAL:
        return values
    offsets = quantity.offsets
    physical = {
        EValAxis.POINT: point,
        EValAxis.COMPONENT: component,
        EValAxis.TIME: time,
    }
    bound = offsets.lower if offset == EOffsetScenario.LOWER else offsets.upper
    magnitude = np.asarray(bound)[tuple(physical[axis] for axis in offsets.axes)]
    sign = -1.0 if offset == EOffsetScenario.LOWER else 1.0
    return values + sign * magnitude


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

    def run(self, case: ValCase) -> ValResult:
        """Return all physical, sampled epistemic, and offset scenarios."""
        results: dict[str, dict[EMetricType, ValMetricField]] = {}
        for name, observable in case.domain.observables.items():
            sim = case.sim.quantities[name]
            exp = case.exp.quantities[name]
            axis_sizes = {
                EResultAxis.POINT: (
                    1
                    if observable.geometry.coords is None
                    else len(observable.geometry.coords)
                ),
                EResultAxis.COMPONENT: len(observable.components),
                EResultAxis.TIME: 1
                if observable.times is None
                else len(observable.times),
                EResultAxis.SIM_TRANS: _axis_size(sim, EValAxis.EPISTEMIC_TRANS),
                EResultAxis.SIM_NUIS: _axis_size(sim, EValAxis.EPISTEMIC_NUIS),
                EResultAxis.EXP_TRANS: _axis_size(exp, EValAxis.EPISTEMIC_TRANS),
                EResultAxis.EXP_NUIS: _axis_size(exp, EValAxis.EPISTEMIC_NUIS),
                EResultAxis.SIM_OFFSET: 3 if sim.offsets is not None else 1,
                EResultAxis.EXP_OFFSET: 3 if exp.offsets is not None else 1,
            }
            axes = tuple(
                axis
                for axis in RESULT_AXES
                if axis in RESULT_AXES[:3] or axis_sizes[axis] > 1
            )
            shape = tuple(axis_sizes[axis] for axis in axes)
            scenarios = {}
            for axis in (EResultAxis.SIM_OFFSET, EResultAxis.EXP_OFFSET):
                if axis in axes:
                    scenarios[axis] = tuple(EOffsetScenario)
            fields = {}
            for metric in self.metrics:
                directional = metric.get_kind() is EMetricType.MAVM
                fields[metric.get_kind()] = ValMetricField(
                    axes=axes,
                    value=np.full(shape, np.nan),
                    d_plus=np.full(shape, np.nan) if directional else None,
                    d_minus=np.full(shape, np.nan) if directional else None,
                    scenarios=scenarios or None,
                )
            for full_index in product(*(range(axis_sizes[a]) for a in RESULT_AXES)):
                selected = dict(zip(RESULT_AXES, full_index, strict=True))
                sim_samples = select_aleatory(
                    sim,
                    selected["point"],
                    selected["component"],
                    selected["time"],
                    selected["sim_trans"],
                    selected["sim_nuis"],
                    selected["sim_offset"],
                )
                exp_samples = select_aleatory(
                    exp,
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
                    field.value[out_index] = metric_result.value
                    if isinstance(metric_result, MAVMMetricResult):
                        assert field.d_plus is not None
                        assert field.d_minus is not None
                        field.d_plus[out_index] = metric_result.d_plus
                        field.d_minus[out_index] = metric_result.d_minus
            results[name] = fields
        configs = {metric.get_kind(): metric for metric in self.metrics}
        return ValResult(domain=case.domain, fields=results, metric_configs=configs)


def _axis_size(quantity: ValArray, axis: EValAxis) -> int:
    if axis not in quantity.axes:
        return 1
    return quantity.data.shape[quantity.axes.index(axis)]

# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Select labelled metric arrays and construct metric labels."""

import numpy as np

from pyvale.valid.metricstrategy import EMetricType
from pyvale.valid.validtypes import (
    EMetricOutput,
    EValidResAxis,
    ValidMetricField,
    resolve_metric_output,
)


def select_metric_output(
    field: ValidMetricField,
    output: EMetricOutput | None = None,
    *,
    selection_idxs: tuple[int | slice, ...] = (),
) -> np.ndarray:
    """Borrow an explicit output, or the sole output when no choice is supplied."""
    return field.outputs[resolve_metric_output(field, output)][selection_idxs]


def extract_metric_values(
    field: ValidMetricField,
    *,
    keep: EValidResAxis = EValidResAxis.POINT,
    selectors: dict[EValidResAxis, int] | None = None,
    output: EMetricOutput | None = None,
) -> np.ndarray:
    """Retain one axis and select every other non-singleton result axis."""
    if not isinstance(keep, EValidResAxis) or keep not in field.axes:
        raise ValueError(f"Result has no {keep} axis.")
    selected = selectors or {}
    if keep in selected:
        raise ValueError("Selectors cannot name the retained axis.")
    selection_idxs = build_metric_selection(field, selectors, keep=keep)
    return select_metric_output(field, output, selection_idxs=tuple(selection_idxs))


def build_metric_selection(
    field: ValidMetricField,
    selectors: dict[EValidResAxis, int] | None,
    *,
    keep: EValidResAxis | None = None,
) -> tuple[int | slice, ...]:
    selected = selectors or {}
    for axis in selected:
        if not isinstance(axis, EValidResAxis) or axis not in field.axes:
            raise ValueError("Selectors must name stored result axes.")
    selection = []
    for axis, size in zip(
        field.axes, next(iter(field.outputs.values())).shape, strict=True
    ):
        if axis == keep or (keep is None and axis not in selected):
            selection.append(slice(None))
            continue
        if axis not in selected and size != 1:
            raise ValueError(f"Select a {axis} index before plotting.")
        index = selected.get(axis, 0)
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError(f"{axis} index must be an integer.")
        if not 0 <= index < size:
            raise IndexError(f"{axis} index {index} is outside [0, {size}).")
        selection.append(index)
    return tuple(selection)


def create_metric_label(metric: EMetricType, output: EMetricOutput) -> str:
    output_label = {
        EMetricOutput.D_PLUS: "d+",
        EMetricOutput.D_MINUS: "d−",
        EMetricOutput.DISTANCE: "distance",
        EMetricOutput.STATISTIC: "statistic",
        EMetricOutput.ERROR: "error",
    }[output]
    return f"{metric.name} {output_label}"

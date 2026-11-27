# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Select labelled metric arrays and construct metric labels."""

import numpy as np

from pyvale.validation.imetric import EMetricOut, EMetricType
from pyvale.validation.validtypes import EValidResAxis, MetricField


def select_metric_output(
    field: MetricField,
    output: EMetricOut | None = None,
    *,
    selection_idxs: tuple[int | slice, ...] = (),
) -> np.ndarray:
    """Borrow an explicit output, or the sole output when no choice is supplied.

    Parameters
    ----------
    field : MetricField
        Metric field with arrays in named result-axis order.
    output : EMetricOut or None, optional
        Metric output to select. Use None to infer a single available output.
    selection_idxs : tuple[int or slice, ...], optional
        Index tuple in field axis order. The default selects the complete stored
        array.

    Returns
    -------
    ndarray
        Selected metric values, borrowing the stored array for basic slicing. Shape
        follows the unsliced dimensions.
    """
    return field.metric_vals[field._resolve_output(output)][selection_idxs]


def extract_metric_values(
    field: MetricField,
    *,
    keep: EValidResAxis = EValidResAxis.POINT,
    selectors: dict[EValidResAxis, int] | None = None,
    output: EMetricOut | None = None,
) -> np.ndarray:
    """Retain one axis and select every other non-singleton result axis.

    Parameters
    ----------
    field : MetricField
        Metric field with arrays in named result-axis order.
    keep : EValidResAxis, optional
        Result axis left as a full slice; other non-singleton dimensions need
        selectors.
    selectors : dict[EValidResAxis, int] or None, optional
        Zero-based indices keyed by result axis; omitted axes follow the operation's
        defaults.
    output : EMetricOut or None, optional
        Metric output to select. Use None to infer a single available output.

    Returns
    -------
    ndarray
        One-dimensional metric values of shape (n_kept,), in the retained axis
        order.
    """

    if not isinstance(keep, EValidResAxis) or keep not in field.axes:
        raise ValueError(f"Result has no {keep} axis.")

    selected = selectors or {}

    if keep in selected:
        raise ValueError("Selectors cannot name the retained axis.")

    selection_idxs = _build_metric_selection(field, selectors, keep=keep)
    return select_metric_output(field, output, selection_idxs=tuple(selection_idxs))


def _build_metric_selection(
    field: MetricField,
    selectors: dict[EValidResAxis, int] | None,
    *,
    keep: EValidResAxis | None = None,
) -> tuple[int | slice, ...]:
    """Resolve fixed indices and the retained result dimension.

    Parameters
    ----------
    field : MetricField
        Metric field with arrays in named result-axis order.
    selectors : dict[EValidResAxis, int] or None
        Zero-based indices keyed by result axis; omitted axes follow the operation's
        defaults.
    keep : EValidResAxis or None, optional
        Axis to retain as a full slice, or None to leave unselected axes as slices.

    Returns
    -------
    tuple[int or slice, ...]
        One zero-based index or full slice per axis in the metric field.
    """
    selected = selectors or {}

    for axis in selected:
        if not isinstance(axis, EValidResAxis) or axis not in field.axes:
            raise ValueError("Selectors must name stored result axes.")

    selection = []

    for axis, size in zip(
        field.axes, next(iter(field.metric_vals.values())).shape, strict=True
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


def _create_default_metric_labels() -> dict[EMetricType, str]:
    """Create a new mapping of default metric label overrides.

    Returns
    -------
    dict[EMetricType, str]
        Metric types mapped to custom display labels.
    """
    return {EMetricType.REL_ERR: "Rel Err"}


def _create_metric_label(
    metric: EMetricType,
    output: EMetricOut | None = None,
    *,
    metric_labels: dict[EMetricType, str],
) -> str:
    """Format a display label for a metric or directional output.

    Parameters
    ----------
    metric : EMetricType
        Metric to select or evaluate.
    output : EMetricOut or None, optional
        Metric output to select. Use None to infer a single available output.
    metric_labels : dict[EMetricType, str]
        Optional label overrides keyed by metric type.

    Returns
    -------
    str
        Metric label, with a directional suffix where applicable.
    """
    label = metric_labels.get(metric, metric.name)

    if output is None or metric in (
        EMetricType.RMSE,
        EMetricType.CVM,
        EMetricType.REL_ERR,
    ):
        return label

    output_label = {
        EMetricOut.D_PLUS: "d+",
        EMetricOut.D_MINUS: "d−",
        EMetricOut.DISTANCE: "distance",
        EMetricOut.STATISTIC: "statistic",
        EMetricOut.ERROR: "error",
    }[output]
    return f"{label} {output_label}"

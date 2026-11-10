# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Extract conditional empirical distributions without plotting or pooling axes."""

from itertools import product

import numpy as np
from scipy import stats

from pyvale.valid.metricstrategy import EMetricType
from pyvale.valid.valengine import EValResAxis, ValMetricField
from pyvale.valid.valtypes import (
    EEpisOffset,
    EValAxis,
    EValMetricPart,
    ValArray,
    ValECDF,
)


def calc_ecdf(samples: np.ndarray) -> ValECDF:
    """Calculate a 1D ECDF, omitting NaNs and preserving duplicate mass."""
    observations = np.asarray(samples, dtype=np.float64)
    if observations.ndim != 1 or np.any(np.isinf(observations)):
        raise ValueError("ECDF samples must be 1D and cannot contain infinity.")
    observations = observations[~np.isnan(observations)]
    if not len(observations):
        raise ValueError("ECDF needs at least one finite observation.")
    cdf = stats.ecdf(observations).cdf
    return ValECDF(cdf.quantiles, cdf.probabilities)


def extract_ecdf(
    quantity: ValArray,
    *,
    point: int = 0,
    component: int = 0,
    time: int = 0,
    epis_trans: int = 0,
    epis_nuis: int = 0,
    offset: EEpisOffset = EEpisOffset.NOMINAL,
) -> ValECDF:
    """Extract one conditional ECDF using physical and epistemic axis indices.

    Only ALEATORY supplies observations. Epistemic scenarios are selected, never
    pooled. The offset coherently shifts every observation in the selected cell.
    """
    samples = quantity.select_aleatory(
        point,
        component,
        time,
        epis_trans=epis_trans,
        epis_nuis=epis_nuis,
        offset=offset,
    )
    return calc_ecdf(samples)


def calc_ecdf_envelope(
    samples: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise ECDF bounds over coherent, supplied sample scenarios."""
    if not samples:
        raise ValueError("At least one sample scenario is required.")
    if any(sample.ndim != 1 or np.any(np.isinf(sample)) for sample in samples):
        raise ValueError("ECDF scenarios need finite or NaN 1D samples.")
    clean = [np.sort(s[np.isfinite(s)]) for s in samples]
    if any(len(s) == 0 for s in clean):
        raise ValueError("Each ECDF scenario needs a finite observation.")
    support = np.unique(np.concatenate(clean))
    cdfs = np.stack([np.searchsorted(s, support, side="right") / len(s) for s in clean])
    return support, np.min(cdfs, axis=0), np.max(cdfs, axis=0)


def extract_ecdf_envelope(
    quantity: ValArray,
    *,
    point: int = 0,
    component: int = 0,
    time: int = 0,
    epis_trans: int | None = None,
    epis_nuis: int | None = None,
    offset: EEpisOffset | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract pointwise ECDF bounds, fixing or spanning epistemic axes.

    An integer fixes that axis index; None spans all indices on that axis.
    offset=None includes nominal/lower/upper when offsets exist, otherwise only
    nominal. Returns (support, lower_probability, upper_probability), each (n,).
    """
    selections = []
    for axis, selection in (
        (EValAxis.EPISTEMIC_TRANS, epis_trans),
        (EValAxis.EPISTEMIC_NUIS, epis_nuis),
    ):
        size = (
            quantity.data.shape[quantity.axes.index(axis)]
            if axis in quantity.axes
            else 1
        )
        selections.append(range(size) if selection is None else (selection,))
    offsets = (
        (tuple(EEpisOffset) if quantity.offsets is not None else (EEpisOffset.NOMINAL,))
        if offset is None
        else (offset,)
    )
    samples = [
        quantity.select_aleatory(
            point,
            component,
            time,
            epis_trans=trans,
            epis_nuis=nuis,
            offset=selected_offset,
        )
        for trans, nuis, selected_offset in product(*selections, offsets)
    ]
    return calc_ecdf_envelope(samples)


def select_metric_part(
    field: ValMetricField,
    part: EValMetricPart = EValMetricPart.TOTAL,
    *,
    index: tuple[int | slice, ...] = (),
) -> np.ndarray:
    """Select a metric array; balance is d_plus - d_minus, not total mismatch."""
    if not isinstance(part, EValMetricPart):
        raise TypeError("Metric part must be EValMetricPart.")
    if part is EValMetricPart.TOTAL:
        return field.metric[index]
    if field.d_plus is None or field.d_minus is None:
        raise ValueError("Selected metric has no directional results.")
    if part is EValMetricPart.D_PLUS:
        return field.d_plus[index]
    if part is EValMetricPart.D_MINUS:
        return field.d_minus[index]
    return field.d_plus[index] - field.d_minus[index]


def extract_metric_values(
    field: ValMetricField,
    *,
    keep: EValResAxis = EValResAxis.POINT,
    selectors: dict[EValResAxis, int] | None = None,
    part: EValMetricPart = EValMetricPart.TOTAL,
) -> np.ndarray:
    """Retain one axis and select every other non-singleton result axis."""
    selected = selectors or {}
    if keep not in field.axes:
        raise ValueError(f"Result has no {keep} axis.")
    if any(axis not in field.axes or axis == keep for axis in selected):
        raise ValueError(
            "Selectors must name stored axes other than the retained axis."
        )
    indices: list[int | slice] = []
    for axis, size in zip(field.axes, field.metric.shape, strict=True):
        if axis == keep:
            indices.append(slice(None))
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
        indices.append(index)
    return select_metric_part(field, part, index=tuple(indices))


def create_metric_label(metric: EMetricType, part: EValMetricPart) -> str:
    """Name the plotted quantity, including MAVM direction and cancellation."""
    if metric is EMetricType.MAVM:
        return {
            EValMetricPart.TOTAL: "MAVM d+ + d−",
            EValMetricPart.D_PLUS: "MAVM d+",
            EValMetricPart.D_MINUS: "MAVM d−",
            EValMetricPart.BALANCE: "MAVM directional balance (d+ − d−)",
        }[part]
    return metric.name

import numpy as np
import numpy.typing as npt

from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefunc import IVectorObjectiveFunction


def _finite_vector(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Flatten residual-like values and remove entries masked with NaN."""
    flat_values = np.asarray(values, dtype=np.float64).ravel()
    return flat_values[np.isfinite(flat_values)]


def _apply_temporal_weights(
    values: npt.NDArray[np.float64],
    temporal_weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply weights along the first axis of a metric residual."""
    weights = np.asarray(temporal_weights, dtype=np.float64)
    if values.shape[0] != weights.shape[0]:
        raise ValueError(
            "Temporal weights must match the first residual dimension: "
            f"{weights.shape[0]} vs {values.shape[0]}."
        )
    weight_shape = weights.shape + (1,) * (values.ndim - 1)
    return values * np.sqrt(weights.reshape(weight_shape))


def _apply_spatial_weights(
    values: npt.NDArray[np.float64],
    spatial_weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply weights over the trailing dimensions of a metric residual."""
    weights = np.asarray(spatial_weights, dtype=np.float64)
    weight_shape = (1,) * (values.ndim - weights.ndim) + weights.shape
    return values * np.sqrt(weights.reshape(weight_shape))


def _resolve_metric_result_vector(
    metric_result: MetricResult,
    *,
    use_normalised_residual: bool,
    use_temporal_weighting: bool,
    use_spatial_weighting: bool,
) -> npt.NDArray[np.float64]:
    if metric_result.residual is None:
        raise ValueError("Metric residual doesn't exist.")

    metadata = metric_result.additional_fields or {}

    if use_normalised_residual:
        residual = metadata.get("normalised_residual", metric_result.residual)
    else:
        residual = metadata.get("raw_residual", metric_result.residual)

    resolved = np.asarray(residual, dtype=np.float64)

    if use_temporal_weighting:
        temporal_weights = metadata.get("temporal_weights")
        if temporal_weights is None:
            raise TypeError(
                "Temporal weighting was requested, but the metric did not "
                "provide temporal weights."
            )
        resolved = _apply_temporal_weights(resolved, temporal_weights)

    if use_spatial_weighting:
        spatial_weights = metadata.get("spatial_weights")
        if spatial_weights is None:
            raise TypeError(
                "Spatial weighting was requested, but the metric did not "
                "provide spatial weights."
            )
        resolved = _apply_spatial_weights(resolved, spatial_weights)

    return _finite_vector(resolved)


class VectorFirstResultPassthrough(IVectorObjectiveFunction):
    """
    Vector objective that passes the first metric's residual straight through
    to the optimiser.

    The most common choice for a single-metric least-squares identification
    """

    def evaluate(
        self,
        metric_results: list[MetricResult],
    ) -> npt.NDArray[np.float64]:
        if metric_results[0].residual is None:
            raise ValueError("Metric residual doesn't exist")

        return _finite_vector(metric_results[0].residual)

class VectorConcatenateObjective(IVectorObjectiveFunction):
    """
    Vector objective that concatenates the residuals of every metric into a
    single residual vector.

    Use this to drive a least-squares optimiser with more than one metric
    """

    def evaluate(
        self,
        metric_results: list[MetricResult],
    ) -> npt.NDArray[np.float64]:
        if not metric_results:
            return np.array([], dtype=np.float64)

        residuals = []
        for metric_result in metric_results:
            if metric_result.residual is None:
                raise ValueError("Metric residual doesn't exist")

            residuals.append(_finite_vector(metric_result.residual))

        return np.concatenate(residuals)


class VectorWeightedObjective(IVectorObjectiveFunction):
    """Apply optional normalisation and metric-provided weights to vector residuals."""

    def __init__(
        self,
        *,
        use_normalised_residual: bool = True,
        use_temporal_weighting: bool = True,
        use_spatial_weighting: bool = True,
    ) -> None:
        self.use_normalised_residual = use_normalised_residual
        self.use_temporal_weighting = use_temporal_weighting
        self.use_spatial_weighting = use_spatial_weighting

    def evaluate(
        self,
        metric_results: list[MetricResult],
    ) -> npt.NDArray[np.float64]:
        if not metric_results:
            return np.array([], dtype=np.float64)
        return np.concatenate(
            [
                _resolve_metric_result_vector(
                    metric_result,
                    use_normalised_residual=self.use_normalised_residual,
                    use_temporal_weighting=self.use_temporal_weighting,
                    use_spatial_weighting=self.use_spatial_weighting,
                )
                for metric_result in metric_results
            ]
        )

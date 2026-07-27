import numpy as np
import numpy.typing as npt

from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefunc import IVectorObjectiveFunction


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
            raise TypeError("Temporal weighting was requested, but the metric did not provide temporal weights.")
        resolved = resolved * np.sqrt(np.asarray(temporal_weights, dtype=np.float64))

    if use_spatial_weighting:
        spatial_weights = metadata.get("spatial_weights")
        if spatial_weights is None:
            raise TypeError("Spatial weighting was requested, but the metric did not provide spatial weights.")
        resolved = resolved * np.sqrt(np.asarray(spatial_weights, dtype=np.float64))

    return resolved


class VectorFirstResultPassthrough(IVectorObjectiveFunction):
    def evaluate(
        self,
        metric_results: list[MetricResult],
    ) -> npt.NDArray[np.float64]:
        if metric_results[0].residual is None:
            raise ValueError("Metric residual doesn't exist")

        return metric_results[0].residual.ravel()

class VectorConcatenateObjective(IVectorObjectiveFunction):
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

            residuals.append(metric_result.residual.ravel())

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
                ).ravel()
                for metric_result in metric_results
            ]
        )

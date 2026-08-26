import numpy as np

from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefunc import IScalarObjectiveFunction


class ScalarFirstResultPassthrough(IScalarObjectiveFunction):
    """
    Scalar objective that returns the first entry of the first metric's
    residual unchanged.

    A minimal objective for single-metric identifications driven by a scalar
    optimiser
    """

    def evaluate(
        self,
        metric_results: list[MetricResult],
    ) -> float:
        if metric_results[0].residual is None:
            raise ValueError("Metric residual doesn't exist")

        # TODO: only valid for 1D arrays
        return metric_results[0].residual[0]


class ScalarFirstResultRms(IScalarObjectiveFunction):
    """Return the RMS of the first metric residual."""

    def evaluate(self, metric_results: list[MetricResult]) -> float:
        residual = metric_results[0].residual
        if residual is None:
            raise ValueError("Metric residual doesn't exist")
        finite = np.asarray(residual, dtype=np.float64).ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("Metric residual does not contain finite values")
        return float(np.sqrt(np.mean(finite**2)))

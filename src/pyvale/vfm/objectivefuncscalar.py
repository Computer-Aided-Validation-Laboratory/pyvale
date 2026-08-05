from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefunc import IScalarObjectiveFunction


class ScalarFirstResultPassthrough(IScalarObjectiveFunction):
    def evaluate(
        self,
        metric_results: list[MetricResult],
    ) -> float:
        if metric_results[0].residual is None:
            raise ValueError("Metric residual doesn't exist")

        # TODO: only valid for 1D arrays
        return metric_results[0].residual[0]

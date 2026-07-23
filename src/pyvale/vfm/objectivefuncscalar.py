import numpy as np
import numpy.typing as npt

from pyvale.vfm.objectivefunc import IScalarObjectiveFunction, MetricResult
from pyvale.vfm.objectivefuncvector import _resolve_metric_result_vector


class ScalarFirstResultPassthrough(IScalarObjectiveFunction):
    def evaluate(
        self,
        metric_results: list[MetricResult],
    ) -> float:
        # TODO: only valid for 1D arrays
        return float(
            _resolve_metric_result_vector(
                metric_results[0],
                use_normalised_residual=False,
                use_temporal_weighting=False,
                use_spatial_weighting=False,
            )[0]
        )

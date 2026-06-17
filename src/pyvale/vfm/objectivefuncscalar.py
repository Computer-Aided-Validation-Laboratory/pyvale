import numpy as np
import numpy.typing as npt

from pyvale.vfm.objectivefunc import IScalarObjectiveFunction


class ScalarFirstResultPassthrough(IScalarObjectiveFunction):
    def evaluate(
        self,
        metric_results: list[npt.NDArray[np.float64]],
    ) -> float:
        # TODO: only valid for 1D arrays
        return metric_results[0][0]

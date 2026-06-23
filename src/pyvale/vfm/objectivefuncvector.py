import numpy as np
import numpy.typing as npt

from pyvale.vfm.objectivefunc import IVectorObjectiveFunction


class VectorFirstResultPassthrough(IVectorObjectiveFunction):
    def evaluate(
        self,
        metric_results: list[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        return metric_results[0]


class VectorConcatenateObjective(IVectorObjectiveFunction):
    def evaluate(
        self,
        metric_results: list[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        if not metric_results:
            return np.array([], dtype=np.float64)
        return np.concatenate(metric_results)

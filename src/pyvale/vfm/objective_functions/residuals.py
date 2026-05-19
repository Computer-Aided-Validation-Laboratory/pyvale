import numpy as np
import numpy.typing as npt
from pyvale.vfm.objective_functions.objective_function import (
    IVectorObjectiveFunction,
)

class Residuals(IVectorObjectiveFunction):
    def evaluate(
        self,
        metric_results: list[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        return metric_results[0]

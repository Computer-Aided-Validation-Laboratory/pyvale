import numpy as np
import numpy.typing as npt
from pyvale.vfm.objective_functions.objective_function import (
    VectorObjectiveFunction,
)

class Residuals(VectorObjectiveFunction):
    def evaluate(
        self,
        metric_results: list[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        x = metric_results[0]
        print(x)
        return x

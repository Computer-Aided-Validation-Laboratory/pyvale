import numpy as np
import numpy.typing as npt
from pyvale.vfm.objective_functions.objective_function import (
    ScalarObjectiveFunction,
)

class RMS(ScalarObjectiveFunction):
    def evaluate(
        self,
        metric_results: list[npt.NDArray[np.float64]],
    ) -> float:
        return 0

from scipy.optimize import least_squares

from pyvale.vfm.optimisers.optimiser import Optimiser


class LeastSquares(Optimiser):
    def optimise(self) -> None:
        ...

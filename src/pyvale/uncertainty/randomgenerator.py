'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from abc import ABC, abstractmethod
import numpy as np


class IRandomGenerator(ABC):
    @abstractmethod
    def generate(self, size: tuple[int,...]) -> np.ndarray:
        pass


class NormalGenerator(IRandomGenerator):
    def __init__(self,
                 std: float = 1.0,
                 mean: float = 0.0,
                 seed: int | None = None) -> None:

        self._std =std
        self._mean = mean
        self._rng = np.random.default_rng(seed)

    def generate(self, size: tuple[int,...]) -> np.ndarray:
        return self._rng.normal(loc = self._mean,
                                scale = self._std,
                                size = size)

class LogNormalGenerator(IRandomGenerator):
    def __init__(self,
                 std: float = 1.0,
                 mean: float = 0.0,
                 seed: int | None = None) -> None:

        self._std =std
        self._mean = mean
        self._rng = np.random.default_rng(seed)

    def generate(self, size: tuple[int,...]) -> np.ndarray:
        return self._rng.lognormal(mean = self._mean,
                                   sigma = self._std,
                                   size = size)


class UniformGenerator(IRandomGenerator):
    def __init__(self,
                 low: float = -1.0,
                 high: float = 1.0,
                 seed: int | None = None) -> None:

        self._low = low
        self._high = high
        self._rng = np.random.default_rng(seed)

    def generate(self, size: tuple[int,...]) -> np.ndarray:
        return self._rng.uniform(low = self._low,
                                 high = self._high,
                                 size = size)


class ExponentialGenerator(IRandomGenerator):
    def __init__(self,
                 scale: float = 1.0,
                 seed: int | None = None) -> None:

        self._scale = scale
        self._rng = np.random.default_rng(seed)

    def generate(self, size: tuple[int,...]) -> np.ndarray:
        return self._rng.exponential(scale = self._scale,
                                     size = size)


class ChiSquareGenerator(IRandomGenerator):
    def __init__(self,
                 dofs: float,
                 seed: int | None = None) -> None:

        self._dofs = np.abs(dofs)
        self._rng = np.random.default_rng(seed)

    def generate(self, size: tuple[int,...]) -> np.ndarray:
        return self._rng.chisquare(df = self._dofs,
                                   size = size)


class DirichletGenerator(IRandomGenerator):
    def __init__(self,
                 alpha: float,
                 seed: int | None = None) -> None:

        self._alpha = alpha
        self._rng = np.random.default_rng(seed)

    def generate(self, size: tuple[int,...]) -> np.ndarray:
        return self._rng.dirichlet(alpha = self._alpha, size = size)


class FGenerator(IRandomGenerator):
    def __init__(self,
                 dofs: float,
                 seed: int | None = None) -> None:

        self._dofs = np.abs(dofs)
        self._rng = np.random.default_rng(seed)

    def generate(self, size: tuple[int,...]) -> np.ndarray:
        return self._rng.f(dfnum = self._dofs, size = size)


class GammaGenerator(IRandomGenerator):
    def __init__(self,
                 shape: float,
                 scale: float = 1.0,
                 seed: int | None = None) -> None:

        self._shape = np.abs(shape)
        self._scale = scale
        self._rng = np.random.default_rng(seed)

    def generate(self, size: tuple[int,...]) -> np.ndarray:
        return self._rng.gamma(scale = self._scale,
                                     size = size)

class StudentTGenerator(IRandomGenerator):
    def __init__(self,
                 dofs: float,
                 seed: int | None = None) -> None:

        self._dofs = np.abs(dofs)
        self._rng = np.random.default_rng(seed)

    def generate(self, size: tuple[int,...]) -> np.ndarray:
        return self._rng.standard_t(df = self._dofs,
                                   size = size)

class BetaGenerator(IRandomGenerator):
    def __init__(self,
                 a: float,
                 b: float,
                 seed: int | None = None) -> None:

        self._a = np.abs(a)
        self._b = np.abs(b)
        self._rng = np.random.default_rng(seed)

    def generate(self, size: tuple[int,...]) -> np.ndarray:
        return self._rng.beta(a = self._a,
                              b = self._b,
                              size = size)


class TriangularGenerator(IRandomGenerator):
    def __init__(self,
                 left: float = -1.0,
                 mode: float = 0.0,
                 right: float = 1.0,
                 seed: int | None = None) -> None:

        self._left = left
        self._mode = mode
        self._right = right

        self._rng = np.random.default_rng(seed)

    def generate(self, size: tuple[int,...]) -> np.ndarray:
        return self._rng.triangular(left = self._left,
                                    mode = self._mode,
                                    right = self._right,
                                    size = size)
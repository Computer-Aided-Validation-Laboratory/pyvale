# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from abc import ABC, abstractmethod
import numpy as np


class IGenRandom(ABC):
    """Interface (abstract base class) for wrapping numpy random number
    generation to allow probability distribution parameters to be specified in
    the initialiser whereas the generation of random numbers has a common
    method that just takes the require shape to return. Allows for easy
    subsitution of different probability distributions.
    """

    @abstractmethod
    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Abstract method. Generates an array of random numbers with the shape
        specified by the input.

        Parameters
        ----------
        shape : tuple[int,...]
            Shape of the array of random numbers to be returned.

        Returns
        -------
        np.ndarray
            Array of random numbers with shape specified by the input shape.
        """
        pass


class GenNormal(IGenRandom):
    """Class wrapping the numpy normal random number generator. Implements the
    IGeneratorRandom interface to allow for interchangeability with other random
    number generators.
    """
    __slots__ = ("_std","_mean","_rng")

    def __init__(self,
                 std: float = 1.0,
                 mean: float = 0.0,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        std : float, optional
            Standard deviation of the normal distribution to sample, by default
            1.0
        mean : float, optional
            Mean of the normal distribution to sample, by default 0.0
        seed : int | None, optional
            Optional seed for the random generator to allow for reproducibility
            and testing, by default None
        """
        self._std =std
        self._mean = mean
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Generates an array of random numbers with the shape specified by the
        input.

        Parameters
        ----------
        shape : tuple[int,...]
            Shape of the array to return.

        Returns
        -------
        np.ndarray
            Array of random numbers with the specified shape.
        """
        return self._rng.normal(loc = self._mean,
                                scale = self._std,
                                size = shape)


class GenLogNormal(IGenRandom):
    """Class wrapping the numpy lognormal random generator. Implements the
    IGeneratorRandom interface to allow for interchangeability with other random
    number generators.
    """
    __slots__ = ("_std","_mean","_rng")

    def __init__(self,
                 std: float = 1.0,
                 mean: float = 0.0,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        std : float, optional
            Standard deviation of the normal distribution to sample, by default
            1.0
        mean : float, optional
            Mean of the normal distribution to sample, by default 0.0
        seed : int | None, optional
            Optional seed for the random generator to allow for reproducibility
            and testing, by default None
        """
        self._std =std
        self._mean = mean
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Generates an array of random numbers with the shape specified by the
        input.

        Parameters
        ----------
        shape : tuple[int,...]
            Shape of the array to return.
        Returns
        -------
        np.ndarray
            Array of random numbers with the specified shape.
        """
        return self._rng.lognormal(mean = self._mean,
                                   sigma = self._std,
                                   size = shape)


class GenUniform(IGenRandom):
    """Class wrapping the numpy uniform random number generator. Implements the
    IGeneratorRandom interface to allow for interchangeability with other random
    number generators.
    """
    __slots__ = ("_low","_high","_rng")

    def __init__(self,
                 low: float = -1.0,
                 high: float = 1.0,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        low : float, optional
            Lower bound of the uniform dsitribution., by default -1.0
        high : float, optional
            Upper bound of the uniform distribution, by default 1.0
        seed : int | None, optional
            Optional seed for the random generator to allow for reproducibility
            and testing, by default None
        """

        self._low = low
        self._high = high
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Generates an array of random numbers with the shape specified by the
        input.

        Parameters
        ----------
        shape : tuple[int,...]
            Shape of the array to return.
        Returns
        -------
        np.ndarray
            Array of random numbers with the specified shape.
        """
        return self._rng.uniform(low = self._low,
                                 high = self._high,
                                 size = shape)


class GenExponential(IGenRandom):
    """Class wrapping the numpy exponential random generator. Implements the
    IGeneratorRandom interface to allow for interchangeability with other random
    number generators.
    """
    __slots__ = ("_scale","_rng")

    def __init__(self,
                 scale: float = 1.0,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        scale : float, optional
            Scale parameter of the distribution which must be positive, by
            default 1.0
        seed : int | None, optional
            Optional seed for the random generator to allow for reproducibility
            and testing, by default None
        """
        self._scale = np.abs(scale)
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Generates an array of random numbers with the shape specified by the
        input.

        Parameters
        ----------
        shape : tuple[int,...]
            Shape of the array to return.
        Returns
        -------
        np.ndarray
            Array of random numbers with the specified shape.
        """
        return self._rng.exponential(scale = self._scale,
                                     size = shape)


class GenChiSquare(IGenRandom):
    """Class wrapping the numpy chi square random generator. Implements the
    IGeneratorRandom interface to allow for interchangeability with other random
    number generators.
    """
    __slots__ = ("_dofs","_rng")

    def __init__(self,
                 dofs: float,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        dofs : float
            Number of degrees of freedom of the distribution must be greater
            than zero.
        seed : int | None, optional
            Optional seed for the random generator to allow for reproducibility
            and testing, by default None
        """
        self._dofs = np.abs(dofs)
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Generates an array of random numbers with the shape specified by the
        input.

        Parameters
        ----------
        shape : tuple[int,...]
            Shape of the array to return.
        Returns
        -------
        np.ndarray
            Array of random numbers with the specified shape.
        """
        return self._rng.chisquare(df = self._dofs,
                                   size = shape)


class GenDirichlet(IGenRandom):
    """Class wrapping the numpy dirichlet random generator. Implements the
    IGeneratorRandom interface to allow for interchangeability with other random
    number generators.
    """
    __slots__ = ("_alpha","_rng")

    def __init__(self,
                 alpha: float,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        alpha : float
            Alpha parameter of the distribution
        seed : int | None, optional
            Optional seed for the random generator to allow for reproducibility
            and testing, by default None
        """
        self._alpha = alpha
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Generates an array of random numbers with the shape specified by the
        input.

        Parameters
        ----------
        shape : tuple[int,...]
            Shape of the array to return.
        Returns
        -------
        np.ndarray
            Array of random numbers with the specified shape.
        """
        return self._rng.dirichlet(alpha = self._alpha, size = shape)


class GenF(IGenRandom):
    """Class wrapping the numpy F distribution random generator. Implements the
    IGeneratorRandom interface to allow for interchangeability with other random
    number generators.
    """
    __slots__ = ("_dofs","_rng")

    def __init__(self,
                 dofs: float,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        dofs : float
            Number of degrees of freedom of the distribution must be greater
            than zero
        seed : int | None, optional
            Optional seed for the random generator to allow for reproducibility
            and testing, by default None
        """
        self._dofs = np.abs(dofs)
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Generates an array of random numbers with the shape specified by the
        input.

        Parameters
        ----------
        shape : tuple[int,...]
            Shape of the array to return.
        Returns
        -------
        np.ndarray
            Array of random numbers with the specified shape.
        """
        return self._rng.f(dfnum = self._dofs, size = shape)


class GenGamma(IGenRandom):
    """Class wrapping the numpy gamma random generator. Implements the
    IGeneratorRandom interface to allow for interchangeability with other random
    number generators.
    """
    __slots__ = ("_shape","_scale","_rng")

    def __init__(self,
                 shape: float,
                 scale: float = 1.0,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        shape : float
            Shape parameter of the gamma distribution, must be greater than zero
        scale : float, optional
            Scale parameter of the gamma distribution which must be greater than
            zero, by default 1.0
        seed : int | None, optional
            Optional seed for the random generator to allow for reproducibility
            and testing, by default None
        """
        self._shape = np.abs(shape)
        self._scale = np.abs(scale)
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Generates an array of random numbers with the shape specified by the
        input.

        Parameters
        ----------
        shape : tuple[int,...]
            Shape of the array to return.
        Returns
        -------
        np.ndarray
            Array of random numbers with the specified shape.
        """
        return self._rng.gamma(scale = self._scale,
                                     size = shape)


class GenStandardT(IGenRandom):
    """Class wrapping the numpy t distribution random generator. Implements the
    IGeneratorRandom interface to allow for interchangeability with other random
    number generators.
    """
    __slots__ = ("_dofs","_rng")

    def __init__(self,
                 dofs: float,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        dofs : float
            Number of degrees of freedom of the distribution must be greater
            than zero.
        seed : int | None, optional
            Optional seed for the random generator to allow for reproducibility
            and testing, by default None
        """
        self._dofs = np.abs(dofs)
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Generates an array of random numbers with the shape specified by the
        input.

        Parameters
        ----------
        shape : tuple[int,...]
            Shape of the array to return.
        Returns
        -------
        np.ndarray
            Array of random numbers with the specified shape.
        """
        return self._rng.standard_t(df = self._dofs,
                                   size = shape)


class GenBeta(IGenRandom):
    """Class wrapping the numpy beta distribution random generator. Implements
    the IGeneratorRandom interface to allow for interchangeability with other
    random number generators.
    """
    __slots__ = ("_a","_b","_rng")

    def __init__(self,
                 a: float,
                 b: float,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        a : float
            Alpha parameter of the distribution which must be greater than zero
        b : float
            Beta parameter of the distribution which must be greater than zero
        seed : int | None, optional
            Optional seed for the random generator to allow for reproducibility
            and testing, by default None
        """
        self._a = np.abs(a)
        self._b = np.abs(b)
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Generates an array of random numbers with the shape specified by the
        input.

        Parameters
        ----------
        shape : tuple[int,...]
            Shape of the array to return.
        Returns
        -------
        np.ndarray
            Array of random numbers with the specified shape.
        """
        return self._rng.beta(a = self._a,
                              b = self._b,
                              size = shape)


class GenTriangular(IGenRandom):
    """Class wrapping the numpy triangular random generator. Implements the
    IGeneratorRandom interface to allow for interchangeability with other random
    number generators.
    """
    __slots__ = ("_left","_mode","_right","_rng")

    def __init__(self,
                 left: float = -1.0,
                 mode: float = 0.0,
                 right: float = 1.0,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        left : float, optional
            Left (min) corner of the triangular distribution, by default -1.0
        mode : float, optional
            Central peak of the triangular distribution, by default 0.0
        right : float, optional
            Right (max) corner of the triangular distribution , by default 1.0
        seed : int | None, optional
            Optional seed for the random generator to allow for reproducibility
            and testing, by default None
        """
        self._left = left
        self._mode = mode
        self._right = right

        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Generates an array of random numbers with the shape specified by the
        input.

        Parameters
        ----------
        shape : tuple[int,...]
            Shape of the array to return.
        Returns
        -------
        np.ndarray
            Array of random numbers with the specified shape.
        """
        return self._rng.triangular(left = self._left,
                                    mode = self._mode,
                                    right = self._right,
                                    size = shape)
# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from abc import ABC, abstractmethod
import numpy as np


class IDriftCalculator(ABC):
    """Interface (abstract base class) for applying a function to cause a sensor
    array measurement to drift (normally over time). The initialiser for the
    concrete implementation of this class specifies the function parameters and
    a unified method is provided to calculate the drift based on the input.
    """

    @abstractmethod
    def calc_drift(self, drift_var: np.ndarray) -> np.ndarray:
        """Abstract method. Used to calculate the drift function based on the
        input drift variable (useually the time steps).

        Parameters
        ----------
        drift_var : np.ndarray
            Array of values (normally time steps) that are used to calculate the
            drift errors.

        Returns
        -------
        np.ndarray
            Array of drift errors having the same shape as the input drift_var
            array.
        """


class DriftConstant(IDriftCalculator):
    """Class for applying a constant drift error over time.

    Implements the IDriftCalculator interface.
    """
    __slots__ = ("_offset",)

    def __init__(self, offset: float) -> None:
        """
        Parameters
        ----------
        offset : float
            Constant drift offset.
        """
        self._offset = offset

    def calc_drift(self, drift_var: np.ndarray) -> np.ndarray:
        """Calculates the drift errors based on the input drift variable array.

        Parameters
        ----------
        drift_var : np.ndarray
            Array of values (normally time steps) that are used to calculate the
            drift errors.

        Returns
        -------
        np.ndarray
            Array of drift errors having the same shape as the input drift_var
            array.
        """
        return self._offset*np.ones_like(drift_var)


class DriftLinear(IDriftCalculator):
    """Class for applying a linear drift error over time.

    Implements the IDriftCalculator interface.
    """
    __slots__ = ("_slope","_offset")

    def __init__(self, slope: float, offset: float = 0.0) -> None:
        """
        Parameters
        ----------
        slope : float
            Slope of the drift error function.
        offset : float, optional
            Offset (intercept) of the drift error function, by default 0.0.
        """
        self._slope = slope
        self._offset = offset

    def calc_drift(self, drift_var: np.ndarray) -> np.ndarray:
        """Calculates the drift errors based on the input drift variable array.

        Parameters
        ----------
        drift_var : np.ndarray
            Array of values (normally time steps) that are used to calculate the
            drift errors.

        Returns
        -------
        np.ndarray
            Array of drift errors having the same shape as the input drift_var
            array.
        """
        return self._slope*drift_var + self._offset


class DriftPolynomial(IDriftCalculator):
    """Class for applying a polynomial drift error over time. The coefficients
    of the polynomial are specified with a numpy array from constant term to
    highest power.

    Implements the IDriftCalculator interface.
    """

    __slots__ = ("_coeffs",)

    def __init__(self, coeffs: np.ndarray) -> None:
        """
        Parameters
        ----------
        coeffs : np.ndarray
            Array of polynomial coefficients from constant to highest power.
        """
        self._coeffs = coeffs

    def calc_drift(self, drift_var: np.ndarray) -> np.ndarray:
        """Calculates the drift errors based on the input drift variable array.

        Parameters
        ----------
        drift_var : np.ndarray
            Array of values (normally time steps) that are used to calculate the
            drift errors.

        Returns
        -------
        np.ndarray
            Array of drift errors having the same shape as the input drift_var
            array.
        """
        poly = np.zeros_like(drift_var)
        for ii,cc in enumerate(self._coeffs):
            poly += cc*drift_var**ii

        return poly
import enum
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class EIdentificationType(enum.Enum):
    """
    Identifies whether a constitutive law's parameters should be
    identified with linear on nonlinear identification
    """

    Linear = enum.auto()
    """Identify parameters with linear identification"""
    Nonlinear = enum.auto()
    """Identify parameters with nonlinear identification"""


class IConstitutiveLaw(ABC):
    """
    Interface (abstract base class) for a constitutive law.

    Provides the material model that relates strain to stress. Concrete
    implementations define the specific constitutive equations and report
    whether identification is linear or nonlinear
    """

    @abstractmethod
    def get_identification_type(self) -> EIdentificationType:
        """
        Indicate whether this law is linear or nonlinear in its parameters.

        Returns
        -------
        EIdentificationType
            ``Linear`` or ``Nonlinear``
        """
        pass

    @abstractmethod
    def get_required_parameters(self) -> list[str]:
        """
        Return the list of required constitutive parameters for this law.

        Concrete implementations should combine their own parameter names
        with those from any nested hardening law

        Returns
        -------
        list[str]
            All parameter name strings this law requires
        """
        pass

    @abstractmethod
    def calculate_stress(
        self,
        strain: npt.NDArray[np.float64],
        constitutive_parameter_maps: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """
        Compute stress from the current strain and parameter maps.

        Parameters
        ----------
        strain : npt.NDArray[np.float64]
            Full-field strain history, shape ``(timesteps, components, y, x)``
        constitutive_parameter_maps : dict[str, npt.NDArray[np.float64]]
            Dictionary of current parameter values as 2D maps,
            keyed by parameter name

        Returns
        -------
        npt.NDArray[np.float64]
            Stress field with the same shape as ``strain``
        """
        pass

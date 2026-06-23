from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class ConstitutiveParameter:
    """
    A single constitutive parameter with a spatially varying map of values and
    associated value bounds.

    The ``value`` may be supplied as a scalar (int or float) together with a
    ``map_size`` to create a homogeneous 2D field, or as a full 2D array
    directly
    """

    map: npt.NDArray[np.float64]
    """Parameter map, shape ``(y, x)``"""

    lower_bound: float
    """Lower bound for the parameter value"""

    upper_bound: float
    """Upper bound for the parameter value"""

    def __init__(
        self,
        value: int | float | npt.NDArray[np.float64],
        lower_bound: float,
        upper_bound: float,
        map_size: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        value : int | float | npt.NDArray[np.float64]
            Parameter value.  If scalar, ``map_size`` must also be provided
            and the value is broadcast to create a homogeneous 2D field
        lower_bound : float
            Lower bound for the parameter
        upper_bound : float
            Upper bound for the parameter
        map_size : npt.NDArray[np.float64] | None, optional
            Shape ``(y, x)`` of the spatial parameterisation when ``value``
            is a scalar.  Ignored when ``value`` is already an array
        """
        if isinstance(value, (int, float)):
            if map_size is None:
                raise ValueError(
                    "map_size must be defined if "
                    "parameter value is int or float"
                )

            self.map = np.full((map_size[0], map_size[1]), value)

        else:
            self.map = value

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

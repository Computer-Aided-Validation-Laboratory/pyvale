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
        map_size: npt.NDArray[np.uint32] | None = None,
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
        map_size : npt.NDArray[np.uint32] | None, optional
            Shape ``(y, x)`` of the spatial parameterisation when ``value``
            is a scalar.  Ignored when ``value`` is already an array

        Raises
        ------
        ValueError
            If ``lower_bound >= upper_bound``, or if any value in the
            parameter map lies outside ``[lower_bound, upper_bound]``, or
            if ``value`` is a scalar and ``map_size`` is ``None``
        """
        if lower_bound >= upper_bound:
            raise ValueError(
                f"lower_bound ({lower_bound}) must be less than "
                f"upper_bound ({upper_bound})"
            )

        if isinstance(value, (int, float)):
            if map_size is None:
                raise ValueError(
                    "map_size must be defined if "
                    "parameter value is int or float"
                )

            self.map = np.full(map_size, value)

        else:
            self.map = value

        if np.any((self.map < lower_bound) | (self.map > upper_bound)):
            raise ValueError(
                f"parameter values must be within provided bounds "
                f"[{lower_bound}, {upper_bound}]"
            )

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

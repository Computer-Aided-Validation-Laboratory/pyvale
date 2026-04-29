from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class Parameter:
    value: npt.NDArray[np.float64]
    lower_bound: float
    upper_bound: float

    # map_size must be in form (y, x)
    def __init__(
        self,
        value: int | float | npt.NDArray[np.float64],
        lower_bound: float,
        upper_bound: float,
        map_size: npt.NDArray[np.float64] | None = None
    ) -> None:
        if isinstance(value, (int, float)):
            if map_size is None:
                raise ValueError(
                    "map_size must be defined if "
                    "parameter value is int or float"
                )

            self.value = np.full((map_size[0], map_size[1]), value)

        else:
            self.value = value

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


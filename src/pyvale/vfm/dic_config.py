from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


# TODO: should this be merged/replaced with dicresults.py?
@dataclass(slots=True)
class DICConfig:
    # number of DIC x points
    x_dimension: int
    # number of DIC y points
    y_dimension: int
    timesteps: npt.NDArray[np.float64]

    def calculate_timestep_deltas(self) -> npt.NDArray[np.float64]:
        delta_time = np.zeros_like(self.timesteps)
        delta_time[0] = self.timesteps[0]
        delta_time[1:] = np.diff(self.timesteps)

        return delta_time

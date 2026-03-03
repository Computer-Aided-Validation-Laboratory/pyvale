from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm import DICConfig


# TODO: more specific class name?
# TODO: for non linear geometries might need an extra stress component
# TODO: sometimes von mises can be empty
@dataclass(slots=True)
class Stress:
    xx: npt.NDArray[np.float64]
    xy: npt.NDArray[np.float64]
    yy: npt.NDArray[np.float64]
    von_mises: npt.NDArray[np.float64]


# TODO: do we need eqvStress like original impl?
# Convert stress to 4D tensor of the shape (x_points, y_points, components, timestep]
def convert_stress_to_4d(
    stress: Stress,
    dic_config: DICConfig
) -> npt.NDArray[np.float64]:
    # TODO: might be different for non linear geometry
    num_stress_components = 3

    stress_4d = np.zeros((
        dic_config.x_dimension,
        dic_config.y_dimension,
        num_stress_components,
        dic_config.timesteps.size
    ))

    component_dimensions = (
        dic_config.x_dimension,
        dic_config.y_dimension,
        # 1,
        dic_config.timesteps.size
    )

    stress_4d[:, :, 0, :] = np.reshape(
        stress.xx,
        component_dimensions,
        order="F"
    )

    stress_4d[:, :, 1, :] = np.reshape(
        stress.yy,
        component_dimensions,
        order="F"
    )

    stress_4d[:, :, 2, :] = np.reshape(
        stress.xy,
        component_dimensions,
        order="F"
    )

    return stress_4d

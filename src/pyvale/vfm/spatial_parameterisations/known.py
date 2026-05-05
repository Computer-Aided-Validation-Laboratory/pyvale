from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    SpatialParameterisation,
)
from pyvale.vfm.parameter import ConstitutiveParameter


@dataclass(slots=True)
class KnownSpatialParameterisation(SpatialParameterisation):
    parameter_name: str

    def to_map(
        self,
        params: dict[str, ConstitutiveParameter],
        size: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.float64]:
        parameter = params[self.parameter_name]

        # TODO: value error if param value not the right size

        return parameter.value

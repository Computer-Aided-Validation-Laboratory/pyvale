from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.metrics.metric import Metric
from pyvale.vfm.virtual_fields_mesh import VirtualFieldsMesh


@dataclass(slots=True)
class SensitivityBasedVitualFieldsMetric(Metric):
    virtual_fields_mesh: VirtualFieldsMesh

    def __init__(self, mesh_size: npt.NDArray[np.uint32]) -> None:
        # TODO: input shape and size checking where appropriate
        # build sbvf mesh
        return


    def evaluate(
        self,
        stress: npt.NDArray[np.float64]
    ) -> float:
        # calculate stress sensitivity
        # generate sbvfs
        # perform metric evaluation
        return 0

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.experiment_data import ExperimentData
from pyvale.vfm.metrics.generate_sensitivity_based_virtual_fields import (
    generate_sensitivity_based_virtual_fields,
)
from pyvale.vfm.metrics.metric import Metric
from pyvale.vfm.metrics.stress_sensitivity import calculate_stress_sensitivity
from pyvale.vfm.metrics.virtual_fields_mesh import (
    VirtualFieldsMesh,
    generate_virtual_fields_mesh,
)


@dataclass(slots=True)
class SensitivityBasedVirtualFieldsMetric(Metric):
    virtual_fields_mesh: VirtualFieldsMesh

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        region_of_interest: npt.NDArray[np.uint32],
        boundary_conditions: npt.NDArray[np.uint32],
        mesh_size: npt.NDArray[np.uint32],
    ) -> None:
        # TODO: input shape and size checking where appropriate

        self.virtual_fields_mesh = generate_virtual_fields_mesh(
            x, y, region_of_interest, boundary_conditions, mesh_size
        )

    def evaluate(
        self,
        stress: npt.NDArray[np.float64],
        experiment_data: ExperimentData
    ) -> float:
        # for each dof compute its stress sensitivity
        stress_sensitivities = calculate_stress_sensitivity()
        sbvfs = generate_sensitivity_based_virtual_fields()
        # perform metric evaluation
        return 0

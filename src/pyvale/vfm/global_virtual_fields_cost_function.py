import numpy as np
import numpy.typing as npt

from pyvale.vfm.mechanical_properties import (
    EDOFLabel,
    EParameterLabel,
)
from pyvale.vfm.sensitivity_based_virtual_fields import (
    SensitivityBasedVirtualFields,
)

# Combine:
# globalVirtualFieldObjectiveFunction.m
# globalVirtualFieldCostFunction.m


## Define virtual fields (either update or use existing)

# sensitivity based virtual fields or manual virtual fields

## Evaluate cost function

# Inputs:
#  - stress
#  - our SBVFs dict[EParameterLabel, list[dict[EDOFLabel, SensitivityBasedVirtualFields]]]
#  - force
#  - area
#  - thickness
# Outputs:
#  - scalar float cost?
def global_vf_cost_function(
    stress: npt.NDArray[np.float64], #(timesteps, components, y, x)
    sensitivity_based_virtual_fields: dict[
        EParameterLabel,
        list[dict[EDOFLabel, SensitivityBasedVirtualFields]]
    ],
    force: npt.NDArray[np.float64], #(timesteps, components)
    # TODO: is this the right form for area? or can we take a scalar and blow it up to size
    area: npt.NDArray[np.float64], #(timesteps, components, y, x)
    thickness: float
):
    # For each dof
    #  - calculate internal virtual work
    #  - calculate external virtual work

    sbvfs = [
        sbvf
        for parameterisations in sensitivity_based_virtual_fields.values()
        for p in parameterisations
        for sbvf in p.values()
    ]

    for sbvf in sbvfs:
        internal_virtual_work = (
            stress * sbvf.virtual_strain * area * thickness
        )

        # This is the edge that the clamp force acts on
        # TODO: pull this out of some config object? maybe
        # dic_config should exist to store this kind of thing?
        edge = 0

        external_virtual_work = (
            (force[:, 0] * sbvf.virtual_displacement[:, 0, edge])
            + (force[:, 1] * sbvf.virtual_displacement[:, 1, edge])
        )

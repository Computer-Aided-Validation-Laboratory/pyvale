import numpy as np

from pyvale.vfm.experiment_data import ExperimentData
from pyvale.vfm.identification import EIdentificationType, Identification
from pyvale.vfm.spatial_parameterisations.known import (
     KnownSpatialParameterisation,
)


# TODO: return type
# TODO: config validation
#   - no forward referencing in phases list
#   - individual weights cant be greater than 1.0 in total
#   - sum of weights must be 1.0
#   - optimiser is compatible with objective function
def vfm(
    experiment_data: ExperimentData,
    identification: Identification
):
     match identification.constitutive_law.identification_type:
        # TODO: implement linear case
        case EIdentificationType.Linear:
            ...
        case EIdentificationType.Nonlinear:
            parameter_map_size = np.array(
                experiment_data.specimen_geometry.x.shape,
                dtype=np.uint32
            )

            for phase in identification.phases:
                optimisation_result = phase.optimiser.optimise(
                    identification.constitutive_law,
                    parameter_map_size,
                    phase.spatial_parameterisations,
                    phase.metrics,
                    phase.objective_function,
                    experiment_data
                )

                # update parameter maps
                # perform refinement?

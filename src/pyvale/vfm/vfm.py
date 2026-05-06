from pyvale.vfm.experiment_data import ExperimentData
from pyvale.vfm.identification import EIdentificationType, Identification


# TODO: return type
# TODO: config validation
#   - no forward referencing in phases list
#   - individual weights cant be greater than 1.0 in total
#   - sum of weights must be 1.0
def vfm(
    experiment_data: ExperimentData,
    identification: Identification
):
     match identification.constitutive_law.identification_type:
        # TODO: implement linear case
        case EIdentificationType.Linear:
            ...
        case EIdentificationType.Nonlinear:
            for phase in identification.phases:
                # Collect unknown parameterisations
                # run optimiser

    # check identification type
    # for non linear
    # run each phase sequentially
    # get unknown params
    # build parameterisations?
    # build metrics?
    # build optimiser?
    # (Above might all be already build and passed to vfm)
    # Run optimiser
    # Collect best result
    # Optionally perform refinement
    # Collect best result
    # Output result from all phases

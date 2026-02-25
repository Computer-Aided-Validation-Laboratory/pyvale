import numpy as np
from scipy.io import loadmat

from pyvale.vfm import radial_return


# To compute stress sensitivity we need to
# - take the original stress we calculated
# - perform some normalisation on the degrees of freedom (ignore if only 1 dof?)
# - calculate the perturbed stress from a slightly perturbed parameter map
# - calculate their differences
# Should take as input:
# - a referene stress to compare against
# - parameter map (is this a misnomer? are these elements of the param map K or smth else?)
#   - taking a single parameter for now rather than a whole map
# - strain + material properties (needed to calculate perturbed param strain)
# Should return:
# - total stress sensitivity
# - incremental stress sensitivity
# Assumptions:
# - only 1 degree of freedom
#   - do we need to care about exactly what these are?
# - only 1 parameter to perturb?
#   - how many params will we expect in a real scenario?
# - (do these conditions mean it's a homogeneous material?)
# - linear scaling when purturbing params (what does this mean physically?)
# TODO: arg/return types
def calculate_stress_sensitivities(
    stress_reference,
    parameter,
    strain,
    material_properties
):
    # TODO: should this be and input, and what will change this value if so?
    perturbation_factor = 0.15
    perturbed_parameter = parameter * (1 - perturbation_factor)

    # TODO: need to pass in perturbed parameter to be used in hardening calcs
    perturbed_stress = radial_return(strain, material_properties)
    print("break")


input = loadmat("/Users/chris/work/vfmap-numerical-paper/test_data/compute_stress_sensetivity_input.mat")

parameter_map = input["spatialParamData"][0][0]
stress = input["stressRef"][0][0]
# test_data = input["testData"][0][0]
# options = input["options"][0][0]
# print(spatial_param_data["parDof"].shape)

# calculate_stress_sensitivities(stress, parameter_map)

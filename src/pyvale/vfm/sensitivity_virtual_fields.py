import numpy as np
from scipy.io import loadmat

from pyvale.vfm.virtual_fields_mesh import VirtualFieldsMesh

# Generate the sensitivity based virtual fields based on the
# total/incremental stress sensitivity that we generate in
# calculate_stress_sensitivity and the virtual fields mesh
# we generate in generate_virtual_fields_mesh
# Input:
# - stress_sensitivity
#   - can be either total/incremental
#   - will have the same 4d convention as stress/strain?
# - virtual_fields_mesh
# Output:
# - virtual_fields (TODO: naming?)
#   - the virtual strain
#   - the virtual displacements on the edges (TODO: is this the edges of the whole specimen)
# - virtual displacements (TODO: matlab comment says maybe not needed since we have the virtual displacements above)
# Assumptions:

# TODO: args/return type
# TODO: vectorise
# TODO: matlab calls this function for each degree of freedom, but we could compute all dofs in this func
def generate_sensitivity_based_virtual_fields(
    stress_sensitivity, # (timestep, component, y, x)
    virtual_fields_mesh
):
    # For each timestep
    # - set virtual strains equal to stress sensitivity (refmap)
    # - matrix multiply virtual strains with Binv from the virtual fields mesh to get your virtual displacements
    #   - since the derivative of displacement is strain
    #   - Bglob (strain displacement matrix) is the operator that does this, and Binv is its inverse
    # - at this point we have defined the virtual strains to equal the stress
    #   sensitivity (but boundary conditions have not been applied yet)
    # - apply boundary conditions to virtual displacements
    # - at this point we need to recompute virtual strains that correspond to the new virtual displacements
    # - matrix multiply Bglob by virtual displacements to generate virtual strain(?)
    # - store this timestep's virtual strain for each component
    # - if we need to output virtual displacements at all datapoints, use global shape function (Nglob)
    # - apply mean virtual displacement to loading edges

    num_timesteps = stress_sensitivity.shape[0]

    for t in range(num_timesteps):
        # currExx = stress_sensitivity[t, 0, :, :]
        # test = stress_sensitivity[t, 0, :, :].flatten(order='F')
        virtual_strains = np.concatenate([
            stress_sensitivity[t, 0, :, :].flatten(order='F')[virtual_fields_mesh["indexlist"] - 1],
            stress_sensitivity[t, 1, :, :].flatten(order='F')[virtual_fields_mesh["indexlist"] - 1],
            stress_sensitivity[t, 2, :, :].flatten(order='F')[virtual_fields_mesh["indexlist"] - 1]
        ])
        print("for loop break")

    print("break")

data = loadmat(
    "/Users/chris/work/vfmap-numerical-paper/test_data/sensitivity_virtual_fields_input.mat",
    struct_as_record=False,
    squeeze_me=True,
    simplify_cells=True
)

# TODO: will this contains nans? matblab replaces nans with zeros
stress_sensitivity = data["refmap"]
stress_sensitivity = np.transpose(stress_sensitivity, (3, 2, 0, 1))
mesh_data = data["meshData"]

generate_sensitivity_based_virtual_fields(stress_sensitivity, mesh_data)

from dataclasses import dataclass
import enum

import numpy as np
import numpy.typing as npt
from scipy.io import loadmat

from pyvale.vfm.virtual_fields_mesh import (
    EBoundaryConditionSetting,
    BoundaryConditionSettings,
    EEdge,
    VirtualFieldsMesh,
)

@dataclass(slots=True)
class SensitivityBasedVirtualFields:
    virtual_strain: npt.NDArray[np.float64] # (timestep, component, y, x)
    virtual_displacement: npt.NDArray[np.float64] # (timestep, component, edge)

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

# TODO: return type
# TODO: vectorise
# TODO: matlab calls this function for each degree of freedom, but we could compute all dofs in this func
# TODO: support non linear geometries
# TODO: unpack some fields of virtual fields mesh for readability
# TODO: add virtual displacement at all data points to output (for dynamics)?
# TODO: pre allocate memory for anything in the main loop that needs it
# TODO: should we consider creating an enum for components to use in indexing?
def generate_sensitivity_based_virtual_fields(
    stress_sensitivity: npt.NDArray[np.float64], # (timestep, component, y, x)
    virtual_fields_mesh: VirtualFieldsMesh
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
    # x and y components
    num_components = 2
    num_edges = 4

    sensitivity_based_virtual_fields = SensitivityBasedVirtualFields(
        np.empty_like(stress_sensitivity),
        np.zeros((num_timesteps, num_components, num_edges))
    )

    num_degrees_of_freedom = (
        virtual_fields_mesh.x.shape[0] * virtual_fields_mesh.x.shape[1] * num_components
    )

    # TODO: need to understand the structure of b_inv to do the below
    # In Matlab's virtual displacement the layout is (x_component, y_component)
    # repeated times the size of the virtual mesh
    # I think we can assume that indexing increases column wise in the matlab
    #
    # Shape (num_components, y, x)
    # component 0 is x
    # component 1 is y
    virtual_displacement = np.zeros(
        (num_components, virtual_fields_mesh.y.shape[0], virtual_fields_mesh.x.shape[1])
    )
    virtual_displacement_flattened = np.zeros(
        num_components * virtual_fields_mesh.y.shape[0] * virtual_fields_mesh.x.shape[1]
    )

    for t in range(num_timesteps):
        # TODO: will we need to flatten with order F when our SS data comes from python?
        virtual_strain = np.concatenate([
            stress_sensitivity[t, 0, :, :].flatten(order='F')[virtual_fields_mesh.indices],
            stress_sensitivity[t, 1, :, :].flatten(order='F')[virtual_fields_mesh.indices],
            stress_sensitivity[t, 2, :, :].flatten(order='F')[virtual_fields_mesh.indices]
        ])

        virtual_displacement_flattened[virtual_fields_mesh.act_dofs] = virtual_fields_mesh.b_inv.dot(virtual_strain)

        virtual_displacement = unpack_virtual_displacement(virtual_displacement_flattened)

        virtual_displacement = apply_boundary_conditions(virtual_displacement)

        virtual_displacement_flattened = flatten_virtual_displacement(virtual_displacement)

        virtual_strain = virtual_fields_mesh.b_glob.dot(virtual_displacement_flattened)

        virtual_strain_x = np.full((stress_sensitivity.shape[2] * stress_sensitivity.shape[3]), np.nan)
        virtual_strain_y = np.full_like(virtual_strain_x, np.nan)
        virtual_strain_z = np.full_like(virtual_strain_x, np.nan)

        virtual_strain_x[virtual_fields_mesh.indices] = virtual_strain[0:virtual_fields_mesh.indices.size]
        virtual_strain_x = virtual_strain_x.reshape(
            (stress_sensitivity.shape[2], stress_sensitivity.shape[3]), order="F"
        )

        virtual_strain_y[virtual_fields_mesh.indices] = virtual_strain[virtual_fields_mesh.indices.size:2 * virtual_fields_mesh.indices.size]
        virtual_strain_y = virtual_strain_y.reshape(
            (stress_sensitivity.shape[2], stress_sensitivity.shape[3]), order="F"
        )

        virtual_strain_z[virtual_fields_mesh.indices] = virtual_strain[2 * virtual_fields_mesh.indices.size:3 * virtual_fields_mesh.indices.size]
        virtual_strain_z = virtual_strain_z.reshape(
            (stress_sensitivity.shape[2], stress_sensitivity.shape[3]), order="F"
        )
        
        sensitivity_based_virtual_fields.virtual_strain[t, 0, :, :] = virtual_strain_x
        sensitivity_based_virtual_fields.virtual_strain[t, 1, :, :] = virtual_strain_y
        sensitivity_based_virtual_fields.virtual_strain[t, 2, :, :] = virtual_strain_z

        # TODO: use Edge enum to index into this array for edges
        sensitivity_based_virtual_fields.virtual_displacement[t, 0, EEdge.Top.value] = virtual_displacement[0, 0, :].mean()
        sensitivity_based_virtual_fields.virtual_displacement[t, 0, EEdge.Bottom.value] = virtual_displacement[0, -1, :].mean()
        sensitivity_based_virtual_fields.virtual_displacement[t, 0, EEdge.Left.value] = virtual_displacement[0, :, 0].mean()
        sensitivity_based_virtual_fields.virtual_displacement[t, 0, EEdge.Right.value] = virtual_displacement[0, :, -1].mean()

        sensitivity_based_virtual_fields.virtual_displacement[t, 1, EEdge.Top.value] = virtual_displacement[1, 0, :].mean()
        sensitivity_based_virtual_fields.virtual_displacement[t, 1, EEdge.Bottom.value] = virtual_displacement[1, -1, :].mean()
        sensitivity_based_virtual_fields.virtual_displacement[t, 1, EEdge.Left.value] = virtual_displacement[1, :, 0].mean()
        sensitivity_based_virtual_fields.virtual_displacement[t, 1, EEdge.Right.value] = virtual_displacement[1, :, -1].mean()

        # End of loop var resets
        virtual_displacement.fill(0)
        virtual_displacement_flattened.fill(0)

    return sensitivity_based_virtual_fields


# TODO: verify the validity of this with Rob
def unpack_virtual_displacement(flattened_virtual_displacement):
    # TODO: remove the order F when we are getting python data
    # Convert to the form (degrees_of_freedom, y, x)
    # TODO: in the current form the x dof is 0 and y is 1,
    #       is that correct or should we swap them to be y then x?
    virtual_displacement_x = flattened_virtual_displacement[0::2].reshape(
        virtual_fields_mesh.y.shape[0],
        virtual_fields_mesh.x.shape[1],
        order="F"
    )

    virtual_displacement_y = flattened_virtual_displacement[1::2].reshape(
        virtual_fields_mesh.y.shape[0],
        virtual_fields_mesh.x.shape[1],
        order="F"
    )

    virtual_displacement = np.stack(
        (virtual_displacement_x, virtual_displacement_y),
        axis=0
    )

    return virtual_displacement


def flatten_virtual_displacement(virtual_displacement):
    virtual_displacement_x = virtual_displacement[0]
    virtual_displacement_y = virtual_displacement[1]

    # TODO: remove the order F when we are getting python data
    flattened_x = virtual_displacement_x.ravel(order="F")
    flattened_y = virtual_displacement_y.ravel(order="F")

    flattened_virtual_displacement = np.empty(flattened_x.size * 2, dtype=flattened_x.dtype)

    flattened_virtual_displacement[0::2] = flattened_x
    flattened_virtual_displacement[1::2] = flattened_y

    return flattened_virtual_displacement


# Apply boundary conditions
# seems like edge convention in the matlab is
#   - 1 = top
#   - 2 = left
#   - 3 = bottom
#   - 4 = right
# for each edge (top, bottom, left, right)
#   - collect the indices in virtual displacements that corresponds to the edge dofs
#   - get the "master value"
#     - top/bottom master value is top left of grid
#     - left/right master value is bottom right of grid
#   - based on our boundary condition settings either
#     - do nothing to virtual displacements
#     - set the virtual displacement to 0 (clamped edge?)
#     - set the virtual displacement for  all edge nodes
#       to the same value (master dof in the matlab)
def apply_boundary_conditions(virtual_displacements):
    for e in EEdge:
        match e:
            case EEdge.Top:
                edge_elements = virtual_fields_mesh.virtual_elements[0, :]
                constant_value_x = virtual_displacements[0, 0, 0]
                constant_value_y = virtual_displacements[1, 0, 0]
            case EEdge.Bottom:
                edge_elements = virtual_fields_mesh.virtual_elements[-1, :]
                constant_value_x = virtual_displacements[0, -1, -1]
                constant_value_y = virtual_displacements[1, -1, -1]
            case EEdge.Left:
                edge_elements = virtual_fields_mesh.virtual_elements[:, 0]
                constant_value_x = virtual_displacements[0, 0, 0]
                constant_value_y = virtual_displacements[1, 0, 0]
            case EEdge.Right:
                edge_elements = virtual_fields_mesh.virtual_elements[:, -1]
                constant_value_x = virtual_displacements[0, -1, -1]
                constant_value_y = virtual_displacements[1, -1, -1]

        bc_setting_x = virtual_fields_mesh.boundary_condition_settings.x[e]
        bc_setting_y = virtual_fields_mesh.boundary_condition_settings.y[e]

        # TODO: flip operators when we accept the row major virtual elements
        x_indices = edge_elements // virtual_fields_mesh.x.shape[1]
        y_indices = edge_elements % virtual_fields_mesh.y.shape[0]

        match bc_setting_x:
            case EBoundaryConditionSetting.Fixed:
                virtual_displacements[0, y_indices, x_indices] = 0
            case EBoundaryConditionSetting.Constant:
                virtual_displacements[0, y_indices, x_indices] = constant_value_x

        match bc_setting_y:
            case EBoundaryConditionSetting.Fixed:
                virtual_displacements[1, y_indices, x_indices] = 0
            case EBoundaryConditionSetting.Constant:
                virtual_displacements[0, y_indices, x_indices] = constant_value_y

    return virtual_displacements

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

x = mesh_data["vXcoords"]
y = mesh_data["vYcoords"]
b_glob = mesh_data["Bglob"].toarray()
b_inv = mesh_data["Binv"].toarray()
# -1 on each element as matlab indexing starts at 1
active_degrees_of_freedom = mesh_data["actDofs"] - 1
# -1 on each element as matlab indexing starts at 1
virtual_element_connectivity = mesh_data["virtConnectivity"] - 1
# TODO: Transposing so indices increase in a row major ordering, which is
# idiomatic for numpy
# virtual_elements = mesh_data["vCoordsGrid"].T - 1
# elements need to be 0 indexed
virtual_elements = mesh_data["vCoordsGrid"] - 1
boundary_condition_settings = mesh_data["BC_settings"]
boundary_condition_settings = BoundaryConditionSettings(
    {
        EEdge.Top: EBoundaryConditionSetting.Free,
        EEdge.Bottom: EBoundaryConditionSetting.Fixed,
        EEdge.Left: EBoundaryConditionSetting.Free,
        EEdge.Right: EBoundaryConditionSetting.Constant,
    },
    {
        EEdge.Top: EBoundaryConditionSetting.Free,
        EEdge.Bottom: EBoundaryConditionSetting.Fixed,
        EEdge.Left: EBoundaryConditionSetting.Free,
        EEdge.Right: EBoundaryConditionSetting.Fixed,
    }
)

# -1 on each element as matlab indexing starts at 1
indices = mesh_data["indexlist"] - 1
n_glob = mesh_data["NGlob"].toarray()
virtual_element_point_mapping = mesh_data["ptElemAss"]
free_degrees_of_freedom = mesh_data["freeDof"]

virtual_fields_mesh = VirtualFieldsMesh(
    x,
    y,
    b_glob,
    b_inv,
    active_degrees_of_freedom,
    virtual_element_connectivity,
    virtual_elements,
    boundary_condition_settings,
    indices,
    n_glob,
    virtual_element_point_mapping,
    free_degrees_of_freedom
)

generate_sensitivity_based_virtual_fields(stress_sensitivity, virtual_fields_mesh)

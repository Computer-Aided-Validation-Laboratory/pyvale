from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.stress_sensitivity import StressSensitivity
from pyvale.vfm.virtual_fields_mesh import VirtualFieldsMesh


@dataclass(slots=True)
class SensitivityBasedVirtualFields:
    """Virtual strains and edge displacements generated from one sensitivity map."""

    virtual_strain: npt.NDArray[np.float64]
    edge_displacement: npt.NDArray[np.float64]
    full_displacement: npt.NDArray[np.float64]


def generate_sensitivity_based_virtual_fields(
    stress_sensitivities: dict[str, StressSensitivity],
    virtual_fields_mesh: VirtualFieldsMesh,
    use_incremental: bool = False,
) -> dict[str, SensitivityBasedVirtualFields]:
    """Generate one SBVF set per active DOF.

    This follows the MATLAB `sensitivityVFs.m` idea fairly directly:
    project a stress-sensitivity field onto the virtual mesh, recover the
    nodal virtual displacements, enforce the edge boundary conditions, then
    rebuild virtual strains from those displacements.
    """

    virtual_fields: dict[str, SensitivityBasedVirtualFields] = {}

    for dof_uid, sensitivity in stress_sensitivities.items():
        sensitivity_map = sensitivity.incremental if use_incremental else sensitivity.total
        virtual_fields[dof_uid] = _generate_for_one_dof(
            sensitivity_map,
            virtual_fields_mesh,
        )

    return virtual_fields


def _generate_for_one_dof(
    sensitivity_map: npt.NDArray[np.float64],
    virtual_fields_mesh: VirtualFieldsMesh,
) -> SensitivityBasedVirtualFields:
    num_timesteps, _, size_y, size_x = sensitivity_map.shape
    num_measured_points = int(virtual_fields_mesh.indices.size)
    num_dofs = int(virtual_fields_mesh.b_glob.shape[1])

    virtual_strain = np.full((num_timesteps, 3, size_y, size_x), np.nan, dtype=np.float64)
    edge_displacement = np.zeros((num_timesteps, 2, 4), dtype=np.float64)
    full_displacement = np.full((num_timesteps, 2, size_y, size_x), np.nan, dtype=np.float64)

    for timestep in range(num_timesteps):
        target_strain = np.concatenate(
            [
                sensitivity_map[timestep, 0, :, :].flatten(order="F")[virtual_fields_mesh.indices],
                sensitivity_map[timestep, 1, :, :].flatten(order="F")[virtual_fields_mesh.indices],
                sensitivity_map[timestep, 2, :, :].flatten(order="F")[virtual_fields_mesh.indices],
            ]
        )
        target_strain = np.nan_to_num(target_strain, nan=0.0)

        virtual_displacement_vector = np.zeros(num_dofs, dtype=np.float64)
        virtual_displacement_vector[virtual_fields_mesh.act_dofs] = (
            virtual_fields_mesh.b_inv @ target_strain
        )
        virtual_displacement_vector = _apply_boundary_conditions(
            virtual_displacement_vector,
            virtual_fields_mesh.boundary_condition_settings,
            virtual_fields_mesh.virtual_elements,
        )

        reconstructed_virtual_strain = (
            virtual_fields_mesh.b_glob @ virtual_displacement_vector
        )

        for component in range(3):
            component_map = np.full(size_x * size_y, np.nan, dtype=np.float64)
            start = component * num_measured_points
            stop = (component + 1) * num_measured_points
            component_map[virtual_fields_mesh.indices] = reconstructed_virtual_strain[start:stop]
            virtual_strain[timestep, component, :, :] = component_map.reshape(
                (size_y, size_x),
                order="F",
            )

        x_displacement = virtual_fields_mesh.n_glob @ virtual_displacement_vector[0::2]
        y_displacement = virtual_fields_mesh.n_glob @ virtual_displacement_vector[1::2]

        flat_x = np.full(size_x * size_y, np.nan, dtype=np.float64)
        flat_y = np.full(size_x * size_y, np.nan, dtype=np.float64)
        flat_x[virtual_fields_mesh.indices] = x_displacement
        flat_y[virtual_fields_mesh.indices] = y_displacement
        full_displacement[timestep, 0, :, :] = flat_x.reshape((size_y, size_x), order="F")
        full_displacement[timestep, 1, :, :] = flat_y.reshape((size_y, size_x), order="F")

        edge_displacement[timestep, 0, 0] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[0, :]])
        edge_displacement[timestep, 0, 1] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[:, 0]])
        edge_displacement[timestep, 0, 2] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[-1, :]])
        edge_displacement[timestep, 0, 3] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[:, -1]])

        edge_displacement[timestep, 1, 0] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[0, :] + 1])
        edge_displacement[timestep, 1, 1] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[:, 0] + 1])
        edge_displacement[timestep, 1, 2] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[-1, :] + 1])
        edge_displacement[timestep, 1, 3] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[:, -1] + 1])

    return SensitivityBasedVirtualFields(
        virtual_strain=virtual_strain,
        edge_displacement=edge_displacement,
        full_displacement=full_displacement,
    )


def _apply_boundary_conditions(
    virtual_displacement: npt.NDArray[np.float64],
    settings: npt.NDArray[np.uint32],
    virtual_elements: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    updated = virtual_displacement.copy()

    for edge in range(4):
        if edge == 0:
            edge_nodes = virtual_elements[0, :]
            master_node = virtual_elements[0, 0]
            slave_nodes = edge_nodes[1:]
        elif edge == 1:
            edge_nodes = virtual_elements[:, 0]
            master_node = virtual_elements[0, 0]
            slave_nodes = edge_nodes[1:]
        elif edge == 2:
            edge_nodes = virtual_elements[-1, :]
            master_node = virtual_elements[-1, -1]
            slave_nodes = edge_nodes[:-1]
        else:
            edge_nodes = virtual_elements[:, -1]
            master_node = virtual_elements[-1, -1]
            slave_nodes = edge_nodes[:-1]

        edge_dofs_x = 2 * edge_nodes
        edge_dofs_y = edge_dofs_x + 1
        master_dof_x = 2 * master_node
        master_dof_y = master_dof_x + 1
        slave_dofs_x = 2 * slave_nodes
        slave_dofs_y = slave_dofs_x + 1

        if settings[0, edge] == 1:
            updated[edge_dofs_x] = 0.0
        elif settings[0, edge] == 2:
            updated[slave_dofs_x] = updated[master_dof_x]

        if settings[1, edge] == 1:
            updated[edge_dofs_y] = 0.0
        elif settings[1, edge] == 2:
            updated[slave_dofs_y] = updated[master_dof_y]

    return updated

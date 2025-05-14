# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
import numpy as np
from pyvale.rendermesh import RenderMeshData

class SimTools:
    """Namespace for tools required for analysing simulation results.
    """

    @staticmethod
    def centre_mesh_nodes(nodes: np.ndarray, spat_dim: int) -> np.ndarray:
        """A method to centre the nodes of a mesh around the origin.

        Parameters
        ----------
        nodes : np.ndarray
            An array containing the node locations of the mesh.
        spat_dim : int
            The spatial dimension of the mesh.

        Returns
        -------
        np.ndarray
            An array containing the mesh node locations, but centred around
            the origin.
        """
        max = np.max(nodes, axis=0)
        min = np.min(nodes, axis=0)
        middle = max - ((max - min) / 2)
        if spat_dim == 3:
            middle[2] = 0
        centred = np.subtract(nodes, middle)
        return centred

    @staticmethod
    def get_deformed_nodes(timestep: int,
                           render_mesh: RenderMeshData) -> np.ndarray | None:
        """A method to obtain the deformed locations of all the nodes at a given
            timestep.

        Parameters
        ----------
        timestep : int
            The timestep at which to find the deformed nodes.
        render_mesh: RenderMeshData
            A dataclass containing the skinned mesh and simulation results.

        Returns
        -------
        np.ndarray | None
            An array containing the deformed values of all the components at
            each node location. Returns None if there are no deformation values.
        """
        if render_mesh.fields_disp is None:
            return None

        added_disp = render_mesh.fields_disp[:, timestep]
        if added_disp.shape[1] == 2:
            added_disp = np.hstack((added_disp,np.zeros([added_disp.shape[0],1])))
        coords = np.delete(render_mesh.coords, 3, axis=1)
        deformed_nodes = coords + added_disp
        return deformed_nodes



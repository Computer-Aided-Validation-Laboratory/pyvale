# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
from typing import Any
import dataclasses
import numpy as np
import pyvale.mooseherder as mh
from pyvale.sensorsim.rendermesh import RenderMesh


class SimTools:
    @staticmethod
    def print_dataclass_fields(in_data: Any) -> None:
        """Diagnostic function

        Parameters
        ----------
        in_data : Any
            A data class to print the type and fields for as well as the type of
            each of the fields.
        """

        print(f"Data class fields for: {type(in_data)}")
        for field in dataclasses.fields(in_data):
            if not field.name.startswith('__'):
                print(f"    {field.name}: {field.type}")
        print()

    @staticmethod
    def print_sim_data(sim_data: mh.SimData) -> None:
        """Diagnostic function for inspecting a sim data object to work out shapes
        of time, coordinates, connectivity tables, node vars, elem vars as well as
        the associated keys in the dicttionaries for the connectivity,
        node/elem/glob vars.

        Parameters
        ----------
        sim_data : mh.SimData
            SimData to print shapes of numpy arrays.
        """
        print()
        if sim_data.time is not None:
            print(f"{sim_data.time.shape=}")
        print()

        if sim_data.coords is not None:
            print(f"{sim_data.coords.shape=}")
        print()

        def print_dict(in_dict: dict | None) -> None:
            if in_dict is None:
                print("    None\n")
                return

            print(f"keys={in_dict.keys()}")
            for kk in in_dict:
                print(f"    {kk}.shape={in_dict[kk].shape}")

            print()

        print("sim_data.connect")
        print_dict(sim_data.connect)
        print("sim_data.node_vars")
        print_dict(sim_data.node_vars)
        print("sim_data.elem_vars")
        print_dict(sim_data.elem_vars)
        print("sim_data.glob_vars")
        print_dict(sim_data.glob_vars)


    @staticmethod
    def print_dimensions(sim_data: mh.SimData) -> None:
        """Diagnostic function for quickly finding the coordinate limits for from a
        given simulation.

        Parameters
        ----------
        sim_data : mh.SimData
            Simulation data object containing the nodal coordinates.
        """
        print(80*"-")
        print("SimData Dimensions:")
        print(f"x [min,max] = [{np.min(sim_data.coords[:,0])}," + \
            f"{np.max(sim_data.coords[:,0])}]")
        print(f"y [min,max] = [{np.min(sim_data.coords[:,1])}," + \
            f"{np.max(sim_data.coords[:,1])}]")
        print(f"z [min,max] = [{np.min(sim_data.coords[:,2])}," + \
            f"{np.max(sim_data.coords[:,2])}]")
        print(f"t [min,max] = [{np.min(sim_data.time)},{np.max(sim_data.time)}]")
        print(80*"-")

    @staticmethod
    def get_sim_dims(sim_data: mh.SimData) -> dict[str,tuple[float,float]]:
        """Diagnostic function for extracting the dimensional limits in space and
        time from a SimData object. Useful for finding the spatial dimensions over
        which simulated sensors can be placed as well as the times over which they
        can sampled the underlying field.

        Parameters
        ----------
        sim_data : mh.SimData
            Simulation data object containing the coordinates and time steps.

        Returns
        -------
        dict[str,tuple[float,float]]
            Dictionary of space and time coordinate limits keyed as 'x','y','z' for
            the spatial dimensions and 't' for time. The dictionary will return a
            tuple with the (min,max) of the given dimension.
        """
        sim_dims = {}
        sim_dims["x"] = (np.min(sim_data.coords[:,0]),np.max(sim_data.coords[:,0]))
        sim_dims["y"] = (np.min(sim_data.coords[:,1]),np.max(sim_data.coords[:,1]))
        sim_dims["z"] = (np.min(sim_data.coords[:,2]),np.max(sim_data.coords[:,2]))
        sim_dims["t"] = (np.min(sim_data.time),np.max(sim_data.time))
        return sim_dims

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
                            render_mesh: RenderMesh) -> np.ndarray | None:
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



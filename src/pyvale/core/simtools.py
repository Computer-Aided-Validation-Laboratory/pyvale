"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import numpy as np
import pyvista as pv
import mooseherder as mh

class SimTools():
    """Namespace for tools required for handling simulation results.
    """

    @staticmethod
    def surf_mesh_elements_per_face(pv_surf: pv.PolyData) -> int:
        """A method to obtain the number of elements per face in a pyvista surface
            mesh

        Args:
            pv_surf (pv.PolyData): A pyvista surface mesh

        Returns:
            int: The number of elements per face of the mesh
        """
        elements_per_face = int((pv_surf.faces.shape[0] / pv_surf.n_cells))
        return elements_per_face

    @staticmethod
    def get_mesh_spat_dim(sim_data: mh.SimData) -> int:
        """A method to obtain the spatial dimension of the mesh

        Args:
            sim_data (mh.SimData): A SimData object containing the mesh

        Returns:
            int: The spat dimension of the mesh
        """
        nodes = sim_data.coords
        check_if_2d = np.count_nonzero(nodes, axis=0)
        if check_if_2d[2] == 0:
            spat_dim = 2
        else:
            spat_dim = 3
        return spat_dim

    @staticmethod
    def get_simulation_components(sim_data: mh.SimData) -> tuple | None:
        """A method to obtain the measured simulation components from the SimData
            object.

        Args:
            sim_data (mh.SimData): A SimData object containing the simulation
                results

        Returns:
            tuple | None: A tuple of the variable names present in the
                simulation results. Returns None if none exist.
        """
        node_vars = sim_data.node_vars
        node_vars_names = list(node_vars.keys())
        components = []
        if 'disp_x' in node_vars_names:
            components.append('disp_x')
        if 'disp_y' in node_vars_names:
            components.append('disp_y')
        if 'disp_z' in node_vars_names:
            components.append('disp_z')
        components = tuple(components)
        if len(components) == 0:
            components = None
        return components

    @staticmethod
    def centre_mesh_nodes(nodes: np.ndarray, spat_dim: int) -> np.ndarray:
        """A method to centre the nodes of a mesh, given the mesh as a numpy array

        Args:
            nodes (np.ndarray): The node locations of the mesh
            spat_dim (int): The spatial dimension of the mesh

        Returns:
            np.ndarray: The mesh node locations, but centred around (0, 0, 0)
        """
        max = np.max(nodes, axis=0)
        min = np.min(nodes, axis=0)
        middle = max - ((max - min) / 2)
        if spat_dim == 3:
            middle[2] = 0
        centred = np.subtract(nodes, middle)
        return centred

    @staticmethod
    def conv_pvgrid_to_pvsurf(pv_grid: pv.UnstructuredGrid) -> pv.PolyData:
        """A method to convert a pyvista grid object into a pyvista surface mesh

        Args:
            pv_grid (pv.UnstructuredGrid): The pyvista grid mesh object

        Returns:
            pv.PolyData: A pyvista surface mesh
        """
        pv_surf = pv_grid.extract_surface()
        return pv_surf

    @staticmethod
    def triangulate_pv_surf_mesh(pv_surf: pv.PolyData) -> pv.PolyData:
        """A method to triangulate a pyvista surface mesh

        Args:
            pv_surf (pv.PolyData): A pyvista surface mesh

        Returns:
            pv.PolyData: A triangulated pyvista surface mesh
        """
        tri_surf = pv_surf.triangulate()
        return tri_surf

    @staticmethod
    def get_deformed_nodes(timestep: int,
                           pv_surf: pv.PolyData,
                           spat_dim: int,
                           components: tuple) -> np.ndarray | None:
        """A method to obtain the deformed location of all the nodes at a given
            timestep

        Args:
            timestep (int): The timestep at which to find the deformed nodes
            pv_surf (pv.PolyData): A pyvista surface mesh
            spat_dim (int): The spatial dimension of the mesh
            components (tuple): The simulated component variable names

        Returns:
            np.ndarray | None: The deformed values of all the components at each
                node location. Returns None if simulation results do not contain
                the given components.
        """
        if set(components).issubset(pv_surf.array_names):
            added_disp = np.zeros_like(pv_surf.points)
            dim = 0
            for component in components:
                added_disp_1d = pv_surf.get_array(component)[:, timestep]
                added_disp[:, dim] = added_disp_1d * 1000
                dim += 1
            deformed_nodes = pv_surf.points + added_disp
            return deformed_nodes
        else:
            return None



"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from abc import ABC, abstractmethod
import numpy as np
import pyvista as pv
import mooseherder as mh

class SimTools(ABC):
    """Interface (abstract base class) for tools relating to simulation results
    within pyvale

    #TODO: Add to this

    """

    @abstractmethod
    def surf_mesh_elements_per_face(self, pv_surf: pv.PolyData) -> int:
        elements_per_face = int((pv_surf.faces.shape[0] / pv_surf.n_cells))
        return elements_per_face

    @abstractmethod
    def get_mesh_spat_dim(self, sim_data: mh.SimData) -> int:
        nodes = self.sim_data.coords
        check_if_2d = np.count_nonzero(nodes, axis=0)
        if check_if_2d[2] == 0:
            spat_dim = 2
        else:
            spat_dim = 3
        return spat_dim

    @abstractmethod
    def get_simulation_components(self, sim_data: mh.SimData) -> tuple | None:
        node_vars = self.sim_data.node_vars
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

    @abstractmethod
    def conv_pvgrid_to_pvsurf(self, pv_grid: pv.UnstructuredGrid) -> pv.PolyData:
        pv_surf = pv_grid.extract_surface()
        return pv_surf

    @abstractmethod
    def triangulate_pv_surf_mesh(self, pv_surf: pv.PolyData) -> pv.PolyData:
        tri_surf = pv_surf.triangulate()
        return tri_surf

    @abstractmethod
    def get_deformed_nodes(self,
                           timestep: int,
                           pv_surf: pv.PolyData,
                           spat_dim: int,
                           components: tuple) -> np.ndarray | None:
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



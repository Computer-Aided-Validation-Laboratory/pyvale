"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from abc import ABC, abstractmethod
import pyvista as pv
import numpy as np
from pathlib import Path
import os
from scipy.spatial.transform import Rotation
import bpy
import mooseherder as mh

# NOTE: This module is a feature under development

class BlenderError(Exception):
    pass

class BlenderTools(ABC):
    """Interface (abstract base class) for tools to be used within Blender
    feature of `pyvale`.

    #TODO: Add to this

    """

    @abstractmethod
    def save_blender_file(self, filepath: Path, override: bool = False):
        # Unsure whether this fits best within blendertools or blenderscene
        # TODO: Make this only use Path - .exists()
        if filepath.exists():
                if override is True:
                    filepath.unlink()
                else:
                    raise BlenderError("A file already exists with this filepath")
        filepath = str(filepath)
        bpy.ops.wm.save_as_mainfile(filepath=filepath)

    @abstractmethod
    def move_blender_part(self, pos_world: np.ndarray, part):
        z_location = int(part.dimensions[2])
        part.location = (pos_world[0], pos_world[1], (pos_world[2] - z_location))

    @abstractmethod
    def rotate_blender_part(self, rot_world: Rotation, part):
        part.rotation_mode = 'XYZ'
        rot_euler = Rotation.as_euler(rot_world)
        part.rotation_euler = rot_euler

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
    def centre_mesh_nodes(nodes: np.ndarray) -> np.ndarray:
        max = np.max(nodes, axis=0)
        min = np.min(nodes, axis=0)
        middle = max - ((max - min) / 2)
        centred = np.subtract(nodes, middle)
        return centred




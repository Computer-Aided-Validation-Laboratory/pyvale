# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: this module is a feature under developement.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import pyvale.mooseherder as mh
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_interp
from pyvale.sensorsim.enums import EDim

# TODO:
# - Store the render field keys and match them between meshes?


class RenderMesh:
    __slots__ = ("coords","connectivity","fields_render","fields_disp",
                 "pos_world","rot_world","node_count","elem_count",
                 "nodes_per_elem","mesh_to_world_mat","world_to_mesh_mat")

    def __init__(self,
                 coords: np.ndarray,
                 connectivity: np.ndarray,
                 fields_render: np.ndarray,
                 fields_disp: np.ndarray | None = None,
                 pos_world: np.ndarray | None = None,
                 rot_world: Rotation | None = None) -> None:

        self.coords = coords
        self.connectivity = connectivity
        self.fields_render = fields_render
        self.fields_disp = fields_disp

        self.node_count = self.coords.shape[0]
        self.elem_count = self.connectivity.shape[0]
        self.nodes_per_elem = self.connectivity.shape[1]

        if pos_world is None:
            self.pos_world = np.array((0.0,0.0,0.0),dtype=np.float64)

        if rot_world is None:
            self.rot_world = Rotation.from_euler("zyx",(0.0,0.0,0.0),degrees=True)

        self.mesh_to_world_mat = np.zeros((4,4),dtype=np.float64)
        self.world_to_mesh_mat = np.zeros((4,4),dtype=np.float64)
        self._build_transform_mats()

    def _build_transform_mats(self) -> None:
        self.mesh_to_world_mat = np.zeros((4,4))
        self.mesh_to_world_mat[0:3,0:3] = self.rot_world.as_matrix()
        self.mesh_to_world_mat[-1,-1] = 1.0
        self.mesh_to_world_mat[0:3,-1] = self.pos_world
        self.world_to_mesh_mat = np.linalg.inv(self.mesh_to_world_mat)

    def set_pos(self, pos_world: np.ndarray) -> None:
        self.pos_world = pos_world
        self._build_transform_mats()

    def set_rot(self, rot_world: Rotation) -> None:
        self.rot_world = rot_world
        self._build_transform_mats()



def create_render_mesh(sim_data: mh.SimData,
                       field_render_keys: tuple[str,...],
                       sim_spat_dim: EDim,
                       field_disp_keys: tuple[str,...] | None = None,
                       pos_world: np.ndarray | None  = None,
                       rot_world: Rotation | None = None
                       ) -> RenderMesh:

    extract_keys = field_render_keys
    if field_disp_keys is not None:
        extract_keys = field_render_keys+field_disp_keys

    pv_grid = simdata_to_pyvista_interp(sim_data,
                                        extract_keys,
                                        spatial_dims=sim_spat_dim)

    pv_surf = pv_grid.extract_surface()
    faces = np.array(pv_surf.faces)

    first_elem_nodes_per_face = faces[0]
    nodes_per_face_vec = faces[0::(first_elem_nodes_per_face+1)]

    # TODO: CHECKS
    # - Number of displacement keys match the spat_dim parameter
    assert np.all(nodes_per_face_vec == first_elem_nodes_per_face), \
    "Not all elements in the simdata object have the same number of nodes per element"

    nodes_per_face = first_elem_nodes_per_face
    num_faces = int(faces.shape[0] / (nodes_per_face+1))

    # Reshape the faces table and slice off the first column which is just the
    # number of nodes per element and should be the same for all elements
    connectivity = np.reshape(faces,(num_faces,nodes_per_face+1))
    # shape=(num_elems,nodes_per_elem), C format
    connectivity = np.ascontiguousarray(connectivity[:,1:],dtype=np.uintp)

    # shape=(num_nodes,3), C format
    coords_world = np.array(pv_surf.points)

    # Add w coord=1, shape=(num_nodes,3+1)
    coords_world= np.hstack((coords_world,np.ones([coords_world.shape[0],1])))

    # shape=(num_nodes,num_time_steps,num_components)
    field_render_shape = np.array(pv_surf[field_render_keys[0]]).shape
    fields_render_by_node = np.zeros(field_render_shape+(len(field_render_keys),),
                                     dtype=np.float64)
    for ii,cc in enumerate(field_render_keys):
        fields_render_by_node[:,:,ii] = np.ascontiguousarray(
            np.array(pv_surf[cc]))


    field_disp_by_node = None
    if field_disp_keys is not None:
        field_disp_shape = np.array(pv_surf[field_disp_keys[0]]).shape
        # shape=(num_nodes,num_time_steps,num_components)
        field_disp_by_node = np.zeros(field_disp_shape+(len(field_disp_keys),),
                                       dtype=np.float64)
        for ii,cc in enumerate(field_disp_keys):
            field_disp_by_node[:,:,ii] = np.ascontiguousarray(
                np.array(pv_surf[cc]))

    return RenderMesh(coords=coords_world,
                          connectivity=connectivity,
                          fields_render=fields_render_by_node,
                          fields_disp=field_disp_by_node,
                          pos_world=pos_world,
                          rot_world=rot_world)


"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""
#from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import mooseherder as mh
from pyvale.core.fieldconverter import simdata_to_pyvista

# NOTE: This module is a feature under developement.

# TODO:
# - Store the render field keys and match them between meshes?

@dataclass(slots=True)
class RenderMeshData:
    coords: np.ndarray
    connectivity: np.ndarray
    fields_render: np.ndarray
    # If this is None then the mesh is not deformable
    fields_disp: np.ndarray | None = None

    node_count: int = field(init=False)
    elem_count: int = field(init=False)
    nodes_per_elem: int = field(init=False)

    coord_cent: np.ndarray = field(init=False)
    coord_bound_min: np.ndarray = field(init=False)
    coord_bound_max: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # C format: num_nodes/num_elems first as it is the largest dimension
        self.node_count = self.coords.shape[0]
        self.elem_count = self.connectivity.shape[0]
        self.nodes_per_elem = self.connectivity.shape[1]

        self.coord_bound_min = np.min(self.coords,axis=0)
        self.coord_bound_max = np.max(self.coords,axis=0)
        self.coord_cent = (self.coord_bound_max + self.coord_bound_min)/2.0

        if self.fields_disp is None:
            self.fields_disp = np.zeros((self.node_count,),dtype=np.float64)



def create_render_mesh(sim_data: mh.SimData,
                       field_render_keys: tuple[str,...],
                       sim_spat_dim: int,
                       field_disp_keys: tuple[str,...] | None = None,
                       ) -> RenderMeshData:

    extract_keys = field_render_keys
    if field_disp_keys is not None:
        extract_keys = field_render_keys+field_disp_keys

    (pv_grid,_) = simdata_to_pyvista(sim_data,
                                     extract_keys,
                                     spat_dim=sim_spat_dim)

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

    return RenderMeshData(coords=coords_world,
                          connectivity=connectivity,
                          fields_render=fields_render_by_node,
                          fields_disp=field_disp_by_node)


# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: this module is a feature under developement.
"""

from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import mooseherder as mh
from pyvale.fieldconverter import simdata_to_pyvista


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
                                     elem_dims=sim_spat_dim)

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


def slice_mesh_data_by_elem(coords_world: np.ndarray,
                            connectivity: np.ndarray,
                            field_by_node: np.ndarray,
                            ) -> tuple[np.ndarray,np.ndarray]:
    """_summary_

    Parameters
    ----------
    coords_world : np.ndarray
        _description_
    connectivity : np.ndarray
        _description_
    field_by_node : np.ndarray
        _description_

    Returns
    -------
    tuple[np.ndarray,np.ndarray]
        _description_
    """
    # shape=(coord[X,Y,Z,W],node_per_elem,elem_num)
    elem_world_coords = np.copy(coords_world[connectivity,:])

    # shape=(elem_num,nodes_per_elem,coord[X,Y,Z,W]), C memory format
    # elem_world_coords = np.ascontiguousarray(np.swapaxes(elem_world_coords,0,2))
    elem_world_coords = np.ascontiguousarray(elem_world_coords)

    # shape=(nodes_per_elem,elem_num,time_steps)
    field_by_elem = np.copy(field_by_node[connectivity,:])

    # shape=(elem_num,nodes_per_elem,time_steps), C memory format
    # field_by_elem = np.ascontiguousarray(np.swapaxes(field_by_elem,0,1))
    field_by_elem = np.ascontiguousarray(field_by_elem)

    return (elem_world_coords,field_by_elem)
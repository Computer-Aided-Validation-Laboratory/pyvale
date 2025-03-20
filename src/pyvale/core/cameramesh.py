"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import mooseherder as mh

from pyvale.core.fieldconverter import conv_simdata_to_pyvista


@dataclass(slots=True)
class CameraMeshData:
    name: str

    coords: np.ndarray
    connectivity: np.ndarray
    fields_by_node: np.ndarray

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


def create_camera_mesh(sim_path: Path,
                       field_key: str,
                       components: tuple[str,...],
                       spat_dim: int
                       ) -> CameraMeshData:
    # TODO: fix the unit conversion here, this should probably not be done
    sim_data = mh.ExodusReader(sim_path).read_all_sim_data()
    sim_data.coords = sim_data.coords*1000.0 # scale to mm

    (pv_grid,_) = conv_simdata_to_pyvista(sim_data,
                                          components,
                                          spat_dim=spat_dim)

    pv_surf = pv_grid.extract_surface()
    faces = np.array(pv_surf.faces)

    first_elem_nodes_per_face = faces[0]
    nodes_per_face_vec = faces[0::(first_elem_nodes_per_face+1)]
    assert np.all(nodes_per_face_vec == first_elem_nodes_per_face), \
    "Not all elements in the simdata object have the same number of nodes per element"

    nodes_per_face = first_elem_nodes_per_face
    num_faces = int(faces.shape[0] / (nodes_per_face+1))

    # Reshape the faces table and slice off the first column which is just the
    # number of nodes per element and should be the same for all elements
    connectivity = np.reshape(faces,(num_faces,nodes_per_face+1))
    # shape=(num_elems,nodes_per_elem), C format
    connectivity = connectivity[:,1:]

    # shape=(num_nodes,3), C format
    coords_world = np.array(pv_surf.points)

    # Add w coord =1, shape=(num_nodes,1)
    coords_world= np.hstack((coords_world,np.ones([coords_world.shape[0],1])))

    # shape=(num_nodes,num_time_steps,num_components)
    field_shape = np.array(pv_surf[field_key]).shape
    fields_by_node = np.zeros(field_shape+(len(components),),dtype=np.float64)
    for ii,cc in enumerate(components):
        fields_by_node[:,:,ii] = np.ascontiguousarray(np.array(pv_surf[cc]))

    image_mesh_world = CameraMeshData(name=sim_path.name,
                                      coords=coords_world,
                                      connectivity=connectivity,
                                      fields_by_node=fields_by_node)

    return image_mesh_world


def slice_mesh_data_by_elem(coords_world: np.ndarray,
                            connectivity: np.ndarray,
                            field_by_node: np.ndarray,
                            ) -> tuple[np.ndarray,np.ndarray]:

    # shape=(coord[X,Y,Z,W],node_per_elem,elem_num)
    elem_world_coords = np.copy(coords_world[:,connectivity])
    # shape=(elem_num,nodes_per_elem,coord[X,Y,Z,W]), C memory format
    elem_world_coords = np.ascontiguousarray(np.swapaxes(elem_world_coords,0,2))

    # shape=(nodes_per_elem,elem_num,time_steps)
    field_by_elem = np.copy(field_by_node[connectivity,:])
    # shape=(elem_num,nodes_per_elem,time_steps), C memory format
    field_by_elem = np.ascontiguousarray(
                               np.swapaxes(field_by_elem,0,1))

    return (elem_world_coords,field_by_elem)
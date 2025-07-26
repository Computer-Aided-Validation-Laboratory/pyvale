#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import numpy as np
import pyvale.mooseherder as mh
import pyvale as pyv

# # TODO: make this work for sim_data with multiple connectivity
# def extract_surf_mesh(sim_data: mh.SimData) -> mh.SimData:

#     # NOTE: need to fix exodus 1 indexing for now and put it back at the end
#     # shape=(nodes_per_elem,num_elems)
#     connect = np.copy(sim_data.connect["connect1"])-1
#     num_elems = connect.shape[1]

#     assert "connect2" not in sim_data.connect, \
#         "Multiple connectivity tables not supported yet."

#     # Mapping of node numbers to faces for each element face
#     face_map = _get_surf_map(nodes_per_elem=connect.shape[0])
#     faces_per_elem = face_map.shape[0]
#     nodes_per_face = face_map.shape[1]

#     # shape=(faces_per_elem,nodes_per_face,num_elems)
#     faces_wound = connect[face_map,:]
#     # shape=(num_elems,faces_per_elem,nodes_per_face)
#     faces_wound = faces_wound.transpose((2,0,1))

#     # Create an array of all faces with shape=(total_faces,nodes_per_face)
#     faces_total = faces_per_elem*num_elems
#     faces_flat_wound = faces_wound.reshape((faces_total,nodes_per_face))
#     # Sort the rows so nodes are in the same order when comparing them
#     faces_flat_sorted = np.copy(np.sort(faces_flat_wound,axis=1))

#     # Count each unique face in the list of faces, faces that appear only once
#     # must be external faces
#     (_,
#      faces_unique_inds,
#      faces_unique_counts) = np.unique(faces_flat_sorted,
#                                       axis=0,
#                                       return_counts=True,
#                                       return_index=True)

#     # Indices of the external faces in faces_flat
#     faces_ext_inds_in_unique = np.where(faces_unique_counts==1)[0]

#     # shape=(num_ext_faces,nodes_per_face)
#     faces_ext_inds = faces_unique_inds[faces_ext_inds_in_unique]

#     faces_ext_wound = faces_flat_wound[faces_ext_inds]

#     faces_coord_inds = np.unique(faces_ext_wound.flatten())
#     faces_coords = np.copy(sim_data.coords[faces_coord_inds])

#     faces_shape = faces_ext_wound.shape
#     faces_ext_wound_flat = faces_ext_wound.flatten()
#     faces_ext_remap_flat = np.copy(faces_ext_wound_flat)

#     # Remap coordinates in the connectivity to match the trimmed list of coords
#     # that belong to the external faces
#     for mm,cc in enumerate(faces_coord_inds):
#         if mm == cc:
#             continue

#         ind_to_map = np.where(faces_ext_wound_flat == cc)[0]
#         faces_ext_remap_flat[ind_to_map] = mm

#     faces_ext_remap = faces_ext_remap_flat.reshape(faces_shape)
#     faces_ext_remap = faces_ext_remap + 1 # back to exodus 1 index

#     # Now we build the SimData object and slice out the node and element
#     # variables using the coordinate indexing.
#     face_data = mh.SimData(coords=faces_coords,
#                            connect={"connect1":faces_ext_remap.T},
#                            time=sim_data.time)

#     if sim_data.node_vars is not None:
#         face_data.node_vars = {}
#         for nn in sim_data.node_vars:
#             face_data.node_vars[nn] = sim_data.node_vars[nn][faces_coord_inds,:]

#     if sim_data.elem_vars is not None:
#         face_data.elem_vars = {}
#         for ee in sim_data.node_vars:
#             face_data.elem_vars[ee] = sim_data.elem_vars[ee][faces_coord_inds,:]

#     return face_data


# def _get_surf_map(nodes_per_elem: int) -> np.ndarray:

#     if nodes_per_elem == 4: # TET4
#        return np.array(((0,1,2),
#                         (0,3,1),
#                         (0,2,3),
#                         (1,3,2)))

#     if nodes_per_elem == 8: # HEX8
#         return np.array(((0,1,2,3),
#                          (0,3,7,4),
#                          (4,7,6,5),
#                          (1,5,6,2),
#                          (0,4,5,1),
#                          (2,6,7,3)))

#     if nodes_per_elem == 10: # TET10
#        return np.array(((0,1,2,4,5,6),
#                         (0,3,1,4,8,7),
#                         (0,2,3,6,9,7),
#                         (1,3,2,8,9,5)))

#     if nodes_per_elem == 20: # HEX20
#         return np.array(((0,1,2,3,8,9,10,11),
#                          (0,3,7,4,11,15,19,12),
#                          (4,7,6,5,19,18,17,16),
#                          (1,5,6,2,13,17,14,9),
#                          (0,4,5,1,12,16,13,8),
#                          (2,6,7,3,14,18,15,10)))

#     if nodes_per_elem == 27: # HEX27
#         return np.array(((0,1,2,3,8,9,10,11,21),
#                          (0,3,7,4,11,15,19,12,23),
#                          (4,7,6,5,19,18,17,16,22),
#                          (1,5,6,2,13,17,14,9,24),
#                          (0,4,5,1,12,16,13,8,25),
#                          (2,6,7,3,14,18,15,10,26)))

#     raise ValueError("Number of nodes does not match a 3D element type for surface extraction.")


def main() -> None:
    sim_path = pyv.DataSet.element_case_path(pyv.EElemTest.HEX27)
    #sim_path = pyv.DataSet.thermomechanical_3d_path()
    sim_data = mh.ExodusReader(sim_path).read_all_sim_data()

    print(80*"-")
    print(f"{sim_data.coords.shape=}")
    print(f"{sim_data.connect['connect1'].shape=}")
    print(80*"-")

    field_keys = ("temperature","disp_y")
    disp_keys = ("disp_x","disp_y","disp_z")
    check_mesh = pyv.create_render_mesh(sim_data=sim_data,
                                        field_render_keys=field_keys,
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_keys)

    print(80*"-")
    print(f"{check_mesh.coords.shape=}")
    print(f"{check_mesh.connectivity.shape=}")
    print(80*"-")

    face_data = pyv.extract_surf_mesh(sim_data)

    print(80*"-")
    print(f"{face_data.coords.shape=}")
    print(f"{face_data.connect['connect1'].shape=}")
    print(80*"-")

    face_data = pyv.scale_length_units(face_data,("disp_x","disp_y","disp_z"),1000.0)
    pv_plot = pyv.plot_sim_data(face_data,"disp_y",elem_dims=2)
    pv_plot.show()



if __name__ == "__main__":
    main()
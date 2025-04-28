#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import numpy as np
import mooseherder as mh
import pyvale as pyv

def extract_surface_mesh() -> tuple[np.ndarray,np.ndarray]:
    pass

def main() -> None:
    sim_path = pyv.DataSet.element_case_path(pyv.EElemTest.TET4)
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

    # Create array with shape=(num_elems,num_faces,nodes_per_face)
    # Find the unique faces as these are the external ones, count of 1
    # Need unique mapping for each face for each element type

    coords = np.copy(sim_data.coords)
    connect = np.copy(sim_data.connect["connect1"])-1 # fixing zero indexing

    num_elems = connect.shape[1]
    num_faces = 4
    nodes_per_face = 3

    # Tets: Mapping:
    # Face 0: 0,1,2
    # Face 1: 0,1,3
    # Face 2: 0,2,3
    # Face 4: 1,2,3
    tet4_face_map = ((0,1,2),(0,1,3),(0,2,3),(1,2,3))
    faces = np.zeros((num_elems,num_faces,nodes_per_face),dtype=np.uintp)
    faces = connect[tet4_face_map,:]


    print(f"{connect.shape=}")
    print(f"{faces.shape=}")
    print()
    print(f"{connect[:,0]}")
    print()
    print(f"{faces[:,:,0]}")
    print()


if __name__ == "__main__":
    main()
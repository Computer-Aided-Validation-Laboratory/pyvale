# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np

import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.raytracer.rtscene 

def simdata_to_mesh(pypath, field_components, fields_to_render, scale):
    # Convert the simulation output into a SimData object
    sim_data = mh.ExodusReader(pypath).read_all_sim_data()
    # Scale the coordinates and displ. fields to mm
    sim_data = sens.scale_length_units(scale=scale,sim_data=sim_data,disp_comps=field_components)
    render_mesh = sens.create_render_mesh(sim_data, fields_to_render ,sim_spat_dim=3,field_disp_keys=field_components)
    return render_mesh

def add_mesh_to_scene(scene, pypath, field_components=("disp_x","disp_y", "disp_z"), fields_to_render = ("disp_y", "disp_x"), world_position = None, scale = 100.0) -> None:
    '''Adds a mesh to the scene dataclass.'''
    render_mesh = simdata_to_mesh(pypath, field_components, fields_to_render, scale)
    if world_position is not None:
        render_mesh.set_pos(world_position)
    connectivity = render_mesh.connectivity
    coords = np.ascontiguousarray(render_mesh.coords[:,:3])
    #node_coords = coords[connectivity,:3]
    x_disp_node_vals = render_mesh.fields_render[:,1, 1] # Field displacement_x at timestep 1 for all nodes. Use this for coloring somehow
    x_disp_node_norm = (x_disp_node_vals - x_disp_node_vals.min())/(x_disp_node_vals.max()-x_disp_node_vals.min()) # Normalize displacement values, scaling them to range [0,1] so they can map to color intensities
    # Approach 2 - taking averages and stacking them together
    node_colors = np.column_stack((x_disp_node_norm, x_disp_node_norm, x_disp_node_norm))  # Convert each scalar to an RGB triplet
    face_colors = np.mean(node_colors[connectivity],axis=1)  # Compute each face's colour as the average of its 3 node colours
    # Approach 1 - using a colour map to assign an rgb value
    # cmap = plt.get_cmap('viridis')
    # face_colors = cmap(x_disp_node_norm)[:,:3]
    scene.add_mesh(connectivity, coords, face_colors)

# Function to get mesh data; partially deprecated due to the introduction of add_mesh_to_scene.
def get_mesh_data(pypath, field_components=("disp_x","disp_y", "disp_z"), fields_to_render = ("disp_y", "disp_x"), world_position = None, scale = 100.0) -> dict:
    '''Returns the mesh data as a numpy array.'''
    render_mesh = simdata_to_mesh(pypath, field_components, fields_to_render, scale)
    if world_position is not None:
        render_mesh.set_pos(world_position)
    connectivity = render_mesh.connectivity
    coords = np.ascontiguousarray(render_mesh.coords[:,:3])
    #node_coords = coords[connectivity,:3]
    x_disp_node_vals = render_mesh.fields_render[:,1, 1] # Field displacement_x at timestep 1 for all nodes. Use this for coloring somehow
    x_disp_node_norm = (x_disp_node_vals - x_disp_node_vals.min())/(x_disp_node_vals.max()-x_disp_node_vals.min()) # Normalize displacement values, scaling them to range [0,1] so they can map to color intensities
    # Approach 2 - taking averages and stacking them together
    node_colors = np.column_stack((x_disp_node_norm, x_disp_node_norm, x_disp_node_norm))  # Convert each scalar to an RGB triplet
    face_colors = np.mean(node_colors[connectivity],axis=1)  # Compute each face's colour as the average of its 3 node colours
    # Approach 1 - using a colour map to assign an rgb value
    # cmap = plt.get_cmap('viridis')
    # face_colors = cmap(x_disp_node_norm)[:,:3]
    return {"connectivity": connectivity, "coords": coords, "face_colors": face_colors}
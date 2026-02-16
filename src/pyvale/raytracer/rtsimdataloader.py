# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np
from pathlib import Path
#import matplotlib as plt # for cmap face color determination

import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.sensorsim.simtools as simtools
from pyvale.raytracer.rtscene import Scene

def simdata_to_mesh(pypath: Path, field_components, fields_to_render, scale):
    # Convert the simulation output into a SimData object
    #sim_data = mh.ExodusReader(pypath).read_all_sim_data() # Pyvale 2025.8.1
    sim_data = mh.ExodusLoader(pypath).load_all_sim_data() # Pyvale 2026.1.0
    # Scale the coordinates and displ. fields to mm
    #sim_data = sens.scale_length_units(scale=scale,sim_data=sim_data,disp_comps=field_components) # Pyvale 2025.8.1
    sim_data = sens.scale_length_units(scale=scale,sim_data=sim_data,disp_keys=field_components) # Pyvale 2026.1.0
    #render_mesh = sens.create_render_mesh(sim_data, fields_to_render ,sim_spat_dim=3,field_disp_keys=field_components) # Pyvale 2025.8.1. Still works, but now we use enum for spatial dim, not a number
    render_mesh = sens.create_render_mesh(sim_data, fields_to_render ,sim_spat_dim=sens.EDim.THREED,field_disp_keys=field_components) # Pyvale 2026.1.0
    return render_mesh

def get_timesteps(pypath, field_components=("disp_x","disp_y", "disp_z"), fields_to_render = ("disp_y", "disp_x"), scale = 100.0):
    '''Returns the number of timesteps for the given simdata object with datapath provided.'''
    render_mesh = simdata_to_mesh(pypath, field_components, fields_to_render, scale)
    return render_mesh.fields_render.shape[1]

def get_timesteps_rm(render_mesh):
    '''Returns the number of timesteps for the given RenderMesh object.'''
    return render_mesh.fields_render.shape[1]

def compute_face_colors_averages(field_nodal_values: np.ndarray, connectivity: np.ndarray):
    '''Calculates face colors based on the nodal values for the chosen field. Approach 2 - taking averages and stacking them together'''
    field_node_norm = (field_nodal_values - field_nodal_values.min())/(field_nodal_values.max()-field_nodal_values.min()) # Normalize displacement values, scaling them to range [0,1] so they can map to color intensities
    node_colors = np.column_stack((field_node_norm, field_node_norm, field_node_norm)) # Convert each scalar to an RGB triplet
    face_colors=np.mean(node_colors[connectivity],axis=1)
    #print(f"face_colors_shape: {face_colors.shape}")
    return  face_colors # Compute each face's colour as the average of its 3 node colours

#def compute_face_colors_cmap(field_nodal_values: np.ndarray):
    '''Approach 1 - using a colour map to assign an rgb value'''
 #   field_node_norm = (field_nodal_values - field_nodal_values.min())/(field_nodal_values.max()-field_nodal_values.min()) # Normalize displacement values, scaling them to range [0,1] so they can map to color intensities
#    cmap = plt.get_cmap('viridis')
 #   return cmap(field_node_norm)[:,:3]

def add_mesh_to_scene(scene: Scene, pypath: Path, field_components=("disp_x","disp_y", "disp_z"), fields_to_render = ("disp_y", "disp_x"), world_position: np.ndarray = None, scale: float = 100.0) -> None:
    '''Adds a mesh to the scene dataclass, including the data for all timesteps.'''
    render_mesh = simdata_to_mesh(pypath, field_components, fields_to_render, scale)
    if world_position is not None:
        render_mesh.set_pos(world_position)
    connectivity = render_mesh.connectivity
    
    # Handle nodal coordinates
    coords_world = np.matmul(render_mesh.coords, render_mesh.mesh_to_world_mat.T) # Convert to world coordinates
    render_mesh.coords = coords_world # Replace nodal coordinates in RenderMesh with their world coordinate equivalents. We can do that since for deformed nodes, we just add values
    element_count = connectivity.shape[0]
    timestep_count = render_mesh.fields_render.shape[1]
    node_coords_expanded_over_time = np.ndarray(shape=(timestep_count, element_count, 3, 3), dtype=np.float64) # Store nodal coordinates over all timesteps
    face_colors_over_time = np.ndarray(shape=(timestep_count, element_count, 3), dtype=np.float64) # Store face colors over all timesteps

    # Process data for the 0th element - always the same for deformable and static images
    coords = np.ascontiguousarray(render_mesh.coords[:,:3])
    node_coords_expanded_over_time[0] = coords[connectivity,:3] # Expanded nodal coords, so we do not need the connectivity array
    # Calculate face colors
    #x_disp_node_vals = render_mesh.fields_render[:,1,1] # Field displacement_x at timestep 1 for all nodes. Not 0 because then we get NaN
    #face_colors = compute_face_colors_averages(x_disp_node_vals, connectivity)
    face_colors_over_time[0] = np.ones(shape=(element_count, 3)) * 0.5 # Default - no deformation in the beginning, so set to 0.5 uniform color. Would this be true for non-displacement fields though? Check

    if timestep_count != 1:
        for timestep in range(1, timestep_count):
            # Get deformed nodal coordinates and expand them
            node_coords = simtools.get_deformed_nodes(timestep, render_mesh)
            coords = np.ascontiguousarray(node_coords)
            #node_coords_expanded = coords[connectivity,:3]
            node_coords_expanded_over_time[timestep] = coords[connectivity,:3] # Expand nodal coords, so we do not need the connectivity array and add to the array storing data for all timesteps
            # Calculate face colors
            x_disp_node_vals = render_mesh.fields_render[:,timestep,1] # Hard-coded for the development, but user should be able to select the field - enum?
            face_colors_over_time[timestep] = compute_face_colors_averages(x_disp_node_vals, connectivity)

    #print("Node_coords_expanded_over_time: \n")
    #print(node_coords_expanded_over_time)
    #print("face_colors_over_time: \n")
    #print(face_colors_over_time)
    # Add mesh to scene
    scene.add_mesh(node_coords_expanded_over_time, face_colors_over_time, timestep_count)


# Functions to load quadratic elements from .vol mesh file with uniform colour.

class Mesh:
    def __init__(self):
        # self.elements contains indices; self.points contains coordinates
        self.points = np.empty((0, 3))
        self.elements = np.empty((0, 10), dtype=int)
        self.elem_coords = np.empty((0, 10, 3))

    def loadVolFile(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Cannot open .vol file: {filename}")
            return

        # Use an iterator to move through lines linearly
        line_iter = iter(lines)
        
        for line in line_iter:
            # Load points
            if "points" in line:
                num_points = int(next(line_iter).strip())
                # Read next num_points lines and convert to float array
                point_data = []
                for _ in range(num_points):
                    point_data.append(list(map(float, next(line_iter).split())))
                self.points = np.array(point_data)
            
            # Load volume elements
            elif "volumeelements" in line:
                num_elements = int(next(line_iter).strip())
                element_list = []
                for _ in range(num_elements):
                    parts = list(map(int, next(line_iter).split()))
                    # parts[0]: mat, parts[1]: np (num nodes)
                    # elements start from index 2 to the end
                    # Convert 1-based indexing to 0-based
                    nodes = [node - 1 for node in parts[2:]]
                    element_list.append(nodes)
                
                self.elements = np.array(element_list)

    def getElementCoords(self):
        """
        Map point coordinates to elements.
        """
        if self.points.size == 0 or self.elements.size == 0:
            print("Mesh data not loaded.")
            return

        self.elem_coords = self.points[self.elements]


def add_vol_mesh_to_scene(scene: Scene, pypath: Path, world_position: np.ndarray = None, scale: float = 100.0) -> None:
    '''Adds a .vol mesh to the scene with uniform element colour.'''

    mesh = Mesh()
    mesh.loadVolFile(pypath)
    mesh.getElementCoords()
    print(mesh.elem_coords.shape)

    print(f"Number of points: {mesh.points.shape[0]}.")
    print(f"Number of elements: {mesh.elements.shape[0]}.")

    element_count = mesh.elements.shape[0]
    timestep_count = 1

    node_coords_expanded_over_time = np.ndarray(shape=(timestep_count, element_count, 10, 3), dtype=np.float64) # Store nodal coordinates over all timesteps
    face_colors_over_time = np.ndarray(shape=(timestep_count, element_count, 3), dtype=np.float64) # Store face colors over all timesteps

    node_coords_expanded_over_time[0, :, :, :] = mesh.elem_coords
    face_colors_over_time[:, :] = [1.0, 0.078, 0.57]


    node_coords_expanded_over_time = node_coords_expanded_over_time * scale



    scene.add_mesh(node_coords_expanded_over_time, face_colors_over_time, timestep_count)


    # print(face_colors_over_time)






 ################################################ DEBUG/DEPRECATED ###############################################

def add_still_mesh_to_scene(scene, pypath,field_components=("disp_x","disp_y", "disp_z"), fields_to_render = ("disp_y", "disp_x"), world_position = None, scale = 100.0) -> None:
    ''' Retired function, keeping it for static tests - retrieves mesh data only for the first timestep, then adds it to the scene dataclass.
    Might expand it to get passed a specific timestep and render that to justify retaining it.'''
    render_mesh = simdata_to_mesh(pypath, field_components, fields_to_render, scale)
    if world_position is not None:
        render_mesh.set_pos(world_position)
    coords_world = np.matmul(render_mesh.coords, render_mesh.mesh_to_world_mat.T) # Convert to world coordinates
    coords = np.ascontiguousarray(coords_world[:,:3])
    #coords = np.ascontiguousarray(render_mesh.coords[:,:3])
    connectivity = render_mesh.connectivity
    node_coords_expanded = coords[connectivity,:3] # Expanded nodal coords, so we do not need the connectivity array. Comment out to test rtbvh_stack, rtbvh_recursion, or no BVH
    x_disp_node_vals = render_mesh.fields_render[:,1, 1] # Field displacement_x at timestep 1 for all nodes.
    face_colors = compute_face_colors_averages(x_disp_node_vals, connectivity)
    scene.add_mesh(node_coords_expanded, face_colors, 1)

def get_mesh_data(pypath, field_components=("disp_x","disp_y", "disp_z"), fields_to_render = ("disp_y", "disp_x"), world_position = None, scale = 100.0) -> dict:
    '''Returns the mesh data as a numpy array.
    Function to get mesh data; partially deprecated due to the introduction of add_mesh_to_scene.'''
    render_mesh = simdata_to_mesh(pypath, field_components, fields_to_render, scale)
    if world_position is not None:
        render_mesh.set_pos(world_position)
    coords_world = np.matmul(render_mesh.coords, render_mesh.mesh_to_world_mat.T) # Convert to world coordinates
    coords = np.ascontiguousarray(coords_world[:,:3])
    connectivity = render_mesh.connectivity
    node_coords_expanded = coords[connectivity,:3] # Expanded nodal coords, so we do not need the connectivity array.
    x_disp_node_vals = render_mesh.fields_render[:,1, 1] # Field displacement_x at timestep 1 for all nodes.
    x_disp_node_norm = (x_disp_node_vals - x_disp_node_vals.min())/(x_disp_node_vals.max()-x_disp_node_vals.min()) # Normalize displacement values, scaling them to range [0,1] so they can map to color intensities
    # Approach 2 - taking averages and stacking them together
    node_colors = np.column_stack((x_disp_node_norm, x_disp_node_norm, x_disp_node_norm))  # Convert each scalar to an RGB triplet
    face_colors = np.mean(node_colors[connectivity],axis=1)  # Compute each face's colour as the average of its 3 node colours
    # Approach 1 - using a colour map to assign an rgb value
    # cmap = plt.get_cmap('viridis')
    # face_colors = cmap(x_disp_node_norm)[:,:3]
    return {"node_coords_expanded": node_coords_expanded, "face_colors": face_colors, "connectivity": connectivity, "coords": coords}
    #return {"connectivity": connectivity, "coords": coords, "face_colors": face_colors}
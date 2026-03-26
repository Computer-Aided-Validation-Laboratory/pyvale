# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from dataclasses import dataclass, field
from enum import Enum
import numpy as np
# from pyvale.raytracer.rtsimdataloader import MaterialType

# Enum to specify render type to be able to let user pick between static and dynamic images
# Would make more sense to be in rtmain, but then we suffer from circular imports
class RenderType(Enum):
    STATIC = 0
    DYNAMIC = 1

class MaterialType(str, Enum):
    NOT_DEFINED = "NOT_DEFINED"
    DIFFUSE = "DIFFUSE"
    SPECULAR = "SPECULAR"
    REFRACTIVE = "REFRACTIVE"

    @property
    def as_int(self) -> int:
        mapping = {
            MaterialType.NOT_DEFINED: 0,
            MaterialType.DIFFUSE: 1,
            MaterialType.SPECULAR: 2,
            MaterialType.REFRACTIVE: 3
        }
        return mapping[self]

@dataclass(slots=True)
class Scene:
    '''WIP: Dataclass for storing camera, mesh, and light data in a format that should work best with C++
    while preserving user-friendly interface.'''
    #scene_connectivity: list[np.ndarray] = field(default_factory=list) # Uncomment to test rtbvh_stack, rtbvh_recursion, or no BVH
    #scene_coords: list[np.ndarray] = field(default_factory=list) # Uncomment to test rtbvh_stack, rtbvh_recursion, or no BVH
    coords_expanded: list[np.ndarray] = field(default_factory=list)
    deform_vals: list[np.ndarray] = field(default_factory=list)
    face_colors: list[np.ndarray] = field(default_factory=list)
    materials: list[np.ndarray] = field(default_factory=list)
    camera_center: list[np.ndarray] = field(default_factory=list)
    pixel_00_center: list[np.ndarray] = field(default_factory=list)
    matrix_pixel_spacing: list[np.ndarray] = field(default_factory=list)
    timestep_count: int = 1 # Number of timesteps with the default value being 1 for static images
    mesh_count: int = 0 # Store the number of meshes in the scene simply because it is used quite a lot

    def add_camera (self, camera_center: np.ndarray, pixel_00_center: np.ndarray, matrix_pixel_spacing: np.ndarray) -> None:
        '''Adds a camera to the scene.'''
        self.camera_center.append(camera_center)
        self.pixel_00_center.append(pixel_00_center)
        self.matrix_pixel_spacing.append(matrix_pixel_spacing)

    def add_mesh(self, node_coords_expanded: np.ndarray, face_colors: np.ndarray, timestep_count: int, material: MaterialType) -> None:
        '''Adds a mesh to the scene.'''
        self.coords_expanded.append(node_coords_expanded)
        self.face_colors.append(face_colors)
        self.materials.append(material.as_int)
        self.mesh_count += 1
        if timestep_count > self.timestep_count: # Keep the highest timestep count (should be the same for all meshes, but you never know)
            self.timestep_count = timestep_count

    def fill_empty_timesteps(self):
        '''Verifies that all meshes in the scene contain data for the defined number of timesteps. If there is missing data for some meshes,
         it fills the nodal coordinates with the repeats of the last known position, and the face colors with white by default. '''
        COORDS_PER_NODE = 3 # Number of coordinates per single node of mesh element

        for mesh in range(self.mesh_count):
            mesh_timesteps = self.coords_expanded[mesh].shape[0]
            mesh_elements = self.coords_expanded[mesh].shape[1]
            #mesh_nodes_per_elem = self.coords_expanded[mesh].shape[2]
            timestep_difference = self.timestep_count - mesh_timesteps # Number of timesteps not accounted for
            if timestep_difference > 0:
                # Expand arrays to fill the missing data
                # Create an array that tells numpy.repeat to to keep all data the same, and only repeat the values
                # for the last known timestep. Add +1 to account for the existing row, i.e., get the total number of 
                # times this data appears in the array, not just how much we want to add.
                repeat_counts = [1] * (mesh_timesteps - 1) + [timestep_difference + 1] 
                self.coords_expanded[mesh] = np.ascontiguousarray(np.repeat(self.coords_expanded[mesh], repeat_counts, axis=0)) # Should be C-contiguous by default, but we need to be extra sure
                # For face colors, we just fill the blanks with 1s (RGB white)
                # TO DO: Give user choice if they want it white or filled with the last known values as well. Or if the mesh should just magically vanish once we run out of timesteps.

                # Option 1: Same as with nodal coordinates; repeat last known values
                #repeat_counts = [1] * (mesh_timesteps - 1) + [timestep_difference + 1] 
                #self.face_colors[mesh] = np.ascontiguousarray(np.repeat(self.face_colors[mesh], repeat_counts, axis=0)) # Should be C-contiguous by default, but we need to be extra sure

                # Option 2: Fill with uniform color (mid on the scale by default since all white/black stood out... a lot)
                filler_data = np.ones(shape=(timestep_difference, mesh_elements, COORDS_PER_NODE)) * 0.5
                self.face_colors[mesh] = np.ascontiguousarray(np.concatenate((self.face_colors[mesh], filler_data), axis=0)) # Should be C-contiguous by default, but we need to be extra sure

    def clip_scene(self, frames_to_render: int, render_type: RenderType):
        '''Clips the data to render only :
            Dynamic renders - the passed number of frames; or
            Static renders - the frame with the passed index.'''
        if render_type == RenderType.DYNAMIC:
            if frames_to_render == self.timestep_count:
                return # No need to change anything if we are rendering all possible frames
            else:
                self.timestep_count = frames_to_render
                for mesh in range(self.mesh_count):
                    self.coords_expanded[mesh] = self.coords_expanded[mesh][:frames_to_render]
                    self.face_colors[mesh] = self.face_colors[mesh][:frames_to_render]
                    #self.deform_vals = self.deform_vals[mesh][:frames_to_render]
        elif render_type == RenderType.STATIC:
            # Split this into two loops to avoid branching out in the main loop, although not sure if this changes much performance-wise in Python. To be tested
            for mesh in range(self.mesh_count):
                # Check if we have enough timestep data for all meshes to render the desired frame number
                if (self.coords_expanded[mesh].shape[0] < frames_to_render):
                    # If there is missing data for any mesh, fill it only up to the required frame to enable rendering
                    self.timestep_count = frames_to_render
                    self.fill_empty_timesteps()
                    break
            for mesh in range(self.mesh_count):
                    self.coords_expanded[mesh] = self.coords_expanded[mesh][:frames_to_render]
                    self.face_colors[mesh] = self.face_colors[mesh][:frames_to_render]
            self.timestep_count = 1
        #print(self.coords_expanded[0].shape)

def find_max_displacements(scene: Scene, render_type: RenderType):
    '''Finds the maximum displacement amongst all nodes for each mesh, so it can be compared against a characteristic length (e.g., edge length or element area)
        to pre-determine whether BLAS/TLAS should be rebuilt or updated.
        Not a part Scene as dataclass can't have ndarrays as a field without pre-allocated shape, which we cannot do.
        WIP: Currently not in use as these metrics are likely to change a lot with the introduction of new element types.'''
    max_displacement_per_step = np.zeros(shape=(scene.mesh_count, scene.timestep_count)) # Set default displacements to zero. Stores max displacement of a single element node for each mesh. Shape is (mesh count, timesteps)
    if render_type == RenderType.STATIC:
        return max_displacement_per_step # Return 0 displacements for static renders since TLAS/BLAS will be only built once anyway
    for mesh in range(scene.mesh_count):
        for timestep in range(1, scene.timestep_count - 1):
            displacement_between_timesteps = scene.coords_expanded[mesh][timestep+1] - scene.coords_expanded[mesh][timestep]
            magnitude_displacement = np.linalg.norm(displacement_between_timesteps, axis=2) # Find magnitude of the displacement for every triangle node; shape is (mesh elements x nodes per element)
            max_displacement_per_step[mesh, timestep] = np.max(magnitude_displacement, axis=(0,1)) # Max displacement of a single mesh element node in this timestep
    return max_displacement_per_step
    #print(max_displacement_per_step)
    #max_displacement_per_frame = np.max(max_displacement_per_step, axis=(0)) # Max displacement out of all element nodes in the scene
    #print(max_displacement_per_frame)
        
         # Debug code - use to analytically figure out if the logic and numbers are correct
            #max_displacement = np.max(displacement_between_timesteps, axis=(0,1))
            #if timestep == 1 or timestep == 2:
                #print(f"Displacement_between_timesteps at t {timestep}")
                #print(displacement_between_timesteps)
                #print(f"Max_displacement at t {timestep}")
                #print(max_displacement)
                #print(test2.shape)
                #print(test2)
                #print(np.max((self.coords_expanded[mesh][timestep+1] - self.coords_expanded[mesh][timestep]), axis=2))
                #max_displacement_per_step[mesh,timestep] = np.max((self.coords_expanded[mesh][timestep+1] - self.coords_expanded[mesh][timestep]), axis=2)
        #print(max_displacement_per_step)
              

 ################################################ DEBUG/DEPRECATED ###############################################
     # Uncomment to test rtbvh_stack, rtbvh_recursion, or no BVH    
    #def add_mesh(self, connectivity:np.ndarray, coords: np.ndarray, face_colors: np.ndarray) -> None:
    #    '''Adds a mesh to the scene.'''
    #    self.scene_connectivity.append(connectivity)
    #    self.scene_coords.append(coords)
    #    self.scene_face_colors.append(face_colors)
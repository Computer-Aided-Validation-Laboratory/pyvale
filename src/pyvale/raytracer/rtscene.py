# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import numpy as np
from pyvale.raytracer.rtmesh import RTMesh, ElementNodeCount, SurfType
from pyvale.raytracer.rtcamera import Camera

# Enum to specify render type to be able to let user pick between static and dynamic images
# Would make more sense to be in rtmain, but then we suffer from circular imports
class RenderType(Enum):
    STATIC = 0
    DYNAMIC = 1

# Enum to specify the type of texture sampler type.
# Must match the enum in rtcolorsampling.h on the C++ side
class TextureSampler(IntEnum):
    NEAREST_NEIGHBOUR = 0
    LANCZOS_2 = 1
    LANCZOS_3 = 2
    CATMULL_ROM = 3
    MITCHELL_NETRAVALI = 4
    BSPLINE = 5
    QUINTIC_SPLINE = 6


# ================================================================================
# SCENE
# Old implementations are at the very bottom of the class for compatibility for the time being.
# ================================================================================

@dataclass(slots=True)
class Scene:
    """
    Dataclass for storing camera, mesh, and light data in a format compatible with the C++ backend renderer.
    """
    # Mesh data
    #scene_connectivity: list[np.ndarray] = field(default_factory=list) # Uncomment to test rtbvh_stack, rtbvh_recursion, or no BVH
    #scene_coords: list[np.ndarray] = field(default_factory=list) # Uncomment to test rtbvh_stack, rtbvh_recursion, or no BVH
    coords_expanded: list[np.ndarray] = field(default_factory=list) # # Replace connectivity and coords; dimensioned as [mesh_idx, timestep, mesh_element_count, nodes_per_element, 3 (for x,y,z)]
    #deform_vals: list[np.ndarray] = field(default_factory=list) # May be needed for deciding whether to update or rebuild TLAS later on
    face_colors: list[np.ndarray] = field(default_factory=list) # Would be good to know the size of this bc it can either be the same for all frames (no need to broadcast data) or different. Do we want this much functionality?
    uvs: list[np.ndarray] = field(default_factory=list) # Not over time, so the dimensions here are [mesh_idx, mesh_element_count, nodes_per_element, 2]
    textures: list[np.ndarray] = field(default_factory=list)
    surface_types: list[SurfType] = field(default_factory=list)
    nodes_per_element: list[ElementNodeCount] = field(default_factory=list)
    element_count: list[int] = field(default_factory=list)
    # Camera data
    camera_center: list[np.ndarray] = field(default_factory=list)
    pixel_00_center: list[np.ndarray] = field(default_factory=list)
    matrix_pixel_spacing: list[np.ndarray] = field(default_factory=list)
    matrix_defocus_disc: list[np.ndarray] = field(default_factory=list)
    # Overall scene data
    timestep_count: int = 1 # Number of timesteps with the default value being 1 for static images
    mesh_count: int = 0 # Store the number of meshes in the scene simply because it is used quite a lot

    def add_camera(self, camera: Camera) -> None:
        """
        Adds a camera to the scene.

        Parameters:
        -----------
        camera: Camera
            The camera to add to the scene.

        Returns:
        --------
        None
        """
        self.camera_center.append(camera.camera_center)
        self.pixel_00_center.append(camera.pixel_00_center)
        self.matrix_pixel_spacing.append(camera.matrix_pixel_spacing)
        self.matrix_defocus_disc.append(camera.matrix_defocus_disc)



    def add_rtmesh(self, rtmesh: RTMesh) -> None:
        """
        Adds a RTMesh object to the scene.

        Parameters:
        -----------
        rtmesh: RTMesh
            The RTMesh object to add to the scene.
        
        Returns:
        --------
        None

        Raises:
        -------
        ValueError:
            If the surface type is not set for the mesh before adding it to the scene.
        """

        if rtmesh.surface_type is None:
            raise ValueError("Please set surface type for mesh before adding it to the scene.")
        #self.scene_connectivity.append(rtmesh.connectivity)
        #self.scene_coords.append(rtmesh.node_coords)
        self.coords_expanded.append(rtmesh.node_coords_expanded_over_time)
        if rtmesh.surface_type == SurfType.FIELD_COLOR:
            self.face_colors.append(rtmesh.face_colors_over_time)
            self.textures.append(np.zeros(shape=(1,1))) # Append a small array of zeros, only so we have matching indices but this data should never be accessed. Hacky solution, to be resolved better (probably merging face_colors and textures into one)
            self.uvs.append(np.zeros(shape=(1,1))) # Append a small array of zeros, only so we have matching indices but this data should never be accessed. Hacky solution
        elif rtmesh.surface_type == SurfType.TEXTURE:
            #self.uvs.append(rtmesh.uvs_over_time) # if using uvs_over_time
            self.uvs.append(rtmesh.uvs)
            self.add_texture(rtmesh.texture)
            self.face_colors.append(np.zeros(shape=(1,1))) # Append a small array of zeros, only so we have matching indices but this data should never be accessed. Hacky solution, to be resolved better (probably merging face_colors and textures into one)
        self.mesh_count += 1
        self.surface_types.append(rtmesh.surface_type) # Will be used for determining coloring
        self.nodes_per_element.append(rtmesh.nodes_per_element) # Will help assign appropriate functions in ray_tracer
        self.element_count.append(rtmesh.element_count) # Will be used in C interface
        if rtmesh.timestep_count > self.timestep_count:  # Keep the highest timestep count (should be the same for all meshes, but you never know)
            self.timestep_count = rtmesh.timestep_count

    def _fill_empty_timesteps(self) -> None:
        """
        Verifies that all meshes in the scene contain data for the defined number of timesteps. If there is missing data for some meshes,
        it fills the nodal coordinates and/or face colors with the last known values.

        Parameters:
        -----------
        None
        
        Returns:
        --------
        None

        Raises:
        -------
        ValueError:
            If the surface type is not set for the mesh before adding it to the scene.
        """

        for mesh in range(self.mesh_count):
            mesh_timesteps = self.coords_expanded[mesh].shape[0]
            mesh_elements = self.coords_expanded[mesh].shape[1]
            # mesh_nodes_per_elem = self.coords_expanded[mesh].shape[2]
            timestep_difference = self.timestep_count - mesh_timesteps  # Number of timesteps not accounted for
            if timestep_difference > 0:
                # Expand arrays to fill the missing data
                # Create an array that tells numpy.repeat to to keep all data the same, and only repeat the values
                # for the last known timestep. Add +1 to account for the existing row, i.e., get the total number of
                # times this data appears in the array, not just how much we want to add.
                repeat_counts = [1] * (mesh_timesteps - 1) + [timestep_difference + 1]
                self.coords_expanded[mesh] = np.ascontiguousarray(np.repeat(self.coords_expanded[mesh], repeat_counts,
                                                                            axis=0))  # Should be C-contiguous by default, but we need to be extra sure
                # Case 1: Mesh filled with solid colour (assumes colour changes between frames, i.e., field-value based). Might remove this entirely if we keep one solid color for all timeframes.
                # TO DO: Give user choice if they want it white or filled with the last known values as well. Or if the mesh should just magically vanish once we run out of timesteps.
                if self.surface_types[mesh] == SurfType.FIELD_COLOR:
                    # Option 1: Same as with nodal coordinates; repeat last known values
                    self.face_colors[mesh] = np.ascontiguousarray(np.repeat(self.face_colors[mesh], repeat_counts, axis=0)) # Should be C-contiguous by default, but we need to be extra sure
                    # Option 2: Fill with uniform color (mid on the scale by default since all white/black stood out... a lot)
                    #filler_data = np.ones(shape=(timestep_difference, mesh_elements, NODE_COORDINATES)) * 0.5
                    #self.face_colors[mesh] = np.ascontiguousarray(np.concatenate((self.face_colors[mesh], filler_data),axis=0))
                
                # Case 2: Mesh with texture
                # Uncomment if you use uvs_over_time in rtmesh; otherwise, not necessary as uvs do not change across timeframes.
                elif self.surface_types[mesh] == SurfType.TEXTURE:
                    pass
                    #self.uvs[mesh] = np.ascontiguousarray(np.repeat(self.uvs[mesh], repeat_counts, axis=0))
                # No need to duplicate texture data as we assume it won't change across the frames
                else: # Surface data not set
                    raise ValueError("Surface data not set for mesh " + str(mesh))

    def _clip_scene(self,
                   frames_to_render: int,
                   render_type: RenderType):
            
        """
        Clips the data to render only the passed number of frames for dynamic renders, or the frame with the passed index for static renders.

        Parameters:
        -----------
        frames_to_render: int
            The number of frames to render for dynamic renders, or the frame index for static renders.
        render_type: RenderType
            The type of rendering to perform. Can be either DYNAMIC or STATIC.
        
        Returns:
        --------
        None

        Raises:
        -------
        ValueError:
            If the surface type is not set for the mesh before adding it to the scene.

        """
        if render_type == RenderType.DYNAMIC:
            if frames_to_render == self.timestep_count:
                return  # No need to change anything if we are rendering all possible frames
            else:
                self.timestep_count = frames_to_render
                for mesh in range(self.mesh_count):
                    self.coords_expanded[mesh] = self.coords_expanded[mesh][:frames_to_render]
                    # self.deform_vals = self.deform_vals[mesh][:frames_to_render]
                    if self.surface_types[mesh] == SurfType.FIELD_COLOR:
                        self.face_colors[mesh] = self.face_colors[mesh][:frames_to_render]
                    # Uncomment if you use uvs_over_time in rtmesh; otherwise, not necessary as uvs do not change across timeframes.
                    elif self.surface_types[mesh] == SurfType.TEXTURE:
                        pass
                        #self.uvs[mesh] = self.uvs[mesh][:frames_to_render]
                    else: # Potentially an unnecessary check as meshes without surface type cannot be added to the scene
                        raise ValueError("Surface type not set for mesh " + str(mesh))
        elif render_type == RenderType.STATIC:
            # Split this into two loops to avoid branching out in the main loop, although not sure if this changes much performance-wise in Python. To be tested
            for mesh in range(self.mesh_count):
                # Check if we have enough timestep data for all meshes to render the desired frame number
                if (self.coords_expanded[mesh].shape[0] < frames_to_render):
                    # If there is missing data for any mesh, fill it only up to the required frame to enable rendering
                    self.timestep_count = frames_to_render
                    self._fill_empty_timesteps() # This will raise an exception if SurfType not set, so no need to do it below
                    break
            for mesh in range(self.mesh_count):
                self.coords_expanded[mesh] = self.coords_expanded[mesh][:frames_to_render]
                if self.surface_types[mesh] == SurfType.FIELD_COLOR:
                    self.face_colors[mesh] = self.face_colors[mesh][:frames_to_render]
                # Uncomment if you use uvs_over_time in rtmesh; otherwise, not necessary as uvs do not change across timeframes.
                #elif self.surface_types[mesh] == SurfType.TEXTURE:
                    #self.uvs[mesh] = self.uvs[mesh][:frames_to_render]
            self.timestep_count = 1
        # print(self.coords_expanded[0].shape)


# ================================================================================
# DEV/DEBUG/DEPRECATED
# ================================================================================

    def add_mesh(self, node_coords_expanded: np.ndarray, face_colors: np.ndarray, timestep_count: int) -> None:
        '''Adds a mesh to the scene.'''
        self.coords_expanded.append(node_coords_expanded)
        self.face_colors.append(face_colors)
        self.mesh_count += 1
        if timestep_count > self.timestep_count: # Keep the highest timestep count (should be the same for all meshes, but you never know)
            self.timestep_count = timestep_count
    
    def add_mesh2(self, connectivity, coords, node_coords_expanded: np.ndarray, face_colors: np.ndarray, uvs: np.ndarray, timestep_count: int) -> None:
        '''Adds a mesh to the scene. Second version with texturing.'''
        self.scene_connectivity.append(connectivity)
        self.scene_coords.append(coords)
        self.coords_expanded.append(node_coords_expanded)
        self.face_colors.append(face_colors)
        self.uvs.append(uvs)
        self.mesh_count += 1
        if timestep_count > self.timestep_count:  # Keep the highest timestep count (should be the same for all meshes, but you never know)
            self.timestep_count = timestep_count

    def add_texture(self, texture: np.ndarray) -> None:
        '''Adds a texture to the scene. Only to be used with add_mesh2; RTMeshes take textures as their attribute and there is no need to add them separately.
        Texture is expected to be a 2D numpy array of shape (width, height).
        Values are expected to be between 0 and 1. If they are not, they will be scaled as the renderer converts floats to 0-255 integers later.'''
        if texture.min() > 1.0 or texture.max() > 1.0:
            texture = texture.astype(np.float64) / 255.0
        self.textures.append(texture)

    def fill_empty_timesteps_old(self):
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

    def clip_scene_old(self, frames_to_render: int, render_type: RenderType):
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



# ================================================================================
# DEV/DEBUG
# ================================================================================

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
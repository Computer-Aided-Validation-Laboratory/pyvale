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

# ================================================================================
# ENUMS WITH OPTIONS
# ================================================================================

# Enum to specify render type to be able to let user pick between static and dynamic images
# Would make more sense to be in rtmain, but then we suffer from circular imports
class RenderType(Enum):
    STATIC = 0
    DYNAMIC = 1

# Ray tracer output type
class OutputType(IntEnum):
    IMG_PPM = 0
    IMG_TIFF = 1
    #NP_BUFFER = 2 # Not implemented yet

# Enum to specify the texture sampler type
# Must match the enum in rtcolorsampling.h on the C++ side
class TextureSampler(IntEnum):
    NEAREST_NEIGHBOUR = 0
    LANCZOS_2 = 1
    LANCZOS_3 = 2
    CATMULL_ROM = 3
    MITCHELL_NETRAVALI = 4
    BSPLINE = 5
    QUINTIC_SPLINE = 6

# Enum to specify which normals are used for shading
class ShadingType(IntEnum):
    FLAT = 0 # Shade with geometric normals for all elements
    BLENDED = 1 # Use angle-avg node normals for TRI3 and QUAD4, Jacobians for curved elements
    ANGLE_AVG_BLENDED = 2 # Angle-avg node normals for all elements

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
    coords_expanded: list[np.ndarray] = field(default_factory=list) # Replace connectivity and coords; dimensioned as [mesh_idx, timestep, mesh_element_count, nodes_per_element, 3 (for x,y,z)]
    normals_expanded: list[np.ndarray] = field(default_factory=list) # Node normals for shading; dimensioned as [mesh_idx, timestep, mesh_element_count, nodes_per_element, 3 (for x,y,z)]
    #deform_vals: list[np.ndarray] = field(default_factory=list) # May be needed for deciding whether to update or rebuild TLAS later on
    face_colors: list[np.ndarray] = field(default_factory=list) # Would be good to know the size of this bc it can either be the same for all frames (no need to broadcast data) or different. Do we want this much functionality?
    materials: list[int] = field(default_factory=list)
    uvs: list[np.ndarray] = field(default_factory=list) # Not over time, so the dimensions here are [mesh_idx, mesh_element_count, nodes_per_element, 2]
    textures: list[np.ndarray] = field(default_factory=list)
    surface_types: list[SurfType] = field(default_factory=list)
    nodes_per_element: list[ElementNodeCount] = field(default_factory=list)
    element_count: list[int] = field(default_factory=list)
    refractive_indices: list[float] = field(default_factory=list) # Refractive indices of meshes stored in the scene
    mesh_priorities: list[float] = field(default_factory=list) # Priorities of objects used to determine the intersections if there are nested volumes that are refractive
    mesh_object_types: list[float] = field(default_factory=list) # Tells us if solids or thin shells to adjust refractive behaviour
    mesh_thickness: list[float] = field(default_factory=list) # Mesh thicknesses used if mesh_type is declared as SHELL
    # Camera data
    camera_center: list[np.ndarray] = field(default_factory=list)
    pixel_00_center: list[np.ndarray] = field(default_factory=list)
    matrix_pixel_spacing: list[np.ndarray] = field(default_factory=list)
    matrix_defocus_disc: list[np.ndarray] = field(default_factory=list)
    # Overall scene data
    scene_ri: float = 1.0003 # Refractive index of the material filling the scene. 1.0 set as default for air
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

    def _add_mesh(self, rtmesh: RTMesh) -> None:
        """
        Helper to avoid repeating oneself in add_rtmesh.

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
        node_coords_expanded_over_time, node_normals_expanded_over_time = rtmesh.get_expanded_coords()
        self.coords_expanded.append(node_coords_expanded_over_time)
        self.normals_expanded.append(node_normals_expanded_over_time)
        if rtmesh.surface_type == SurfType.FIELD_COLOR:
            self.face_colors.append(rtmesh.face_colors_over_time)
            self.textures.append(np.zeros(shape=(1,1))) # Append a small array of zeros, only so we have matching indices but this data should never be accessed. Hacky solution, to be resolved better
            self.uvs.append(np.zeros(shape=(1,1))) # Append a small array of zeros, only so we have matching indices but this data should never be accessed. Hacky solution
        elif rtmesh.surface_type == SurfType.TEXTURE:
            #self.uvs.append(rtmesh.uvs_over_time) # if using uvs_over_time
            self.uvs.append(rtmesh.uvs)
            self._add_texture(rtmesh.texture)
            self.face_colors.append(np.zeros(shape=(1,1))) # Append a small array of zeros, only so we have matching indices but this data should never be accessed. Hacky solution, to be resolved better
        self.surface_types.append(rtmesh.surface_type) # Will be used for determining coloring
        self.materials.append(rtmesh.material_type.as_int)
        self.nodes_per_element.append(rtmesh.nodes_per_element) # Will help assign appropriate functions in ray_tracer
        self.element_count.append(rtmesh.element_count) # Will be used in C interface
        if rtmesh.timestep_count > self.timestep_count:  # Keep the highest timestep count (should be the same for all meshes, but you never know)
            self.timestep_count = rtmesh.timestep_count
        self.refractive_indices.append(rtmesh.refractive_index)
        self.mesh_priorities.append(rtmesh.priority)
        self.mesh_object_types.append(rtmesh.mesh_type)
        self.mesh_thickness.append(rtmesh.thickness)
        self.mesh_count += 1

    def add_rtmesh(self, rtmesh: RTMesh | list[RTMesh]):
        """
        Adds a RTMesh object to the scene.

        Note before: If you modify a RTMesh after adding it to the scene, these changes will not be reflected.

        Parameters:
        -----------
        rtmesh: RTMesh | list[RTMesh]
            The RTMesh object to add to the scene, or list of those.
        
        Returns:
        --------
        None
        """
        if isinstance(rtmesh, list):
            for idx, mesh in enumerate(rtmesh):
                try:
                    self._add_mesh(mesh)
                except ValueError: # Propagate and adjust the ValueError to let the user know which mesh is set incorrectly
                    print(f"Mesh ID {idx}: Surface type not set for mesh before adding it to the scene.")
        else:
            self._add_mesh(rtmesh)
        
    def set_refractive_index(self, refractive_index: float):
        """
        Sets the refractive index of the scene.

        Parameters:
        -----------
        refractive_index: float
            The refractive index of the scene. Default is 1.0003 for air in visible light. (https://refractiveindex.info/?shelf=other&book=air&page=Ciddor)
        
        Returns:
        --------
        None

        Raises:
        -------
        ValueError:
            If the refractive index is negative.
        """
        if refractive_index > 0.0:
            self.scene_ri = refractive_index
        else:
            raise ValueError("Refractive index can be negative only for metamaterials, and it is highly unlikely that the entire scene is filled with one.")

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
                self.normals_expanded[mesh] = np.ascontiguousarray(np.repeat(self.normals_expanded[mesh], repeat_counts,
                                                                            axis=0))
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
                    self.normals_expanded[mesh] = self.normals_expanded[mesh][:frames_to_render]
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
            # Split this into two loops to avoid branching out in the main loop, although not sure if this changes much performance-wise in Python
            for mesh in range(self.mesh_count):
                # Check if we have enough timestep data for all meshes to render the desired frame number
                if (self.coords_expanded[mesh].shape[0] < frames_to_render):
                    # If there is missing data for any mesh, fill it only up to the required frame to enable rendering
                    self.timestep_count = frames_to_render
                    self._fill_empty_timesteps() # This will raise an exception if SurfType not set, so no need to do it below
                    break
            for mesh in range(self.mesh_count):
                self.coords_expanded[mesh] = self.coords_expanded[mesh][:frames_to_render]
                self.normals_expanded[mesh] = self.normals_expanded[mesh][:frames_to_render]
                if self.surface_types[mesh] == SurfType.FIELD_COLOR:
                    self.face_colors[mesh] = self.face_colors[mesh][:frames_to_render]
                # Uncomment if you use uvs_over_time in rtmesh; otherwise, not necessary as uvs do not change across timeframes.
                #elif self.surface_types[mesh] == SurfType.TEXTURE:
                    #self.uvs[mesh] = self.uvs[mesh][:frames_to_render]
            self.timestep_count = 1   

    def _add_texture(self, texture: np.ndarray) -> None:
        """
        Adds a texture to the scene.

        Values are expected to be between 0 and 1. If they are not, they will be scaled as the renderer converts floats to 0-255 integers later.

        Parameters:
        -----------
        texture: np.ndarray
            The texture to add to the scene. A 2D numpy array of shape (width, height).
        """
        if texture.min() > 1.0 or texture.max() > 1.0:
            texture = texture.astype(np.float64) / 255.0
        self.textures.append(texture)
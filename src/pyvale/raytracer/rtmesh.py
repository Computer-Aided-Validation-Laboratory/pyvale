# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import itertools
import meshio
import pandas as pd
import numpy as np
import pyvista as pv
# import matplotlib as plt # for cmap face color determination
from pathlib import Path
from scipy.spatial.transform import Rotation
from enum import StrEnum, IntEnum
from dataclasses import dataclass, field
from enum import Enum
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
from copy import deepcopy

import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
from pyvale.sensorsim import RenderMesh, EDim, simdata_to_pyvista_interp, extract_surf_mesh
from pyvale.raytracer.rtcamera import Camera
from pyvale.raytracer.rtpresets import Material

# ================================================================================
# CONSTANTS AND ENUMS
# ================================================================================

COORDS_PER_NODE = 3
RGB_VALS = 3
FEATURE_ANGLE = 60.0 # For calculating surface normals; edges sharper than this will not be smoothed

# Axes for orienting/resizing the meshes
class Axis(IntEnum):
    """
    Axes for orienting/resizing the meshes:
    X - Left/right
    Y - Up/down. +Y is up if camera vector_view_up = [0,1,0], which is currently the hard-coded default
    Z - Forward/backwards
    """
    X = 0
    Y = 1
    Z = 2 

class Anchor(StrEnum):
    """
    Specifies which point on a mesh's bounding box the position refers to:
    CENTER - Center
    MIN - Min corner on all 3 axes
    MAX - Max corner on all 3 axes
    BASE - Centered in X/Z, min in Y  (sits on a floor)
    TOP - Centered in X/Z, max in Y
    """
    CENTER = "center"
    MIN = "min"
    MAX = "max"
    BASE = "base"
    TOP = "top"


class SurfType(IntEnum): # IntEnum so it can be passed to C++ nicely
    """
    Type of coloring that goes onto the mesh surface:
    FIELD_COLOR - Solid RGB colour value
    TEXTURE - Texture (requires UV unwrapping)
    """
    FIELD_COLOR = 0,
    TEXTURE = 1

class MaterialType(str, Enum):
    #NOT_DEFINED = "NOT_DEFINED" # Nothing stopping the ray, i.e. empty space
    DIFFUSE = "DIFFUSE"
    SPECULAR = "SPECULAR"
    REFRACTIVE = "REFRACTIVE"
    UNLIT = "UNLIT" # Just surface color, ignoring all lighting calculations, i.e. no shadows, no shading, no reflections

    @property
    def as_int(self) -> int:
        mapping = {
            #MaterialType.NOT_DEFINED: 0,
            MaterialType.DIFFUSE: 1,
            MaterialType.SPECULAR: 2,
            MaterialType.REFRACTIVE: 3,
            MaterialType.UNLIT: 4
        }
        return mapping[self]

# Number of nodes per element
class ElementNodeCount(IntEnum):
    """
    Specifies the number of nodes per mesh element.
    """
    TRI3 = 3,
    TRI6 = 6,
    QUAD4 = 4,
    QUAD8 = 8,
    QUAD9 = 9,
    TET4 = 4,
    TET10 = 10,
    TET14 = 14,
    HEX8 = 8,
    HEX20 = 20,
    HEX27 = 27

# Mesh type - used for adjusting refractive behaviour
class MeshType(IntEnum):
    """
    Specifies the type of mesh.
    SOLID - Solid body
    Shell - Either thick or thin shell of a material
    """
    SOLID = 0, 
    SHELL = 1

# Meshio mapping
# Enum with surface elements (so 2D that we get after skinning the mesh) and their names in meshio cell types
# List of all meshio cell types: https://www.binyang.fun/meshio-cell-type/
SURFACE_ELEMENTS = {"triangle", "triangle6", "quad", "quad8", "quad9"} # Surface element names as defined in meshio
MESHIO_ACCEPTED_VOL_ELEMS = {"tetra10", "tetra", "hexahedron", "hexahedron20", "hexahedron27"} # Set of currently accepted volume elements from meshio
MESHIO_BAD_TYPES = {"empty", "vertex", "poly_vertex", "line", "line3", "line4", "voxel", "poly_line", "triangle_strip"} # Set of meshio cell types that we can't render so will always be removed

# Mapping between 3D elements and (surface element type, faces per 3D element)
SURFACE_TO_VOLUME_ELEM_MAPPING = {
    "triangle6": ("tetra10", 4),
    "triangle": ("tetra", 4),
    "quad": ("hexahedron", 6),
    "quad8": ("hexahedron20", 6),
    "quad9": ("hexahedron27", 6)
}
# Mapping between volume elements and (surface element type, faces per 3D element)
VOLUME_TO_SURFACE_ELEM_MAPPING = {
    "tetra10": ("triangle6", 4),
    "tetra": ("triangle", 4),
    "hexahedron": ("quad", 6),
    "hexahedron20": ("quad8", 6),
    "hexahedron27": ("quad9", 6)
}
# Mapping between meshio element types and element node counts
MESHIO_TO_ELEMENTNODECOUNT = {
    "triangle": ElementNodeCount.TRI3,
    "triangle6": ElementNodeCount.TRI6,
    "quad": ElementNodeCount.QUAD4,
    "quad8": ElementNodeCount.QUAD8,
    "quad9": ElementNodeCount.QUAD9,
    "tetra": ElementNodeCount.TET4,
    "tetra10": ElementNodeCount.TET10,
    "hexahedron": ElementNodeCount.HEX8,
    "hexahedron20": ElementNodeCount.HEX20,
    "hexahedron27": ElementNodeCount.HEX27
}

# ================================================================================
# Drop-in (for compatibility) that should probably go to simtools
# ================================================================================

def get_displacement_at_timestep(timestep: int,
                                  render_mesh: RenderMesh) -> np.ndarray | None:
    """
    Returns only the displacement field (not absolute position) for all nodes
    at a given timestep. 
    
    This is preferable to get_deformed_nodes for the new orienting system, when the caller needs to compute a *delta* between two timesteps,
    because the raw displacement values are independent of the coordinate scaling / translation / rotation applied to the RTMesh in world space.

    Parameters:
    -----------
    timestep: int
        Timestep at which we want to get the displacement field.
    render_mesh: RenderMesh
        RenderMesh object whose displacement field we want to extract.

    Returns:
    --------
    np.ndarray
        Displacement field at the given timestep.
    
    """
    if render_mesh.fields_disp is None:
        return None

    disp = render_mesh.fields_disp[:, timestep]  # shape (N, num_components)
    if disp.shape[1] == 2:
        disp = np.hstack((disp, np.zeros([disp.shape[0], 1], dtype=np.float64)))
    return disp  # shape (node_count, 3)


# ================================================================================
# RTMESH CLASS
# ================================================================================

@dataclass
class RTMesh:
    """
    A class representing a ray-traceable mesh.

    Used as a temporary common interface between linear RenderMeshes native to pyvale and Mesh from curved element implementation in any other format.
    """
    node_coords: np.ndarray = field(default=None)
    connectivity: np.ndarray = field(default=None)
    node_coords_over_time: np.ndarray = field(default=None) # Not expanded (so we still need connectivity); needed to enable changing mesh after it is initialised; shape (timesteps, node_count, 3)
    face_colors_over_time: np.ndarray = field(default=None)
    #uvs_over_time: np.ndarray = field(default=None) # Temporary used in development. But uvs should not change over time and they can be massive, so this was deprecated to reduce memory consumption. Keeping it just in case
    uvs: np.ndarray = field(default=None)
    seams: list = field(default_factory=list)
    texture: np.ndarray = field(default=None)
    material_type: MaterialType = field(default=None)
    refractive_index: float = field(default=1.0003) # Refractive index of the material comprising the mesh; defaults to 1.003 for air, i.e., no refraction
    priority: int = field(default=0) # Mesh priority to determine refraction in nested volumes. 0 is the default, value not used for non-refractive cases
    mesh_type: MeshType = field(default=MeshType.SOLID) # Type of mesh determining the refractive behaviour
    thickness: float = field(default=1.0) # Mesh thickness used for SHELLs
    #mesh_to_world_mat: np.ndarray = field(default=None)
    pyvista_surface: pv.UnstructuredGrid | pv.PolyData = field(default=None) # For SeamSplitter; TRIANGULATED mesh surface - used for UV-unwrapping procedures, not for rendering
    tri_face_mapping: np.ndarray = field(default=None) # To map triangulated faces back to original elements; needed for Blender UV unwrapping
    tri_node_mapping: np.ndarray = field(default=None) # To map triangulated vertex v to original higher order node/vertex
    surface_type: SurfType = field(default=None)
    spatial_dimensions: sens.EDim = field(default=None)
    timestep_count: int = field(default=1)
    element_count: int = field(default=0)
    node_count: int = field(default=0)
    nodes_per_element: ElementNodeCount = field(default=ElementNodeCount.TRI3)
    avg_element_length: float = field(default=0.0) # Average edge length of mesh elements in this mesh
    rm_point_ids: np.ndarray | None = field(default=None) # For SimData/RenderMesh submeshes: maps local node index -> render_mesh global node index
    # Store unoriented AABB and transformations for object-oriented bounding box 
    resting_aabb: dict = field(default_factory=dict)
    translation: np.ndarray = field(default_factory=lambda:np.zeros(3)) # Need to use a lambda, otherwise this will return 'numpy.ndarray' object is not callable because it expects a zero-argument callable
    rotation: Rotation | np.ndarray = field(default_factory=lambda:np.eye(3)) # Default is identity = no rotation
    scale: float = field(default=1.0)

    def get_mesh_data_over_time(self, render_mesh: RenderMesh = None) -> None:
        """
        Fetches mesh data over the number of timesteps.
        Currently works only for simdata-based meshes - for others, the timestep count is always assumed to be 1.

        Parameters:
        -----------
        render_mesh: RenderMesh
            Optional. RenderMesh object used to fetch deformed nodal coordinates. Defaults to None.
        """
        coords = self.node_coords
        
        # Process data for the 0th element - always the same for deformable and static images
        # Node coords over time
        node_coords_over_time = np.ndarray(shape=(self.timestep_count, self.node_count, COORDS_PER_NODE),dtype=np.float64)  # Store nodal coords over all timesteps
        node_coords_over_time[0] = coords

        # Get data over multiple timesteps
        if self.timestep_count != 1:
            if render_mesh is not None:
                base_displacement = get_displacement_at_timestep(0, render_mesh)  # shape (node_count, 3)
                for timestep in range(1, self.timestep_count):
                    current_displacement = get_displacement_at_timestep(timestep, render_mesh)
                    delta_disp = (current_displacement - base_displacement) * self.scale  # shape (node_count, 3)

                    # Map render_mesh-global node indices back to submesh-local indices
                    # When the RTMesh was built from a submesh, sub_ugrid was extracted with pass_point_ids=True so self.node_coords rows correspond to
                    # the render_mesh rows stored in self.rm_point_ids (set below). For the full-mesh case rm_point_ids is None and we use all rows.
                    if self.rm_point_ids is not None:
                        local_delta = delta_disp[self.rm_point_ids]  # (N_local, 3)
                    else:
                        local_delta = delta_disp  # (N_rm == N_local, 3)

                    # Apply delta on top of the already world-positioned coords
                    deformed_coords = coords + local_delta  # (N_local, 3)
                    deformed_coords = np.ascontiguousarray(deformed_coords)
                    node_coords_over_time[timestep] = deformed_coords
            else: # Any other mesh
                print("Timesteps aren't currently supported for meshes that aren't imported as SimData.")
                # Data "over time" - currently not supported for non-simdata as other formats don't seem to work with multi-step data, so not sure how this would look like?
                # But data extraction overall should be similar to what is done in the simdata branch

        # Update the rtmesh object
        self.node_coords_over_time = node_coords_over_time

    def _compute_average_element_length(self) -> float:
        """
        Computes the average element edge length for validating thickness for shells and mesh scale.

        Uses a rough approximate from the triangulated surface where need be for now as the length parameter does not work for 2D elements (i.e., not lines),
        so we have to use the area and reverse engineer it.
        It assumes that a ~ h, so A_triangle ~ 0.5 * 2a => A_triangle ~ a, which definitely is not the best method moving forward, but should be an okay-ish ballpark
        cut-off value for non-degenerate elements.

        Returns:
        --------
        float
            The average element edge length.
        """

        pv_temp = self.pyvista_surface.compute_cell_sizes(area = True)
        avg_element_length = np.mean(pv_temp.cell_data['Area'])
        return avg_element_length
    
    def _set_element_length(self, element_length: float) -> None:
        """
        Checks if the element length is valid and unlikely to cause rendering errors and sets its value if it is valid.

        Raises:
        -------
        ValueError:
            If the element length is too small.
        """
        if element_length < 1e-5:
            raise ValueError("Element size {element_length} is too small. Consider changing the world units to a larger magnitude.")
        else:
            self.avg_element_length = element_length
    
    def get_expanded_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Helper function that expands the nodal coordinates and node normals over time, so the connectivity array does not need to be passed.

        It also calculated node normals to ensure these stem from the current coordinate values in case they have changed.

        It is used when the mesh is added to the scene to avoid re-calculating all this data whenever the mesh is rescaled etc.

        Returns
        --------
        tuple[np.ndarray, np.ndarray]
            Expanded nodal coordinates and nodal normals over time, shaped (timestep_count, element_count, nodes_per_element, 3).
        """
        connectivity = self.connectivity
        node_coords_expanded_over_time = np.ndarray(shape=(self.timestep_count, self.element_count, self.nodes_per_element, COORDS_PER_NODE),dtype=np.float64)  # Store nodal coordinates over all timesteps
        node_normals_expanded_over_time = np.ndarray(shape=(self.timestep_count, self.element_count, self.nodes_per_element, COORDS_PER_NODE), dtype=np.float64)  # Store nodal normals over all timesteps

        for timestep in range(self.timestep_count):
            coords_at_t = self.node_coords_over_time[timestep]
            node_coords_expanded_over_time[timestep] = coords_at_t[connectivity]
            node_normals_expanded_over_time[timestep]  = find_node_normals(coords_at_t, connectivity, self.element_count)[connectivity]

        return np.ascontiguousarray(node_coords_expanded_over_time, dtype=np.float64), np.ascontiguousarray(node_normals_expanded_over_time, dtype = np.float64)

    def translate(self, delta: np.ndarray) -> None:
        """
        In-place world-space translation by a given vector delta.

        Parameters:
        -----------
        delta: np.ndarray
            The translation vector. Expected to be shaped (3).
        """
        # This will be None when we _orient_in_world at the moment of converting SimData/other meshes to RTMesh, so to avoid errors
        if self.node_coords_over_time is not None:
            self.node_coords_over_time[..., :] += delta
        self.node_coords[...] = self.node_coords + delta
        self.pyvista_surface.points = self.pyvista_surface.points + delta # Update pyvista surface, so it displays properly in SeamSelector and UV unwrapping
        self.translation = self.translation + delta # Update translation

    def place_at(self, target_position: np.ndarray,
             anchor: Anchor = Anchor.CENTER,
             timestep: int = 0) -> None:
        """
        Moves the mesh so that its anchor point ends up at world position.
        
        Parameters:
        -----------
        target_position: np.ndarray
            Target world position of the anchor point after the placement.
        anchor: Anchor
            Selected point on the mesh whose target is specified. Defaults to CENTER.
        timestep: int
            Optional. Timestep at which the bounding box is calculated .
        """
        aabb = self._get_bounding_box(timestep)
        self.translate(np.ascontiguousarray(target_position, dtype=np.float64) - get_anchor_point(aabb, anchor))

    def fit_size(self, target_size: float,
             axis: Axis | None = None,
             timestep: int = 0) -> float:
        """
        Uniformly rescales the mesh so that its AABB measures target world units along axis (or along its longest axis if axis is None).
        
        Parameters:
        -----------
        target_size: np.ndarray
            Target size along the given axis following the rescaling, in the specified world units.
        axis: Axis | None
            Axis alongisde which the target_size is specified. Defaults to the longest axis at timestep = 0.
        timestep: int
            Optional. Timestep at which the longest axis is found.

        Returns
        --------
        factor: float
            The scaling factor applied to the mesh.
        """

        # Check that the size is within the bounds for which we can expect that the renders will still be correct with the given tolerances
        # NB4: this will not be enough if the mesh is very fine, so this would be best updated to check for the avg_element_length 
        #if target_size <= 0.001:
        #    return ValueError(f"The size {target_size} is too small and the render may feature artefacts that should not be there. Consider changing your world units to a larger magnitude.")

        size = self.get_size(timestep)
        current = size.max() if axis is None else size[axis.value]
        if current <= 0:
            raise ValueError("Degenerate mesh.")
        factor = target_size / current
        scaled_element_length = self.avg_element_length * factor
        self._set_element_length(scaled_element_length) # This also checks the validity of mesh dimensions
        centre = self._get_bounding_box(timestep)["center"]
        # This will be None when we _orient_in_world at the moment of converting SimData/other meshes to RTMesh, so to avoid errors
        if self.node_coords_over_time is not None:
            self.node_coords_over_time[..., :] = (self.node_coords_over_time - centre) * factor + centre
        self.node_coords[...] = (self.node_coords - centre) * factor + centre
        self.pyvista_surface.points = (self.pyvista_surface.points - centre) * factor + centre # Update pyvista surface, so it displays properly in SeamSelector and UV unwrapping
        self.scale *= factor # Update scaling
        return factor
    
    def rotate(self, rotation: Rotation | np.ndarray,
           pivot: np.ndarray | None = None,
           timestep: int = 0) -> None:
        """
        Applies the passed rotation about the specified pivot to the mesh.
        
        Parameters:
        -----------
        rotation: Rotation | np.ndarray
            The rotation to apply.
        pivot: np.ndarray | None
            Point used as a pivot around which the location is applied. Defaults to the bounding box center at timestep = 0.
        timestep: int
            Optional. Timestep at which the pivot position is found.
        """
        if pivot is None:
            pivot = self._get_bounding_box(timestep)["center"]
        R = rotation.as_matrix() if not isinstance(rotation, np.ndarray) else rotation
        self.node_coords[...] = ((self.node_coords - pivot) @ R.T) + pivot
        # This will be None when we _orient_in_world at the moment of converting SimData/other meshes to RTMesh, so to avoid errors
        if self.node_coords_over_time is not None:
            self.node_coords_over_time[...,:] = ((self.node_coords_over_time - pivot) @ R.T) + pivot
        self.pyvista_surface.points = ((self.pyvista_surface.points - pivot) @ R.T) + pivot # Update pyvista surface, so it displays properly in SeamSelector and UV unwrapping
        self.rotation = R @ self.rotation # Update rotation
        self.translation = R @ (self.translation - pivot) + pivot

    def _orient_in_world(self, world_position: np.ndarray | None = None,
                          world_rotation: Rotation | None = None,
                          target_size: float | None = None,
                          size_axis: Axis | None = None,
                          anchor: Anchor = Anchor.CENTER,
                          rotation_pivot: np.ndarray | None = None):
        """
        Orients the mesh in world space and records the resulting transforms on the RTMesh. It assumes that the RTMesh.node_coords have been assigned
        to the (node_count, 3) array of nodal coordinates in file/simulation units.

        Applies scaling, rotation, and translation to the supplied raw coordinate array in that order, then stores the computed transforms
        and the axis-aligned bounding box resting_aabb so that oriented bounding box (OBB) queries can reconstruct the world-space box
        without re-scanning all nodes.

        Processing order:
        1. fit_size — If target_size is given: uniform scale about centre.
        2. rotate — Rotation about rotation_pivot (defaults to post-scale bounding box centre).
        3. place_at — Translates anchor point to world_position.

        After the call the following RTMesh fields are populated:
        - scale: Cumulative uniform scale factor applied (1.0 if target_size is None).
        - rotation: Scipy Rotation that was applied (identity if world_rotation is None).
        - resting_aabb: AABB dict (min_corner, max_corner, center) of the mesh *after*
                        scaling and rotation but *before* the final translation, i.e., the
                        box in the rotated/scaled frame at the origin.  Combined with
                        translation this fully describes the OBB in world space.
        - translation: World-space translation vector that moved the anchor to
                       world_position.

        Parameters
        ----------
        world_position : np.ndarray
            Optional. Target world-space location for the chosen anchor. Defaults to the origin.
        world_rotation : Rotation
            Optional. Scipy Rotation to apply. Use make_axis_rotation(...) for single-axis spins.
        target_size : float
            Optional. World-space size along size_axis (or longest axis when None).
        size_axis : Axis
            Optional. Axis along which target_size is measured.
        anchor : Anchor
            Which point on the AABB is placed at world_position. Defaults to CENTER.
        rotation_pivot : np.ndarray, optional
            Overrides the rotation pivot. Defaults to the post-scale bounding box centre.
        """

        # 1. Set the element length  and scale
        self.avg_element_length = self._compute_average_element_length()
        if target_size is not None:
            self.fit_size(target_size, axis=size_axis)
        
        # 2. Find resting AABB before translation and rotation
        self.resting_aabb = self._get_bounding_box()
        
        # 3. Rotate
        if world_rotation is not None:
            self.rotate(world_rotation, pivot=rotation_pivot)

        # 4. Translate
        if world_position is not None:
            self.place_at(world_position, anchor=anchor)


    def _set_refractive_index(self, refractive_index: float) -> None:
        """
        Sets the refractive index of the mesh. 

        Helper function to clean up set_surface a little bit.

        Parameters:
        -----------
        refractive_index: float
            The refractive index to set for the mesh. Defaults to 1.0003 for air. 

        Raises:
        -------
        ValueError:
            If the refractive index is less than 0.
        """
        if refractive_index > 0.0:
                self.refractive_index = refractive_index
        else: 
            raise ValueError("Refractive index can be negative only for metamaterials, and these are not supported yet.")
    
    def _set_reference_thickness(self, reference_thickness: float,
                                 mesh_type: MeshType) -> float:
        """
        Validates and sets the reference thickness of a (refractive) mesh.

        Helper function to clean up set_surface a little bit.

        Parameters:
        -----------
        reference_thickness: float
            The relative thickness to set for the mesh. If passed as none, we assign:
            - For MeshType.SHELL: reference_thickness = thickness
            - For MeshType.SOLID: reference_thickness = bounding box diagonal
        mesh_type: MeshType
            The type of mesh, which will affect the refractive behaviour, where applicable.

        Returns:
        --------
        reference_thickness: float
            The updated reference thickness.

        Raises:
        -------
        ValueError:
            If the reference thickness is negative.
        """
        if reference_thickness is not None:
            if reference_thickness < 0.0:
                raise ValueError("Reference thickness cannot be negative.")
        else:
            if mesh_type == MeshType.SHELL:
                # x2 to account for thickness from both sides of the mesh, otherwise the shells come out far darker than the desired colour
                reference_thickness = self.thickness*2
            elif mesh_type == MeshType.SOLID:
                # For solids, set to the diagonal of the bounding box as a ballpark value that should give us decent enough values
                # Below assumption is valid for previous mesh positioning/orienting where we were scaling the meshes massively, but in the new implementation, they are so small that they turn black
                bounding_box = self._get_bounding_box()
                diagonal = np.linalg.norm(bounding_box["max_corner"] - bounding_box["min_corner"])
                reference_thickness = diagonal
        return reference_thickness
        
    def _set_thickness(self, thickness: float,
                       mesh_type: MeshType) -> None:
        """
        Sets the thickness of the shell mesh.

        Helper function to clean up set_surface a little bit.

        Parameters:
        -----------
        thickness: float
             The thickness of the mesh shell, counted INWARDS from the actual boundary. Optional. Defaults to None or the maximum allowed thickness for a given shell, given as 1/10 of the average element length.
        mesh_type: MeshType
            The type of mesh, which will affect the refractive behaviour, where applicable.

        Raises:
        -------
        ValueError:
            If the reference thickness is negative or too small.
        """
        if mesh_type == MeshType.SOLID:
            if thickness is not None:
                print("Thickness value is ignored for solid meshes.")
        elif mesh_type == MeshType.SHELL:
            # Reissner-Mindlin cut-off for thickness that makes sense physically: 1/10 of a planar dimension; we have 2 options, to be determined which one makes more sense for a ray-tracer
            shell_thickness_cutoff_elem = 0.1 * self.avg_element_length # Mesh element edge length - might be too small if mesh is very fine
            min_dimension_length = min(self.resting_aabb["max_corner"] - self.resting_aabb["min_corner"])
            shell_thickness_cutoff_geom = 0.1 * min_dimension_length # Based on the smallest dimension of the mesh bounding box
            if thickness is None:
                #print(f"Thickness not set for a thin shell. Setting it to the maximum allowed value: {shell_thickness_cutoff_elem:.5f}.")
                #self.thickness = shell_thickness_cutoff_shell
                print(f"Thickness not set for a thin shell. Setting it to the maximum allowed value: {shell_thickness_cutoff_geom:.5f}.")
                self.thickness = shell_thickness_cutoff_geom
            elif thickness <= 0.0:
                raise ValueError("Thickness of a shell cannot be negative or zero.")
            elif thickness > shell_thickness_cutoff_elem:
                pass # Pass for now, because I found this cut-off to be too small
                #raise ValueError(f"Thickness of the shell should not exceed 1/10th of the average edge length. The cut-off value is {shell_thickness_cutoff_elem:.6f}. Current thickness is {thickness:.6f}.")
            elif thickness > shell_thickness_cutoff_geom: 
                raise ValueError(f"Thickness of the shell should not exceed 1/10th of the smallest bounding box dimension. The cut-off value is {shell_thickness_cutoff_geom:.6f}. Current thickness is {thickness:.6f}.")
            else:
                self.thickness = thickness

    def set_surface(self, surface_type: SurfType = SurfType.FIELD_COLOR,
                    surface_fill: np.ndarray = None,
                    material_type: MaterialType = MaterialType.UNLIT,
                    priority: int = 0,
                    refractive_index: float = 1.0003,
                    mesh_type: MeshType = MeshType.SOLID,
                    thickness: float | None = None,
                    reference_thickness: float | None = None, 
                    material: Material | None = None) -> None:
        """
        Sets the surface type and fill for the mesh.
        
        Surface fill can be either a solid color, an array of colors, or a texture array, depending on the surface type.
        For field-based coloring, the surface fill can also be based on field values, which will be mapped to colors using a colormap (not integrated here yet).
        The method also resets any existing surface data if the surface type is changed.

        Parameters:
        -----------
        surface_type: SurfType
            The type of surface to apply to the mesh. Can be either FIELD_COLOR or TEXTURE.
        surface_fill: np.ndarray
            The fill to apply to the mesh, clamped to the [0,1] intensity range (linear sRGB). The expected format depends on the surface type:
            - For FIELD_COLOR:
                - If shape is (3,), it is interpreted as a single RGB color applied to the entire mesh.
                - If shape is (element_count, 3), it is interpreted as an RGB color for each element, applied to the entire time series.
                - If shape is (timestep_count, element_count, 3), it is interpreted as an RGB color for each element at each timestep.
            - For TEXTURE:
                - The surface_fill should be a 2D array representing the texture image. The UV coordinates must be set for the mesh to apply the texture correctly.
        material_type: MaterialType
            The material type to apply to the mesh. Defaults to UNLIT for no shading effects.
        priority: int
            The priority of the mesh. Used only if the material_type is set to refractive. The higher the priority, the more internal the mesh is, so e.g., for for a glass of water with ice cubes in it, you'd want to set glass = 0, water = 1, ice cubes = 2.
            Defaults to 0.
        refractive_index: float
            Optional. The refractive index to apply to the mesh. Applicable only for refractive materials. Defaults to 1.0003 for air. 
        mesh_type: MeshType
            Optional. The type of mesh, which will affect the refractive behaviour, where applicable. Defaults to SOLID.
        thickness: float
            Optional. The thickness of the mesh shell, counted INWARDS from the actual boundary. Defaults to None or the maximum allowed thickness for a given shell given as 1/10 of the average element length.
        reference_thickness: float | None
            Optional. Used only for refractive materials. Defines how thick the slab of a medium should be to see exactly the colour passed in surface_fil to determine the absorption via Beer-Lambert's law. Defaults to None, and then we assign:
            - For MeshType.SHELL: reference_thickness = thickness
            - For MeshType.SOLID: reference_thickness = bounding box diagonal
        material: Material | None
            Optional. Material class containing the data about colour and RI. Defaults to None.
            
        Raises:
        -------
        ValueError:
            If the surface_fill does not match the expected format for the given surface_type, or;
            - UV coordinates are required but not provided for texture mapping,
            - Refractive_index is less than 0; or
            - Mesh thickness/relative_thickness has an unphysical value.
        TypeError:
            If the material_type is set to REFRACTIVE and texture is passed, as this is currently not supported.
        """
        # Reset everything if user is changing the surface type
        if self.surface_type is not None and surface_type != self.surface_type:
            self.face_colors_over_time = None
            self.texture = None
            self.uvs = None
            self.material_type = None
            self.refractive_index = 1.0003
            self.priority = 0
            self.mesh_type = MeshType.SOLID
            self.thickness = 1.0
            #self.uvs_over_time = None
        self.surface_type = surface_type
        self.material_type = material_type
        self.mesh_type = mesh_type
        
        # Priority
        if priority >= 0:
            self.priority = priority
        else: 
            raise ValueError("Priority should be greater or equal to 0.")
        
        self._set_thickness(thickness, mesh_type)

        # If using a material preset
        if material is not None:
            refractive_index = material.RI
            surface_fill = material.color
                
        # Solid colors
        if surface_type == SurfType.FIELD_COLOR:
            #if material_type == MaterialType.REFRACTIVE:
            #    print("Tinting for refractive materials is currently not supported, so the colour data will be ignored.")
            if surface_fill is None:
                print("No colour data passed. Pre-filling automatically with grey.")
            else:
                # Validate input values
                if np.any(surface_fill < 0.0):
                    raise ValueError("Surface fill cannot be negative.")
                elif np.any(surface_fill > 1.0):
                    print("Passed colour data contains values exceeding 1.0. It is assumed that it was given as regular RBG values in range [0, 255], so they will be clamped.")
                    surface_fill = np.clip(surface_fill/255, 0.0, 1.0)
                # For refractive materials, we want to turn tint into absorption based on the Beer-Lambert law to get tinting and set all other quantities that we need
                if material_type == MaterialType.REFRACTIVE:
                    self._set_refractive_index(refractive_index)
                    reference_thickness = self._set_reference_thickness(reference_thickness, mesh_type) # Validate and update value if needed
                    if reference_thickness is None:
                        raise ValueError("Reference thickness is required for refractive materials.")
                    surface_fill = -np.log(surface_fill) / reference_thickness # This is the sigma_a in the equation, given in (length unit)^-1
                    print(f"Absorption: {surface_fill}")
                # Process data based on the shape
                if surface_fill.shape == (RGB_VALS,): 
                    # Populate with passed solid color, e.g., [0.5, 0.2, 0.45]
                    self.face_colors_over_time = np.ones((self.timestep_count, self.element_count, RGB_VALS)) * surface_fill
                    return
                elif surface_fill.shape == (self.element_count, RGB_VALS):
                    # One avg. RGB colour value per element, given only for one timestep
                    self.face_colors_over_time =  np.broadcast_to(surface_fill[np.newaxis, ...], (self.timestep_count, self.element_count, RGB_VALS))
                    return
                elif surface_fill.shape == (self.timestep_count, self.element_count, RGB_VALS):
                    # One avg. RGB colour value per element, given for each timestep
                    self.face_colors_over_time = surface_fill
                    return
                else:
                    print("Surface fill must be of shape (3,) or (element_count, 3) or (timestep_count, element_count, 3).\nPre-filling automatically with grey.")
            # Create face colors over time of appropriate size and pre-populate with grey (this ignores absorption)
            self.face_colors_over_time = np.ones((self.timestep_count, self.element_count, RGB_VALS), dtype=np.float64) * 0.5
        # Texture
        elif surface_type == SurfType.TEXTURE:
            # Might move this logic elsewhere, so we could use the mesh texture in BlenderUnwrapper without passing it as an argument?
            if self.uvs is None:
                raise ValueError("UV coordinates are required to append texture.")
            if surface_fill.ndim != 2:
                raise ValueError("Wrong number of dimensions. The array containing the texture should be two-dimensional.")
            if material_type == MaterialType.REFRACTIVE:
                raise TypeError("Textures with refractive materials are currently not supported.") # They might be in the future if we use transparency etc.
            # Convert UVs to the format similar to node_coords_expanded: (element_count, nodes_per_element, 2)
            # NOTE: UVS should **NOT** change across the frames, so we do not need that. If you need to use it, uncomment relevant lines in rtscene.py and copy_data_to_blas_tex in rtbvh.cpp
            #self.uvs_over_time = np.broadcast_to(self.uvs[np.newaxis, ...], (self.timestep_count, self.element_count, self.nodes_per_element, 2))
            # TO DO: Add check for shape of texture array
            self.texture = surface_fill

    def set_custom_uvs(self, uv_coords: np.ndarray = None,
                       face_mapping: np.ndarray = None) -> None:
        """
        Allows user to set custom UV coordinates for texture mapping.

        Parameters:
        -----------
            uv_coords: np.ndarray
                The UV coordinates to set for the mesh. The expected format can be either:
                - (new_node_count, 2): Standard UV format where each row corresponds to a unique node in the mesh. The face_mapping argument is required in this case to map the UVs to the correct nodes in the triangulated mesh.
                - (element_count, nodes_per_element, 2): Expanded UV format where each row corresponds to a unique element in the mesh and each column corresponds to a node in that element. This format is already expanded to match
                the triangulated mesh, so the face_mapping argument is not required in this case.
            face_mapping: np.ndarray
                An array of shape (element_count, nodes_per_element) that maps the original nodes of the mesh to the nodes in the triangulated mesh. This is required if the uv_coords are provided in standard format (new_node_count, 2)
                to correctly assign UVs to the triangulated mesh nodes. The values in face_mapping should be indices that correspond to the rows in uv_coords.
        Raises:
        -------
        ValueError:
            If the uv_coords are not provided, if they contain only zero values, if they do not have the correct shape, or if the face_mapping is required but not provided or has the wrong shape.
         """
        # Set custom UVs that aren't acquired from the Blender module
        # Expect either standard UV format or already expanded
        if uv_coords is None:
            raise ValueError("UV coordinates are required to append texture.")
        if not np.any(uv_coords):
            raise ValueError("UV coordinates cannot contain only zero values.")
        uv_coords_shape = uv_coords.shape
        if uv_coords_shape[-1] != 2:
            raise ValueError(f"Invalid uv coordinate array shape: {uv_coords.shape}. UV coordinates must be of shape (new_node_count, 2) or (element_count, nodes_per_element, 2).")

        if uv_coords.ndim == 2: # UVs in standard format (u,v) - we need to know the face mapping to use that
            if face_mapping is None:
                raise ValueError("Face mapping is required to set custom UVs.")
            if face_mapping.shape != (self.element_count, self.nodes_per_element):
                raise ValueError(f"Face mapping must be of shape (element_count, nodes_per_element). Got {face_mapping.shape}. If you triangulated your mesh independently, you need to map the uvs back to the original surface mesh.")
            self.uvs = np.ascontiguousarray(uv_coords[face_mapping], dtype=np.float64)
        elif uv_coords.ndim == 3: # UVs in expanded format (element_count, nodes_per_element, 2)
            if uv_coords_shape[0] != self.element_count or uv_coords_shape[1] != self.nodes_per_element: # Check that the dimensions match expectations
                raise ValueError(f"UV coordinates must be of shape (element_count, nodes_per_element, 2). Got {uv_coords.shape}. If you triangulated your mesh independently, you need to map the uvs back to the original surface mesh.")
            self.uvs = np.ascontiguousarray(uv_coords, dtype=np.float64)

    def import_seams_from_csv(self, filepath: str) -> None:
        """
        Imports seams from a CSV file and stores them in the RTMesh object, without having to go through SeamSplitter.

        The method reads the CSV file, processes each row to extract the seam ID and corresponding node IDs, and stores them in the seams attribute of the RTMesh object as a list of lists, where each inner list represents a seam with
        its associated node IDs.
        
        Parameters:
        -----------
        filepath: str
            The path to the CSV file containing the seam data.
        
        Notes:
        -------
        - The CSV file is expected to have the following format:
            SeamID, NodeID1, NodeID2, ..., NodeIDn
            Where each row corresponds to a seam, the first column contains the seam ID, and the subsequent columns contain the node IDs that belong to that seam.
            The number of node ID columns can vary for each seam, and empty values should be left blank (or filled with NaN if using pandas).
            I.e., the same format as exported from SeamSplitter.
        """
        try:
            temp_df = pd.read_csv(filepath, sep=",", header=None,
                                  dtype="Int64")  # Pandas turns integers to floats if some rows have NaN for the same columns, so we need to force integers
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return
        temp_df = temp_df.fillna(-1).astype(int)  # Replace <NA> with -1
        list_of_seams = temp_df.values.tolist()  # Convert to list of lists to match the formatting in the rest of the workflow
        one_seam = list()
        for seam in list_of_seams:
            one_seam.append(seam[0])
            for i in range(1, len(seam)):
                node_id = seam[i]
                if node_id == -1:  # -1 marks start of empty values for that seam (pandas fills it with NaN), so break
                    break
                one_seam.append(node_id)
            self.seams.append(one_seam.copy())
            one_seam.clear()
        self.seams = list_of_seams

    def get_size(self, timestep: int = 0) -> np.ndarray:
        """
        Returns the size of the mesh in world units at the given timestep.

        Parameters
        ----------
        timestep : int, optional
            The timestep to get the mesh size for (default is 0 for static meshes).
        Returns
        -------
        np.ndarray
            The size of the mesh in world units at the given timestep.
        """
        # node_coords_over_time[timestep] and node_coords are both (node_count, 3),
        # so we reduce over the node axis (axis 0) only to get per-axis spans.
        if self.node_coords_over_time is not None:
            coords = self.node_coords_over_time[timestep, :, :]
        else:
            coords = self.node_coords
        spans = np.abs(coords.max(axis=0) - coords.min(axis=0))
        return spans

    def _get_bounding_box(self, timestep: int = 0) -> dict:
        """
        Returns mesh position information as a dictionary containing the minimum and maximum x, y, and z coordinates,
        and the center (mean nodal position; not an actual nodal position) in world coordinates at the given timestep.

        Parameters
        ----------
        timestep : int, optional
            The timestep to get the bounding box for (default is 0 for static meshes).
        Returns
        -------
        dict
            A dictionary containing the minimum and maximum x, y, and z coordinates, and the center in world coordinates at the given timestep.
        """
        if self.node_coords_over_time is not None:
            coords = self.node_coords_over_time[timestep, :]
        else:
            coords = self.node_coords
        minimal_coords = coords.min(axis=0)
        maximal_coords = coords.max(axis=0)
        center = 0.5 * (minimal_coords + maximal_coords)
        return {"min_corner": minimal_coords, "max_corner": maximal_coords, "center": center}
    
    def is_visible_in_viewport(self, camera: Camera, timestep: int = 0):
        """
        Checks if a mesh is visible in the viewport of a camera.

        Parameters
        ----------
        camera : Camera
            The Camera object containing the camera data.
        timestep : int, optional
            The timestep to check the mesh visibility for (default is 0 for static meshes).
        Returns
        -------
        bool
            True if the mesh is fully or partially visible in the viewport, False if the mesh is completely outside the viewport or behind the camera,
            along with hints on how to adjust the mesh position to make it visible, and an estimate of how much of the viewport it should cover if it is visible.
        """
        # Get mesh bounds
        mesh_aabb = self._get_bounding_box(timestep)
        mesh_center = mesh_aabb["center"]
        half_extents = self.get_size(timestep)/2

        # Extract camera basis vectors from the camera-to-world matrix
        c2w = camera.matrix_camera_to_world
        cam_right = c2w[0,:3]
        cam_up = c2w[1, :3]
        cam_forward = -c2w[2, :3]  # Camera looking down -z hence negative

        # Perspective projection parameters
        # tan(FOV/2) gives us the boundary of the frustum at distance 1.0; or, half of the viewport height
        h_temp = np.tan(camera.angle_vertical_view / 2)
        aspect_ratio = camera.image_width / camera.image_height
        tan_half_fov_h = h_temp * aspect_ratio

        # Check the 8 corners of the AABB
        corners_in_front = 0
        any_in_view = False

        # Track NDC bounds to see how much of the screen it covers
        ndc_bounds = {"x": [np.inf, -np.inf], "y": [np.inf, -np.inf]}

        for signs in itertools.product([-1, 1], repeat=3):
            p_world = mesh_center + (np.array(signs) * half_extents)
            delta = p_world - camera.camera_center

            # Project world point onto camera axes (transform to camera space)
            z_cam = np.dot(delta, cam_forward)
            x_cam = np.dot(delta, cam_right)
            y_cam = np.dot(delta, cam_up)

            # Skip points behind the camera (near plane clipping)
            if z_cam <= 0.001:
                continue

            corners_in_front += 1
            # Project to Normalized Device Coordinates (NDC)
            # Range will be [-1, 1] if the point is inside the frustum
            x_ndc = x_cam / (z_cam * h_temp)
            y_ndc = y_cam / (z_cam * h_temp)

            ndc_bounds["x"][0] = min(ndc_bounds["x"][0], x_ndc)
            ndc_bounds["x"][1] = max(ndc_bounds["x"][1], x_ndc)
            ndc_bounds["y"][0] = min(ndc_bounds["y"][0], y_ndc)
            ndc_bounds["y"][1] = max(ndc_bounds["y"][1], y_ndc)

            if -1 <= x_ndc <= 1 and -1 <= y_ndc <= 1:
                any_in_view = True

        if corners_in_front == 0:
            print(f"Mesh is entirely behind the camera.")
            return False

        if not any_in_view:
            # Check if the mesh is so large it spans the whole viewport, but it's corners are outside
            if (ndc_bounds["x"][0] < -1 and ndc_bounds["x"][1] > 1 and
                    ndc_bounds["y"][0] < -1 and ndc_bounds["y"][1] > 1):
                print(f"Mesh spans the whole viewport, but its corners are outside.")
                return True
            # Positioning guidance if the mesh is outside the frustum
            hints = []
            # Horizontal positioning
            if ndc_bounds["x"][1] < -1:
                hints.append("too far LEFT (increase X or pan camera left)")
            elif ndc_bounds["x"][0] > 1:
                hints.append("too far RIGHT (decrease X or pan camera right)")

            # Vertical positioning
            if ndc_bounds["y"][1] < -1:
                hints.append("too far DOWN (increase Y or tilt camera down)")
            elif ndc_bounds["y"][0] > 1:
                hints.append("too far UP (decrease Y or tilt camera up)")

            # Depth positioning (if all corners were behind the camera)
            if corners_in_front == 0:
                hints.append("BEHIND the camera (check Z-positioning)")

            direction_str = " and ".join(hints) if hints else "out of frustum range"
            print(f"Mesh is invisible: {direction_str}.")

            # Debugging values
            print(f"   NDC X-range: [{ndc_bounds['x'][0]:.2f}, {ndc_bounds['x'][1]:.2f}]")
            print(f"   NDC Y-range: [{ndc_bounds['y'][0]:.2f}, {ndc_bounds['y'][1]:.2f}]")

            return False

        # Estimate screen coverage - useful for scaling
        # Clamp bounds to screen for area calculation
        x_range = min(1, ndc_bounds["x"][1]) - max(-1, ndc_bounds["x"][0])
        y_range = min(1, ndc_bounds["y"][1]) - max(-1, ndc_bounds["y"][0])

        # Area in NDC is 2x2=4. Percentage of screen:
        pct_coverage = (max(0, x_range) * max(0, y_range) / 4.0) * 100

        print(f"Mesh is fully visible in viewport. It should occupy about {pct_coverage:.2f}% of the viewport.")
        return True

# ================================================================================
# POSITIONING HELPERS
# ================================================================================

def get_anchor_point(aabb: dict, anchor: Anchor) -> np.ndarray:
    """
    Gets the coordinates of the specified anchor point of the bounding box.

    Parameters:
    -----------
    aabb: dict
        Mesh bounding box. A dictionary containing the minimum and maximum x, y, and z coordinates, and the center (mean nodal position; not an actual nodal position) in world coordinates at the given timestep.
    anchor: Anchor
        Specified anchor point of the bounding box.
    """
    low, high, center = aabb["min_corner"], aabb["max_corner"], aabb["center"]
    if anchor is Anchor.CENTER: return center
    if anchor is Anchor.MIN: return low
    if anchor is Anchor.MAX: return high
    if anchor is Anchor.BASE: return np.array([center[0], low[1], center[2]])
    if anchor is Anchor.TOP: return np.array([center[0], high[1], center[2]])
    raise ValueError(f"Invalid anchor: {anchor}")

def snap_to(move_rtmesh: RTMesh,
            support_rtmesh: RTMesh,
            axis: Axis = Axis.Y,
            align: tuple[Axis, ...] = (Axis.X, Axis.Z),
            gap: float = 0.0,
            stack_above: bool = True,
            rotate = False,
            timestep: int = 0) -> None:
    """
    Puts one RTMesh in contact with another along a specified axis, e.g., a ball on a table using their Oriented Bounding Boxes (OBB).
    axis and align are interpreted in the SUPPORT's local frame — i.e. Axis.Y means "up relative to the support". The moved mesh's contact
    point is its world-space AABB extreme along the support's local axis.

    Note: This is a very basic version of this function, which doesn't work for concave meshes. Snapping will also not work
    entirely well if they have very different rotations.
    In these cases, you may find SceneVisualiser more useful as it will allow you to select the anchor points and translate meshes directly.

    Parameters:
    -----------
    move_rtmesh: RTMesh
        Mesh to be aligned.
    support_rtmesh: RTMesh
        Mesh to be aligned to.
    axis: Axis
        Axis along which to align the meshes.
    align: tuple[Axis,...]
        Axes on which to center move_rtmesh over support_rtmesh. Pass () to keep the mover's lateral position unchanged. Defaults to (Axis.X, Axis.Z).
    gap: float
        Signed clearance in world units. 0 = touching (should be best for ray tracing, with no hover-shadow or hidden geometry).
        Use a tiny positive value (e.g. 1e-4 * scene size) if you see z-fighting at the contact surface; negative values intentionally intersect.
    stack_above: bool
        - True: moving sits on top of support along +axis.
        - False: moving hangs below support along -axis.
    rotate: bool
        Optional.Whether to apply the rotation of the support RTMesh to the moved object. Defaults to False.
    timestep: int
        Optional. Timestep at which we consider the bounding boxes of the meshes, used for alignment. Defaults to 0.
    """
    # Get local frame for the support RTMesh: a unit basis written in world coordinates
    R_support = support_rtmesh.rotation
    support_local_axis_world = R_support[:, axis.value] # (3,) unit vector
    # Support the local AABB (rest-frame) and its corresponding world face
    support_rest = support_rtmesh.resting_aabb
    support_face_local = support_rest["center"].copy()
    if stack_above:
        support_face_local[axis.value] = support_rest["max_corner"][axis.value]
    else:
        support_face_local[axis.value] = support_rest["min_corner"][axis.value]
    
    # Map that face-centre point from local to world
    sup_face_world = R_support @ support_face_local + support_rtmesh.translation

    # Process moved RTMesh: take its world AABB extreme along the SUPPORT's local axis
    if rotate:
        move_rtmesh.rotate(support_rtmesh.rotation)
    move_coords = move_rtmesh.node_coords_over_time[timestep].reshape(-1, 3)
    proj = move_coords @ support_local_axis_world  # (faces,) signed distances
    if stack_above:
        move_extreme_world = move_coords[np.argmin(proj)]  # lowest along support-up
    else:
        move_extreme_world = move_coords[np.argmax(proj)]

    # Find delta and project gap onto the support-axis only
    along_gap = (gap if stack_above else -gap) * support_local_axis_world
    delta_full = (sup_face_world - move_extreme_world) + along_gap

    # Restrict lateral movement to the requested align axes (in support frame)
    # Decompose delta into support-frame components, zero-out the unwanted ones
    delta_local = R_support.T @ delta_full
    keep = np.zeros(3, dtype=bool)
    keep[axis.value] = True
    # This is in case we align to one axis (e.g., (Axis.X)) because for this to work, we would need to type (Axis.X,) and not all users might know about that
    if isinstance(align, Axis):
        align = (align,)  # normalize to tuple
    for a in align:
        keep[a.value] = True
    delta_local = np.where(keep, delta_local, 0.0)
    delta = R_support @ delta_local

    move_rtmesh.translate(delta)

def make_axis_rotation(axis: "Axis | np.ndarray",
                       angle: float,
                       degrees: bool = True) -> Rotation:
    """
    Builds a Scipy Rotation about an arbitrary axis.

    Parameters
    ----------
    axis: Axis | np.ndarray
        Either an Axis enum (X/Y/Z) or a (3,) np.ndarray — need not be unit length; it is normalised internally.
    angle: float
        Rotation angle.
    degrees: bool
        If True (default), angle is in degrees.

    Returns
    -------
    Rotation
        A Scipy Rotation object.
    """
    if isinstance(axis, Axis):
        v = np.zeros(3)
        v[axis.value] = 1.0
    else:
        axis_normalised = np.linalg.norm(axis)
        if axis_normalised == 0:
            raise ValueError("Rotation axis must be non-zero.")
        v = axis / axis_normalised
    angle_radians = np.radians(angle) if degrees else angle
    return Rotation.from_rotvec(angle_radians * v)

# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def segment_into_charts(connectivity: np.ndarray):
    """
    Segment mesh into charts using sparse adjacency matrix.

    Parameters:
    -----------
    connectivity: ndarray
        Face connectivity after seam cutting. Shape (element_count, nodes_per_element)

    Returns:
    --------
    chart_labels: ndarray
        Chart ID for each face (0, 1, 2, ...). Shape (element_count,)
    chart_count: int
        Total number of charts
    charts: list of ndarray
        List where charts[i] contains face indices belonging to chart i
    adjacency: csr_matrix
        Sparse matrix storing face-to-face adjacency data for the input connectivity array
    
    """

    element_count = len(connectivity)
    nodes_per_element = connectivity.shape[1]

    # Get the number of corners to correctly extract the connectivity
    corner_count = 0
    if nodes_per_element in (ElementNodeCount.QUAD4, ElementNodeCount.QUAD8, ElementNodeCount.QUAD9):
        corner_count = 4
    elif nodes_per_element in (ElementNodeCount.TRI3, ElementNodeCount.TRI6):
        corner_count = 3

    # Slice the connectivity array to only look at the corner nodes
    corner_connectivity = connectivity[:, :corner_count]

    # Build edge -> face mapping
    edge_to_faces = defaultdict(list)

    for face_idx, face_vertices in enumerate(corner_connectivity):
        # Generate all edges of this face
        for i in range(corner_count):
            v1 = face_vertices[i]
            #v2 = face_verts[(i + 1) % nodes_per_element]
            v2 = face_vertices[(i + 1) % corner_count]
            edge = tuple(sorted([v1, v2]))
            edge_to_faces[edge].append(face_idx)

    # Build sparse adjacency matrix (face-to-face)
    row_indices = []
    col_indices = []

    for edge, face_list in edge_to_faces.items():
        # Faces sharing this edge are neighbors
        for i, f1 in enumerate(face_list):
            for f2 in face_list[i + 1:]:
                row_indices.extend([f1, f2])
                col_indices.extend([f2, f1])

    # Create sparse adjacency matrix
    data = np.ones(len(row_indices), dtype=np.uint8)
    adjacency = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(element_count, element_count),
        dtype=np.uint8)

    # Find connected components using scipy
    chart_count, chart_labels = connected_components(
        csgraph=adjacency,
        directed=False,
        return_labels=True)

    # Group faces by chart
    charts = [np.where(chart_labels == i)[0] for i in range(chart_count)]

    return chart_labels, chart_count, charts, adjacency

def extract_submesh(rtmesh: RTMesh,
                    chart_face_indices: np.ndarray,
                    render_mesh: RenderMesh = None) -> RTMesh:
    """
    Extracts a submesh from the passed PyVista UnstructuredGrid or PolyData object based on
    the passed face indices and populates the supplied RTMesh with the result.

    The RTMesh object should have already been created and oriented by the caller via
    _orient_in_world; its transform fields are copied onto the new submesh so that every
    chart shares the same world-space placement as the parent mesh.

    Parameters:
    ----------- 
    rtmesh: RTMesh
        Pre-created, pre-oriented RTMesh from the calling factory function. Its transform fields (translation, rotation, scale, resting_aabb) are copied to the submesh so
        every chart's OBB data stays consistent with the full mesh orientation.
    chart_face_indices: np.ndarray
        The face indices of the submesh to extract.
    render_mesh: RenderMesh
        Optional. RenderMesh object passed for RTMesh generation if the mesh is SimData-based. Defaults to None.
    
    Returns:
    --------
    chart_rtmesh: RTMesh
        The extracted submesh as an RTMesh object.
    """
    # Extract sub-grid based on the chart connectivity
    sub_ugrid = rtmesh.pyvista_surface.extract_cells(chart_face_indices, pass_point_ids=True)
            
    # Explicit copy so node_coords does not share memory with sub_ugrid.points; otherwise any later in-place transform (translate/rotate/fit_size) would
    # mutate the grid twice (see the note in any_mesh_to_rtmesh).
    sub_coords = np.array(sub_ugrid.points, dtype = np.float64)
    sub_connectivity = pyvista_faces_to_connectivity(sub_ugrid)

    # Capture the mapping from local submesh node indices -> global render_mesh node indices so that get_mesh_data_over_time can slice the correct rows of the
    # full-mesh displacement array
    rm_point_ids = None
    if render_mesh is not None and "vtkOriginalPointIds" in sub_ugrid.point_data:
        rm_point_ids = np.ascontiguousarray(
            sub_ugrid.point_data["vtkOriginalPointIds"], dtype=np.intp)

    # Copy the parent RTMesh to preserve its transform fields etc., to share the world-space orientation with the parent,
    # but replace coordinates, connectivity, and pyvista grid with the values appropriate for the submesh.
    # Deep copy to create a new object, so changes do not propagate between the parent and the child.
    chart_rtmesh = deepcopy(rtmesh)
    chart_rtmesh.connectivity = np.ascontiguousarray(sub_connectivity, dtype=np.uint64)
    chart_rtmesh.node_coords = sub_coords
    chart_rtmesh.pyvista_surface = sub_ugrid # Oriented sub-grid; create_rtmesh will (re)triangulate it for UV-unwrapping if needed
    # The over-time series belongs to the parent's node set, so reset it; create_rtmesh rebuilds it for this submesh
    chart_rtmesh.node_coords_over_time = None
    chart_rtmesh.rm_point_ids = rm_point_ids

    # Populate the chart RTMesh with geometry and topology
    create_rtmesh(chart_rtmesh, render_mesh)
    return chart_rtmesh


def pyvista_faces_to_connectivity(pv_grid: pv.UnstructuredGrid | pv.PolyData) -> np.ndarray:
    """
    Converts the faces array from a PyVista UnstructuredGrid or PolyData object into a connectivity array of shape (num_elements, nodes_per_element) in C format (0-based indexing, contiguous in memory).

    This is an adaptation of part of the logic from pyvale's create_render_mesh, but modified to preserve the original higher-order elements instead of triangulating them.

    Parameters:
    -----------
    pv_grid: pv.UnstructuredGrid | pv.PolyData
        The PyVista object to convert.

    Returns:
    -----------
    connectivity: np.ndarray
        Shape (num_elements, nodes_per_element). A 2D array containing the connectivity information for the mesh elements, formatted in C style (0-based indexing, contiguous in memory).
    Raises:
    --------
    TypeError:
        If the input grid is not a PyVista UnstructuredGrid or PolyData object.
    """
    faces = None
    if type(pv_grid) == pv.UnstructuredGrid:
        faces = np.array(pv_grid.cells)  # Assume unstructured grid as default
    elif type(pv_grid) == pv.PolyData:
        faces = np.array(pv_grid.faces) # If we use extract_surface or triangulate, we need to use faces attribute
    if faces is None:
        raise TypeError("Input grid must be a PyVista UnstructuredGrid or PolyData object.")
    first_elem_nodes_per_face = faces[0]
    nodes_per_face_vec = faces[0::(first_elem_nodes_per_face + 1)]

    assert np.all(nodes_per_face_vec == first_elem_nodes_per_face), \
        "Not all elements have the same number of nodes per element"

    nodes_per_face = first_elem_nodes_per_face
    num_faces = int(faces.shape[0] / (nodes_per_face + 1))
    # Reshape the faces table and slice off the first column which is just the number of nodes per element and should be the same for all elements
    connectivity = np.reshape(faces, (num_faces, nodes_per_face + 1))
    # shape=(num_elems,nodes_per_elem), C format
    connectivity = np.ascontiguousarray(connectivity[:, 1:], dtype=np.uint64)
    return connectivity


def find_node_normals(node_coords: np.ndarray, 
                      connectivity: np.ndarray, 
                      element_count: int) -> np.ndarray:
    """
    Finds angle-weighted node normals for surface meshes. Should handle quadratic elements (TRI6, QUAD8, QUAD9) as well.

    For each element we compute a normal per corner from the two edges incident to that corner, weighted by the interior angle at that corner (https://www.tandfonline.com/doi/abs/10.1080/10867651.1999.10487501).
    Corner contributions are accumulated to corner nodes only. Mid-edge nodes get the average of their two adjacent corners' weighted normals, and QUAD9 center node gets the average of all 4 corners.
    
    Parameters:
    -----------
    node_coords: np.ndarray
        Shape (node_count, 3). The coordinates of the nodes in the mesh.
    connectivity: np.ndarray
        Shape (element_count, nodes_per_element). The connectivity of the mesh elements.
    element_count: int
        Number of elements in the mesh.
    
     Returns:
    -----------
    node_normals: np.ndarray
        Shape (node_count, 3). The angle-averaged normal vector for each node.

    Raises:
    --------
    ValueError:
        If the input element type is unsupported.
    """

    node_count = node_coords.shape[0]
    node_normals = np.zeros((node_count, 3), dtype=np.float64)
    nodes_per_element = connectivity.shape[1]

    # 1. Define corner topology per element type
    # corner_local_idx[i] gives the local index (column in `connectivity`) of the i-th corner. edge_mid maps a mid-edge local index to the two
    # corner local indices that bracket it
    if nodes_per_element in (ElementNodeCount.TRI3, ElementNodeCount.TRI6): # Triangles
        corner_local = [0, 1, 2]
        # Local prev/next corner for each corner (for the two incident edges)
        corner_neighbors = [(2, 1), (0, 2), (1, 0)]
        # Mid-edge node -> (corner_a_local, corner_b_local) for TRI6
        edge_mid = {3: (0, 1), 4: (1, 2), 5: (2, 0)} if nodes_per_element == ElementNodeCount.TRI6 else {}
        center_local = None
    elif nodes_per_element in (ElementNodeCount.QUAD4, ElementNodeCount.QUAD8, ElementNodeCount.QUAD9): # Quads
        corner_local = [0, 1, 2, 3]
        corner_neighbors = [(3, 1), (0, 2), (1, 3), (2, 0)]
        if nodes_per_element in (ElementNodeCount.QUAD8, ElementNodeCount.QUAD9):
            edge_mid = {4: (0, 1), 5: (1, 2), 6: (2, 3), 7: (3, 0)}
        else:
            edge_mid = {}
        center_local = 8 if nodes_per_element == ElementNodeCount.QUAD9 else None
    else:
        raise ValueError(f"Unsupported nodes_per_element: {nodes_per_element}")

    # 2. Per-corner angle-weighted normals
    # corner_normals[c] has shape (element_count, 3): the contribution that corner c of every element makes to its own node
    corner_normals = np.empty((len(corner_local), element_count, 3), dtype=np.float64)

    eps = 1e-30
    # corner_idx - Corner index in the corner list (corner_normals, corner_neightbours) etc.
    # local_corner_idx - Local (to the element) corner node index; indexes into connectivity for that corner, e.g., (0,1,2) for TRI3 
    for corner_idx, local_corner_idx in enumerate(corner_local):
        previous_idx, next_idx = corner_neighbors[corner_idx] # Get indices of the local neighbours of current corner
        # Get node coordinates
        coords_current = node_coords[connectivity[:, local_corner_idx]] # Current corner
        coords_previous = node_coords[connectivity[:, previous_idx]] # Previous corner
        coords_next = node_coords[connectivity[:, next_idx]] # Next corner

        edge_1 = coords_next - coords_current # Outgoing edge to "next"
        edge_2 = coords_previous - coords_current # Outgoing edge to "previous"

        n = np.cross(edge_1, edge_2) # Raw normal at this corner
        n_len = np.linalg.norm(n, axis=1, keepdims=True)
        n_unit = np.divide(n, n_len, out=np.zeros_like(n), where=n_len > eps)

        # Interior angle at the corner (via atan2(|a x b|, a . b)).
        length_e1 = np.linalg.norm(edge_1, axis=1)
        length_e2 = np.linalg.norm(edge_2, axis=1)
        dot = np.einsum('ij,ij->i', edge_1, edge_2)
        cross_len = n_len[:, 0]
        angle = np.arctan2(cross_len, dot) # in [0, pi]
        # Guard against degenerate edges
        good = (length_e1 > eps) & (length_e2 > eps)
        angle = np.where(good, angle, 0.0)

        corner_normals[corner_idx] = n_unit * angle[:, None]

        # Accumulate to the corner node
        np.add.at(node_normals, connectivity[:, local_corner_idx], corner_normals[corner_idx]) # Add corner normals to node normals, using indices from connectivity 

    # 3. Mid-edge nodes: average of the two bracketing corners' contributions
    # corner_a, corner_b are element-local node indices (the same system as corner_local and connectivity) of the nodes at the ends of the edge, where the given mid-node is
    for mid_local, (corner_a, corner_b) in edge_mid.items():
        # Find which entry of corner_local each refers to
        idx_a = corner_local.index(corner_a)
        idx_b = corner_local.index(corner_b)
        contrib = 0.5 * (corner_normals[idx_a] + corner_normals[idx_b]) # Get the contribution
        np.add.at(node_normals, connectivity[:, mid_local], contrib)

    # QUAD9 center node: mean of the four corner contributions
    if center_local is not None:
        contrib = 0.25 * corner_normals.sum(axis=0)
        np.add.at(node_normals, connectivity[:, center_local], contrib)

    # 4. Normalize
    magnitudes = np.linalg.norm(node_normals, axis=1, keepdims=True)
    magnitudes = np.where(magnitudes > eps, magnitudes, 1.0)
    node_normals /= magnitudes

    return node_normals

def display_pyvista_grid_with_indices(pv_grid: pv.UnstructuredGrid | pv.PolyData) -> None:
    """
    Displays a pyvista UnstructuredGrid or PolyData object with point labels showing the node indices.

    Helper function meant to be inserted to meshtype_to_rtmesh functions to help debug and visualize the internal mesh representation/winding, so it can be compared to algorithms used for intersection
    testing etc. as sometimes they differ.

    Parameters:
    -----------
    pv_grid: pv.UnstructuredGrid | pv.PolyData
        The PyVista object to convert.

    Returns:
    -----------
    None. Displays the PyVista grid with point labels showing the node indices.

    Raises:
    --------
    TypeError:
        If the input grid is not a PyVista UnstructuredGrid or PolyData object.
    """
    #if not type(pv_grid) == pv.UnstructuredGrid or not type(pv_grid) == pv.PolyData:
    #    raise TypeError("Input grid must be a PyVista UnstructuredGrid or PolyData object.")
    
    pv_grid.point_data['NodeID'] = np.arange(pv_grid.n_points)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_grid, point_size=10, color='lightblue', show_edges=True)
    #plotter.add_point_labels(pv_grid.points, pv_grid.point_data['NodeID'], scale=10, font_size=10)
    plotter.add_point_labels(pv_grid.points, [str(i) for i in range(pv_grid.n_points)],
                    font_size=14, point_size=8, shape_opacity=0.7)
    plotter.show()

def triangulate_and_map(pv_grid: pv.UnstructuredGrid | pv.PolyData) -> tuple[pv.UnstructuredGrid, np.ndarray, np.ndarray]:
    """
    Triangulates a PyVista UnstructuredGrid or PolyData object and returns the triangulated grid, face mapping, and node mapping.

    Parameters:
    -----------
    pv_grid: pv.UnstructuredGrid | pv.PolyData
        The PyVista grid to triangulate. This can be either an UnstructuredGrid or a PolyData object.

    Returns:
    --------
    tuple[pv.UnstructuredGrid, np.ndarray, np.ndarray]:
        The triangulated grid, face mapping, and node mapping.
    
    """
    # Assign a unique ID to every original cell/coordinate
    pv_grid.cell_data["original_face_ids"] = np.arange(pv_grid.n_cells)
    pv_grid.point_data["original_node_ids"] = np.arange(pv_grid.n_points)
    pv_triangulated = pv_grid.triangulate()
    # Retrieve the mapped IDs
    mapped_face_ids = pv_triangulated.cell_data["original_face_ids"]  # Contains duplicate IDs where an element was split
    mapped_coords = pv_triangulated.point_data["original_node_ids"]
    return pv_triangulated, mapped_face_ids, mapped_coords

def get_mesh_to_world_matrix(world_position: np.ndarray,
                             world_rotation: Rotation) -> np.ndarray:
    """
    Computes the transformation matrix from mesh coordinates to world coordinates.

    Parameters:
    -----------
    world_position: np.ndarray
        3D vector. The position of the mesh in world coordinates.
    world_rotation: Rotation
        The rotation of the mesh in world coordinates.

    Returns:
    --------
    np.ndarray
        The 4x4 transformation matrix.
    """
    mesh_to_world_mat = np.zeros((4,4),dtype=np.float64)
    mesh_to_world_mat[0:3, 0:3] = world_rotation.as_matrix()
    mesh_to_world_mat[-1, -1] = 1.0
    mesh_to_world_mat[0:3, -1] = world_position
    return mesh_to_world_mat

def prune_internal_nodes(node_coords: np.ndarray,
                         connectivity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes internal nodes that are not used in the connectivity array, and remaps the connectivity to match the new node indices.

    This removes some excess data that comes from the pyvista surface extraction, which gives us the full original mesh with all nodes, even those that are not used in the surface connectivity.
    This is not strictly necessary, but it reduces memory usage and speeds up the ray tracing.

    Parameters:
    -----------
    node_coords: np.ndarray
        Shape (node_count, 3). The coordinates of the nodes in the mesh.
    connectivity: np.ndarray
        Shape (element_count, nodes_per_element). The connectivity of the mesh elements.
    
    Returns:
    --------
    tuple[np.ndarray, np.ndarray]:
        The pruned node coordinates and connectivity.
    """
    used_nodes = np.unique(connectivity.ravel())
    remap = -np.ones(node_coords.shape[0], dtype=np.int64)
    remap[used_nodes] = np.arange(used_nodes.size, dtype=np.int64)
    pruned_node_coords = node_coords[used_nodes]
    pruned_connectivity = remap[connectivity]
    return pruned_node_coords, pruned_connectivity

# ================================================================================
# SIMDATA -> RTMESH
# ================================================================================

def create_rtmesh(rtmesh: RTMesh,
                  render_mesh: RenderMesh = None) -> RTMesh:
    """
    Populates and returns the passed RTMesh object with the supplied mesh data.

    The caller is responsible for creating the RTMesh and running _orient_in_world on it before passing it here; this function only fills in the geometry,
    topology, and time-series fields that depend on the resolved coordinates.

    Factored out to support the pipeline for SimData- and any other mesh type-data, while allowing simple creation of submeshes via extract_submesh.

    Parameters:
    -----------
    rtmesh: RTMesh
        Pre-created RTMesh object (from any_mesh_to_rtmesh or simdata_to_rtmesh). Its
        transform fields (translation, rotation, scale, resting_aabb) are already set by
        _orient_in_world and are left untouched here.
    render_mesh: RenderMesh
        Optional. RenderMesh object used to fetch deformed nodal coordinates. Defaults to None.

    Returns:
    --------
    RTMesh
        The populated RTMesh object (same instance that was passed in).
    """
    # Ensure topology/coords are in the expected contiguous formats before any downstream use
    rtmesh.connectivity = np.ascontiguousarray(rtmesh.connectivity, dtype=np.uint64)
    rtmesh.node_coords = np.ascontiguousarray(rtmesh.node_coords, dtype=np.double)

    # Triangulation and mapping for everything that is not a TRI3 for UV unwrapping (QUAD4 would pass in Blender, but not SeamSplitter).
    # The pyvista_surface was already oriented in-place by _orient_in_world, so the triangulated surface inherits the world-space coordinates,
    # keeping it usable for SceneVisualiser and the UV-unwrapping pipeline.
    if rtmesh.nodes_per_element != ElementNodeCount.TRI3:
        pv_triangulated, mapped_face_ids, mapped_coords = triangulate_and_map(rtmesh.pyvista_surface)
        rtmesh.pyvista_surface = pv_triangulated
        rtmesh.tri_face_mapping = np.ascontiguousarray(mapped_face_ids, dtype=np.int64)
        rtmesh.tri_node_mapping = np.ascontiguousarray(mapped_coords, dtype=np.int64)

    # RenderMesh passed = processing SimData object
    if render_mesh is not None:
        rtmesh.timestep_count = render_mesh.fields_render.shape[1]
        rtmesh.element_count = render_mesh.elem_count
        rtmesh.node_count = render_mesh.node_count
        rtmesh.get_mesh_data_over_time(render_mesh) # Pass render_mesh to extract deformation data
    else: # No RenderMesh = not SimData
        timestep_count = 1 # Temporarily they only have data for static renders
        rtmesh.timestep_count = timestep_count  
        rtmesh.element_count = rtmesh.connectivity.shape[0]
        # DEBUG NOTES: If this breaks (particularly by trying to pass an invalid value like 1)
        # -> Meshio probably detected some weird element types in your mesh; where applicable, updating MESHIO_BAD_TYPES with whatever was found should help
        rtmesh.node_count = rtmesh.node_coords.shape[0]
        rtmesh.get_mesh_data_over_time()

    return rtmesh

def create_render_mesh_higher_order(sim_data: mh.SimData,
                       field_render_keys: tuple[str,...],
                       sim_spat_dim: EDim,
                       field_disp_keys: tuple[str,...] | None = None) -> tuple[RenderMesh, pv.UnstructuredGrid]:
    """
    Creates the RenderMesh object from the passed SimData.

    Adapted create_render_mesh function that preserves the original higher-order elements instead of triangulating them.

    Parameters:
    -----------
    sim_data: mh.SimData
        The SimData object to create the RenderMesh from.
    field_render_keys: tuple[str,...]
        The keys of the fields to render.
    sim_spat_dim: sens.EDim
        The spatial dimension of the simulation.
    field_disp_keys: tuple[str,...] | None
        The keys of the fields to display.
    pos_world: np.ndarray | None
        The position of the mesh in world coordinates.
    rot_world: Rotation | None
        The rotation of the mesh in world coordinates.

    Returns:
    --------
    tuple[RenderMesh, pv.UnstructuredGrid]:
        The RenderMesh object and the PyVista grid used to create it (for temporary use in triangulation and mapping).
    """
    extract_keys = field_render_keys
    if field_disp_keys is not None:
        extract_keys = field_render_keys + field_disp_keys

    # Key change vs. original create_render mesh to preserve the higher order elements
    # pv_surf returns Unstructured Grid, so we keep that instead of triangulating via extract_surface()
    pv_grid = simdata_to_pyvista_interp(sim_data, extract_keys, sim_spat_dim)
    connectivity = pyvista_faces_to_connectivity(pv_grid)
    coords_world = np.array(pv_grid.points)
    # shape=(num_nodes,3), C format
    #print(f"Pyvale grid points shape: {pv_grid.points.shape}")
    #print(f"Pyvale grid cells: {pv_grid.celltypes}")
    #print(f"Pyvale grid faces: {pv_grid.cells.shape}")
    #print(f"Pyvale surface points shape: {pv_surf.points.shape}")
    #print(f"Pyvale surface cells: {pv_surf.faces}")
    #print(f"Pyvale surface faces: {pv_surf.faces.shape}")
    #print(connectivity)

    # Alternative if we want to keep TET10 or other volume elems for some reason instead of just using a higher-order surface: prune beforehand to remove internal values
    #coords_world, connectivity = prune_internal_nodes(coords_world, connectivity)
    # Add w coord=1, shape=(num_nodes,3+1)
    coords_world = np.hstack((coords_world, np.ones([coords_world.shape[0], 1])))

    # shape=(num_nodes,num_time_steps,num_components)
    field_render_shape = np.array(pv_grid [field_render_keys[0]]).shape
    fields_render_by_node = np.zeros(field_render_shape + (len(field_render_keys),),
                                     dtype=np.float64)
    for ii, cc in enumerate(field_render_keys):
        fields_render_by_node[:, :, ii] = np.ascontiguousarray(np.array(pv_grid [cc]))

    field_disp_by_node = None
    if field_disp_keys is not None:
        field_disp_shape = np.array(pv_grid [field_disp_keys[0]]).shape
        # shape=(num_nodes,num_time_steps,num_components)
        field_disp_by_node = np.zeros(field_disp_shape + (len(field_disp_keys),),
                                      dtype=np.float64)
        for ii, cc in enumerate(field_disp_keys):
            field_disp_by_node[:, :, ii] = np.ascontiguousarray(np.array(pv_grid[cc]))

# Return pv_grid so we can use it for TEMPORARY triangulation later on
    return RenderMesh(coords=coords_world,
                      connectivity=connectivity,
                      fields_render=fields_render_by_node,
                      fields_disp=field_disp_by_node,
                      pos_world=None, # We set these to none because RTMesh has its own workflow that positions meshes a bit more intuitively in the scene
                      rot_world=None), pv_grid

def simdata_to_rtmesh(pypath: Path,
                    field_components: tuple = ("disp_x", "disp_y", "disp_z"),
                    fields_to_render: tuple = ("disp_y", "disp_x"),
                    spatial_dim: sens.EDim = sens.EDim.TWOD,
                    world_position: np.ndarray = None,
                    world_rotation: Rotation = None,
                    target_size: float | None = None,
                    size_axis: Axis | None = None,
                    anchor: Anchor = Anchor.CENTER,
                    rotation_axis: "Axis | np.ndarray | None" = None,
                    rotation_angle_deg: float = 0.0,
                    rotation_pivot: np.ndarray | None = None) -> RTMesh | list[RTMesh]:
    """
    Converts a SimData object to an RTMesh.

    Parameters:
    -----------
    pypath: Path
        The path to the mesh to convert.
    fields_component: tuple
        Component fields.
    fields_to_render: tuple
        Fields that are supposed to be rendered.
    pypath: Path
        The path to the mesh to convert.
    spatial_dim: sens.EDim
        The spatial dimension of the mesh.
    world_position: np.cc
        Optional. The position of the mesh in world coordinates.
    world_rotation: Rotation
        Optional. The rotation of the mesh in world coordinates.
    target_size:
        Optional. Target mesh size in world units.
    size_axis: Axis | None
        Optional. The axis about which the target size is defined. Defaults to None (longest axis).
    anchor: Anchor
        Optional. Anchor on the bounding box (CENTER, BASE, TOP, MIN, MAX) used for positioning.  E.g. anchor=Anchor.BASE, world_position=[0,-1,-3] puts the bottom
    centre of the bbox at (0, -1, -3) regardless of where the file's origin is.
    rotation_axis: Axis | np.ndarray | None
        Optional. The axis about which the mesh is rotated.
    rotation_angle_deg: float
        Optional. Rotation angle about the specified axis in degrees.
    rotation_pivot: np.ndarray | None
        Pivot about which the rotation is applied.

    Returns:
    --------
    RTMesh | list[RTMesh]
        The converted RTMesh object or a list of two RTMesh objects in order [outer, inner] based on their bounding box size.

    Raises:
    -------
    ValueError:
        If the element type is not supported.
    """
     # There are two ways to pass the rotation, so they need to be reconciled
    if world_rotation is not None and rotation_axis is not None:
        raise ValueError("Pass either world_rotation or rotation_axis/rotation_angle_deg, not both.")
    if rotation_axis is not None:
        world_rotation = make_axis_rotation(rotation_axis, rotation_angle_deg, degrees=True)

    # Convert the simulation output into a SimData object
    sim_data = mh.ExodusLoader(pypath).load_all_sim_data()  # Pyvale 2026.1.0
    # Scale the coordinates and displacement fields to mm - Deprecated in this pipeline, it makes positioning counterintuitive
    #sim_data = sens.scale_length_units(scale=scale, sim_data=sim_data, disp_keys=field_components)
    #render_mesh, pv_surf = sens.create_render_mesh(sim_data, fields_to_render, sim_spat_dim=spatial_dim,
                                          #field_disp_keys=field_components)
    # Extract surface mesh only
    sim_data = extract_surf_mesh(sim_data)
    # Create RenderMesh and triangulated surface. This function preserves the higher order elements
    render_mesh, pv_grid = create_render_mesh_higher_order(sim_data, fields_to_render, sim_spat_dim=spatial_dim,
                                          field_disp_keys=field_components)

    # Set world position and rotation (where applicable)
    if world_position is None:
        world_position = np.zeros(3, dtype=np.float64)


    # World positioning - handle nodal coordinates (scaling, positioning)
    # IMPORTANT: node_coords and pyvista_surface.points are transformed independently (with the same pivot/factor) inside fit_size/rotate/translate,
    # so they MUST be backed by separate buffers. Both np.ascontiguousarray(...) on a contiguous array AND PyVista's grid.points = arr setter return
    # memory-sharing views, so we make explicit copies to fully decouple node_coords from render_mesh.coords and from the grid; otherwise the first
    # in-place write also mutates the grid, which then gets transformed a second time (e.g. scaled by factor**2).
    coords_file = np.array(render_mesh.coords[:, :COORDS_PER_NODE], dtype=np.float64) # Independent copy of the SimData nodal coordinates

    # Create RTMesh early so _orient_in_world can record transforms directly onto it
    rtmesh = RTMesh()
    try:
        rtmesh.nodes_per_element = ElementNodeCount(render_mesh.nodes_per_elem)
    except ValueError:
        print(f"Error: Invalid nodes_per_elem value: {render_mesh.nodes_per_elem}.")
    rtmesh.node_coords = coords_file # Assign the nodal coordinates as extracted from the file; they will be updated in-place
    rtmesh.pyvista_surface = pv_grid # Grid also will be updated in place within _orient_in_world
    rtmesh.spatial_dimensions = spatial_dim
    rtmesh.connectivity = np.ascontiguousarray(render_mesh.connectivity, dtype=np.uint64)
    # The interpolated pv_grid and render_mesh.coords are the same node set in the same order, but they may not be bit-identical (interpolation, dtype).
    # Sync the grid to node_coords up front so the in-place transforms applied by _orient_in_world keep pyvista_surface and node_coords in lockstep
    # (matching the proven coords-version behaviour where pv_grid.points is overwritten with the oriented coordinates). Assign a COPY so the grid does
    # not share memory with node_coords (the PyVista points setter aliases its input).
    rtmesh.pyvista_surface.points = np.array(rtmesh.node_coords, dtype=np.float64)

    # Fit the mesh size, rotate, and place in the world
    rtmesh._orient_in_world(world_position=world_position,
        world_rotation=world_rotation,
        target_size=target_size,
        size_axis=size_axis,
        anchor=anchor,
        rotation_pivot=rotation_pivot)

    # Check whether mesh has more than 1 surface and get its adjacency matrix
    chart_labels, chart_count, charts, adjacency_matrix = segment_into_charts(rtmesh.connectivity)
    if chart_count == 2:
        print("Detected mesh with more than 1 surface. It will be treated as a hollowed solid, and returned as 2 separate RTMesh objects: [outer, inner]. \nAssign surface data to the inner surface only if the object is supposed to be refractive.")
        sub_rtmesh_1 = extract_submesh(rtmesh, charts[0], render_mesh)
        sub_rtmesh_2 = extract_submesh(rtmesh, charts[1], render_mesh)
        # Compare bounding boxes; bigger => this RTMesh is the outer shell
        if sub_rtmesh_1.get_size(0).min() > sub_rtmesh_2.get_size(0).max():
            return [sub_rtmesh_1, sub_rtmesh_2] # Outer, inner
        else:
            return [sub_rtmesh_2, sub_rtmesh_1] #Outer, inner
    elif chart_count == 1:
        return create_rtmesh(rtmesh, render_mesh)
    else:
        raise IOError(f"Detected mesh with more than 2 surfaces: {chart_count}. This is currently not supported.") # More than 2 meshes - cannot guarantee what it is or why

# ================================================================================
# ANY MESH -> RTMESH
# ================================================================================

def _convert_netgen_mesh(mesh_path, converted_filepath) -> None:
    """
    Converts a Netgen .vol mesh to VTK format, which is stored in the passed converted_filepath.

    WORK IN PROGRESS:
    The conversion is done by first loading it into Netgen, exporting it in Gmsh2 format, and then converting that to VTK using Meshio. This is necessary because Gmsh doesn't support second-order
    and Meshio does not work with .vol meshes, so Netgen is used to convert to a compatible format. However, this process can lead to issues such as missing surface elements (especially for a single
    volume elements) and then to memory access violations.

    Parameters:
    -----------
    mesh_path: Path
        The path to the .vol mesh to convert.
    converted_filepath: Path
        The path to save the converted mesh.
    """
    print("Converting from netgen mesh format. Note: currently not working for meshes without surface elements.")
    # Netgen .vol meshes need special treatment to work
    import netgen.meshing as ngmeshing
    # Load .vol mesh into NGSolve/Netgen and export it in Gmsh2 format (Gmsh format doesn't support second-order elements)
    ngmesh = ngmeshing.Mesh()
    ngmesh.Load(str(mesh_path)) # Path needs to be converted to string, otherwise ngmesh complains about arg type

    # Check the mesh (conversion to gmsh returns memory access violation if there are issues, which we can't catch with try/except)
    #print("Points:", len(ngmesh.Points())) # Number of points (vertices)
    volume_elements = list(ngmesh.Elements3D()) # Iterate volumes (dimension = 3)
    surf_elements = list(ngmesh.Elements2D())  # Iterate surfaces (dim=2)
    #print("Volume elements:", len(volume_elements))
    #print("Surface elements:", len(surf_elements))

    # Check if single volume
    if len(volume_elements) == 1:
        print("Single volume element detected - likely crash cause")
        return # Temp until the fix below is expanded
    # Check if surface elements are missing (they likely are for gmsh, and it needs them to convert)
    if len(list(ngmesh.Elements2D())) == 0:
        print("Missing surface elements detected. Reconstructing boundary...")
        return # Temp until the fix below is expanded
    # Below is an initial implementation of adding surfaces to enable export - it works, but with caveats
    # TO DO to make it functional:
    # - Write it for cases other than TET10 (below is hard code)
    # - Fix index winding (current is incorrect and gives bad results, even after converting to VTK)
    # - Test for other cases than single TET10 element
    """
     # Create a boundary FaceDescriptor (Gmsh needs this to group the surface)
    fd_outside = ngmesh.Add(ngmeshing.FaceDescriptor(surfnr=1, bc=1, domin=1))

    # Dictionaries to track face occurrences and their actual PointId objects
    face_counts = {}
    face_to_points = {}

    # Extract all faces from all volume tetrahedrons
    for vol in ngmesh.Elements3D():
        v = vol.vertices

        # The first 4 vertices of a Tet are its corners.
        faces = [
            [v[0], v[1], v[2]],
            [v[0], v[1], v[3]],
            [v[1], v[2], v[3]],
            [v[0], v[2], v[3]]
        ]
        for f in faces:
            # Create a sorted tuple of the integer IDs (using .nr) for the dictionary key
            f_key = tuple(sorted([p.nr for p in f]))
            # Count how many times this geometric face appears
            face_counts[f_key] = face_counts.get(f_key, 0) + 1
            # Store the actual PointId objects so we can use them later
            if f_key not in face_to_points:
                face_to_points[f_key] = f
    # 3. Any face that appears exactly ONCE is on the outer boundary
    for f_key, count in face_counts.items():
        if count == 1:
            # Add the missing 2D surface element using the original PointId objects
            ngmesh.Add(ngmeshing.Element2D(fd_outside, face_to_points[f_key]))
    print(f"Successfully added {len(list(ngmesh.Elements2D()))} surface elements.")
    """

    ngmesh.Export(str(converted_filepath.with_suffix(".gmsh2")), "Gmsh2 Format")
    # Load .gmsh2 mesh into Meshio and export it in Exodus format
    mesh = meshio.read(converted_filepath.with_suffix(".gmsh2"), file_format='gmsh')
    mesh.write(converted_filepath.with_suffix(".vtk"), file_format='vtk')


def _convert_any_to_vtk_mesh(mesh_path, converted_filepath) -> None:
    """
    Converts any mesh to VTK format.

    This is a general function that can be used to convert any mesh format supported by Meshio to VTK format. It uses Meshio to read the input mesh and write it in VTK format,
    which is compatible with the rest of the pipeline.
    It should work in principle, but extra functions might be needed with time to handle specific cases, e.g., like .vol files.

    Parameters:
    -----------
    mesh_path: Path
        The path to the mesh to convert.
    converted_filepath: Path
        The path to save the converted mesh.

    """
    mesh = meshio.read(mesh_path)
    mesh.write(converted_filepath.with_suffix(".vtk"), file_format='vtk')


def extract_surface_faces(volume_mesh: meshio.Mesh, volume_element_type: str) -> np.ndarray:
    """
    Extracts the surface faces from a volume mesh.
    
    Supported element types: TET4, TET10, HEX8, HEX20, HEX27. The corresponding surface element types are TRI3, TRI6, QUAD4, QUAD8, QUAD9 respectively.
    The mapping is defined in VOLUME_TO_SURFACE_ELEM_MAPPING.
    
    Parameters:
    -----------
    volume_mesh: meshio.Mesh
        The volume mesh to extract the surface faces from.
    volume_element_type: str
        The element type of the volume mesh.
    
    Returns:
    --------
    np.ndarray
        Shape (num_surf_faces, nodes_per_surf_face). The connectivity of the surface faces.
    
    Raises:
    -------
    ValueError:
        If the volume element type is not supported or if the surface element type is not supported.
    """

    # NOTE: Not tested, so I have no idea if it works correctly. Index winding might be wrong.
    print("Extracting mesh surface... Note: This may yield incorrect results. Check the index winding.")
    # Get the expected surface element type and number of faces per element
    corresp_surf_elem, faces_per_vol_elem = VOLUME_TO_SURFACE_ELEM_MAPPING[volume_element_type]
    # Extract volume mesh cells
    volume_cells = volume_mesh.get_cells_type(volume_element_type)
    # Get the faces - this is element-specific
    match corresp_surf_elem:
        case "triangle":
            # TET4 -> TRI3
            # Corners: 0, 1, 2, 3
            faces = np.vstack([
                volume_cells[:, [0, 1, 3]],  # Face 0
                volume_cells[:, [1, 2, 3]],  # Face 1
                volume_cells[:, [2, 0, 3]],  # Face 2
                volume_cells[:, [0, 2, 1]],  # Face 3 (Bottom)
                # volume_cells[:, [0,1,2]], # Face 1
                # volume_cells[:, [0,1,3]], # Face 2
                # volume_cells[:, [0,2,3]], # Face 3
                # volume_cells[:, [1,2,3]], # Face 4
            ])
        case "triangle6":
            # TET10 -> TRI6
            # Corners: 0-3
            # Mid-edges: 4:0-1, 5:1-2, 6:2-0, 7:0-3, 8:1-3, 9:2-3
            faces = np.vstack([
                volume_cells[:, [0, 1, 3, 4, 8, 7]],  # Face 0
                volume_cells[:, [1, 2, 3, 5, 9, 8]],  # Face 1
                volume_cells[:, [2, 0, 3, 6, 7, 9]],  # Face 2
                volume_cells[:, [0, 2, 1, 6, 5, 4]],  # Face 3 (Bottom)
                # volume_cells[:, [0,1,2,4,7,5]],
                # volume_cells[:, [0,1,3,4,8,6]],
                # volume_cells[:, [0,2,3,5,9,6]],
                # volume_cells[:, [1,2,3,7,9,8]],
            ])
        case "quad":
            # HEX8 -> QUAD4
            # Corners: 0-3 (bottom), 4-7 (top)
            faces = np.vstack([
                volume_cells[:, [0, 3, 2, 1]],  # Face 0 (Bottom)
                volume_cells[:, [0, 1, 5, 4]],  # Face 1 (Front)
                volume_cells[:, [1, 2, 6, 5]],  # Face 2 (Right)
                volume_cells[:, [2, 3, 7, 6]],  # Face 3 (Back)
                volume_cells[:, [3, 0, 4, 7]],  # Face 4 (Left)
                volume_cells[:, [4, 5, 6, 7]],  # Face 5 (Top)
            ])
        case "quad8":
            # HEX20 -> QUAD8
            # Corners: 0-7
            # Mid-edges: 8-19
            faces = np.vstack([
                volume_cells[:, [0, 3, 2, 1, 11, 10, 9, 8]],  # Face 0 (Bottom)
                volume_cells[:, [0, 1, 5, 4, 8, 17, 12, 16]],  # Face 1 (Front)
                volume_cells[:, [1, 2, 6, 5, 9, 18, 13, 17]],  # Face 2 (Right)
                volume_cells[:, [2, 3, 7, 6, 10, 19, 14, 18]],  # Face 3 (Back)
                volume_cells[:, [3, 0, 4, 7, 11, 16, 15, 19]],  # Face 4 (Left)
                volume_cells[:, [4, 5, 6, 7, 12, 13, 14, 15]],  # Face 5 (Top)
            ])
        case "quad9":
            # HEX27 -> QUAD9 (Often used as HEX28 without a volume center)
            # Corners: 0-7
            # Mid-edges: 8-19
            # Mid-faces: 20-25
            faces = np.vstack([
                volume_cells[:, [0, 3, 2, 1, 11, 10, 9, 8, 24]],  # Face 0 (Bottom)
                volume_cells[:, [0, 1, 5, 4, 8, 17, 12, 16, 20]],  # Face 1 (Front)
                volume_cells[:, [1, 2, 6, 5, 9, 18, 13, 17, 21]],  # Face 2 (Right)
                volume_cells[:, [2, 3, 7, 6, 10, 19, 14, 18, 22]],  # Face 3 (Back)
                volume_cells[:, [3, 0, 4, 7, 11, 16, 15, 19, 23]],  # Face 4 (Left)
                volume_cells[:, [4, 5, 6, 7, 12, 13, 14, 15, 25]],  # Face 5 (Top)
            ])
        case _:
            raise ValueError(f"Unsupported volume element type: {volume_element_type}")

    # Sort each face to ensure consistent ordering; copy to use as a comparison key, but keep unsorted faces for output to keep the winding
    faces_key = np.sort(faces, axis=1)
    # Find unique faces (boundary faces)
    # A face is on the surface if it appears only once
    unique_faces, face_counts = np.unique(faces_key, axis=0, return_counts=True)
    surface_key_mask = face_counts == 1

    # Recover the original oriented faces that correspond to the unique boundary keys
    # For each boundary key, find the first matching oriented face
    surface_faces = []
    for key in unique_faces[surface_key_mask]:
        match_idx = np.where(np.all(faces_key == key, axis=1))[0][0]
        surface_faces.append(faces[match_idx])

    surface_faces = np.ascontiguousarray(surface_faces, dtype=volume_cells.dtype)
    return surface_faces

def process_mesh_elements(mmesh: meshio.Mesh) -> tuple [None, None] | tuple[meshio.Mesh, str]:
    """
    Processes the elements in the mesh to determine surface and volume elements, and extracts the surface mesh if necessary.
    
    Parameters:
    -----------
    mmesh: meshio.Mesh
        The mesh to process.
    
    Returns:
    --------
    tuple [None, None] | tuple[meshio.Mesh, str]:
        If the mesh is valid and can be processed, returns a tuple containing the surface mesh and the surface element type.
        If the mesh is invalid or cannot be processed, returns (None, None).
    """
    # Use sets since they operate on hash maps so membership checks are fast and easy
    element_types = set(mmesh.cells_dict.keys())
    surface_types =  element_types & SURFACE_ELEMENTS
    # Remaining elements in the mesh must be volume. Remove artifacts that aren't volumetric and we can't render (lines, etc.)
    volume_types_all = element_types.difference(surface_types).difference(MESHIO_BAD_TYPES)
    # Check if the remaining elements are something we currently support
    volume_types = volume_types_all & MESHIO_ACCEPTED_VOL_ELEMS
    if len(volume_types) != len(volume_types_all):
        print(f"Unsupported element types detected: {volume_types_all - volume_types}.")
        print(f"Volume elements: {volume_types_all}")
        print(f"Surface elements: {surface_types}")
        return None, None
    surface_element_count = len(surface_types)
    volume_element_count = len(volume_types)
    print(f"Surface elements: {surface_element_count} of type {surface_types}")
    print(f"Volume elements: {volume_element_count} of type {volume_types}")
    if volume_element_count == 1: # Only one volume element in the mesh
        volume_elem_type = volume_types.pop()
        corresp_surf_elem, faces_per_vol_elem = VOLUME_TO_SURFACE_ELEM_MAPPING[volume_elem_type]
        if surface_element_count == 0:
            print("No surface elements detected in the mesh. Extracting surface.")
            print(f"Volume elements: {len(mmesh.cells_dict[volume_elem_type])} of type {volume_elem_type}")
            # Find boundary/surface faces
            surface_faces = extract_surface_faces(mmesh, volume_elem_type)
            # Prune the mesh to remove unused points and remap connectivity
            #print(f"Before pruning: {mmesh.points.shape[0]} nodes, {surface_faces.shape[0]} elements")
            surf_node_coords, surf_connectivity = prune_internal_nodes(mmesh.points, surface_faces)
            surface_mesh = meshio.Mesh(points=surf_node_coords, cells=[(corresp_surf_elem, surf_connectivity)])
            node_coords = surface_mesh.points
            connectivity = surface_mesh.cells_dict[corresp_surf_elem]
            #print(f"After pruning: {node_coords.shape[0]} nodes, {connectivity.shape[0]} elements")
            return surface_mesh, corresp_surf_elem
        elif surface_element_count == 1:  # One surface element in the mesh
            surface_elem_type = surface_types.pop()
            # Check if we have a surface-to-volume mapping
            if corresp_surf_elem == surface_elem_type:
                print("Using surface data from the mesh instead of skinning.")
                print(f"Volume elements: {len(mmesh.cells_dict[volume_elem_type])} of type {volume_elem_type}")
                print(f"Surface elements: {len(mmesh.cells_dict[surface_elem_type])} of type {surface_elem_type}")
                # We have mapping, so likely we can just use this surface and ignore volume elements (this is the case with meshes from gmsh)
                node_coords = mmesh.points
                connectivity = mmesh.cells_dict[surface_elem_type]
                #print(f"Before pruning: {node_coords.shape[0]} nodes, {connectivity.shape[0]} elements")
                # Pruning
                node_coords, connectivity = prune_internal_nodes(node_coords, connectivity)
                surface_mesh = meshio.Mesh(points=node_coords, cells=[(surface_elem_type, connectivity)])
                #print(f"After pruning: {node_coords.shape[0]} nodes, {connectivity.shape[0]} elements")
                # Return the same either way
                return surface_mesh, corresp_surf_elem
            else:
                # Weird case of mixed elements - don't expect to run into this often
                # Might need updating in the future to handle more cases
                print("Mixed element type meshes are currently not supported.")
                return None, None
        else: # Probably a pyramid or wedge - currently not supported
            print("Mixed element type meshes are currently not supported.")
            return None, None
    elif volume_element_count == 0: # No volume elements in the mesh - check if it is a surface mesh
        if surface_element_count == 1: # One surface element in the mesh - perfect, we can just take that
            surface_elem_type = surface_types.pop()
            print("Surface mesh detected. No changes needed.")
            print(f"Surface elements: {len(mmesh.cells_dict[surface_elem_type])} of type {surface_elem_type}")
            node_coords = mmesh.points
            connectivity = mmesh.cells_dict[surface_elem_type]
            #print(f"Before pruning: {node_coords.shape[0]} nodes, {connectivity.shape[0]} elements")
            # Pruning - shouldn't be necessary for most cases unless we have a 3D mesh made out of linear elements
            node_coords, connectivity = prune_internal_nodes(node_coords, connectivity)
            surface_mesh = meshio.Mesh(points=node_coords, cells=[(surface_elem_type, connectivity)])
            #print(f"After pruning: {node_coords.shape[0]} nodes, {connectivity.shape[0]} elements")
            surface_mesh = meshio.Mesh(points=node_coords, cells=[(surface_elem_type, connectivity)])
            return surface_mesh, surface_elem_type
        else:
            print("Mixed element type meshes are currently not supported.")
            return None, None
    else: # More than one type of volume elements in the mesh
        print("Mixed element type meshes are currently not supported.")
        return None, None

def any_mesh_to_rtmesh(pypath: Path,
                       world_position: np.ndarray = None,
                       world_rotation: Rotation = None,
                       target_size: float | None = None,
                       size_axis: Axis | None = None,
                       anchor: Anchor = Anchor.CENTER,
                       rotation_axis: "Axis | np.ndarray | None" = None,
                       rotation_angle_deg: float = 0.0,
                       rotation_pivot: np.ndarray | None = None) -> RTMesh | list[RTMesh]:
    """Converts any mesh to an RTMesh object.

    Two equivalent ways to express rotation:
      - world_rotation=<scipy Rotation>
      - rotation_axis=<Axis or 3-vector> + rotation_angle_deg=<float> - convenience for the common "spin about one axis" case.

    Parameters:
    -----------
    pypath: Path
        The path to the mesh to convert.
    world_position: np.ndarray
        Optional. The position of the mesh in world coordinates.
    world_rotation: Rotation
        Optional. The rotation of the mesh in world coordinates.
    target_size:
        Optional. Target mesh size in world units.
    size_axis: Axis | None
        Optional. The axis about which the target size is defined. Defaults to None (longest axis).
    anchor: Anchor
        Optional. Anchor on the bounding box (CENTER, BASE, TOP, MIN, MAX) used for positioning.  E.g. anchor=Anchor.BASE, world_position=[0,-1,-3] puts the bottom
    centre of the bbox at (0, -1, -3) regardless of where the file's origin is.
    rotation_axis: Axis | np.ndarray | None
        Optional. The axis about which the mesh is rotated.
    rotation_angle_deg: float
        Optional. Rotation angle about the specified axis in degrees.
    rotation_pivot: np.ndarray | None
        Pivot about which the rotation is applied.

    Returns:
    --------
    RTMesh | list[RTMesh]
        The converted RTMesh object or a list of two RTMesh objects in order [outer, inner] based on their bounding box size.

    Raises:
    -------
    IOError
        If the mesh cannot be processed or converted.
    ValueError:
        If the element type is not supported.
    """
    # There are two ways to pass the rotation, so they need to be reconciled
    if world_rotation is not None and rotation_axis is not None:
        raise ValueError("Pass either world_rotation or rotation_axis/rotation_angle_deg, not both.")
    if rotation_axis is not None:
        world_rotation = make_axis_rotation(rotation_axis, rotation_angle_deg, degrees=True)

    # File-type checking and conversions to ensure compatible index winding
    file_type = pypath.suffix.lower()
    # Check if mesh is .vtk and convert if not to ensure winding compatibility
    if file_type != ".vtk" and file_type != ".vtu":
        print(f"Non-VTK mesh. Converting to ensure index winding compatibility...")
        converted_filepath = Path.joinpath(pypath.parent, "temp") # Path to save the converted mesh in the same directory
        # Implement as "switch" statement in case we discover that other types also need special treatment to convert
        match file_type:
            case ".vol":
                _convert_netgen_mesh(pypath, converted_filepath)
                pypath = converted_filepath.with_suffix(".vtk") # Update path to read the mesh from
            case _:
                _convert_any_to_vtk_mesh(pypath, converted_filepath)
                pypath = converted_filepath.with_suffix(".vtk") # Update path to read the mesh from
    mesh = meshio.read(pypath)
    # Check what element types are in the mesh
    element_types = len(mesh.cells_dict)
    if element_types >= 1:
        # Determine element types in the mesh and extract the surface for rendering
        surface_mesh, element_type = process_mesh_elements(mesh)
        if surface_mesh is None:
            raise IOError("Mesh processing failed. Please check the mesh format and element types.")
    else: # 0 elements
        raise IOError("No elements detected in the mesh. Please check the mesh format and element types.")
    # Rest of logic for RTMesh

    # We will need pyvista format for SeamSplitter and texturing, so convert to pyvista unstructured grid
    # Normally, we would go pv_surf = pv_ugrid.extract_surface() but this triangulates, so we will keep on using grid
    pv_ugrid = pv.from_meshio(surface_mesh) # Convert meshio to pyvista unstructured grid

    # Set world position
    if world_position is None:
        world_position = np.array((0.0, 0.0, 0.0), dtype=np.float64)

    # World positioning - handle nodal coordinates (scaling, positioning)
    # IMPORTANT: np.ascontiguousarray(pv_ugrid.points) returns a VIEW that shares memory with the grid when the points are already contiguous float64.
    # node_coords and pyvista_surface.points are transformed independently (but with the same pivot/factor) inside fit_size/rotate/translate, so they MUST be
    # separate buffers - otherwise the first in-place write also mutates the grid, and the grid then gets transformed a second time (e.g. scaled by factor**2).
    # Use an explicit copy to fully decouple them.
    coords_file = np.array(pv_ugrid.points, dtype=np.float64) # Independent copy of the input-file coordinates (which might be positioned randomly)

    # Create RTMesh early so _orient_in_world can record transforms directly onto it
    rtmesh = RTMesh()
    element_node_count = MESHIO_TO_ELEMENTNODECOUNT[element_type] # Convert meshio element type to ElementNodeCount to keep the same interface as simdata pipeline
    try:
        rtmesh.nodes_per_element = ElementNodeCount(element_node_count) 
    except ValueError:
        print(f"Error: Invalid nodes_per_elem value: {element_node_count}.")
    rtmesh.node_coords = coords_file
    rtmesh.pyvista_surface = pv_ugrid
    rtmesh.spatial_dimensions = EDim.TWOD
    rtmesh.connectivity = pyvista_faces_to_connectivity(pv_ugrid)

    # Fit the mesh size, rotate, and place in the world; transforms stored on rtmesh
    rtmesh._orient_in_world(world_position=world_position,
        world_rotation=world_rotation,
        target_size=target_size,
        size_axis=size_axis,
        anchor=anchor,
        rotation_pivot=rotation_pivot)

    # Helper to display mesh with node indices in case there are winding issues:
    #display_pyvista_grid_with_indices(rtmesh.pyvista_surface)

    # Check whether mesh has more than 1 surface and get its adjacency matrix
    chart_labels, chart_count, charts, adjacency_matrix = segment_into_charts(rtmesh.connectivity)

    # Below we always set the spatial dim in create_rtmesh to TWOD since we skin to surface mesh + this is only for compatibility with SimData anyway.
    if chart_count == 2:
        print("Detected mesh with more than 1 surface. It will be treated as a hollowed solid, and returned as 2 separate RTMesh objects: [outer, inner]. \nAssign surface data to the inner surface only if the object is supposed to be refractive.")
        sub_rtmesh_1 = extract_submesh(rtmesh, charts[0])
        sub_rtmesh_2 = extract_submesh(rtmesh, charts[1])
        # Compare bounding boxes; bigger => this RTMesh is the outer shell
        if sub_rtmesh_1.get_size(0).min() > sub_rtmesh_2.get_size(0).max():
            sub_rtmesh_1.mesh_type = MeshType.SHELL
            sub_rtmesh_2.mesh_type = MeshType.SHELL
            return [sub_rtmesh_1, sub_rtmesh_2]
        else:
            sub_rtmesh_1.mesh_type = MeshType.SHELL
            sub_rtmesh_2.mesh_type = MeshType.SHELL
            return [sub_rtmesh_2, sub_rtmesh_1]
    elif chart_count == 1:
        return create_rtmesh(rtmesh, render_mesh = None)
    else:
        raise IOError(f"Detected mesh with more than 2 surfaces: {chart_count}. This is currently not supported.") # More than 2 meshes - cannot guarantee what it is or why
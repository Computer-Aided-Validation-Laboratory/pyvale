# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================
import meshio
import pandas as pd
import numpy as np
import pyvista as pv
from pathlib import Path
from scipy.spatial.transform import Rotation
from enum import StrEnum, IntEnum
from dataclasses import dataclass, field
from enum import Enum
# import matplotlib as plt # for cmap face color determination

import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.sensorsim.simtools as simtools
from pyvale.sensorsim import RenderMesh, EDim, simdata_to_pyvista_interp, extract_surf_mesh

# ================================================================================
# CONSTANTS AND ENUMS
# ================================================================================

COORDS_PER_NODE = 3
RGB_VALS = 3

# Type of coloring that goes onto the mesh surface
class SurfType(IntEnum): # IntEnum so it can be passed to C++ nicely
    FIELD_COLOR = 0,
    TEXTURE = 1

class MaterialType(str, Enum):
    NOT_DEFINED = "NOT_DEFINED" # Nothing stopping the ray, i.e. empty space
    DIFFUSE = "DIFFUSE"
    SPECULAR = "SPECULAR"
    REFRACTIVE = "REFRACTIVE"
    UNLIT = "UNLIT" # Just surface color, ignoring all lighting calculations, i.e. no shadows, no shading, no reflections

    @property
    def as_int(self) -> int:
        mapping = {
            MaterialType.NOT_DEFINED: 0,
            MaterialType.DIFFUSE: 1,
            MaterialType.SPECULAR: 2,
            MaterialType.REFRACTIVE: 3,
            MaterialType.UNLIT: 4
        }
        return mapping[self]

# Number of nodes per element
class ElementNodeCount(IntEnum):
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
    #node_coords_over_time: np.ndarray = field(default=None)
    node_coords_expanded_over_time: np.ndarray = field(default=None)
    face_colors_over_time: np.ndarray = field(default=None)
    #uvs_over_time: np.ndarray = field(default=None) # Temporary used in development. But uvs should not change over time and they can be massive, so this was deprecated to reduce memory consumption. Keeping it just in case
    uvs: np.ndarray = field(default=None)
    seams: list = field(default_factory=list)
    texture: np.ndarray = field(default=None)
    # mesh_to_world_mat: np.ndarray = field(default=None)
    pyvista_surface: pv.UnstructuredGrid | pv.PolyData = field(default=None) # For SeamSplitter
    tri_face_mapping: np.ndarray = field(default=None) # To map triangulated faces back to original elements; needed for Blender UV unwrapping
    tri_node_mapping: np.ndarray = field(default=None) # To map triangulated vertex v to original higher order node/vertex
    surface_type: SurfType = field(default=None)
    spatial_dimensions: sens.EDim = field(default=None)
    timestep_count: int = field(default=1)
    element_count: int = field(default=0)
    node_count: int = field(default=0)
    nodes_per_element: ElementNodeCount = field(default=ElementNodeCount.TRI3)

    def set_surface(self,
                    surface_type: SurfType = SurfType.FIELD_COLOR,
                    surface_fill: np.ndarray = None,
                    material = MaterialType.NOT_DEFINED) -> None:
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
            The fill to apply to the mesh. The expected format depends on the surface type:
            - For FIELD_COLOR:
                - If shape is (3,), it is interpreted as a single RGB color applied to the entire mesh.
                - If shape is (element_count, 3), it is interpreted as an RGB color for each element, applied to the entire time series.
                - If shape is (timestep_count, element_count, 3), it is interpreted as an RGB color for each element at each timestep.
            - For TEXTURE:
                - The surface_fill should be a 2D array representing the texture image. The UV coordinates must be set for the mesh to apply the texture correctly.
        Raises:
        -------
        ValueError:
            If the surface_fill does not match the expected format for the given surface_type, or if UV coordinates are required but not provided for texture mapping.
        """
        # Reset everything if user is changing the surface type
        if self.surface_type is not None and surface_type != self.surface_type:
            self.face_colors_over_time = None
            self.texture = None
            self.uvs = None
            #self.uvs_over_time = None
        self.surface_type = surface_type
        # Solid colors
        if surface_type == SurfType.FIELD_COLOR:
            if surface_fill is None:
                print("No colour data passed. Pre-filling automatically with grey.")
            elif surface_fill.shape == (RGB_VALS,):
                # Populate with passed solid color
                self.face_colors_over_time = np.ones((self.timestep_count, self.element_count, RGB_VALS)) * surface_fill
                return
            elif surface_fill.shape == (self.element_count, RGB_VALS):
                # One avg. RGB colour value per element, given only for one timestep
                self.face_colors_over_time =  np.broadcast_to(surface_fill[np.newaxis, ...], (self.timestep_count, self.element_count, RGB_VALS))
            elif surface_fill.shape == (self.timestep_count, self.element_count, RGB_VALS):
                # One avg. RGB colour value per element, given for each timestep
                self.face_colors_over_time = surface_fill
                return
            else:
                print("Surface fill must be of shape (3,) or (element_count, 3) or (timestep_count, element_count, 3).\nPre-filling automatically with grey.")
            # Create face colors over time of appropriate size and pre-populate with grey
            self.face_colors_over_time = np.ones((self.timestep_count, self.element_count, RGB_VALS), dtype=np.float64) * 0.5
        # Texture
        elif surface_type == SurfType.TEXTURE:
            # Might move this logic elsewhere, so we could use the mesh texture in BlenderUnwrapper without passing it as an argument?
            if self.uvs is None:
                raise ValueError("UV coordinates are required to append texture.")
            if surface_fill.ndim != 2:
                raise ValueError("Wrong number of dimensions. The array containing the texture should be two-dimensional.")
            # Convert UVs to the format similar to node_coords_expanded: (element_count, nodes_per_element, 2)
            # NOTE: UVS should **NOT** change across the frames, so we do not need that. If you need to use it, uncomment relevant lines in rtscene.py and copy_data_to_blas_tex in rtbvh.cpp
            #self.uvs_over_time = np.broadcast_to(self.uvs[np.newaxis, ...], (self.timestep_count, self.element_count, self.nodes_per_element, 2))
            # TO DO: Add check for shape of texture array
            self.texture = surface_fill
        
        self.material = material

    def set_custom_uvs(self,
                       uv_coords: np.ndarray = None,
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
            self.uvs = np.ascontiguousarray(uv_coords[face_mapping], dtype=np.double)
        elif uv_coords.ndim == 3: # UVs in expanded format (element_count, nodes_per_element, 2)
            if uv_coords_shape[0] != self.element_count or uv_coords_shape[1] != self.nodes_per_element: # Check that the dimensions match expectations
                raise ValueError(f"UV coordinates must be of shape (element_count, nodes_per_element, 2). Got {uv_coords.shape}. If you triangulated your mesh independently, you need to map the uvs back to the original surface mesh.")
            self.uvs = np.ascontiguousarray(uv_coords, dtype=np.double)

    def import_seams_from_csv(self, filepath) -> None:
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

# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

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
    connectivity = np.ascontiguousarray(connectivity[:, 1:], dtype=np.uintp)
    return connectivity

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
    if not type(pv_grid) == pv.UnstructuredGrid or not type(pv_grid) == pv.PolyData:
        raise TypeError("Input grid must be a PyVista UnstructuredGrid or PolyData object.")
    
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

def orient_mesh_in_world(node_coords: np.ndarray,
                        world_position: np.ndarray,
                        world_rotation: Rotation) -> np.ndarray:
    """
    Orient the mesh in world coordinates.

    Parameters:
    -----------
    node_coords: np.ndarray
        Shape (node_count, 3).The coordinates of the nodes in the mesh.
    world_position: np.ndarray
        3D vector. The position of the mesh in world coordinates.
    world_rotation: Rotation
        The rotation of the mesh in world coordinates.

    Returns:
    --------
    np.ndarray
        The oriented node coordinates in world coordinates.
    """
    mesh_to_world_mat = get_mesh_to_world_matrix(world_position, world_rotation)
    node_count = node_coords.shape[0]
    # Stack horizontally (column-wise) so we have (node_count,4) matrix that can be multiplied by transformation matrix
    coords_stack = np.column_stack([node_coords, np.ones(node_count, dtype=np.float64)])
    node_coords_world = np.matmul(coords_stack, mesh_to_world_mat.T)
    return np.ascontiguousarray(node_coords_world[:,:COORDS_PER_NODE])

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

def create_render_mesh_higher_order(sim_data: mh.SimData,
                       field_render_keys: tuple[str,...],
                       sim_spat_dim: EDim,
                       field_disp_keys: tuple[str,...] | None = None,
                       pos_world: np.ndarray | None  = None,
                       rot_world: Rotation | None = None
                       ) -> tuple[RenderMesh, pv.UnstructuredGrid]:
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
        fields_render_by_node[:, :, ii] = np.ascontiguousarray(
            np.array(pv_grid [cc]))

    field_disp_by_node = None
    if field_disp_keys is not None:
        field_disp_shape = np.array(pv_grid [field_disp_keys[0]]).shape
        # shape=(num_nodes,num_time_steps,num_components)
        field_disp_by_node = np.zeros(field_disp_shape + (len(field_disp_keys),),
                                      dtype=np.float64)
        for ii, cc in enumerate(field_disp_keys):
            field_disp_by_node[:, :, ii] = np.ascontiguousarray(
                np.array(pv_grid[cc]))

# Return pv_grid so we can use it for TEMPORARY triangulation later on
    return RenderMesh(coords=coords_world,
                      connectivity=connectivity,
                      fields_render=fields_render_by_node,
                      fields_disp=field_disp_by_node,
                      pos_world=pos_world,
                      rot_world=rot_world), pv_grid


# Linear meshes - use existing pyvale RenderMesh class and functions
def simdata_to_rtmesh(pypath: Path,
                    field_components: tuple = ("disp_x", "disp_y", "disp_z"),
                    fields_to_render: tuple = ("disp_y", "disp_x"),
                    spatial_dim: sens.EDim = sens.EDim.THREED,
                    scale: float = 100.0,
                    world_position: np.ndarray = None,
                    world_rotation: Rotation = None) -> RTMesh:
    # Convert the simulation output into a SimData object
    sim_data = mh.ExodusLoader(pypath).load_all_sim_data()  # Pyvale 2026.1.0
    # Scale the coordinates and displacement fields to mm
    sim_data = sens.scale_length_units(scale=scale, sim_data=sim_data, disp_keys=field_components)
    #render_mesh, pv_surf = sens.create_render_mesh(sim_data, fields_to_render, sim_spat_dim=spatial_dim,
                                          #field_disp_keys=field_components)
    # Extract surface mesh only
    sim_data = extract_surf_mesh(sim_data)
    # Create RenderMesh and triangulated surface. This function preserves the higher order elements
    render_mesh, pv_grid = create_render_mesh_higher_order(sim_data, fields_to_render, sim_spat_dim=sens.EDim.TWOD,
                                          field_disp_keys=field_components)

    # Set world position and rotation (where applicable)
    if world_position is not None:
        render_mesh.set_pos(world_position)
    if world_rotation is not None:
        render_mesh.set_rot(world_rotation)

    # Handle nodal coordinates
    coords_world = np.matmul(render_mesh.coords, render_mesh.mesh_to_world_mat.T)  # Convert to world coordinates
    render_mesh.coords = coords_world  # Replace nodal coordinates in RenderMesh with their world coordinate equivalents. We can do that since for deformed nodes, we just add values
    coords = np.ascontiguousarray(render_mesh.coords[:, :COORDS_PER_NODE])

    # Create RTMesh object and assign data appropriately
    rtmesh = RTMesh()
    try:
        rtmesh.nodes_per_element = ElementNodeCount(render_mesh.nodes_per_elem)
    except ValueError:
        print(f"Error: Invalid nodes_per_elem value: {render_mesh.nodes_per_elem}.")
    rtmesh.node_coords = np.ascontiguousarray(coords, dtype=np.double)
    connectivity = np.ascontiguousarray(render_mesh.connectivity, dtype=np.uint64)
    rtmesh.connectivity = connectivity
    rtmesh.spatial_dimensions = spatial_dim
    timestep_count = render_mesh.fields_render.shape[1]
    rtmesh.timestep_count = timestep_count
    element_count = render_mesh.elem_count
    rtmesh.element_count = element_count
    node_count = render_mesh.node_count
    rtmesh.node_count = render_mesh.node_count
    rtmesh.nodes_per_element = render_mesh.nodes_per_elem
    if rtmesh.nodes_per_element == 3: # Triangular mesh doesn't need to be triangulated
        rtmesh.pyvista_surface = pv_grid
    else:
        pv_triangulated, mapped_face_ids, mapped_coords = triangulate_and_map(pv_grid)
        rtmesh.pyvista_surface = pv_triangulated
        rtmesh.tri_face_mapping = np.ascontiguousarray(mapped_face_ids, dtype=np.int64)
        rtmesh.tri_node_mapping = np.ascontiguousarray(mapped_coords, dtype=np.int64)

    # Nodal coordinates over time
    # Process data for the 0th element - always the same for deformable and static images
    coords = np.ascontiguousarray(render_mesh.coords[:, :COORDS_PER_NODE])
    coords_over_time = np.ndarray(shape=(timestep_count, node_count, COORDS_PER_NODE), dtype=np.float64)
    coords_over_time[0] = coords
    # This may stay, or may not. TBD
    node_coords_expanded_over_time = np.ndarray(shape=(timestep_count, element_count, rtmesh.nodes_per_element, COORDS_PER_NODE),
                                                dtype=np.float64)  # Store nodal coordinates over all timesteps
    node_coords_expanded_over_time[0] = coords[
        connectivity, :COORDS_PER_NODE]  # Expanded nodal coords, so we do not need the connectivity array

    # Get data over multiple timesteps
    if rtmesh.timestep_count != 1:
        for timestep in range(1, timestep_count):
            # Get deformed nodal coordinates and expand them
            node_coords = simtools.get_deformed_nodes(timestep, render_mesh)
            coords = np.ascontiguousarray(node_coords)
            #coords_over_time[timestep] = coords
            node_coords_expanded_over_time[timestep] = coords[connectivity]  # Expand nodal coords,
            #node_coords_expanded_over_time[timestep] = coords[connectivity, :COORDS_PER_NODE]  # Expand nodal coords,
    #rtmesh.node_coords_over_time = coords_over_time
    rtmesh.node_coords_expanded_over_time = node_coords_expanded_over_time
    return rtmesh

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
                       scale: float = 100.0,
                       spatial_dim: sens.EDim = sens.EDim.THREED,
                       world_position: np.ndarray = None,
                       world_rotation: Rotation = None) -> RTMesh:
    """Converts any mesh to an RTMesh object.

    Parameters:
    -----------
    pypath: Path
        The path to the mesh to convert.
    scale: float
        The scale factor to apply to the mesh.
    spatial_dim: sens.EDim
        The spatial dimension of the mesh.
    world_position: np.ndarray
        The position of the mesh in world coordinates.
    world_rotation: Rotation
        The rotation of the mesh in world coordinates.

    Returns:
    --------
    RTMesh
        The converted RTMesh object.

    Raises:
    -------
    IOError
        If the mesh cannot be processed or converted.
    """

    # Set world position and rotation (where applicable)
    if world_position is None:
        world_position = np.array((0.0, 0.0, 0.0), dtype=np.float64)
    if world_rotation is None:
        world_rotation = Rotation.from_euler("zyx", (0.0, 0.0, 0.0), degrees=True)

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
    pv_ugrid = pv.from_meshio(surface_mesh) # Convert meshio to pyvista unstructured grid
    #print(f"Original ugrid: {pv_ugrid}")
    # Normally, we would go pv_surf = pv_ugrid.extract_surface() but this triangulates, so we will keep on using grid
    #print(pv_ugrid.cells) # Equivalent of pv_surface.faces, but preserves the element type
    #pv_surf = pv_ugrid.extract_surface() # this triangulates. May also remove points etc.
    #print(f"Extracted surface: {pv_surf}")
    #nosubd = pv_ugrid.extract_surface(nonlinear_subdivision=0)
    #print(f"Extracted surface with nonlinear subdivision = 0: {nosubd}")
    #print(f"No subdivision points: {nosubd.points}")
    #print(f"Triangulate: {pv_surf.triangulate()}")
    #print(pv_surf.point_data["vtkOriginalPointIds"]) # Could be used for mapping back?
    # grid = mesh.merge_points(tolerance=1e-5)  # Might change node/point ID, so need old->new mapping
    # pv_surf = grid.extract_surface()

    # World positioning - handle nodal coordinates (scaling, positioning)
    #coords_world = np.array(pv_ugrid.points) * scale
    coords_world = np.array(pv_ugrid.points)
    coords_mesh = orient_mesh_in_world(coords_world, world_position, world_rotation)
    coords_mesh *= scale
    pv_ugrid.points = coords_mesh # Replace with oriented points - this is for triangulated surface for texturing

    # Helper to display mesh with node indices in case there are winding issues:
    #display_pyvista_grid_with_indices(pv_ugrid)

    # Connectivity
    connectivity = pyvista_faces_to_connectivity(pv_ugrid)
    #faces = np.array(pv_ugrid.cells)
    #first_elem_nodes_per_face = faces[0]
    #nodes_per_face_vec = faces[0::(first_elem_nodes_per_face + 1)]
    #nodes_per_face = first_elem_nodes_per_face
    #num_faces = int(faces.shape[0] / (nodes_per_face + 1))
    #connectivity = np.reshape(faces, (num_faces, nodes_per_face + 1))
    # shape=(num_elems,nodes_per_elem), C format
    #connectivity = np.ascontiguousarray(connectivity[:, 1:], dtype=np.uintp)

    # Create RTMesh object and assign data appropriately
    rtmesh = RTMesh()
    rtmesh.node_coords = np.ascontiguousarray(coords_mesh, dtype=np.float64)
    rtmesh.connectivity = np.ascontiguousarray(connectivity, dtype=np.uint64)
    # Triangulation for everything that is not a triangle (quad would pass in Blender, but not SeamSplitter)
    if element_type == "triangle":
        rtmesh.pyvista_surface = pv_ugrid
    else: # Everything else - triangulate and create mappings for UV-unwrapping
        pv_triangulated, mapped_face_ids, mapped_coords = triangulate_and_map(pv_ugrid)
        rtmesh.pyvista_surface = pv_triangulated
        rtmesh.tri_face_mapping = np.ascontiguousarray(mapped_face_ids, dtype=np.int64)
        rtmesh.tri_node_mapping = np.ascontiguousarray(mapped_coords, dtype=np.int64)
    rtmesh.spatial_dimensions = spatial_dim
    timestep_count = 1
    rtmesh.timestep_count = timestep_count  # Temporarily they only have data for static renders
    element_count = connectivity.shape[0]
    print(f"Connectivity shape: {connectivity.shape}")
    print(f"Node coords shape: {coords_mesh.shape}")
    rtmesh.element_count = element_count
    # DEBUG NOTES: If this breaks (particularly by trying to pass an invalid value like 1)
    # -> Meshio probably detected some weird element types in your mesh; where applicable, updating MESHIO_BAD_TYPES with whatever was found should help
    rtmesh.nodes_per_element = ElementNodeCount(connectivity.shape[1])

    # Data "over time"
    node_count = coords_mesh.shape[0]
    rtmesh.node_count = node_count
    coords_over_time = np.ndarray(shape=(timestep_count, node_count, COORDS_PER_NODE), dtype=np.float64)
    coords_over_time[0, :, :] = coords_mesh
    #rtmesh.node_coords_over_time = coords_over_time
    # rtmesh.face_colors_over_time = np.ones((rtmesh.timestep_count, rtmesh.element_count, COORDS_PER_NODE)) * [1.0, 0.078, 0.57]

    # Node coords expanded. TBD if they stay
    node_coords_expanded_over_time = np.ndarray(
        shape=(timestep_count, element_count, rtmesh.nodes_per_element, COORDS_PER_NODE),
        dtype=np.float64)  # Store nodal coordinates over all timesteps
    # face_colors_over_time = np.ndarray(shape=(timestep_count, element_count, RGB_VALS), dtype=np.float64)  # Store face colors over all timesteps
    # face_colors_over_time[:, :] = [1.0, 0.078, 0.57]
    node_coords_expanded_over_time[0, :, :, :] = coords_mesh[connectivity]
    rtmesh.node_coords_expanded_over_time = node_coords_expanded_over_time
    return rtmesh


# ================================================================================
# TESTS
# ================================================================================

def volume_meshes_conversion_test():
    """
    Test function to test reading and converting various volume meshes to RTMesh objects.

    """
    import time
    # Test Wiera's TET10 sphere (.vtk format)
    name = "sphere_1"
    mesh_path = Path.joinpath(Path.cwd(), name + ".vtk")
    test_rtmesh = any_mesh_to_rtmesh(mesh_path, scale=500, world_position=np.array([1.0, -23, -1.0]))
    print("\nWiera's TET10 sphere in .vtk format:")
    print(f"Node count: {test_rtmesh.node_count}")
    print(f"Element count: {test_rtmesh.element_count}")
    print(f"Node coords shape: {test_rtmesh.node_coords.shape}")
    print(f"Connectivity shape: {test_rtmesh.connectivity.shape}")
    print(f"Nodes per element: {test_rtmesh.nodes_per_element}")

    time.sleep(2)
    # Test Wiera's TET10 sphere (.vol format)
    name = "sphere_1"
    mesh_path = Path.joinpath(Path.cwd(), name + ".vol")
    test_rtmesh = any_mesh_to_rtmesh(mesh_path, scale=500, world_position=np.array([1.0, -23, -1.0]))
    print("\nWWiera's TET10 sphere in .vol format:")
    print(f"Node count: {test_rtmesh.node_count}")
    print(f"Element count: {test_rtmesh.element_count}")
    print(f"Node coords shape: {test_rtmesh.node_coords.shape}")
    print(f"Connectivity shape: {test_rtmesh.connectivity.shape}")
    print(f"Nodes per element: {test_rtmesh.nodes_per_element}")

    time.sleep(2)
    # Test quadratic 3d sphere from gmsh
    name = "curved_sphere_3d_verycoarse"
    mesh_path = Path.joinpath(Path.cwd(), name + ".vtk")
    test_rtmesh = any_mesh_to_rtmesh(mesh_path, scale=500, world_position=np.array([1.0, -23, -1.0]))
    print("\nWcurved_sphere_3d_verycoarse:")
    print(f"Node count: {test_rtmesh.node_count}")
    print(f"Element count: {test_rtmesh.element_count}")
    print(f"Node coords shape: {test_rtmesh.node_coords.shape}")
    print(f"Connectivity shape: {test_rtmesh.connectivity.shape}")
    print(f"Nodes per element: {test_rtmesh.nodes_per_element}")

    time.sleep(2)
    # Test linear 3d sphere from gmsh
    name = "sphere_gmsh_vtk_test_3d_rough"
    mesh_path = Path.joinpath(Path.cwd(), name + ".vtk")
    test_rtmesh = any_mesh_to_rtmesh(mesh_path, scale=500, world_position=np.array([1.0, -23, -1.0]))
    print("\n3D sphere made up of linear elements: sphere_gmsh_vtk_test_3d_rough")
    print(f"Node count: {test_rtmesh.node_count}")
    print(f"Element count: {test_rtmesh.element_count}")
    print(f"Node coords shape: {test_rtmesh.node_coords.shape}")
    print(f"Connectivity shape: {test_rtmesh.connectivity.shape}")
    print(f"Nodes per element: {test_rtmesh.nodes_per_element}")
    time.sleep(2)
    # Test linear 2d sphere from gmsh (triangles)
    name = "sphere_gmsh_vtk_test_2d_simpler"
    mesh_path = Path.joinpath(Path.cwd(), name + ".vtk")
    test_rtmesh = any_mesh_to_rtmesh(mesh_path, scale=500, world_position=np.array([1.0, -23, -1.0]))
    print("\nW2D (surface) sphere made up of linear elements: sphere_gmsh_vtk_test_2d_simpler")
    print(f"Node count: {test_rtmesh.node_count}")
    print(f"Element count: {test_rtmesh.element_count}")
    print(f"Node coords shape: {test_rtmesh.node_coords.shape}")
    print(f"Connectivity shape: {test_rtmesh.connectivity.shape}")
    print(f"Nodes per element: {test_rtmesh.nodes_per_element}")

#volume_meshes_conversion_test()



"""
def test_vtk_mesh_loader():
    # Seems to pass for both 2d and 3d vtk meshes so keep it for now
    data_path = Path(Path().resolve().joinpath("cyl_gmsh_vtk_test_2d.vtk"))
    rtmesh = vtk_mesh_to_rtmesh(data_path, scale=500, world_position=np.array([1.0, -23, -1.0]))
    assert rtmesh.nodes_per_element == ElementNodeCount(3)
    assert rtmesh.timestep_count == 1
    assert rtmesh.node_coords.shape == (rtmesh.node_count, COORDS_PER_NODE)
    assert rtmesh.connectivity.shape == (rtmesh.element_count, rtmesh.nodes_per_element)
    assert rtmesh.node_coords_over_time.shape == (rtmesh.timestep_count, rtmesh.node_count, COORDS_PER_NODE)
    assert rtmesh.node_coords_expanded_over_time.shape == (rtmesh.timestep_count, rtmesh.element_count, rtmesh.nodes_per_element, COORDS_PER_NODE)

test_vtk_mesh_loader()

def compare_indexing():
    # Check if we need mesh.GetElementCoords() or if we can use the same indexing as for everything else - Yes, we can
    # Quadratic sphere
    data_path_sph = Path(Path().resolve().joinpath("sphere_1.vol"))
    rtmesh_sph = vol_mesh_to_rtmesh(data_path_sph, scale=500, world_position=np.array([1.0, -23, -1.0]))

    # Single curved tet
    #data_path_cur = Path(Path().resolve().joinpath("one_tet_1.vol"))
    #rtmesh_cur_tet = vol_mesh_to_rtmesh(data_path_cur, scale=500, world_position=np.array([1.0, -23, -1.0]))

compare_indexing()


def test_rtmesh_conversion():
    # Test if RTMesh is created correctly for both "regular" (simdata) meshes and volume meshes
    # Simdata example
    import pyvale.dataset as dataset
    data_path = dataset.render_mechanical_3d_path()  # Test mesh 2
    rtmesh_lin = simdata_to_rtmesh(data_path, scale=500, world_position=np.array([1.0, -23, -1.0]))
    # Assertions for known data
    print(rtmesh_lin.nodes_per_element)
    assert rtmesh_lin.timestep_count == 11
    assert rtmesh_lin.node_coords_over_time.shape == (11, rtmesh_lin.node_count, COORDS_PER_NODE)
    assert rtmesh_lin.element_count == rtmesh_lin.connectivity.shape[0]
    assert rtmesh_lin.connectivity.shape == (rtmesh_lin.element_count, rtmesh_lin.nodes_per_element)
    assert rtmesh_lin.node_coords_expanded_over_time.shape == (rtmesh_lin.timestep_count, rtmesh_lin.element_count, rtmesh_lin.nodes_per_element, COORDS_PER_NODE)

    # Single curved tet
    data_path_cur = Path(Path().resolve().joinpath(
        "one_tet_1.vol"))
    rtmesh_cur_tet = vol_mesh_to_rtmesh(data_path_cur, scale=500, world_position=np.array([1.0, -23, -1.0]))
    assert rtmesh_cur_tet.timestep_count == 1
    assert rtmesh_cur_tet.node_coords_over_time.shape == (1, rtmesh_cur_tet.node_count, COORDS_PER_NODE)
    assert rtmesh_cur_tet.element_count == 1 # Single tet, so we expect one element
    assert rtmesh_cur_tet.connectivity.shape == (rtmesh_cur_tet.element_count, rtmesh_cur_tet.nodes_per_element)
    assert rtmesh_cur_tet.node_coords_expanded_over_time.shape == (rtmesh_cur_tet.timestep_count, rtmesh_cur_tet.element_count, rtmesh_cur_tet.nodes_per_element, COORDS_PER_NODE)

    # Quadratic sphere
    data_path_sph = Path(Path().resolve().joinpath("sphere_1.vol"))
    rtmesh_sph = vol_mesh_to_rtmesh(data_path_sph, scale=500, world_position=np.array([1.0, -23, -1.0]))
    assert rtmesh_sph.timestep_count == 1
    assert rtmesh_sph.node_coords_over_time.shape == (1, rtmesh_sph.node_count, COORDS_PER_NODE)
    assert rtmesh_sph.connectivity.shape == (rtmesh_sph.element_count, rtmesh_sph.nodes_per_element)
    assert rtmesh_sph.node_coords_expanded_over_time.shape == (rtmesh_sph.timestep_count, rtmesh_sph.element_count, rtmesh_sph.nodes_per_element, COORDS_PER_NODE)

test_rtmesh_conversion() # All passed - sweet
"""


####################################################### OLD FUNCTIONS ##########################################################

"""
def get_pyvista_surface(rtmesh: RTMesh):
    Helper function to get a PyVista surface mesh for anything that isn't a mesh in the VTK format or a SimData object
    as these have their own dedicated functions embedded within the converters.
    # VTK format: [node_count, id0, id1, ..., node_count, id0, id1, ...]
    connectivity = np.empty((rtmesh.element_count, rtmesh.nodes_per_element + 1), dtype=np.int32)
    connectivity[:, 0] = rtmesh.nodes_per_element
    connectivity[:, 1:] = np.arange(rtmesh.element_count * rtmesh.nodes_per_element).reshape(rtmesh.element_count, rtmesh.nodes_per_element)

    pv_cell_type = sens.fieldconverter._get_pyvista_cell_type(rtmesh.nodes_per_element, rtmesh.spatial_dimensions) # Accessing protected member might be bad practice but it's perfect here, so
    print("Sphere cell type: ", pv_cell_type, "")
    cell_type = np.full(rtmesh.element_count, [pv_cell_type])  # Update cell type (see pyvale field converter)
    raw_vertices = rtmesh.node_coords_expanded_over_time[0].reshape(-1, COORDS_PER_NODE)  # view, not a copy. Need to reshape it to work with pyvista. Shape is # (number_of_elements * 3, 3)
    grid = pv.UnstructuredGrid(connectivity.ravel(), cell_type, raw_vertices)
    # merge_points finds nodes within 'tolerance' and fuses them
    #grid = grid.merge_points(tolerance=1e-5)  # Might change node/point ID, so need old->new mapping. Test if need to uncomment this or not
    # Now, extract the boundary surface
    surf = grid.extract_surface()
    return surf
"""

def get_pyvista_surface(rtmesh: RTMesh):
    """Helper function to get a PyVista surface mesh."""
    # VTK format: [node_count, id0, id1, ..., node_count, id0, id1, ...]
    connectivity = np.empty((rtmesh.element_count, rtmesh.nodes_per_element + 1), dtype=np.int32)
    connectivity[:, 0] = rtmesh.nodes_per_element

    # FIX: Use the actual shared connectivity to maintain topology
    connectivity[:, 1:] = rtmesh.connectivity

    pv_cell_type = sens.fieldconverter._get_pyvista_cell_type(rtmesh.nodes_per_element, rtmesh.spatial_dimensions)
    cell_type = np.full(rtmesh.element_count, [pv_cell_type])

    # FIX: Use the original unique node coordinates, NOT the expanded view
    #raw_vertices = rtmesh.node_coords_over_time[0]
    raw_vertices = rtmesh.node_coords
    #grid = pv.UnstructuredGrid(connectivity.ravel(), cell_type, raw_vertices)
    cells = np.hstack([np.full((rtmesh.connectivity.shape[0],1), 10, np.int32), rtmesh.connectivity]).ravel()
    cell_types = np.full(rtmesh.connectivity.shape[0], [pv_cell_type], np.int32)
    grid = pv.UnstructuredGrid(cells, cell_types, rtmesh.node_coords)

    # Now that nodes are natively shared, this will correctly extract only the true boundary
    surf = grid.extract_surface()
    return surf

def vol_mesh_to_rtmesh(pypath: Path,
                    scale: float = 100.0,
                    spatial_dim: sens.EDim = sens.EDim.THREED,
                    world_position: np.ndarray = None,
                    world_rotation: Rotation = None):
    '''Converts a .vol mesh to a RenderMesh object and returns it.'''
    mesh = Mesh()
    mesh.loadVolFile(pypath)
    mesh.getElementCoords()

    # Set world position and rotation (where applicable)
    if world_position is None:
        world_position = np.array((0.0, 0.0, 0.0), dtype=np.float64)
    if world_rotation is None:
        world_rotation = Rotation.from_euler("zyx", (0.0, 0.0, 0.0), degrees=True)

    # Handle nodal coordinates (scaling, positioning)
    coords_mesh = mesh.points * scale
    coords_mesh = orient_mesh_in_world(coords_mesh, world_position, world_rotation)

    # Create RTMesh object and assign data appropriately
    rtmesh = RTMesh()
    rtmesh.node_coords = coords_mesh
    rtmesh.connectivity = mesh.elements
    #rtmesh.connectivity = rtmesh.connectivity[:, [0,1,2,3,4,7,5,6,9,8]]  # Netgen -> VTK quad tet order
    rtmesh.spatial_dimensions = spatial_dim
    timestep_count = 1
    rtmesh.timestep_count = timestep_count  # Temporarily they only have data for static renders
    element_count = mesh.elements.shape[0]
    rtmesh.element_count = element_count
    rtmesh.nodes_per_element = ElementNodeCount(mesh.elements.shape[1]) # Is 10

    # Data "over time"
    node_count = mesh.points.shape[0]
    rtmesh.node_count = node_count
    coords_over_time = np.ndarray(shape=(timestep_count, node_count, COORDS_PER_NODE), dtype=np.float64)
    coords_over_time[0,:,:] = coords_mesh
    #rtmesh.node_coords_over_time = coords_over_time
    #rtmesh.face_colors_over_time = np.ones((rtmesh.timestep_count, rtmesh.element_count, COORDS_PER_NODE)) * [1.0, 0.078, 0.57]

    # Node coords expanded. TBD if they stay
    mesh.getElementCoords() # Not need it anymore
    node_coords_expanded_over_time = np.ndarray(shape=(timestep_count, element_count, rtmesh.nodes_per_element, COORDS_PER_NODE),
                                                dtype=np.float64)  # Store nodal coordinates over all timesteps
    #face_colors_over_time = np.ndarray(shape=(timestep_count, element_count, RGB_VALS), dtype=np.float64)  # Store face colors over all timesteps
    #face_colors_over_time[:, :] = [1.0, 0.078, 0.57]
    node_coords_expanded_over_time[0, :, :, :] = coords_mesh[rtmesh.connectivity]
    rtmesh.node_coords_expanded_over_time = node_coords_expanded_over_time
    rtmesh.pyvista_surface = get_pyvista_surface(rtmesh)

    # For indexing tests
    #print(f"Elements as done by Wiera: {mesh.elem_coords}")
    #print(f"Elements as done by me: {mesh.points[rtmesh.connectivity]}")
    #print(f"Are the same?: {(mesh.elem_coords == mesh.points[rtmesh.connectivity]).all()}")

    # Get data over multiple timesteps - tbd when we actually render something non-static for these elements
    #if rtmesh.timestep_count != 1:
    #    for timestep in range(1, timestep_count):
    #        # Get deformed nodal coordinates and expand them
    #        node_coords = simtools.get_deformed_nodes(timestep, render_mesh)
    #        coords = np.ascontiguousarray(node_coords)
    #        coords_over_time[timestep] = coords
    #rtmesh.node_coords_over_time = coords_over_time
    return rtmesh

# Test to cheat a little bit with loading non-simdata meshes for now
def vtk_mesh_to_rtmesh(pypath: Path,
                    scale: float = 100.0,
                    spatial_dim: sens.EDim = sens.EDim.THREED,
                    world_position: np.ndarray = None,
                    world_rotation: Rotation = None):

    # Set world position and rotation (where applicable)
    if world_position is None:
        world_position = np.array((0.0, 0.0, 0.0), dtype=np.float64)
    if world_rotation is None:
        world_rotation = Rotation.from_euler("zyx", (0.0, 0.0, 0.0), degrees=True)

    mesh = pv.read(pypath)
    print(mesh)
    #print(f"Cell type:{mesh.celltypes}")
    surf = mesh.extract_surface()
    #grid = mesh.merge_points(tolerance=1e-5)  # Might change node/point ID, so need old->new mapping
    #surf = grid.extract_surface()

    # World positioning - handle nodal coordinates (scaling, positioning)
    coords_world = np.array(surf.points) * scale
    coords_mesh = orient_mesh_in_world(coords_world, world_position, world_rotation)

    # Connectivity
    connectivity = pyvista_faces_to_connectivity(surf)

    #faces = np.array(surf.faces)
    #first_elem_nodes_per_face = faces[0]
    #nodes_per_face_vec = faces[0::(first_elem_nodes_per_face + 1)]
    #nodes_per_face = first_elem_nodes_per_face
    #num_faces = int(faces.shape[0] / (nodes_per_face + 1))
    #connectivity = np.reshape(faces, (num_faces, nodes_per_face + 1))
    ## shape=(num_elems,nodes_per_elem), C format
    #connectivity = np.ascontiguousarray(connectivity[:, 1:], dtype=np.uintp)

    # Create RTMesh object and assign data appropriately
    rtmesh = RTMesh()
    rtmesh.node_coords = coords_mesh
    rtmesh.connectivity = connectivity
    rtmesh.pyvista_surface = surf
    rtmesh.spatial_dimensions = spatial_dim
    timestep_count = 1
    rtmesh.timestep_count = timestep_count  # Temporarily they only have data for static renders
    element_count = connectivity.shape[0]
    rtmesh.element_count = element_count
    rtmesh.nodes_per_element = ElementNodeCount(connectivity.shape[1])

    # Data "over time"
    node_count = coords_mesh.shape[0]
    rtmesh.node_count = node_count
    coords_over_time = np.ndarray(shape=(timestep_count, node_count, COORDS_PER_NODE), dtype=np.float64)
    coords_over_time[0, :, :] = coords_mesh
    #rtmesh.node_coords_over_time = coords_over_time
    # rtmesh.face_colors_over_time = np.ones((rtmesh.timestep_count, rtmesh.element_count, COORDS_PER_NODE)) * [1.0, 0.078, 0.57]

    # Node coords expanded. TBD if they stay
    node_coords_expanded_over_time = np.ndarray(
        shape=(timestep_count, element_count, rtmesh.nodes_per_element, COORDS_PER_NODE),
        dtype=np.float64)  # Store nodal coordinates over all timesteps
    # face_colors_over_time = np.ndarray(shape=(timestep_count, element_count, RGB_VALS), dtype=np.float64)  # Store face colors over all timesteps
    # face_colors_over_time[:, :] = [1.0, 0.078, 0.57]
    node_coords_expanded_over_time[0, :, :, :] = coords_mesh[connectivity]
    rtmesh.node_coords_expanded_over_time = node_coords_expanded_over_time
    return rtmesh

def simdata_to_mesh(pypath: Path, field_components, fields_to_render, scale):
    # Convert the simulation output into a SimData object
    # sim_data = mh.ExodusReader(pypath).read_all_sim_data() # Pyvale 2025.8.1
    sim_data = mh.ExodusLoader(pypath).load_all_sim_data()  # Pyvale 2026.1.0
    # Scale the coordinates and displ. fields to mm
    # sim_data = sens.scale_length_units(scale=scale,sim_data=sim_data,disp_comps=field_components) # Pyvale 2025.8.1
    sim_data = sens.scale_length_units(scale=scale, sim_data=sim_data, disp_keys=field_components)  # Pyvale 2026.1.0
    # render_mesh = sens.create_render_mesh(sim_data, fields_to_render ,sim_spat_dim=3,field_disp_keys=field_components) # Pyvale 2025.8.1. Still works, but now we use enum for spatial dim, not a number
    render_mesh = sens.create_render_mesh(sim_data, fields_to_render, sim_spat_dim=sens.EDim.THREED,
                                          field_disp_keys=field_components)  # Pyvale 2026.1.0
    return render_mesh


def compute_face_colors_averages(field_nodal_values: np.ndarray, connectivity: np.ndarray):
    '''Calculates face colors based on the nodal values for the chosen field. Approach 2 - taking averages and stacking them together'''
    field_node_norm = (field_nodal_values - field_nodal_values.min()) / (
                field_nodal_values.max() - field_nodal_values.min())  # Normalize displacement values, scaling them to range [0,1] so they can map to color intensities
    node_colors = np.column_stack(
        (field_node_norm, field_node_norm, field_node_norm))  # Convert each scalar to an RGB triplet
    face_colors = np.mean(node_colors[connectivity], axis=1)
    # print(f"face_colors_shape: {face_colors.shape}")
    return face_colors  # Compute each face's colour as the average of its 3 node colours

    # def compute_face_colors_cmap(field_nodal_values: np.ndarray):
    '''Approach 1 - using a colour map to assign an rgb value'''
#   field_node_norm = (field_nodal_values - field_nodal_values.min())/(field_nodal_values.max()-field_nodal_values.min()) # Normalize displacement values, scaling them to range [0,1] so they can map to color intensities
#    cmap = plt.get_cmap('viridis')
#   return cmap(field_node_norm)[:,:3]
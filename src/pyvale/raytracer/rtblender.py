# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from enum import StrEnum
import numpy as np
import bpy
import vedo

from pyvale.raytracer.rtmesh import RTMesh, pyvista_faces_to_connectivity

# Enum for different UV unwrapping algorithms available in Blender
class UnwrapAlgo(StrEnum):
    ABF = "ANGLE_BASED" # Angle-based flattening
    LSCM = "CONFORMAL" # Least squares conformal mapping
    SLIM = "MINIMUM_STRETCH" # Scalable locally injective mappings

class BlenderUnwrapper:
    __slots__ = ("rtmesh", "vertices", "faces", "seams", "blender_mesh", "blender_obj", "uvs", "vertex_map", "faces_cut")
    def __init__(self):
        self.rtmesh: RTMesh | None = None
        self.vertices: np.ndarray | None = None
        self.faces: np.ndarray | None = None
        self.seams: list | None = None
        self.blender_mesh: bpy.types.Mesh | None = None
        self.blender_obj: bpy.types.Object | None = None
        self.uvs: np.ndarray | None = None
        self.vertex_map: np.ndarray | None = None
        self.faces_cut: np.ndarray | None = None

    def add_rtmesh(self, rtmesh: RTMesh) -> None:
        """
        Loads the RTMesh data into Blender for UV unwrapping.

        Checks if a mesh is already loaded, and if so, removes it before loading the new one.
        For higher-order meshes, it uses the triangulated surface's connectivity and node coordinates to ensure UV mapping works correctly in Blender.
        For already triangulated meshes, it uses the original data directly.

        Parameters
        ----------
        rtmesh : RTMesh
            The RTMesh to be loaded into Blender.
        """
        # Check if we already have a mesh loaded, and if so, remove it
        if self.blender_mesh is not None or self.blender_obj is not None:
            self._blender_remove_mesh()
            self.rtmesh = None
            self.vertices = None
            self.faces = None
            self.seams = None
            self.uvs = None
            self.vertex_map = None
            self.faces_cut = None
        self.rtmesh = rtmesh
        # If we have mapping, we need to use the connectivity and node coords from triangulated surface
        if rtmesh.tri_face_mapping is not None:
            self.faces = pyvista_faces_to_connectivity(rtmesh.pyvista_surface)
            self.vertices = rtmesh.pyvista_surface.points
        else: # If mesh is TRI3 to begin with, we can just use its starting data
            self.vertices = rtmesh.node_coords
            self.faces = rtmesh.connectivity
        self.seams = rtmesh.seams
        self._blender_load_mesh()
        
    def _blender_load_mesh(self) -> None:
        """
        Creates a new mesh and object in Blender, populates it with the vertices and faces from the RTMesh, and links it to the scene.
        """
        self.blender_mesh = bpy.data.meshes.new("BMesh")
        self.blender_mesh.from_pydata(self.vertices, [], self.faces)  # Empty list for edges, which will be inferred from faces
        self.blender_mesh.update()
        # Create object and link to the scene
        self.blender_obj = bpy.data.objects.new("ObjMesh", self.blender_mesh)
        bpy.context.collection.objects.link(self.blender_obj)
        # Set active object
        bpy.context.view_layer.objects.active = self.blender_obj
        
    def _blender_remove_mesh(self) -> None:
        """
        Removes the mesh and object from Blender. Important if you want to compare multiple UV unwrapping methods in one go to "reset" the data.
        """
        if self.blender_mesh is not None or self.blender_obj is not None:
            print("Nothing to remove.")
            return
        bpy.data.meshes.remove(self.blender_mesh)
        bpy.data.objects.remove(self.blender_obj, do_unlink=True)
        self.blender_mesh = None
        self.blender_obj = None

    def _select_seams(self) -> None:
        """
        Converts the seam information from the RTMesh into Blender's edge selection for UV unwrapping.

        Blender identifies seams on edges, so our seam paths need to be converted (which are lists of vertex indices) into edge selections.
        Blender uses edge keys (v1, v2) where v1 < v2 to identify edges, so our edge keys must be sorted and match the mesh's edge keys.
        After the conversion, selects the corresponding edges in Blender to mark them as seams for the unwrapping process.
        """
        mesh_data = self.blender_obj.data
        # Build a lookup dictionary: (v1, v2) -> edge_index
        edge_key_to_index = {k: i for i, k in enumerate(mesh_data.edge_keys)}
        seam_nodes = set()
        seam_edges = set()
        for seam_path in self.seams:
            seam_nodes.update(seam_path)
            # Build edges from consecutive vertices in seam
            for i in range(len(seam_path) - 1):
                if seam_path[i] < seam_path[i + 1]:
                    v1 = seam_path[i]
                    v2 = seam_path[i + 1]
                else:
                    v1 = seam_path[i + 1]
                    v2 = seam_path[i]
                seam_edges.add((v1, v2))

        for edge_key in seam_edges:
            if edge_key in edge_key_to_index:
                edge_idx = edge_key_to_index[edge_key]
                mesh_data.edges[edge_idx].use_seam = True
            else:
                print(f"Warning: No edge found between vertices {edge_key[0]} and {edge_key[1]}")

    def _get_xatlas_uv_format(self, mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (uvs, vmapping, indices) matching xatlas.parametrize() format.

        Parameters:
        ----------
            mesh: Blender mesh object to extract UVs from
        Returns:
        -------
            uvs: np.ndarray
                Shape (N, 2) - UV coordinates for new vertices
            vmapping: np.ndarray
                Shape (N,) - original vertex index for each new UV vertex
            indices: np.ndarray
                Shape (F, 3) - new triangle indices
        Raises:
        -------
            ValueError:
                If the mesh is not triangulated (i.e., if any face has more than 3 vertices)
        Notes:
        -------
            - F - number of faces (triangles) in the mesh
            - N - number of UV vertices (per-loop), which can be greater than the number of original vertices due to seam splitting
            - Blender's UV data is stored per-loop, meaning each corner of a face can have its own UV coordinates. This allows for seam splitting where a single vertex can have multiple UVs
        """
        nodes_per_element = 3 # Operate only triangulated meshes, so we know we have 3 nodes per face
        uv_layer = mesh.uv_layers.active.data

        # 1. Collect all UV vertices (per-loop)
        uvs_list = []
        original_vertex_indices = []

        # Iterate over polygons (faces) and their loops to extract UVs and corresponding original vertex indices
        for poly in mesh.polygons:
            if len(poly.vertices) != 3:
                raise ValueError("Mesh must be triangulated")

            # Each polygon has loop indices that point to the UV data for each corner of the face. We need to gather these UVs and the corresponding original vertex indices
            for loop_index in poly.loop_indices:
                uv = uv_layer[loop_index].uv
                vertex_idx = mesh.loops[loop_index].vertex_index

                uvs_list.append([uv.x, uv.y])
                original_vertex_indices.append(vertex_idx)

        uvs = np.array(uvs_list, dtype=np.double)  # (num_loops, 2)
        vmapping = np.array(original_vertex_indices, dtype=np.uint64)  # (num_loops,)

        # 2. Build new face indices (sequential: 0,1,2, 3,4,5, etc.)
        face_count = len(mesh.polygons)
        indices = np.arange(face_count * nodes_per_element, dtype=np.uint64).reshape(face_count, nodes_per_element)

        return uvs, vmapping, indices

    def _get_xatlas_uv_selected_format(self, mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (uvs, vmapping, indices) for SELECTED FACES ONLY, i.e, not the entire mesh.

        Parameters:
        ----------
            mesh: Blender mesh object to extract UVs from
        Returns:
        -------
            uvs: np.ndarray
                Shape (N, 2) - UV coordinates for new vertices
            vmapping: np.ndarray
                Shape (N,) - original vertex index for each new UV vertex
            indices: np.ndarray
                Shape (F, 3) - new triangle indices
        Raises:
        -------
            ValueError: If the mesh is not triangulated (i.e., if any face has more than 3 vertices)
        Notes:
        -------
            - F - number of SELECTED faces (triangles) in the mesh
            - N - number of UV vertices (per-loop), which can be greater than the number of original vertices due to seam splitting
            - Blender's UV data is stored per-loop, meaning each corner of a face can have its own UV coordinates. This allows for seam splitting where a single vertex can have multiple UVs
        """
        nodes_per_element = 3 # Operate only triangulated meshes, so we know we have 3 nodes per face
        uv_layer = mesh.uv_layers.active.data

        uvs_list = []
        original_vertex_indices = []
        selected_faces = []  # Track selected face indices

        for poly_idx, poly in enumerate(mesh.polygons):
            # Skip unselected faces
            if not poly.select:
                continue

            if len(poly.vertices) != 3:
                raise ValueError("Selected faces must be triangulated")

            selected_faces.append(poly_idx)

            for loop_index in poly.loop_indices:
                uv = uv_layer[loop_index].uv
                vertex_idx = mesh.loops[loop_index].vertex_index

                uvs_list.append([uv.x, uv.y])
                original_vertex_indices.append(vertex_idx)

        if not selected_faces:
            raise ValueError("No faces selected!")

        uvs = np.array(uvs_list, dtype=np.double)  # (num_loops_selected, 2)
        vmapping = np.array(original_vertex_indices, dtype=np.uint64)  # (num_loops_selected,)

        # Sequential indices for selected faces only
        selected_faces_count = len(selected_faces)
        indices = np.arange(selected_faces_count * nodes_per_element, dtype=np.uint64).reshape(selected_faces_count, nodes_per_element)

        return uvs, vmapping, indices

    def _make_uv_vertex(self,
                        uv: np.ndarray,
                        orig_node: int,
                        uv_vertex_count: int,
                        node_to_uv_vertex_ids: dict[int, list[int]],
                        uv_vertices: list[list[float]],
                        uv_vertex_orig_node: list[int]) -> int:
        """
        Create a new UV vertex, even if the same original node already exists. Use this when a node is seam-split and needs multiple UV instances.

        Parameters
        ----------
        uv : np.ndarray
            UV coordinates for the new vertex.
        orig_node : int
            Original node index for the new vertex.
        uv_vertex_count : int
            Current count of UV vertices.
        node_to_uv_vertex_ids : dict[int, list[int]]
            Mapping from original node index to list of UV vertex indices that represent it.
        uv_vertices : list[list[float]]
            List of UV vertex coordinates.
        uv_vertex_orig_node : list[int]
            List of original node indices corresponding to each UV vertex.

        Returns
        -------
        int
            Index of the new UV vertex.
        """
        uv_vertices.append([float(uv[0]), float(uv[1])])
        uv_vertex_orig_node.append(int(orig_node))
        node_to_uv_vertex_ids.setdefault(int(orig_node), []).append(uv_vertex_count)
        return uv_vertex_count + 1

    def map_uvs_to_higher_order(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Map triangulated UVs back to the original higher-order mesh.

        Returns
        -------
        orig_uvs : np.ndarray
            Shape (N_uv, 2). UV coordinates with seam-aware duplicated vertices mapped back to the original higher-order mesh.
        orig_vmapping : np.ndarray
            Shape (N_uv,), For each UV vertex, the original node index it came from.
        orig_indices : np.ndarray
            Shape (N_elem, nodes_per_element). Per-original-element UV connectivity, referencing orig_uvs.

        Raises
        -------
        ValueError:
            If the RTMesh is not loaded or if UV data has not been generated yet, or if triangulation mappings are missing (required for higher-order meshes).

        Notes
        -------
        - If an original node appears with multiple distinct UVs, it is duplicated.
        - The first UV for a node is reused only if it matches within tolerance.
        - This keeps the connectivity consistent across seams.
        """
        if self.rtmesh is None:
            raise ValueError("No RTMesh loaded.")
        if self.uvs is None or self.vertex_map is None or self.faces_cut is None:
            raise ValueError("UV data has not been generated yet.")
        #if self.rtmesh.tri_face_mapping is None or self.rtmesh.tri_node_mapping is None:
        #    raise ValueError("Triangulation mappings are missing.")

        tri_face_mapping = self.rtmesh.tri_face_mapping
        tri_node_mapping = self.rtmesh.tri_node_mapping

        # Blender/xatlas UV data is per triangulated UV vertex, so map each UV vertex back to the original mesh node index
        tri_uv_to_orig_node = tri_node_mapping[np.ascontiguousarray(self.vertex_map, dtype=np.int64)]

        # Output containers
        uv_vertices: list[list[float]] = []
        uv_vertex_orig_node: list[int] = []

        # For each original node, store all UV-vertex ids that represent it
        node_to_uv_vertex_ids: dict[int, list[int]] = {}

        # For seam consistency: if a node already has a UV that matches, reuse it. Else, duplicate the node in UV space
        uv_lookup_tol = 1e-12

        def get_or_create_uv_vertex(orig_node: int, uv_xy: np.ndarray) -> int:
            candidates = node_to_uv_vertex_ids.get(int(orig_node), [])
            for vertex_idx in candidates:
                if np.allclose(uv_vertices[vertex_idx], uv_xy, atol=uv_lookup_tol, rtol=0.0):
                    return vertex_idx
            vertex_idx = len(uv_vertices)
            uv_vertices.append([float(uv_xy[0]), float(uv_xy[1])])
            uv_vertex_orig_node.append(int(orig_node))
            node_to_uv_vertex_ids.setdefault(int(orig_node), []).append(vertex_idx)
            return vertex_idx

        # Build per-original-element connectivity in UV space, preserving the original element order and local node order
        orig_element_count = self.rtmesh.element_count
        nodes_per_elem = self.rtmesh.nodes_per_element

        # For each original element, gather all triangulated faces that came from it
        elem_to_triangles: list[list[int]] = [[] for _ in range(orig_element_count)]
        for triangle_idx, orig_element_idx in enumerate(tri_face_mapping):
            if 0 <= orig_element_idx < orig_element_count:
                elem_to_triangles[int(orig_element_idx)].append(int(triangle_idx))

        orig_indices = np.empty((orig_element_count, nodes_per_elem), dtype=np.uint64)

        # We need a per-original-node -> UV vertex assignment per element
        # This avoids mixing triangles from different parts of the same higher-order face
        for orig_element_idx in range(orig_element_count):
            orig_node_ids = np.ascontiguousarray(self.rtmesh.connectivity[orig_element_idx], dtype=np.int64)
            triangle_indices = elem_to_triangles[orig_element_idx]

            # Gather all UVs that belong to this original element through its triangulated subfaces
            # If a node appears multiple times with different UVs, we duplicate it
            local_node_to_uv: dict[int, int] = {}

            for triangle_idx in triangle_indices:
                tri_uv_vertex_ids = self.faces_cut[triangle_idx]
                # Go over 3 corners of TRI3 element
                for local_corner in range(3): 
                    tri_uv_vid = int(tri_uv_vertex_ids[local_corner])
                    orig_node = int(tri_uv_to_orig_node[tri_uv_vid])
                    uv_xy = np.ascontiguousarray(self.uvs[tri_uv_vid], dtype=np.double)

                    # Only assign UVs for nodes that actually belong to this original element
                    # (important when multiple elements share triangulated boundary vertices)
                    if orig_node not in orig_node_ids:
                        continue

                    if orig_node in local_node_to_uv:
                        # If already assigned, make sure it's consistent
                        # If it differs, create a duplicate UV vertex and remap this occurrence
                        existing_vid = local_node_to_uv[orig_node]
                        if not np.allclose(uv_vertices[existing_vid], uv_xy, atol=uv_lookup_tol, rtol=0.0):
                            new_vertex_id = get_or_create_uv_vertex(orig_node, uv_xy)
                            local_node_to_uv[orig_node] = new_vertex_id
                    else:
                        local_node_to_uv[orig_node] = get_or_create_uv_vertex(orig_node, uv_xy)

            # Ensure every original node in the element has a UV entry
            # If a node was not seen via triangulated corners, fall back to a default by
            # duplicating the first available UV in this element
            if len(local_node_to_uv) == 0:
                raise ValueError(f"Could not reconstruct any UVs for original element {orig_element_idx}.")

            fallback_vid = next(iter(local_node_to_uv.values()))
            for node in orig_node_ids:
                if int(node) not in local_node_to_uv:
                    # Seam-safe fallback: duplicate a valid UV from the same element
                    uv_xy = np.ascontiguousarray(uv_vertices[fallback_vid], dtype=np.double)
                    local_node_to_uv[int(node)] = get_or_create_uv_vertex(int(node), uv_xy)

            # Now build element connectivity in the original local-node order
            for local_idx, node in enumerate(orig_node_ids):
                orig_indices[orig_element_idx, local_idx] = local_node_to_uv[int(node)]

        orig_uvs = np.ascontiguousarray(np.ascontiguousarray(uv_vertices, dtype=np.double))
        orig_vmapping = np.ascontiguousarray(np.ascontiguousarray(uv_vertex_orig_node, dtype=np.uint64))
        orig_indices = np.ascontiguousarray(orig_indices, dtype=np.uint64)

        return orig_uvs, orig_vmapping, orig_indices

    def _get_uvs(self) -> None:
        """
        Get UVs for the entire mesh after unwrapping.

        UVs get saved back to the RTMesh object, mapped to the original mesh's higher-order connectivity if needed.
        Otherwise, the shape is just changed to match node_coords_expanded, i.e., (element_count, nodes_per_element, 2) for easy indexing during raytracing,
        instead of having to do double index lookup from connectivity to node coords.
        """
        uv_layer = self.blender_mesh.uv_layers.active.data
        uv_coords = []

        for poly in self.blender_mesh.polygons:
           face_uvs = []
           for loop_index in poly.loop_indices:
               uv = uv_layer[loop_index].uv
               face_uvs.append((uv.x, uv.y))
           uv_coords.append(face_uvs)

        print("uvs as from blender directly:")
        print(f"uvs[0] shape: {len(uv_coords[0])}")
        print(f"uvs[0][0] shape: {len(uv_coords[0][0])}")
        print(f"size of list: {len(uv_coords)}")
        self.uvs, self.vertex_map, self.faces_cut = self._get_xatlas_uv_format(self.blender_obj.data)
        print(f"uvs shape: {self.uvs.shape}")
        print(f"vmapping shape: {self.vertex_map.shape}")
        print(f"indices shape: {self.faces_cut.shape}")

        if self.rtmesh.tri_face_mapping is None or self.rtmesh.tri_node_mapping is None: # Triangular mesh
            #self.rtmesh.uvs = np.ascontiguousarray(self.uvs, dtype=np.double)
            #self.rtmesh.connectivity_uv = np.ascontiguousarray(self.faces_cut)
            self.rtmesh.uvs = np.ascontiguousarray(self.uvs[self.faces_cut], dtype=np.double) # To get shape (element count, nodes_per_element, 2)
        else: # Any other mesh - we need to map back to higher order elements
            orig_uvs, orig_vmapping, orig_face_indices = self.map_uvs_to_higher_order()
            print(f"orig_uvs shape: {orig_uvs.shape}")
            print(f"orig_vmapping shape: {orig_vmapping.shape}")
            print(f"orig_indices shape: {orig_face_indices.shape}")
            # Save the seam-aware higher-order UV connectivity
            #self.rtmesh.uvs = np.ascontiguousarray(orig_uvs, dtype=np.double)
            #self.rtmesh.connectivity_uv = np.ascontiguousarray(orig_face_indices, dtype=np.uint64)
            self.rtmesh.uvs = np.ascontiguousarray(orig_uvs[orig_face_indices], dtype=np.double) # To get shape (element count, nodes_per_element, 2)

    def _get_uvs_selected(self):
        """
        Get UVs only for currently selected faces (after loop_to_region) after unwrapping it.

        UVs get saved back to the RTMesh object, mapped to the original mesh's higher-order connectivity if needed.
        Otherwise, the shape is just changed to match node_coords_expanded, i.e., (element_count, nodes_per_element, 2) for easy indexing during raytracing,
        instead of having to do double index lookup from connectivity to node coords.
        """
        # Ensure we're in Object Mode and mesh is updated
        bpy.ops.object.mode_set(mode='OBJECT')
        self.blender_mesh.update()
    
        # Extract only selected faces
        self.uvs, self.vertex_map, self.faces_cut = self._get_xatlas_uv_selected_format(self.blender_obj.data)
        print(
            f"Selected region - uvs: {self.uvs.shape}, vmapping: {self.vertex_map.shape}, indices: {self.faces_cut.shape}")
        if self.rtmesh.tri_face_mapping is None or self.rtmesh.tri_node_mapping is None: # Triangular mesh
            #self.rtmesh.uvs = np.ascontiguousarray(self.uvs)
            #self.rtmesh.connectivity_uv = np.ascontiguousarray(self.faces_cut, dtype=np.uint64)
            self.rtmesh.uvs = np.ascontiguousarray(self.uvs[self.faces_cut], dtype=np.double) # To get shape (element count, nodes_per_element, 2)
        else: # Any other mesh - we need to map back to higher order elements
            orig_uvs, orig_vmapping, orig_face_indices = self.map_uvs_to_higher_order()
            #self.rtmesh.uvs = np.ascontiguousarray(orig_uvs, dtype=np.double)
            #self.rtmesh.connectivity_uv = np.ascontiguousarray(orig_face_indices, dtype=np.uint64)
            self.rtmesh.uvs = np.ascontiguousarray(orig_uvs[orig_face_indices], dtype=np.double) # To get shape (element count, nodes_per_element, 2)

    def smart_unwrap(self,
                     pack_islands=True,
                     angle_limit=None) -> None:
        """
        Performs fully automatic UV unwrapping, where no seams need to be selected, using Blender's smart UV project method.

        Calls Blender's smart UV project unwrapping method, which automatically determines seams based on an angle threshold and projects UVs accordingly.

        Parameters
        ----------
        pack_islands : bool, optional
            Whether to pack UV islands after unwrapping. Default is True.
        angle_limit : float, optional
            Angle limit in radians for determining seams. Must be between 0 and 1.15708 (approximately 66 degrees; this range comes from Blender documentation).
            If None, Blender's default is used.

        Raises
        ------
        ValueError
            If angle_limit is not between 0 and 1.15708 radians.
        """
        if angle_limit is not None:
            if angle_limit < 0 or angle_limit > 1.15708:
                raise ValueError("Angle limit must be between 0 and 1.15708 radians")
            else:
                bpy.context.scene.tool_settings.uv_smart_project.angle_limit = angle_limit
        self.blender_obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project()
        if pack_islands:
            bpy.ops.uv.pack_islands()
        bpy.ops.object.mode_set(mode='OBJECT')
        # Get UVs
        self._get_uvs()

    def unwrap(self,
               algorithm=UnwrapAlgo.LSCM,
               pack_islands=True,
               iterations=10) -> None:
        """
        Performs manual UV unwrapping based on the selected algorithm and seam markings on the mesh.
            
        Calls Blender's unwrap method with the specified algorithm, using the seams marked in the mesh to guide the unwrapping process.

        Parameters
        ----------
        algorithm : UnwrapAlgo, optional
            The algorithm to use for unwrapping. Default is UnwrapAlgo.LSCM (Least Squares Conformal Mapping).
        pack_islands : bool, optional
            Whether to pack UV islands after unwrapping. Default is True.
        iterations : int, optional
            Number of iterations for the unwrapping algorithm, if applicable (e.g., for SLIM). Default is 10. Ignored for algorithms that do not use iterations.
        
        Raises
        ------
        ValueError
            If iterations is not between 1 and 10000 for the SLIM algorithm (Blender's limit for this parameter).
        """
        # To do: ideally include more options to allow full customization like in Blender
        # https://docs.blender.org/api/current/bpy.ops.uv.html#bpy.ops.uv.unwrap
        self._select_seams()
        self.blender_obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        # Set iterations for SLIM only
        if algorithm == UnwrapAlgo.SLIM:
            if iterations <= 0 or iterations > 10000:
                raise ValueError("Iterations for SLIM must be between 1 and 10000")
            bpy.ops.uv.unwrap(method=algorithm.value, iterations=iterations)
        else:
            bpy.ops.uv.unwrap(method=algorithm.value)
        if pack_islands:
            bpy.ops.uv.pack_islands()
        # Set back to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        # Get UVs
        self._get_uvs()

    def unwrap_face(self,
                    algorithm=UnwrapAlgo.LSCM,
                    pack_islands=True,
                    iterations=10) -> None:
        """
        Performs manual UV unwrapping based on the selected algorithm and seam markings on the SELECTED FACES.
            
        Calls Blender's unwrap method with the specified algorithm, using the seams marked in the mesh to guide the unwrapping process.

        Parameters
        ----------
        algorithm : UnwrapAlgo, optional
            The algorithm to use for unwrapping. Default is UnwrapAlgo.LSCM (Least Squares Conformal Mapping).
        pack_islands : bool, optional
            Whether to pack UV islands after unwrapping. Default is True.
        iterations : int, optional
            Number of iterations for the unwrapping algorithm, if applicable (e.g., for SLIM). Default is 10. Ignored for algorithms that do not use iterations.
        
        Raises
        ------
        ValueError
            If iterations is not between 1 and 10000 for the SLIM algorithm (Blender's limit for this parameter).
        """
        self._select_seams()  # Technically, the boundary doesn't have to be a seam for this, but it's easier
        self.blender_obj.select_set(True)
        bpy.context.view_layer.objects.active = self.blender_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')  # Deselect everything
        # print(bpy.context.object.data.count_selected_items()) # Baseline for how many (nodes, edges, faces) there are in the mesh
    
        # Edge select mode
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.object.mode_set(mode='OBJECT')  # Brief switch
        print(bpy.context.object.data.count_selected_items())
    
        # Select all seams we marked before as the loop boundary
        for edge in self.blender_mesh.edges:
            edge.select = edge.use_seam  # Both are boolean, so if an edge is a seam, select it
        bpy.ops.object.mode_set(mode='EDIT')
        print(bpy.context.object.data.count_selected_items())  # Check that edges have been selected
    
        # Convert the boundary loop edge into a face-region selection
        bpy.ops.mesh.loop_to_region(
            select_bigger=False)  # This will have to be passed, depending if our seleted face is bigger than the rest of the mesh or not
        # print(bpy.context.object.data.count_selected_items()) # Check that the number increased, isn't 0, but also shouldn't cover the whole mesh
        if bpy.context.object.data.count_selected_items()[2] == 0:
            raise ValueError("No faces selected - check seam loop")
        # Set iterations for SLIM only
        if algorithm == UnwrapAlgo.SLIM:
            if iterations <= 0 or iterations > 10000:
                raise ValueError("Iterations for SLIM must be between 1 and 10000")
            bpy.ops.uv.unwrap(method=algorithm.value, iterations=iterations)
        else:
            bpy.ops.uv.unwrap(method=algorithm.value)
        bpy.ops.object.mode_set(mode='OBJECT')
        # Get UVs only for selected faces
        # if pack_islands: # Technically shouldn't need packing for a single face?
        # bpy.ops.uv.pack_islands()
        self._get_uvs_selected()

    def display_flat_mesh(self) -> None:
        """
        Visualises the flattened mesh by using uv as x,y coordinates (z=0 for visualization).

        Implemented as a part of the BlenderUnwrapper class since it relies on the UV data generated by the unwrapping process that does not get saved back to the RTMesh in the same format.
        It also uses the faces_cut connectivity which is based on the triangulated mesh, so it needs to be visualized in the same context as the unwrapped UVs due to vedo's expectations.

        Notes:
        ------
            - uvs - nd.ndarray of shape (N,2) containing the UV coordinates for each vertex in the flattened mesh. These coordinates are used as x and y positions for visualization, with z set to 0.
            - faces_cut - nd.ndarray of shape (F, 3) containing the indices of the vertices that form each triangle in the flattened mesh. Equivalent to connectivity, just cut.
        """
        uv_3d = np.hstack((self.uvs, np.zeros((len(self.uvs), 1))))
        flat_mesh = vedo.Mesh([uv_3d, self.faces_cut]).c('tomato').wireframe()
        vedo.show(flat_mesh, "UV unwrapped result", new=True)  # New so it doesn't open in the SeamSplitter

    def display_mesh_with_texture(self, texture: np.ndarray) -> None:
        """
        Visualises the flattened mesh by using uv as x,y coordinates, overlaid on top of the texture to visualise the mapping.

        Implemented as a part of the BlenderUnwrapper class since it relies on the UV data generated by the unwrapping process that does not get saved back to the RTMesh in the same format.

        Parameters:
        ----------
          texture: np.ndarray
            Shape (H, W) - Grayscale image representing the texture to be displayed under the UV-mapped mesh. This should be the output of ImageTools.load_image_grayscale(path) or a similar
            function that loads an image as a 2D numpy array. The UV coordinates will be scaled to match the dimensions of this texture for accurate visualization.
        
        Notes:
        ------
            - uvs - nd.ndarray of shape (N,2) containing the UV coordinates for each vertex in the flattened mesh. These coordinates are used as x and y positions for visualization, with z set to 0.
            - faces_cut - nd.ndarray of shape (F, 3) containing the indices of the vertices that form each triangle in the flattened mesh. Equivalent to connectivity, just cut.

        """
        uvs = self.uvs
        uv_3d = np.hstack((uvs, np.zeros((len(uvs), 1))))
        texture_bg = vedo.Image(texture)
        texture_bg.alpha(0.5)  # Set transparency to the texture, otherwise it is really hard to see the wireframe
        # Output uvs are in the [0,1] space so scale them to match the texture dimensions as scaling down gives terrible interpolation artifacts
        scaled_uvs = uvs * min(texture_bg.dimensions())
        uv_3d = np.insert(scaled_uvs, 2, 0, axis=1)
        packed_mesh = vedo.Mesh([uv_3d, self.faces_cut]).c('tomato').wireframe()
        packed_mesh.linewidth(min(texture_bg.dimensions()) / 1000)
        vedo.show([packed_mesh, texture_bg], "UV unwrapped mesh on the texture", new=True)
        # Example usage with xatlas: vmapping, indices, uvs = xatlas.parametrize(coords_o, connectivity_o)
        # Then call display_mesh_with_texture(uvs, indices, texture)
"""
    def blender_test(self):
        # Smart unwrapping (automatic)
        # self.blender_load_mesh()
        # self.smart_unwrap(pack_islands=True)
        # self._blender_remove_mesh()
    
        # Unwrapping using selected edges and algorithm (full mesh)
        self.blender_load_mesh()
        self.unwrap(algorithm=UnwrapAlgo.SLIM, pack_islands=True, iterations=10000)
        # self._blender_remove_mesh()
    
        # Unwrapping using selected edges and algorithm (one face)
        # self.blender_load_mesh()
        # self.unwrap_face(algorithm=UnwrapAlgo.LSCM, pack_islands=True, iterations=10)
    
        print(self.uvs.shape)
        # Display results overlaid on texture
        # display_mesh_with_texture(self.uvs, self.faces_cut, blender_ex_img)\
"""

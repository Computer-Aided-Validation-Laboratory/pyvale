import numpy as np
import pyvale.mooseherder as mh


def extract_surf_mesh(sim_data: mh.SimData) -> mh.SimData:
    """Extract the exterior surface mesh from a 3D simulation mesh.

    The extracted surface faces are rewound so that their geometric normal
    points away from the parent solid element. This enforces a counterclockwise
    convention when the face is viewed from outside the surface.
    """

    if sim_data.connect is None or len(sim_data.connect) == 0:
        raise ValueError("Simulation data does not contain connectivity tables.")

    if sim_data.coords is None:
        raise ValueError("Simulation data does not contain coordinates.")

    candidates: dict[int, list[dict[str, np.ndarray | int | str]]] = {}

    for connect_key in sorted(sim_data.connect.keys()):
        connect = np.asarray(sim_data.connect[connect_key], dtype=np.int64) - 1
        nodes_per_elem, num_elems = connect.shape
        face_map = _get_surf_map(nodes_per_elem)
        nodes_per_face = face_map.shape[1]

        elem_nodes = connect.T
        elem_centroids = sim_data.coords[elem_nodes].mean(axis=1)

        group = candidates.setdefault(nodes_per_face, [])
        for elem_ind in range(num_elems):
            elem_connect = connect[:, elem_ind]
            elem_centroid = elem_centroids[elem_ind]
            for face_ind, local_face in enumerate(face_map):
                face_nodes = np.copy(elem_connect[local_face])
                group.append({
                    "connect_key": connect_key,
                    "elem_ind": elem_ind,
                    "face_ind": face_ind,
                    "elem_centroid": elem_centroid,
                    "face_nodes": face_nodes,
                })

    ext_face_records: list[dict[str, np.ndarray | int | str]] = []

    for nodes_per_face in sorted(candidates.keys()):
        group = candidates[nodes_per_face]
        faces_flat_wound = np.asarray(
            [rec["face_nodes"] for rec in group],
            dtype=np.int64,
        )
        faces_flat_sorted = np.sort(faces_flat_wound, axis=1)

        (_, unique_inds, unique_counts) = np.unique(
            faces_flat_sorted,
            axis=0,
            return_index=True,
            return_counts=True,
        )
        ext_unique_inds = np.where(unique_counts == 1)[0]
        ext_face_inds = unique_inds[ext_unique_inds]

        ext_faces = np.copy(faces_flat_wound[ext_face_inds])
        for out_ind, flat_face_ind in enumerate(ext_face_inds):
            rec = group[int(flat_face_ind)]
            ext_faces[out_ind] = _orient_face_outward(
                sim_data.coords,
                ext_faces[out_ind],
                np.asarray(rec["elem_centroid"], dtype=np.float64),
            )
            ext_face_records.append({
                "connect_key": str(rec["connect_key"]),
                "nodes_per_face": nodes_per_face,
                "elem_ind": int(rec["elem_ind"]),
                "face_nodes": np.copy(ext_faces[out_ind]),
            })

    if len(ext_face_records) == 0:
        raise ValueError("No exterior faces were found during surface extraction.")
    all_ext_faces = [
        np.asarray([rec["face_nodes"] for rec in ext_face_records], dtype=np.int64),
    ]

    used_coord_inds = np.unique(np.concatenate([faces.reshape(-1) for faces in all_ext_faces]))
    faces_coords = np.copy(sim_data.coords[used_coord_inds])
    coord_remap = np.full(sim_data.coords.shape[0], -1, dtype=np.int64)
    coord_remap[used_coord_inds] = np.arange(used_coord_inds.shape[0], dtype=np.int64)

    grouped_records: dict[tuple[str, int], list[dict[str, np.ndarray | int | str]]] = {}
    for rec in ext_face_records:
        grouped_records.setdefault(
            (str(rec["connect_key"]), int(rec["nodes_per_face"])),
            [],
        ).append(rec)

    connect_out: dict[str, np.ndarray] = {}
    face_table_sources: dict[str, list[tuple[str, int]]] = {}
    for table_ind, group_key in enumerate(sorted(grouped_records.keys()), start=1):
        group_recs = grouped_records[group_key]
        ext_faces = np.asarray([rec["face_nodes"] for rec in group_recs], dtype=np.int64)
        remapped = coord_remap[ext_faces] + 1
        out_key = f"connect{table_ind}"
        connect_out[out_key] = remapped.T
        face_table_sources[out_key] = [
            (str(rec["connect_key"]), int(rec["elem_ind"])) for rec in group_recs
        ]

    face_data = mh.SimData(
        coords=faces_coords,
        connect=connect_out,
        time=sim_data.time,
        num_spat_dims=sim_data.num_spat_dims,
        glob_vars=sim_data.glob_vars,
    )

    if sim_data.node_vars is not None:
        face_data.node_vars = {}
        for name, values in sim_data.node_vars.items():
            face_data.node_vars[name] = values[used_coord_inds, :]

    if sim_data.elem_vars is not None:
        face_data.elem_vars = {}
        for elem_key, values in sim_data.elem_vars.items():
            block_key = f"connect{elem_key[1]}"
            for out_key, sources in face_table_sources.items():
                face_vals = np.asarray(
                    [values[src_elem, :] for src_key, src_elem in sources if src_key == block_key],
                )
                if face_vals.size == 0:
                    continue
                out_elem_key = (elem_key[0], int(out_key.removeprefix("connect")))
                face_data.elem_vars[out_elem_key] = face_vals

    return face_data


def _orient_face_outward(
    coords: np.ndarray,
    face_nodes: np.ndarray,
    elem_centroid: np.ndarray,
) -> np.ndarray:
    face_coords = coords[face_nodes]
    face_centroid = face_coords.mean(axis=0)
    face_normal = _calc_face_normal(face_coords)
    outward_dir = face_centroid - elem_centroid

    if np.dot(face_normal, outward_dir) < 0.0:
        return face_nodes[_get_flip_permutation(face_nodes.shape[0])]
    return face_nodes


def _calc_face_normal(face_coords: np.ndarray) -> np.ndarray:
    corners = _get_corner_coords(face_coords)
    normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    normal_mag = np.linalg.norm(normal)
    if normal_mag == 0.0 and corners.shape[0] == 4:
        normal = np.cross(corners[2] - corners[0], corners[3] - corners[0])
        normal_mag = np.linalg.norm(normal)
    if normal_mag == 0.0:
        raise ValueError("Degenerate face detected while extracting surface mesh.")
    return normal / normal_mag


def _get_corner_coords(face_coords: np.ndarray) -> np.ndarray:
    nodes_per_face = face_coords.shape[0]
    if nodes_per_face in (3, 6):
        return face_coords[:3, :]
    if nodes_per_face in (4, 8, 9):
        return face_coords[:4, :]
    raise ValueError(f"Unsupported surface face size: {nodes_per_face}")


def _get_flip_permutation(nodes_per_face: int) -> np.ndarray:
    if nodes_per_face == 3:
        return np.array((0, 2, 1), dtype=np.int64)
    if nodes_per_face == 4:
        return np.array((0, 3, 2, 1), dtype=np.int64)
    if nodes_per_face == 6:
        return np.array((0, 2, 1, 5, 4, 3), dtype=np.int64)
    if nodes_per_face == 8:
        return np.array((0, 3, 2, 1, 7, 6, 5, 4), dtype=np.int64)
    if nodes_per_face == 9:
        return np.array((0, 3, 2, 1, 7, 6, 5, 4, 8), dtype=np.int64)
    raise ValueError(f"Unsupported surface face size: {nodes_per_face}")


# TODO: make this support triangular prisms in 3D.
def _get_surf_map(nodes_per_elem: int) -> np.ndarray:
    """Map 3D volume-element nodes to wound surface faces."""
    if nodes_per_elem == 4:  # TET4
        return np.array(((0, 1, 2),
                         (0, 3, 1),
                         (0, 2, 3),
                         (1, 3, 2)))

    if nodes_per_elem == 8:  # HEX8
        return np.array(((0, 1, 2, 3),
                         (0, 3, 7, 4),
                         (4, 7, 6, 5),
                         (1, 5, 6, 2),
                         (0, 4, 5, 1),
                         (2, 6, 7, 3)))

    if nodes_per_elem == 10:  # TET10
        return np.array(((0, 1, 2, 4, 5, 6),
                         (0, 3, 1, 4, 8, 7),
                         (0, 2, 3, 6, 9, 7),
                         (1, 3, 2, 8, 9, 5)))

    if nodes_per_elem == 20:  # HEX20
        return np.array(((0, 1, 2, 3, 8, 9, 10, 11),
                         (0, 3, 7, 4, 11, 15, 19, 12),
                         (4, 7, 6, 5, 19, 18, 17, 16),
                         (1, 5, 6, 2, 13, 17, 14, 9),
                         (0, 4, 5, 1, 12, 16, 13, 8),
                         (2, 6, 7, 3, 14, 18, 15, 10)))

    if nodes_per_elem == 27:  # HEX27
        return np.array(((0, 1, 2, 3, 8, 9, 10, 11, 21),
                         (0, 3, 7, 4, 11, 15, 19, 12, 23),
                         (4, 7, 6, 5, 19, 18, 17, 16, 22),
                         (1, 5, 6, 2, 13, 17, 14, 9, 24),
                         (0, 4, 5, 1, 12, 16, 13, 8, 25),
                         (2, 6, 7, 3, 14, 18, 15, 10, 26)))

    raise ValueError(
        "Number of nodes does not match a supported 3D element type for "
        "surface extraction.",
    )

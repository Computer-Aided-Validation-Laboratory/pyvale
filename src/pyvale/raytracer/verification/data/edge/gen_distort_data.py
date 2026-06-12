from pathlib import Path

import numpy as np


EDGE_LENG = 10.0
ELEM_ROT = 20.0

BULGE_TIME_STEPS = 13
BULGE_MIDSIDE_OFFSET_FACTOR = 0.3

TAN_TIME_STEPS = 13
TAN_OFFSET_FACTOR = 0.3

STRETCH = 1.0
SHEAR = 0.5
STEPS = 11

ROT_TIME_STEPS = 13

def save_case(
    base_dir,
    name,
    coords,
    connect,
    disp_x,
    disp_y,
    disp_z,
    uvs,
):
    out_dir = Path(base_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    coords, disp_x, disp_y, disp_z = rotate_case_fields(
        coords,
        disp_x,
        disp_y,
        disp_z,
        ELEM_ROT,
    )
    np.savetxt(out_dir / "coords.csv", coords, delimiter=",")
    np.savetxt(
        out_dir / "connect.csv",
        connect.astype(int),
        delimiter=",",
        fmt="%d",
    )
    np.savetxt(
        out_dir / "connectivity.csv",
        connect.astype(int),
        delimiter=",",
        fmt="%d",
    )
    np.savetxt(out_dir / "field_disp_x.csv", disp_x, delimiter=",")
    np.savetxt(out_dir / "field_disp_y.csv", disp_y, delimiter=",")
    np.savetxt(out_dir / "field_disp_z.csv", disp_z, delimiter=",")
    np.savetxt(out_dir / "uvs.csv", uvs, delimiter=",")


def compute_uvs(coords, u_range=(0.4, 0.6), v_range=(0.4, 0.6)):
    xmin, ymin, _ = np.min(coords, axis=0)
    xmax, ymax, _ = np.max(coords, axis=0)
    xrng = max(xmax - xmin, 1.0)
    yrng = max(ymax - ymin, 1.0)

    return np.array(
        [
            [
                u_range[0] + (u_range[1] - u_range[0]) * (pt[0] - xmin) / xrng,
                v_range[0] + (v_range[1] - v_range[0]) * (pt[1] - ymin) / yrng,
            ]
            for pt in coords
        ]
    )


def build_disp_fields_bulge(coords, midside_info, time_steps, max_offset):
    node_num = coords.shape[0]
    disp_x = np.zeros((node_num, time_steps))
    disp_y = np.zeros((node_num, time_steps))
    disp_z = np.zeros((node_num, time_steps))

    for tt in range(time_steps):
        alpha = tt / (time_steps - 1)
        beta = -2.0 * alpha
        for midside_ind, direction in midside_info:
            delta = beta * max_offset * direction
            disp_x[midside_ind, tt] = delta[0]
            disp_y[midside_ind, tt] = delta[1]
            disp_z[midside_ind, tt] = delta[2]

    return disp_x, disp_y, disp_z


def build_disp_fields_tan(coords, midside_info, time_steps, edge_length, tan_offset_factor):
    node_num = coords.shape[0]
    disp_x = np.zeros((node_num, time_steps))
    disp_y = np.zeros((node_num, time_steps))
    disp_z = np.zeros((node_num, time_steps))

    tan_offsets = np.linspace(
        -tan_offset_factor * edge_length,
        tan_offset_factor * edge_length,
        time_steps,
    )
    initial_offset = tan_offsets[0]

    for tt, tan_offset in enumerate(tan_offsets):
        delta_mag = tan_offset - initial_offset
        for midside_ind, tangent in midside_info:
            delta = delta_mag * tangent
            disp_x[midside_ind, tt] = delta[0]
            disp_y[midside_ind, tt] = delta[1]
            disp_z[midside_ind, tt] = delta[2]

    return disp_x, disp_y, disp_z


def build_disp_fields_to_target(coords_initial, coords_final, time_steps):
    node_num = coords_initial.shape[0]
    disp_x = np.zeros((node_num, time_steps))
    disp_y = np.zeros((node_num, time_steps))
    disp_z = np.zeros((node_num, time_steps))

    delta = coords_final - coords_initial
    for tt in range(time_steps):
        alpha = tt / (time_steps - 1)
        disp_x[:, tt] = alpha * delta[:, 0]
        disp_y[:, tt] = alpha * delta[:, 1]
        disp_z[:, tt] = alpha * delta[:, 2]

    return disp_x, disp_y, disp_z


def build_disp_fields_from_frame_coords(coords_initial, coords_frames):
    node_num = coords_initial.shape[0]
    time_steps = len(coords_frames)
    disp_x = np.zeros((node_num, time_steps))
    disp_y = np.zeros((node_num, time_steps))
    disp_z = np.zeros((node_num, time_steps))

    for tt, coords_frame in enumerate(coords_frames):
        delta = coords_frame - coords_initial
        disp_x[:, tt] = delta[:, 0]
        disp_y[:, tt] = delta[:, 1]
        disp_z[:, tt] = delta[:, 2]

    return disp_x, disp_y, disp_z


def rotate_case_fields(coords, disp_x, disp_y, disp_z, angle_deg):
    if abs(angle_deg) < 1.0e-12:
        return coords, disp_x, disp_y, disp_z

    coords_initial = coords
    coords_frames = []
    time_steps = disp_x.shape[1]

    for tt in range(time_steps):
        coords_frame = coords_initial.copy()
        coords_frame[:, 0] += disp_x[:, tt]
        coords_frame[:, 1] += disp_y[:, tt]
        coords_frame[:, 2] += disp_z[:, tt]
        coords_frames.append(coords_frame)

    center = np.mean(coords_initial, axis=0)
    coords_rot = rotate_points(coords_initial, angle_deg, center)
    coords_frames_rot = [
        rotate_points(coords_frame, angle_deg, center)
        for coords_frame in coords_frames
    ]

    disp_x_rot, disp_y_rot, disp_z_rot = build_disp_fields_from_frame_coords(
        coords_rot,
        coords_frames_rot,
    )
    return coords_rot, disp_x_rot, disp_y_rot, disp_z_rot


def get_midside_direction(v1, v2, centroid):
    midpoint = 0.5 * (v1 + v2)
    direction = midpoint - centroid
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    return direction / direction_norm


def get_edge_tangent(v1, v2):
    tangent = v2 - v1
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    return tangent / tangent_norm


def build_edge_midpoints(vertices, edge_pairs):
    return np.array([0.5 * (vertices[i0] + vertices[i1]) for i0, i1 in edge_pairs])


def rotate_points(points, angle_deg, center):
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot_mat = np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return np.array([rot_mat @ (pp - center) + center for pp in points])


def generate_case(base_dir, name, coords_initial, coords_final, connect):
    disp_x, disp_y, disp_z = build_disp_fields_to_target(
        coords_initial,
        coords_final,
        STEPS,
    )
    save_case(
        base_dir,
        name,
        coords_initial,
        connect,
        disp_x,
        disp_y,
        disp_z,
        compute_uvs(coords_initial),
    )


def tri_stretch_vertices_final(edge_length):
    half_edge = 0.5 * edge_length
    tri_tip_x = np.sqrt(3.0) * edge_length / 2.0
    return np.array(
        [
            [0.0, -half_edge, 0.0],
            [tri_tip_x, 0.0, 0.0],
            [0.0, half_edge, 0.0],
        ]
    )


def tri_stretch_vertices_initial(edge_length, stretch_ratio):
    vertices = tri_stretch_vertices_final(edge_length).copy()
    stretch_disp = edge_length * stretch_ratio
    vertices[0, 0] -= 0.5 * stretch_disp
    vertices[1, 0] += stretch_disp
    vertices[2, 0] -= 0.5 * stretch_disp
    return vertices


def tri_shear_vertices_final(edge_length):
    height = np.sqrt(3.0) * edge_length / 2.0
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [edge_length, 0.0, 0.0],
            [0.5 * edge_length, height, 0.0],
        ]
    )


def tri_shear_vertices_initial(edge_length, shear_ratio):
    vertices = tri_shear_vertices_final(edge_length).copy()
    shear_disp = edge_length * shear_ratio
    vertices[0, 0] -= 0.5 * shear_disp
    vertices[1, 0] -= 0.5 * shear_disp
    vertices[2, 0] += shear_disp
    return vertices


def quad_vertices_final(edge_length):
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [edge_length, 0.0, 0.0],
            [edge_length, edge_length, 0.0],
            [0.0, edge_length, 0.0],
        ]
    )


def quad_stretch_vertices_initial(edge_length, stretch_ratio):
    vertices = quad_vertices_final(edge_length).copy()
    stretch_disp = edge_length * stretch_ratio
    vertices[0, 0] -= stretch_disp
    vertices[3, 0] -= stretch_disp
    vertices[1, 0] += stretch_disp
    vertices[2, 0] += stretch_disp
    return vertices


def quad_shear_vertices_initial(edge_length, shear_ratio):
    vertices = quad_vertices_final(edge_length).copy()
    shear_disp = edge_length * shear_ratio
    vertices[0, 0] -= shear_disp
    vertices[1, 0] -= shear_disp
    vertices[2, 0] += shear_disp
    vertices[3, 0] += shear_disp
    return vertices


def generate_tri3_stretch(base_dir, edge_length, stretch_ratio):
    coords_final = tri_stretch_vertices_final(edge_length)
    coords_initial = tri_stretch_vertices_initial(edge_length, stretch_ratio)
    generate_case(
        base_dir,
        "tri3_distort_stretch",
        coords_initial,
        coords_final,
        np.array([[0, 1, 2]]),
    )


def generate_tri6_stretch(base_dir, edge_length, stretch_ratio):
    coords_vertices_final = tri_stretch_vertices_final(edge_length)
    coords_vertices_initial = tri_stretch_vertices_initial(edge_length, stretch_ratio)
    edge_pairs = [(0, 1), (1, 2), (2, 0)]
    midsides_initial = build_edge_midpoints(coords_vertices_initial, edge_pairs)
    midsides_final = build_edge_midpoints(coords_vertices_final, edge_pairs)
    coords_initial = np.vstack([coords_vertices_initial, midsides_initial])
    coords_final = np.vstack([coords_vertices_final, midsides_final])
    generate_case(
        base_dir,
        "tri6_distort_stretch",
        coords_initial,
        coords_final,
        np.array([[0, 1, 2, 3, 4, 5]]),
    )


def generate_quad4_stretch(base_dir, edge_length, stretch_ratio):
    coords_final = quad_vertices_final(edge_length)
    coords_initial = quad_stretch_vertices_initial(edge_length, stretch_ratio)
    generate_case(
        base_dir,
        "quad4_distort_stretch",
        coords_initial,
        coords_final,
        np.array([[0, 1, 2, 3]]),
    )


def generate_quad8_stretch(base_dir, edge_length, stretch_ratio):
    coords_vertices_final = quad_vertices_final(edge_length)
    coords_vertices_initial = quad_stretch_vertices_initial(edge_length, stretch_ratio)
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    midsides_initial = build_edge_midpoints(coords_vertices_initial, edge_pairs)
    midsides_final = build_edge_midpoints(coords_vertices_final, edge_pairs)
    coords_initial = np.vstack([coords_vertices_initial, midsides_initial])
    coords_final = np.vstack([coords_vertices_final, midsides_final])
    generate_case(
        base_dir,
        "quad8_distort_stretch",
        coords_initial,
        coords_final,
        np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
    )


def generate_quad9_stretch(base_dir, edge_length, stretch_ratio):
    coords_vertices_final = quad_vertices_final(edge_length)
    coords_vertices_initial = quad_stretch_vertices_initial(edge_length, stretch_ratio)
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    midsides_initial = build_edge_midpoints(coords_vertices_initial, edge_pairs)
    midsides_final = build_edge_midpoints(coords_vertices_final, edge_pairs)
    center_initial = np.mean(coords_vertices_initial, axis=0, keepdims=True)
    center_final = np.mean(coords_vertices_final, axis=0, keepdims=True)
    coords_initial = np.vstack([coords_vertices_initial, midsides_initial, center_initial])
    coords_final = np.vstack([coords_vertices_final, midsides_final, center_final])
    generate_case(
        base_dir,
        "quad9_distort_stretch",
        coords_initial,
        coords_final,
        np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]]),
    )


def generate_tri3_shear(base_dir, edge_length, shear_ratio):
    coords_final = tri_shear_vertices_final(edge_length)
    coords_initial = tri_shear_vertices_initial(edge_length, shear_ratio)
    generate_case(
        base_dir,
        "tri3_distort_shear",
        coords_initial,
        coords_final,
        np.array([[0, 1, 2]]),
    )


def generate_tri6_shear(base_dir, edge_length, shear_ratio):
    coords_vertices_final = tri_shear_vertices_final(edge_length)
    coords_vertices_initial = tri_shear_vertices_initial(edge_length, shear_ratio)
    edge_pairs = [(0, 1), (1, 2), (2, 0)]
    midsides_initial = build_edge_midpoints(coords_vertices_initial, edge_pairs)
    midsides_final = build_edge_midpoints(coords_vertices_final, edge_pairs)
    coords_initial = np.vstack([coords_vertices_initial, midsides_initial])
    coords_final = np.vstack([coords_vertices_final, midsides_final])
    generate_case(
        base_dir,
        "tri6_distort_shear",
        coords_initial,
        coords_final,
        np.array([[0, 1, 2, 3, 4, 5]]),
    )


def generate_quad4_shear(base_dir, edge_length, shear_ratio):
    coords_final = quad_vertices_final(edge_length)
    coords_initial = quad_shear_vertices_initial(edge_length, shear_ratio)
    generate_case(
        base_dir,
        "quad4_distort_shear",
        coords_initial,
        coords_final,
        np.array([[0, 1, 2, 3]]),
    )


def generate_quad8_shear(base_dir, edge_length, shear_ratio):
    coords_vertices_final = quad_vertices_final(edge_length)
    coords_vertices_initial = quad_shear_vertices_initial(edge_length, shear_ratio)
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    midsides_initial = build_edge_midpoints(coords_vertices_initial, edge_pairs)
    midsides_final = build_edge_midpoints(coords_vertices_final, edge_pairs)
    coords_initial = np.vstack([coords_vertices_initial, midsides_initial])
    coords_final = np.vstack([coords_vertices_final, midsides_final])
    generate_case(
        base_dir,
        "quad8_distort_shear",
        coords_initial,
        coords_final,
        np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
    )


def generate_quad9_shear(base_dir, edge_length, shear_ratio):
    coords_vertices_final = quad_vertices_final(edge_length)
    coords_vertices_initial = quad_shear_vertices_initial(edge_length, shear_ratio)
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    midsides_initial = build_edge_midpoints(coords_vertices_initial, edge_pairs)
    midsides_final = build_edge_midpoints(coords_vertices_final, edge_pairs)
    center_initial = np.mean(coords_vertices_initial, axis=0, keepdims=True)
    center_final = np.mean(coords_vertices_final, axis=0, keepdims=True)
    coords_initial = np.vstack([coords_vertices_initial, midsides_initial, center_initial])
    coords_final = np.vstack([coords_vertices_final, midsides_final, center_final])
    generate_case(
        base_dir,
        "quad9_distort_shear",
        coords_initial,
        coords_final,
        np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]]),
    )


def generate_tri6_bulge(base_dir, edge_length, time_steps):
    height = np.sqrt(3.0) * edge_length / 2.0
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [edge_length, 0.0, 0.0],
            [0.5 * edge_length, height, 0.0],
        ]
    )
    centroid = np.array([0.5 * edge_length, height / 3.0, 0.0])
    max_offset = BULGE_MIDSIDE_OFFSET_FACTOR * edge_length

    midside_directions = [
        get_midside_direction(vertices[0], vertices[1], centroid),
        get_midside_direction(vertices[1], vertices[2], centroid),
        get_midside_direction(vertices[2], vertices[0], centroid),
    ]

    midsides = np.array(
        [
            0.5 * (vertices[0] + vertices[1]) + max_offset * midside_directions[0],
            0.5 * (vertices[1] + vertices[2]) + max_offset * midside_directions[1],
            0.5 * (vertices[2] + vertices[0]) + max_offset * midside_directions[2],
        ]
    )
    midside_info = [
        (3, midside_directions[0]),
        (4, midside_directions[1]),
        (5, midside_directions[2]),
    ]

    coords = np.vstack([vertices, midsides])
    disp_x, disp_y, disp_z = build_disp_fields_bulge(
        coords,
        midside_info,
        time_steps,
        max_offset,
    )
    save_case(
        base_dir,
        "tri6_distort_bulge",
        coords,
        np.array([[0, 1, 2, 3, 4, 5]]),
        disp_x,
        disp_y,
        disp_z,
        compute_uvs(coords),
    )


def generate_tri6_tan(base_dir, edge_length, time_steps, tan_offset_factor):
    height = np.sqrt(3.0) * edge_length / 2.0
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [edge_length, 0.0, 0.0],
            [0.5 * edge_length, height, 0.0],
        ]
    )

    midside_tangents = [
        get_edge_tangent(vertices[0], vertices[1]),
        get_edge_tangent(vertices[1], vertices[2]),
        get_edge_tangent(vertices[2], vertices[0]),
    ]
    initial_offset = -tan_offset_factor * edge_length

    midsides = np.array(
        [
            0.5 * (vertices[0] + vertices[1]) + initial_offset * midside_tangents[0],
            0.5 * (vertices[1] + vertices[2]) + initial_offset * midside_tangents[1],
            0.5 * (vertices[2] + vertices[0]) + initial_offset * midside_tangents[2],
        ]
    )
    midside_info = [
        (3, midside_tangents[0]),
        (4, midside_tangents[1]),
        (5, midside_tangents[2]),
    ]

    coords = np.vstack([vertices, midsides])
    disp_x, disp_y, disp_z = build_disp_fields_tan(
        coords,
        midside_info,
        time_steps,
        edge_length,
        tan_offset_factor,
    )
    save_case(
        base_dir,
        "tri6_distort_tan",
        coords,
        np.array([[0, 1, 2, 3, 4, 5]]),
        disp_x,
        disp_y,
        disp_z,
        compute_uvs(coords),
    )


def generate_quad_bulge(base_dir, edge_length, time_steps, include_center):
    vertices = quad_vertices_final(edge_length)
    centroid = np.array([0.5 * edge_length, 0.5 * edge_length, 0.0])
    max_offset = BULGE_MIDSIDE_OFFSET_FACTOR * edge_length

    midside_directions = [
        get_midside_direction(vertices[0], vertices[1], centroid),
        get_midside_direction(vertices[1], vertices[2], centroid),
        get_midside_direction(vertices[2], vertices[3], centroid),
        get_midside_direction(vertices[3], vertices[0], centroid),
    ]

    midsides = np.array(
        [
            0.5 * (vertices[0] + vertices[1]) + max_offset * midside_directions[0],
            0.5 * (vertices[1] + vertices[2]) + max_offset * midside_directions[1],
            0.5 * (vertices[2] + vertices[3]) + max_offset * midside_directions[2],
            0.5 * (vertices[3] + vertices[0]) + max_offset * midside_directions[3],
        ]
    )

    midside_info = [
        (4, midside_directions[0]),
        (5, midside_directions[1]),
        (6, midside_directions[2]),
        (7, midside_directions[3]),
    ]

    node_list = [vertices[0], vertices[1], vertices[2], vertices[3]]
    node_list.extend([midsides[0], midsides[1], midsides[2], midsides[3]])
    if include_center:
        node_list.append(centroid)
    coords = np.array(node_list)

    disp_x, disp_y, disp_z = build_disp_fields_bulge(
        coords,
        midside_info,
        time_steps,
        max_offset,
    )
    mesh_name = "quad9_distort_bulge" if include_center else "quad8_distort_bulge"
    connect = np.array([list(range(9 if include_center else 8))])
    save_case(
        base_dir,
        mesh_name,
        coords,
        connect,
        disp_x,
        disp_y,
        disp_z,
        compute_uvs(coords),
    )


def generate_quad_tan(base_dir, edge_length, time_steps, tan_offset_factor, include_center):
    vertices = quad_vertices_final(edge_length)
    centroid = np.array([0.5 * edge_length, 0.5 * edge_length, 0.0])

    midside_tangents = [
        get_edge_tangent(vertices[0], vertices[1]),
        get_edge_tangent(vertices[1], vertices[2]),
        get_edge_tangent(vertices[2], vertices[3]),
        get_edge_tangent(vertices[3], vertices[0]),
    ]
    initial_offset = -tan_offset_factor * edge_length

    midsides = np.array(
        [
            0.5 * (vertices[0] + vertices[1]) + initial_offset * midside_tangents[0],
            0.5 * (vertices[1] + vertices[2]) + initial_offset * midside_tangents[1],
            0.5 * (vertices[2] + vertices[3]) + initial_offset * midside_tangents[2],
            0.5 * (vertices[3] + vertices[0]) + initial_offset * midside_tangents[3],
        ]
    )

    midside_info = [
        (4, midside_tangents[0]),
        (5, midside_tangents[1]),
        (6, midside_tangents[2]),
        (7, midside_tangents[3]),
    ]

    node_list = [vertices[0], vertices[1], vertices[2], vertices[3]]
    node_list.extend([midsides[0], midsides[1], midsides[2], midsides[3]])
    if include_center:
        node_list.append(centroid)
    coords = np.array(node_list)

    disp_x, disp_y, disp_z = build_disp_fields_tan(
        coords,
        midside_info,
        time_steps,
        edge_length,
        tan_offset_factor,
    )
    mesh_name = "quad9_distort_tan" if include_center else "quad8_distort_tan"
    connect = np.array([list(range(9 if include_center else 8))])
    save_case(
        base_dir,
        mesh_name,
        coords,
        connect,
        disp_x,
        disp_y,
        disp_z,
        compute_uvs(coords),
    )


def generate_tri3_rot(base_dir, edge_length, time_steps):
    coords = tri_shear_vertices_final(edge_length)
    centroid = np.mean(coords, axis=0)
    angles_deg = np.linspace(0.0, 360.0, time_steps)
    coords_frames = [rotate_points(coords, angle_deg, centroid) for angle_deg in angles_deg]
    disp_x, disp_y, disp_z = build_disp_fields_from_frame_coords(coords, coords_frames)
    save_case(
        base_dir,
        "tri3_distort_rot",
        coords,
        np.array([[0, 1, 2]]),
        disp_x,
        disp_y,
        disp_z,
        compute_uvs(coords),
    )


def generate_tri6_rot(base_dir, edge_length, time_steps):
    coords_vertices = tri_shear_vertices_final(edge_length)
    edge_pairs = [(0, 1), (1, 2), (2, 0)]
    midsides = build_edge_midpoints(coords_vertices, edge_pairs)
    coords = np.vstack([coords_vertices, midsides])
    centroid = np.mean(coords_vertices, axis=0)
    angles_deg = np.linspace(0.0, 360.0, time_steps)
    coords_frames = [rotate_points(coords, angle_deg, centroid) for angle_deg in angles_deg]
    disp_x, disp_y, disp_z = build_disp_fields_from_frame_coords(coords, coords_frames)
    save_case(
        base_dir,
        "tri6_distort_rot",
        coords,
        np.array([[0, 1, 2, 3, 4, 5]]),
        disp_x,
        disp_y,
        disp_z,
        compute_uvs(coords),
    )


def generate_quad_rot(base_dir, edge_length, time_steps, include_center):
    coords_vertices = quad_vertices_final(edge_length)
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    midsides = build_edge_midpoints(coords_vertices, edge_pairs)
    node_list = [coords_vertices[0], coords_vertices[1], coords_vertices[2], coords_vertices[3]]
    node_list.extend([midsides[0], midsides[1], midsides[2], midsides[3]])
    if include_center:
        node_list.append(np.mean(coords_vertices, axis=0))
    coords = np.array(node_list)
    centroid = np.mean(coords_vertices, axis=0)
    angles_deg = np.linspace(0.0, 360.0, time_steps)
    coords_frames = [rotate_points(coords, angle_deg, centroid) for angle_deg in angles_deg]
    disp_x, disp_y, disp_z = build_disp_fields_from_frame_coords(coords, coords_frames)
    mesh_name = "quad9_distort_rot" if include_center else "quad8_distort_rot"
    connect = np.array([list(range(9 if include_center else 8))])
    save_case(
        base_dir,
        mesh_name,
        coords,
        connect,
        disp_x,
        disp_y,
        disp_z,
        compute_uvs(coords),
    )


def generate_quad4_rot(base_dir, edge_length, time_steps):
    coords = quad_vertices_final(edge_length)
    centroid = np.mean(coords, axis=0)
    angles_deg = np.linspace(0.0, 360.0, time_steps)
    coords_frames = [rotate_points(coords, angle_deg, centroid) for angle_deg in angles_deg]
    disp_x, disp_y, disp_z = build_disp_fields_from_frame_coords(coords, coords_frames)
    save_case(
        base_dir,
        "quad4_distort_rot",
        coords,
        np.array([[0, 1, 2, 3]]),
        disp_x,
        disp_y,
        disp_z,
        compute_uvs(coords),
    )


def main():
    if BULGE_TIME_STEPS % 2 == 0:
        print("Warning: BULGE_TIME_STEPS should be odd to include the square case.")
    if TAN_TIME_STEPS % 2 == 0:
        print("Warning: TAN_TIME_STEPS should be odd to include the exact midpoint case.")
    if STEPS < 2:
        raise ValueError("STEPS must be at least 2")

    base_dir = "data/edge"

    generate_tri6_bulge(base_dir, EDGE_LENG, BULGE_TIME_STEPS)
    generate_quad_bulge(base_dir, EDGE_LENG, BULGE_TIME_STEPS, False)
    generate_quad_bulge(base_dir, EDGE_LENG, BULGE_TIME_STEPS, True)

    generate_tri6_tan(base_dir, EDGE_LENG, TAN_TIME_STEPS, TAN_OFFSET_FACTOR)
    generate_quad_tan(base_dir, EDGE_LENG, TAN_TIME_STEPS, TAN_OFFSET_FACTOR, False)
    generate_quad_tan(base_dir, EDGE_LENG, TAN_TIME_STEPS, TAN_OFFSET_FACTOR, True)

    generate_tri3_stretch(base_dir, EDGE_LENG, STRETCH)
    generate_tri6_stretch(base_dir, EDGE_LENG, STRETCH)
    generate_quad4_stretch(base_dir, EDGE_LENG, STRETCH)
    generate_quad8_stretch(base_dir, EDGE_LENG, STRETCH)
    generate_quad9_stretch(base_dir, EDGE_LENG, STRETCH)

    generate_tri3_shear(base_dir, EDGE_LENG, SHEAR)
    generate_tri6_shear(base_dir, EDGE_LENG, SHEAR)
    generate_quad4_shear(base_dir, EDGE_LENG, SHEAR)
    generate_quad8_shear(base_dir, EDGE_LENG, SHEAR)
    generate_quad9_shear(base_dir, EDGE_LENG, SHEAR)

    generate_tri3_rot(base_dir, EDGE_LENG, ROT_TIME_STEPS)
    generate_tri6_rot(base_dir, EDGE_LENG, ROT_TIME_STEPS)
    generate_quad4_rot(base_dir, EDGE_LENG, ROT_TIME_STEPS)
    generate_quad_rot(base_dir, EDGE_LENG, ROT_TIME_STEPS, False)
    generate_quad_rot(base_dir, EDGE_LENG, ROT_TIME_STEPS, True)

    print(
        "Generated distortion edge data: "
        f"distort_bulge ({BULGE_TIME_STEPS} steps), "
        f"distort_tan ({TAN_TIME_STEPS} steps), "
        f"distort_stretch ({STEPS} steps), "
        f"distort_shear ({STEPS} steps), "
        f"distort_rot ({ROT_TIME_STEPS} steps), "
        f"ELEM_ROT ({ELEM_ROT} deg)."
    )


if __name__ == "__main__":
    main()

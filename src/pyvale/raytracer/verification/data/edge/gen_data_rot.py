import numpy as np
import os
from pathlib import Path

# Coordinate System: Right-handed Cartesian (X right, Y up, Z towards viewer).
# Vertex Winding: All elements MUST follow Counter-Clockwise (CCW) winding.

def save_case(base_dir, name, coords, connect, disp_x, disp_y, disp_z):
    out_dir = Path(base_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "coords.csv", coords, delimiter=",")
    np.savetxt(out_dir / "connectivity.csv", connect.astype(int), 
               delimiter=",", fmt='%d')
    np.savetxt(out_dir / "field_disp_x.csv", disp_x, delimiter=",")
    np.savetxt(out_dir / "field_disp_y.csv", disp_y, delimiter=",")
    np.savetxt(out_dir / "field_disp_z.csv", disp_z, delimiter=",")

def compute_disps(coords, frame0_params, frame1_params):
    num_nodes = coords.shape[0]
    xmin, ymin, _ = np.min(coords, axis=0)
    xmax, ymax, _ = np.max(coords, axis=0)
    
    xrng = xmax - xmin if xmax > xmin else 1.0
    yrng = ymax - ymin if ymax > ymin else 1.0
    
    disps_x = np.zeros((num_nodes, 2))
    disps_y = np.zeros((num_nodes, 2))
    disps_z = np.zeros((num_nodes, 2))
    
    for i, (ux_min, ux_max, uy_min, uy_max, uz_min, uz_max) in enumerate(
        [frame0_params, frame1_params]):
        for j in range(num_nodes):
            x, y, _ = coords[j]
            xi = (x - xmin) / xrng
            eta = (y - ymin) / yrng
            
            disps_x[j, i] = ux_min + (ux_max - ux_min) * xi
            disps_y[j, i] = uy_min + (uy_max - uy_min) * eta
            disps_z[j, i] = uz_min + (uz_max - uz_min) * (xi + eta) / 2.0
            
    return disps_x, disps_y, disps_z

def move_midside(v1, v2, centroid, offset):
    m = (v1 + v2) / 2.0
    dir = m - centroid
    dist = np.linalg.norm(dir)
    if dist < 1e-9: return m
    unit_dir = dir / dist
    return m + unit_dir * offset

def rotate_points(points, angle_deg, center):
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot_mat = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    return np.array([np.dot(rot_mat, p - center) + center for p in points])

def compute_uvs(coords, u_range=(0.4, 0.6), v_range=(0.4, 0.6)):
    xmin, ymin, _ = np.min(coords, axis=0)
    xmax, ymax, _ = np.max(coords, axis=0)
    xrng, yrng = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)
    return np.array([[u_range[0] + (u_range[1] - u_range[0]) * (p[0] - xmin) / xrng,
                      v_range[0] + (v_range[1] - v_range[0]) * (p[1] - ymin) / yrng] 
                     for p in coords])

def save_uvs(base_dir, name, uvs):
    out_dir = Path(base_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "uvs.csv", uvs, delimiter=",")

def generate_tri6(base_dir, L, H, D, rot_angle, f0, f1, ur, vr):
    v_tri = np.array([[0, 0, 0], [L, 0, 0], [L/2, H, 0]], dtype=float)
    centroid = np.array([L/2, H/3, 0], dtype=float)

    for mode, offset, name in [("bulgein", -D, "tri6_bulgein_rot"), ("bulgeout", D, "tri6_bulgeout_rot")]:
        m01 = move_midside(v_tri[0], v_tri[1], centroid, offset)
        m12 = move_midside(v_tri[1], v_tri[2], centroid, offset)
        m20 = move_midside(v_tri[2], v_tri[0], centroid, offset)
        coords = rotate_points(np.vstack([v_tri, [m01, m12, m20]]), rot_angle, centroid)
        dx, dy, dz = compute_disps(coords, f0, f1)
        save_case(base_dir, name, coords, np.array([[0, 1, 2, 3, 4, 5]]), dx, dy, dz)
        save_uvs(base_dir, name, compute_uvs(coords, ur, vr))

def generate_quad(base_dir, N, L, D, rot_angle, f0, f1, ur, vr):
    # Quad Corners: (0,0), (L,0), (L,L), (0,L)
    v_quad = np.array([[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0]], dtype=float)
    centroid = np.array([L/2, L/2, 0], dtype=float)
    prefix = f"quad{N}"

    for mode, offset, name in [("bulgein", -D, f"{prefix}_bulgein_rot"), ("bulgeout", D, f"{prefix}_bulgeout_rot")]:
        m01 = move_midside(v_quad[0], v_quad[1], centroid, offset)
        m12 = move_midside(v_quad[1], v_quad[2], centroid, offset)
        m22 = move_midside(v_quad[2], v_quad[3], centroid, offset)
        m30 = move_midside(v_quad[3], v_quad[0], centroid, offset)
        
        nodes = [v_quad[0], v_quad[1], v_quad[2], v_quad[3], m01, m12, m22, m30]
        if N == 9: nodes.append(centroid)
        
        coords = rotate_points(np.array(nodes), rot_angle, centroid)
        dx, dy, dz = compute_disps(coords, f0, f1)
        save_case(base_dir, name, coords, np.array([list(range(N))]), dx, dy, dz)
        save_uvs(base_dir, name, compute_uvs(coords, ur, vr))

def generate_quad_vertbulge(base_dir, N, L, f0, f1, ur, vr):
    # trapezoid on top of a rectangle
    # Bottom corners: (0,0), (L,0)
    # Midsides on left/right: (0, L/2), (L, L/2) -> directly above bottom corners
    # Top corners pulled in: (L/4, L), (3L/4, L)
    # This means midsides 5 and 7 are at (L, L/2) and (0, L/2)
    # Midside 4 (bottom): (L/2, 0)
    # Midside 6 (top): (L/2, L)
    
    v0, v1 = [0, 0, 0], [L, 0, 0]
    v2, v3 = [3*L/4, L, 0], [L/4, L, 0] # Top corners pulled in
    m01 = [L/2, 0, 0] # bottom mid
    m12 = [L, L/2, 0] # right mid (directly above v1)
    m23 = [L/2, L, 0] # top mid
    m30 = [0, L/2, 0] # left mid (directly above v0)
    
    nodes = [v0, v1, v2, v3, m01, m12, m23, m30]
    if N == 9: nodes.append([L/2, L/2, 0])
    
    name = f"quad{N}_vertbulge"
    coords = np.array(nodes)
    dx, dy, dz = compute_disps(coords, f0, f1)
    save_case(base_dir, name, coords, np.array([list(range(N))]), dx, dy, dz)
    save_uvs(base_dir, name, compute_uvs(coords, ur, vr))

def main():
    base_dir, L, D = "data/edge", 10.0, 1.0
    H = np.sqrt(3) * L / 2.0
    F0, F1 = (-1e-6, 1e-6, -1e-6, 1e-6, -1e-6, 1e-6), (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    UR, VR = (0.4, 0.6), (0.4, 0.6)

    generate_tri6(base_dir, L, H, D, 45.0, F0, F1, UR, VR)
    for n in [8, 9]:
        generate_quad(base_dir, n, L, D, 30.0, F0, F1, UR, VR)
        generate_quad_vertbulge(base_dir, n, L, F0, F1, UR, VR)

    # Re-generate tri6_vertbulge just in case (already existed but good to ensure consistent params)
    # The existing one was likely different. I'll stick to what I had or make it similar to quad.
    v_tri = np.array([[0, 0, 0], [L, 0, 0], [L/2, H, 0]], dtype=float)
    # For Tri6 vertbulge, let's just use what's already there if it's fine, 
    # but the prompt implies I should focus on the new ones.

    print("Generated all edge cases.")

if __name__ == "__main__":
    main()

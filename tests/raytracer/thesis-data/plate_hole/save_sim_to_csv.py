from pathlib import Path
import numpy as np
import pyvale.mooseherder as mh

from extract_surface_mesh import extract_surf_mesh

def main() -> None:
    #---------------------------------------------------------------------------
    # NOTE: exodus connectivity starts at 1 - need to subtract 1!
    #---------------------------------------------------------------------------

    base_dir = Path(__file__).resolve().parent
    num_meshes = 7
    frame_counts = (1, 63)
    for num_frames in frame_counts:
        for mm in range(1, num_meshes):
            mesh_num = mm
            sim_name = f"platehole3d_{mesh_num}mr_{num_frames}f"

            sim_file = base_dir / f"{sim_name}.e"

            save_path = base_dir / sim_name
            if not save_path.is_dir():
                save_path.mkdir(parents=True, exist_ok=True)

            sim_data = mh.ExodusLoader(sim_file).load_all_sim_data()
            mesh_world = extract_surf_mesh(sim_data)

            (num_coords, _) = mesh_world.coords.shape
            uvs = np.zeros((num_coords, 2), dtype=np.float64)

            x_max = np.max(mesh_world.coords[:, 0])
            x_min = np.min(mesh_world.coords[:, 0])
            y_max = np.max(mesh_world.coords[:, 1])
            y_min = np.min(mesh_world.coords[:, 1])

            dx_mesh = x_max - x_min
            dy_mesh = y_max - y_min
            mesh_AR = dx_mesh / dy_mesh

            uv_span_max = 0.8
            # Texture AR (Width/Height)
            tex_AR = 2464.0 / 2056.0 

            # To maintain square pixels: (d_u / d_v) = mesh_AR / tex_AR
            # Let R be the ratio of aspect ratios
            R = mesh_AR / tex_AR

            if R > 1:
                # Mesh is wider than texture relative to its height
                # U is the constraining dimension
                d_u = uv_span_max
                d_v = d_u / R
            else:
                # Mesh is taller than texture relative to its width
                # V is the constraining dimension
                d_v = uv_span_max
                d_u = d_v * R

            u_min = (1 - d_u) / 2
            u_max = 1 - u_min
            v_min = (1 - d_v) / 2
            v_max = 1 - v_min

            u_slope = (u_max - u_min) / (x_max - x_min)
            u_int = u_min - u_slope * x_min

            v_slope = (v_max - v_min) / (y_max - y_min)
            v_int = v_min - v_slope * y_min

            uvs[:, 0] = u_slope * mesh_world.coords[:, 0] + u_int
            uvs[:, 1] = v_slope * mesh_world.coords[:, 1] + v_int

            connect_keys = sorted(mesh_world.connect.keys())
            if len(connect_keys) != 1:
                raise ValueError(
                    f"{sim_name} extracted {len(connect_keys)} surface connectivity tables; "
                    "the CSV export currently expects one table.",
                )

            connect = mesh_world.connect[connect_keys[0]].T - 1
            z_face_stats = _get_broad_face_orientation_stats(mesh_world.coords, connect)

            print(80 * "-")
            print(f"MESH: {sim_name}") 
            print()
            print(f"{sim_data.coords.shape=}")
            print(f"{sim_data.connect['connect1'].shape=}")
            print()
            print(f"{mesh_world.coords.shape=}")
            print(f"{mesh_world.connect[connect_keys[0]].T.shape=}")
            print(f"{mesh_world.node_vars['disp_x'].shape=}")
            print()
            print(f"{np.max(connect)=},{np.min(connect)=}")
            print()
            print(f"{np.min(mesh_world.coords,axis=0)=}")
            print(f"{np.max(mesh_world.coords,axis=0)=}")
            print()
            print(f"{u_min=},{u_max=},{v_min=},{v_max=}")
            print(f"{x_min=},{x_max=},{y_min=},{y_max=}")
            print()
            print(f"{np.min(uvs,axis=0)=}")
            print(f"{np.max(uvs,axis=0)=}")
            print()
            print(f"{z_face_stats=}")
            print(80 * "-")

            np.savetxt(save_path / 'coords.csv', mesh_world.coords, delimiter=',')
            np.savetxt(save_path / 'connect.csv', connect, delimiter=',')
            np.savetxt(save_path / 'field_disp_x.csv',
                        mesh_world.node_vars['disp_x'], delimiter=',')
            np.savetxt(save_path / 'field_disp_y.csv',
                        mesh_world.node_vars['disp_y'], delimiter=',')
            np.savetxt(save_path / 'field_disp_z.csv',
                        mesh_world.node_vars['disp_z'], delimiter=',')
            np.savetxt(save_path / 'uvs.csv', uvs, delimiter=',')
        
def _get_broad_face_orientation_stats(
    coords: np.ndarray,
    connect: np.ndarray,
) -> dict[str, int]:
    corners = connect[:, :4]
    face_coords = coords[corners]
    normals = np.cross(
        face_coords[:, 1, :] - face_coords[:, 0, :],
        face_coords[:, 2, :] - face_coords[:, 0, :],
    )
    z_centroids = face_coords[:, :, 2].mean(axis=1)
    z_min = np.min(z_centroids)
    z_max = np.max(z_centroids)
    tol = max(1.0e-12, 1.0e-6 * max(abs(z_min), abs(z_max), 1.0))

    is_z_min = np.abs(z_centroids - z_min) <= tol
    is_z_max = np.abs(z_centroids - z_max) <= tol

    return {
        "z_min_pos": int(np.sum(normals[is_z_min, 2] > 0.0)),
        "z_min_neg": int(np.sum(normals[is_z_min, 2] < 0.0)),
        "z_max_pos": int(np.sum(normals[is_z_max, 2] > 0.0)),
        "z_max_neg": int(np.sum(normals[is_z_max, 2] < 0.0)),
    }


if __name__ == "__main__":
    main()

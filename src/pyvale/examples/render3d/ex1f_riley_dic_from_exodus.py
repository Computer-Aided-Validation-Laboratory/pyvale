"""Render a stereo DIC experiment directly from an Exodus result file."""

from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
from pyvale import render
from pyvale.mooseherder import ExodusLoader

# %%
# 1. Load Exodus data and build a textured surface mesh
# ------------------------------------------------------------
simulation = ExodusLoader(
    dataset.riley_platehole_exodus_path(),
    enforce_convention=True,
).load_all_sim_data()
texture = riley.load_texture_u8(dataset.riley_speckle_texture_path())
surface_mesh = render.mesh3d_from_simdata(
    simulation,
    shader=None,
    displacement_keys=("disp_x", "disp_y", "disp_z"),
)
surface_mesh.shader = render.RileyTextureShader(
    uvs=riley.project_uvs_planar_centered(
        surface_mesh.coords,
        (2464, 2056),
        uv_span_max=0.8,
    ),
    texture=texture,
)
surface_mesh.displacements = surface_mesh.displacements[[0, -1]]

# %%
# 2. Create and position the stereo cameras
# ------------------------------------------------------------
pixels_num = np.array((2464, 2056))
pixels_size = np.array((3.45e-6, 3.45e-6))
focal_length = 50.0e-3
roi_centre = np.asarray(riley.roi_cent_from_coords(surface_mesh.coords))


def make_camera(angle_degrees: float) -> render.Camera:
    """Create one camera aimed at the extracted surface."""
    rotation = Rotation.from_euler("y", angle_degrees, degrees=True)
    position = riley.pos_fill_frame_from_rot(
        surface_mesh.coords,
        tuple(pixels_num),
        tuple(pixels_size),
        focal_length,
        tuple(rotation.as_euler("xyz")),
        0.65,
    )
    return render.Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=np.asarray(position),
        rot_world=rotation,
        roi_cent_world=roi_centre,
        focal_length=focal_length,
        subsample=2,
    )


cameras = [make_camera(0.0), make_camera(20.0)]

# %%
# 3. Configure and build the renderer
# ------------------------------------------------------------
config = riley.create_raster_config(
    num_frames=surface_mesh.displacements.shape[0],
    total_threads=8,
    save_strategy=riley.SaveStrategy.disk,
)
config.background_value = 128.0
config.save_scaling = riley.ScaleStrategy.none
output_dir = Path.cwd() / "pyvale-output" / "render-riley-exodus"
renderer = render.Riley(config, output_dir)

# %%
# 4. Build the scene and render the extracted surface
# ------------------------------------------------------------
result = renderer.render(render.Scene3D(meshes=[surface_mesh], cameras=cameras))
print(f"Rendered Exodus-driven DIC images to {output_dir}")
print(f"{result.images=}")

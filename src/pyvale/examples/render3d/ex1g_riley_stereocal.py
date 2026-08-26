"""Render a moving stereo-calibration target with Riley."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import riley
from riley.pydemos.common import evenly_spaced_frame_indices

import pyvale.data as dataset
from pyvale import render

# %%
# 1. Load the moving calibration target and its texture
# ------------------------------------------------------------
coords, connectivity, uvs, displacements = riley.load_sim_csvs(
    dataset.riley_stereocal_case_path()
)
frame_indices = evenly_spaced_frame_indices(
    displacements.shape[0],
    8,
)
displacements = displacements[frame_indices]
texture = riley.load_texture_u8(dataset.riley_cal_target_texture_path())

# %%
# 2. Load the stereo pair rendered by the DIC example and re-aim it
# ------------------------------------------------------------
# The cameras come from ex1e (run first), matching Riley's own demo chain.
dicuq_dir = Path.cwd() / "pyvale-output" / "render-riley-dicuq"
camera_0, camera_1 = riley.load_stereo_pair(
    str(dicuq_dir),
    "stereo_data_opengl.csv",
)

roi_pos = np.asarray(riley.roi_cent_from_coords(coords), dtype=np.float64)
target_roi = np.asarray(camera_0.roi_cent_world, dtype=np.float64)
coords = np.ascontiguousarray(coords + (target_roi - roi_pos))
roi_pos = riley.roi_cent_from_coords(coords)
camera_0 = replace(camera_0, roi_cent_world=roi_pos)
camera_1 = replace(camera_1, roi_cent_world=roi_pos)

# %%
# 3. Build the textured calibration-target mesh
# ------------------------------------------------------------
mesh = render.Mesh3D(
    element_type=render.EElementType.TRI3,
    coords=coords,
    connectivity=connectivity,
    shader=render.RileyTextureShader(uvs=uvs, texture=texture),
    displacements=displacements,
)

# %%
# 4. Configure and build the renderer
# ------------------------------------------------------------
config = riley.create_raster_config(
    num_frames=displacements.shape[0],
    total_threads=8,
    save_strategy=riley.SaveStrategy.disk,
)
config.background_value = 128.0
output_dir = Path.cwd() / "pyvale-output" / "render-riley-stereocal"
renderer = render.Riley(config, output_dir)

# %%
# 5. Build the scene and render every calibration-target pose
# ------------------------------------------------------------
result = renderer.render(
    render.Scene3D(meshes=[mesh], cameras=[camera_0, camera_1]),
)
print(f"Rendered stereo-calibration images to {output_dir}")
print(f"{result.images=}")

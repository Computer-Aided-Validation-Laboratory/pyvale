# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================
"""Shared rendering setup for documented Render UV examples."""

from pathlib import Path

import riley

from pyvale import render


def render_uv_example(
    mesh: render.Mesh3D,
    camera: render.Camera,
    output_dir: Path,
) -> render.RenderResult:
    """Render one prepared textured mesh and camera for the UV examples."""

    config = riley.create_raster_config(
        num_frames=1,
        total_threads=4,
        save_strategy=riley.SaveStrategy.disk,
    )
    config.background_value = 128.0
    renderer = render.Riley(config, output_dir)
    scene = render.Scene3D(meshes=[mesh], cameras=[camera])
    return renderer.render(scene)


__all__ = ["render_uv_example"]

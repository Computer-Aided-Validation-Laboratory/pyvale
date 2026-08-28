# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Integration tests for consolidated render setup helper workflows."""

import numpy as np

from scipy.spatial.transform import Rotation

import pyvale.render as render
from pyvale.dataio.simdata import SimData


def _make_quad_simdata() -> SimData:
    coords = np.array([
        [-10.0, -5.0, 0.0],
        [10.0, -5.0, 0.0],
        [10.0, 5.0, 0.0],
        [-10.0, 5.0, 0.0],
    ])
    connect = {"surf": np.array([[0, 1, 2, 3]], dtype=np.uintp)}
    return SimData(coords=coords, connect=connect)


def test_workflow_single_planar_specimen() -> None:
    """SimData -> mesh3d -> center_at -> UVs -> cam_look_at -> cam_frame."""
    sim = _make_quad_simdata()
    mesh = render.mesh3d_from_simdata(sim, shader=None)
    centered_mesh = render.mesh_center_at(mesh, np.zeros(3))

    np.testing.assert_allclose(
        render.mesh_center(centered_mesh), np.zeros(3), atol=1.0e-12
    )

    uvs = render.uv_project_planar(
        centered_mesh.coords,
        plane=render.EUVPlane.XY,
        texture_shape=(512, 512),
    )
    assert uvs.shape == (4, 2)
    assert np.all(uvs >= 0.0) and np.all(uvs <= 1.0)

    cam = render.Camera(
        pixels_num=np.array([512, 512]),
        pixels_size=np.array([0.02, 0.02]),
        pos_world=np.array([0.0, 0.0, 50.0]),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=50.0e-3,
    )

    cam = render.cam_look_at(cam, target=render.mesh_center(centered_mesh))
    cam = render.cam_frame_mesh(cam, centered_mesh, fill=0.9)

    scene = render.Scene3D(meshes=[centered_mesh], cameras=[cam])
    assert len(scene.meshes) == 1
    assert len(scene.cameras) == 1


def test_workflow_multi_specimen_grid() -> None:
    """scene_arrange_grid -> scene_bounds -> cam_frame_scene."""
    sim = _make_quad_simdata()
    meshes = [render.mesh3d_from_simdata(sim, shader=None) for _ in range(4)]

    grid_meshes = render.scene_arrange_grid(
        meshes, columns=2, spacing=np.array((5.0, 5.0)), center=True
    )
    low, high = render.scene_bounds(grid_meshes)
    assert low[0] < high[0]
    assert low[1] < high[1]

    cam = render.Camera(
        pixels_num=np.array([1024, 1024]),
        pixels_size=np.array([0.02, 0.02]),
        pos_world=np.array([0.0, 0.0, 100.0]),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=50.0e-3,
    )
    cam = render.cam_frame_scene(cam, grid_meshes, fill=0.95)
    assert cam.pos_world[2] > 0.0


def test_workflow_stereo_setup() -> None:
    """cam_frame_mesh -> stereo_build_symmetric."""
    sim = _make_quad_simdata()
    mesh = render.mesh3d_from_simdata(sim, shader=None)

    cam = render.Camera(
        pixels_num=np.array([512, 512]),
        pixels_size=np.array([0.02, 0.02]),
        pos_world=np.array([0.0, 0.0, 50.0]),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=50.0e-3,
    )
    cam = render.cam_frame_mesh(cam, mesh, fill=0.8)
    cam_0, cam_1 = render.stereo_build_symmetric(cam, convergence_degrees=15.0)

    assert cam_0.pos_world[0] < cam_1.pos_world[0]
    assert np.isclose(
        cam_0.pos_world[2], cam_1.pos_world[2]
    )

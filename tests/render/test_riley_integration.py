# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Real Riley integration and golden-regression tests."""

from pathlib import Path

import numpy as np
import pytest
import riley
from scipy.spatial.transform import Rotation

from pyvale import render
from pyvale.verif import renderverif
from pyvale.verif.renderverif import assert_render_allclose


def test_common_camera_reorders_scipy_angles_for_riley() -> None:
    """Riley receives its native Z-Y-X angle storage order."""
    camera = render.Camera(
        pixels_num=np.array((32, 32)),
        pixels_size=np.array((0.02, 0.02)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.from_euler(
            "xyz",
            (10.0, 20.0, 30.0),
            degrees=True,
        ),
        roi_cent_world=np.zeros(3),
        focal_length=1.0,
    )

    native = render.to_riley_camera(camera)

    np.testing.assert_allclose(
        native.rot_world,
        np.radians((30.0, 20.0, 10.0)),
    )


def test_riley_returns_canonical_image_layout() -> None:
    """Riley's field-major buffer is normalised by the pyvale adapter."""
    mesh = riley.Mesh(
        riley.MeshType.tri3,
        np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        np.array(((0, 1, 2),)),
        None,
        riley.MeshShaderSet(mono=riley.FunctionShader(
            riley.FuncShaderBuiltin.constant,
            coord_mode=riley.FuncCoordMode.world_reference,
        )),
    )
    camera = render.Camera(
        pixels_num=np.array((32, 32)),
        pixels_size=np.array((0.02, 0.02)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=1.0,
    )
    config = riley.RasterConfig(
        parallel=1,
        save_strategy=riley.SaveStrategy.memory,
    )
    result = render.Riley(config).render(render.Scene3D([mesh], [camera]))
    assert result.images is not None
    assert result.images.shape == (1, 1, 32, 32, 1)


def test_common_mesh_matches_native_riley_mesh() -> None:
    """Mesh3D conversion preserves native Riley rendering exactly."""
    coords = np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)))
    connectivity = np.array(((0, 1, 2),))
    native_mesh = riley.Mesh(
        riley.MeshType.tri3,
        coords,
        connectivity,
        None,
        riley.MeshShaderSet(mono=riley.FunctionShader(
            riley.FuncShaderBuiltin.checker,
            coord_mode=riley.FuncCoordMode.world_reference,
        )),
    )
    common_mesh = render.Mesh3D(
        element_type=render.EElemType.TRI3,
        coords=coords,
        connectivity=connectivity,
        shader=riley.MeshShaderSet(mono=riley.FunctionShader(
            builtin=riley.FuncShaderBuiltin.checker,
            coord_mode=riley.FuncCoordMode.world_reference,
        )),
    )
    camera = render.Camera(
        pixels_num=np.array((32, 32)),
        pixels_size=np.array((0.02, 0.02)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=1.0,
    )

    def render_mesh(mesh: object) -> np.ndarray:
        config = riley.RasterConfig(
            parallel=1,
            save_strategy=riley.SaveStrategy.memory,
        )
        result = render.Riley(config).render(render.Scene3D([mesh], [camera]))
        assert result.images is not None
        return result.images

    np.testing.assert_array_equal(
        render_mesh(common_mesh),
        render_mesh(native_mesh),
    )


def test_riley_rabbit_multimesh_golden_regression() -> None:
    """The Riley rabbit multi-mesh scene remains deterministic end-to-end."""
    result = render.Riley(renderverif.riley_memory_config()).render(
        renderverif.riley_rabbit_scene(),
    )
    assert result.images is not None
    golden_path = Path(__file__).parent / "gold_riley" / "rabbits.npy"
    assert_render_allclose(
        result.images,
        np.load(golden_path),
        "riley_rabbit_multimesh",
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "format_", (riley.RenderFormat.rgb, riley.RenderFormat.multifield),
)
def test_channels_multiframe_multicamera_matches_native(
    format_: riley.RenderFormat,
) -> None:
    """Channel values and all five axes survive the common adapter."""
    field = np.empty((3, 2, 3), dtype=np.float64)
    field[:, 0, :] = (0.2, 0.4, 0.6)
    field[:, 1, :] = (0.3, 0.5, 0.7)
    source = riley.NodalShader(
        field, bits=-1, scaling_type=riley.ScaleStrategy.none,
    )
    shaders = (riley.MeshShaderSet(rgb=source)
               if format_ == riley.RenderFormat.rgb
               else riley.MeshShaderSet(multifield=source))
    mesh = render.Mesh3D(
        element_type=render.EElemType.TRI3,
        coords=np.array(((-2., -2., 0.), (2., -2., 0.), (0., 2., 0.))),
        connectivity=np.array(((0, 1, 2),)),
        shader=shaders,
    )
    camera = render.Camera(
        pixels_num=np.array((24, 16)), pixels_size=np.array((0.02, 0.02)),
        pos_world=np.array((0., 0., 2.)), rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3), focal_length=1.,
        render_format=format_, render_channels=3,
    )
    config = riley.RasterConfig(parallel=1, save_strategy=riley.SaveStrategy.memory)
    cameras = [camera, camera, camera]
    actual = render.Riley(config).render(render.Scene3D([mesh], cameras)).images
    native = riley.raster(
        render.to_riley_mesh(mesh),
        [render.to_riley_camera(c) for c in cameras],
        config,
    )
    assert actual.shape == (2, 3, 16, 24, 3)
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, native.transpose(1, 0, 3, 4, 2))
    np.testing.assert_allclose(
        actual[:, 0, 8, 12], ((0.2, 0.4, 0.6), (0.3, 0.5, 0.7)),
    )


def test_camera_settings_reach_riley() -> None:
    """Extended distortion, PSF and save settings cross the boundary intact."""
    camera = render.Camera(
        pixels_num=np.array((24, 16)), pixels_size=np.array((0.02, 0.02)),
        pos_world=np.array((0., 0., 2.)), rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3), focal_length=1.,
        distortion_s1=0.01, distortion_tau_x=0.02,
        psf_separable=False, background_value=0.125,
        render_format=riley.RenderFormat.multifield, render_channels=4,
        output_files=(riley.ImageSaveOpts(format=riley.ImageFormat.tiff, file_bits=16),),
    )
    native = render.to_riley_camera(camera)
    assert native.distort_s1 == 0.01
    assert native.distort_tau_x == 0.02
    assert not native.psf_separable
    assert native.background_value == 0.125
    assert native.render_channels == 4
    assert native.output_files == camera.output_files


def test_disk_and_memory_save_modes(tmp_path: Path) -> None:
    """Per-camera output settings work with each native save strategy."""
    mesh = render.Mesh3D(
        element_type=render.EElemType.TRI3,
        coords=np.array(((-1., -1., 0.), (1., -1., 0.), (0., 1., 0.))),
        connectivity=np.array(((0, 1, 2),)),
        shader=riley.MeshShaderSet(mono=riley.FunctionShader(
            riley.FuncShaderBuiltin.constant,
        )),
        normal_type=riley.NormalType.averaged,
    )
    assert render.to_riley_mesh(mesh).normal_type == riley.NormalType.averaged
    camera = render.Camera(
        pixels_num=np.array((24, 16)), pixels_size=np.array((0.02, 0.02)),
        pos_world=np.array((0., 0., 2.)), rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3), focal_length=1.,
        output_files=(riley.ImageSaveOpts(format=riley.ImageFormat.tiff, file_bits=16),),
    )
    for strategy in (
        riley.SaveStrategy.memory, riley.SaveStrategy.disk,
        riley.SaveStrategy.both,
    ):
        output = tmp_path / strategy.name
        result = render.Riley(riley.RasterConfig(
            parallel=1, save_strategy=strategy,
        ), output).render(render.Scene3D([mesh], [camera]))
        assert (result.images is not None) == (strategy != riley.SaveStrategy.disk)
        assert bool(list(output.glob('*.tiff'))) == (strategy != riley.SaveStrategy.memory)

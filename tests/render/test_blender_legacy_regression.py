"""Legacy Blender gold regressions through the unified renderer API."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.blender as legacy_blender
import pyvale.render as render
import pyvale.verif.renderverif as renderverif
from pyvale.render.blender.adapter import _triangulate_mesh_for_blender
from pyvale.sensorsim.simtools import centre_mesh_nodes

from pyvale.verif.renderverif import assert_render_allclose


pytestmark = [
    pytest.mark.skipif(
        not render.blender_available(),
        reason="Blender requires Python 3.13 and the optional Blender extra.",
    ),
    pytest.mark.filterwarnings(
        "ignore:Blender support is verified only for Tri3 meshes",
    ),
]

_GOLD = Path(__file__).parents[1] / "blender" / "2D_gold"


def _camera(pixels: tuple[int, int] = (20, 20)) -> render.Camera:
    return render.Camera(
        pixels_num=np.asarray(pixels),
        pixels_size=np.array((0.00345, 0.00345)),
        pos_world=np.array((0.0, 0.0, 500.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=15.0,
    )


def _mesh(camera: render.Camera) -> render.Mesh3D:
    sim_data = renderverif.scaled_mechanical_2d()
    resolution = (
        camera.pixels_size[0] * camera.pos_world[2] / camera.focal_length
    )
    shader = render.BlenderTextureShader(
        dataset.dic_pattern_5mpx_path(), resolution
    )
    return render.mesh3d_from_simdata(sim_data, shader, ("disp_x", "disp_y"))


def _render(
    tmp_path: Path,
    camera: render.Camera,
    light_energy: float = 1.0,
    samples: int = 2,
    bounces: int = 12,
    engine: render.EBlenderEngine = render.EBlenderEngine.CYCLES,
) -> np.ndarray:
    renderer = render.Blender(
        render.BlenderConfig(
            tmp_path,
            engine=engine,
            samples=samples,
            max_bounces=bounces,
            threads=1,
        )
    )
    result = renderer.render(
        render.Scene3D(
            (_mesh(camera),),
            (camera,),
            (
                render.Light(
                    render.ELightType.POINT,
                    np.array((0.0, 0.0, 400.0)),
                    np.zeros(3),
                    light_energy,
                ),
            ),
        )
    )
    assert result.images is not None
    return result.images[0, 0, :, :, 0]


@pytest.mark.parametrize(
    ("energy", "gold"),
    ((0.5, "half_watt_lighting"), (3.0, "three_watt_lighting")),
)
def test_legacy_lighting_gold(tmp_path: Path, energy: float, gold: str) -> None:
    """Common lights reproduce legacy point-light intensity images."""
    assert_render_allclose(
        _render(tmp_path, _camera(), light_energy=energy),
        np.load(_GOLD / f"{gold}.npy"),
        gold,
        rtol=0.0,
        atol=2.0,
    )


@pytest.mark.parametrize(
    ("pixels", "gold"),
    (((10, 20), "vertical_cam"), ((20, 10), "horizontal_cam")),
)
def test_legacy_camera_shape_gold(
    tmp_path: Path,
    pixels: tuple[int, int],
    gold: str,
) -> None:
    """Unified cameras retain legacy orientation and output shape."""
    assert_render_allclose(
        _render(tmp_path, _camera(pixels)),
        np.load(_GOLD / f"{gold}.npy"),
        gold,
        rtol=0.0,
        atol=2.0,
    )


@pytest.mark.parametrize(
    ("samples", "gold"), ((4, "samples_four"), (12, "samples_twelve"))
)
def test_legacy_samples_gold(tmp_path: Path, samples: int, gold: str) -> None:
    """Blender sample counts retain legacy render behaviour."""
    assert_render_allclose(
        _render(tmp_path, _camera(), samples=samples),
        np.load(_GOLD / f"{gold}.npy"),
        gold,
        rtol=0.0,
        atol=2.0,
    )


@pytest.mark.parametrize(
    ("bounces", "gold"), ((2, "bounces_two"), (100, "bounces_hundred"))
)
def test_legacy_bounces_gold(tmp_path: Path, bounces: int, gold: str) -> None:
    """Blender maximum bounces retain legacy render behaviour."""
    assert_render_allclose(
        _render(tmp_path, _camera(), bounces=bounces),
        np.load(_GOLD / f"{gold}.npy"),
        gold,
        rtol=0.0,
        atol=2.0,
    )


@pytest.mark.parametrize(
    ("engine", "gold"),
    (
        (render.EBlenderEngine.CYCLES, "cycles_engine"),
        (render.EBlenderEngine.EEVEE, "eevee_engine"),
    ),
)
def test_legacy_engine_gold(
    tmp_path: Path,
    engine: render.EBlenderEngine,
    gold: str,
) -> None:
    """Blender engine selection retains the committed legacy image output."""
    assert_render_allclose(
        _render(tmp_path, _camera(), engine=engine),
        np.load(_GOLD / f"{gold}.npy"),
        gold,
        rtol=0.0,
        atol=2.0,
    )


def test_legacy_workbench_engine(tmp_path: Path) -> None:
    """The legacy Blender Workbench engine remains selectable."""
    image = _render(tmp_path, _camera(), engine=render.EBlenderEngine.WORKBENCH)
    assert image.shape == (20, 20)


def test_camera_from_resolution_matches_legacy_gold(tmp_path: Path) -> None:
    """The migrated resolution helper creates the legacy Blender camera."""
    camera = render.blender_camera_from_resolution(
        np.array((20, 20)),
        np.array((0.00345, 0.00345)),
        500.0,
        0.1,
    )
    assert_render_allclose(
        _render(tmp_path, camera),
        np.load(_GOLD / "cam_from_resolution.npy"),
        "camera_from_resolution",
        rtol=0.0,
        atol=2.0,
    )
    assert render.blender_mm_per_pixel(camera) == pytest.approx(0.1)


@pytest.mark.parametrize("field", ("samples", "max_bounces", "threads", "seed"))
def test_blender_config_rejects_non_integral_controls(
    tmp_path: Path,
    field: str,
) -> None:
    """Invalid legacy render controls fail before Blender scene construction."""
    values = {"samples": 2, "max_bounces": 12, "threads": 1, "seed": 0}
    values[field] = 2.5
    renderer = render.Blender(render.BlenderConfig(tmp_path, **values))
    with pytest.raises(render.RenderInputError, match="VALUE"):
        renderer.render(render.Scene3D((_mesh(_camera()),), (_camera(),)))


def test_blender_persisted_images_and_scene(tmp_path: Path) -> None:
    """Legacy TIFF and Blender-project output remain available by configuration."""
    camera = _camera()
    result = render.Blender(
        render.BlenderConfig(
            tmp_path,
            threads=1,
            save_images=True,
            save_scene=True,
        )
    ).render(
        render.Scene3D(
            (_mesh(camera),),
            (camera,),
            (
                render.Light(
                    render.ELightType.POINT,
                    np.array((0.0, 0.0, 400.0)),
                    np.zeros(3),
                    1.0,
                ),
            ),
        )
    )
    assert result.images is None
    assert len(result.output_paths) == 1
    assert result.output_paths[0].is_file()
    assert (tmp_path / "blenderfiles" / "projectfile.blend").is_file()


def test_blender_supports_legacy_light_geometries(tmp_path: Path) -> None:
    """The common light enum retains every geometry supported by Blender."""
    camera = _camera()
    for light_type in (
        render.ELightType.SUN,
        render.ELightType.SPOT,
        render.ELightType.AREA,
    ):
        result = render.Blender(
            render.BlenderConfig(
                tmp_path / light_type.value,
                threads=1,
            )
        ).render(
            render.Scene3D(
                (_mesh(camera),),
                (camera,),
                (
                    render.Light(
                        light_type,
                        np.array((0.0, 0.0, 400.0)),
                        np.array((0.0, 0.0, -1.0)),
                        1.0,
                        shadow_soft_size=2.0,
                    ),
                ),
            )
        )
        assert result.images is not None


def test_blender_in_memory_texture(tmp_path: Path) -> None:
    """Legacy in-memory image textures are available through BlenderImageShader."""
    camera = _camera()
    source_mesh = _mesh(camera)
    mesh = render.Mesh3D(
        source_mesh.element_type,
        source_mesh.coords,
        source_mesh.connectivity,
        render.BlenderImageShader(np.indices((64, 64)).sum(axis=0) % 255, 0.1),
        source_mesh.displacements,
    )
    result = render.Blender(render.BlenderConfig(tmp_path, threads=1)).render(
        render.Scene3D(
            (mesh,),
            (camera,),
            (
                render.Light(
                    render.ELightType.POINT,
                    np.array((0.0, 0.0, 400.0)),
                    np.zeros(3),
                    1.0,
                ),
            ),
        ),
    )
    assert result.images is not None


@pytest.mark.parametrize("placement", ("symmetric", "faceon"))
def test_legacy_stereo_gold(
    tmp_path: Path,
    placement: str,
) -> None:
    """Unified stereo helpers render one image for each legacy camera view."""
    camera = _camera()
    if placement == "symmetric":
        stereo = render.symmetric_stereo_cameras(camera, 15.0)
    else:
        stereo = render.faceon_stereo_cameras(camera, 15.0)
    renderer = render.Blender(render.BlenderConfig(tmp_path, threads=1))
    result = renderer.render(
        render.Scene3D(
            (_mesh(camera),),
            (stereo.camera_0, stereo.camera_1),
            (
                render.Light(
                    render.ELightType.POINT,
                    np.array((0.0, 0.0, 400.0)),
                    np.zeros(3),
                    1.0,
                ),
            ),
        )
    )
    assert result.images is not None
    assert result.images.shape == (1, 2, 20, 20, 1)
    assert not np.array_equal(result.images[0, 0], result.images[0, 1])


def test_legacy_deformation_frames(tmp_path: Path) -> None:
    """Unified deformation rendering preserves every simulation timestep."""
    camera = _camera()
    renderer = render.Blender(
        render.BlenderConfig(
            tmp_path,
            threads=1,
            render_deformed=True,
        )
    )
    result = renderer.render(
        render.Scene3D(
            (_mesh(camera),),
            (camera,),
            (
                render.Light(
                    render.ELightType.POINT,
                    np.array((0.0, 0.0, 400.0)),
                    np.zeros(3),
                    1.0,
                ),
            ),
        )
    )
    assert result.images is not None
    assert result.images.shape == (61, 1, 20, 20, 1)
    assert not np.array_equal(result.images[0], result.images[-1])
    assert_render_allclose(
        result.images[10, 0, :, :, 0],
        np.load(_GOLD / "deformed_images.npy"),
        "deformed_images",
        rtol=0.0,
        atol=2.0,
    )


def test_legacy_calibration_round_trip(tmp_path: Path) -> None:
    """Stereo calibration files retain the legacy format and values."""
    stereo = render.faceon_stereo_cameras(_camera(), 15.0)
    stereo.save_calibration(tmp_path)
    output = tmp_path / "calibration" / "calibration.yaml"
    rebuilt = render.CameraStereo.from_calibration(
        output,
        np.array((0.0, 0.0, 500.0)),
        Rotation.identity(),
        15.0,
    )
    assert output.read_text()
    assert np.allclose(rebuilt.stereo_dist, stereo.stereo_dist)
    assert rebuilt.stereo_rotation.approx_equal(stereo.stereo_rotation)
    stereo.save_calibration_mid(tmp_path)
    assert (tmp_path / "calibration" / "calibration.caldat").is_file()


def test_legacy_calibration_image_count() -> None:
    """Blender calibration target-counting remains available in render."""
    data = render.BlenderCalibrationData()
    assert render.calibration_image_count(data) == 675


def test_unified_stereo_matches_legacy_scene(tmp_path: Path) -> None:
    """The adapter reproduces Blender's tessellated legacy scene path."""
    camera = _camera()
    stereo = render.faceon_stereo_cameras(camera, 15.0)
    mesh = _mesh(camera)
    light = render.Light(
        render.ELightType.POINT,
        np.array((0.0, 0.0, 400.0)),
        np.zeros(3),
        1.0,
    )
    unified = render.Blender(
        render.BlenderConfig(tmp_path / "unified", threads=1)
    )
    unified_result = unified.render(
        render.Scene3D(
            (mesh,),
            (stereo.camera_0, stereo.camera_1),
            (light,),
        )
    )
    assert unified_result.images is not None

    legacy_mesh = _triangulate_mesh_for_blender(mesh)
    scene = legacy_blender.Scene()
    part = scene.add_part(legacy_mesh, 3)
    scene.add_speckle(
        part,
        mesh.shader.image_path,
        legacy_blender.MaterialData(),
        mesh.shader.millimetres_per_pixel,
    )
    scene.add_camera(stereo.camera_0)
    scene.add_camera(stereo.camera_1)
    scene.add_light(
        legacy_blender.LightData(
            light.pos_world,
            Rotation.identity(),
            light.intensity,
            legacy_blender.LightType.POINT,
        )
    )
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy = scene.render_single_image(
        legacy_blender.RenderData(
            (stereo.camera_0, stereo.camera_1),
            legacy_dir,
            threads=1,
        )
    )
    expected = np.asarray(legacy).transpose(2, 0, 1)[None, :, :, :, None]
    assert_render_allclose(
        unified_result.images,
        expected,
        "legacy_stereo_adapter",
        rtol=0.0,
        atol=0.0,
    )


def test_unified_deformation_matches_legacy_scene(tmp_path: Path) -> None:
    """The adapter's deformation stack matches Blender's active scene method."""
    camera = _camera()
    mesh = _mesh(camera)
    light = render.Light(
        render.ELightType.POINT,
        np.array((0.0, 0.0, 400.0)),
        np.zeros(3),
        1.0,
    )
    unified_result = render.Blender(
        render.BlenderConfig(
            tmp_path / "unified",
            threads=1,
            render_deformed=True,
        )
    ).render(render.Scene3D([mesh], [camera], [light]))
    assert unified_result.images is not None

    legacy_mesh = _triangulate_mesh_for_blender(mesh)
    scene = legacy_blender.Scene()
    part = scene.add_part(legacy_mesh, 3)
    scene.add_speckle(
        part,
        mesh.shader.image_path,
        legacy_blender.MaterialData(),
        mesh.shader.millimetres_per_pixel,
    )
    scene.add_camera(camera)
    scene.add_light(
        legacy_blender.LightData(
            light.pos_world,
            Rotation.identity(),
            light.intensity,
            legacy_blender.LightType.POINT,
        )
    )
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    centred_mesh = render.Mesh3D(
        legacy_mesh.element_type,
        centre_mesh_nodes(legacy_mesh.coords.copy(), 3),
        legacy_mesh.connectivity,
        legacy_mesh.shader,
        legacy_mesh.displacements,
    )
    legacy = scene.render_deformed_images(
        centred_mesh,
        3,
        legacy_blender.RenderData(camera, legacy_dir, threads=1),
        part,
    )
    expected = np.asarray(legacy).transpose(2, 0, 1)[:, None, :, :, None]
    assert_render_allclose(
        unified_result.images,
        expected,
        "legacy_deformation_adapter",
        rtol=0.0,
        atol=0.0,
    )


def test_calibration_target_rendering(tmp_path: Path) -> None:
    """The migrated calibration target workflow writes stereo TIFF pairs."""
    camera = _camera()
    stereo = render.faceon_stereo_cameras(camera, 15.0)
    result = render.render_calibration_images(
        render.BlenderCalibrationTarget(
            np.array((15.0, 10.0, 1.0)),
            dataset.cal_target(),
            0.1,
        ),
        stereo,
        render.BlenderConfig(tmp_path, threads=1),
        render.BlenderCalibrationData(
            angle_lims=(0.0, 0.0),
            angle_step=1.0,
            plunge_lims=(0.0, 0.0),
            plunge_step=1.0,
            x_limit=0.0,
            y_limit=0.0,
        ),
        [
            render.Light(
                render.ELightType.POINT,
                np.array((0.0, 0.0, 200.0)),
                np.zeros(3),
                1.0,
            )
        ],
    )
    assert result.images is None
    assert len(result.output_paths) == 18
    assert all(path.is_file() for path in result.output_paths)

# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Shared small-image conformance scenes for every render backend."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.render as render
import riley


IMAGE_SIZE = 32
CASE_NAMES = ("tri3_deforming", "tri3_shared_edge", "tri3_clipping")


@dataclass(frozen=True, slots=True)
class RenderConformanceCase:
    """Geometry and two-frame in-plane motion for one conformance case."""

    name: str
    coords: np.ndarray
    connectivity: np.ndarray
    displacements: np.ndarray


def conformance_cases() -> tuple[RenderConformanceCase, ...]:
    """Return the three canonical 32-pixel render conformance cases."""
    return (
        _translation_case(),
        _shared_edge_case(),
        _clipping_case(),
    )


def render_backend_case(
    backend: str,
    case: RenderConformanceCase,
    output_dir: Path,
) -> np.ndarray:
    """Render one canonical case through the requested backend."""
    renderers = {
        "blender": _render_blender,
        "imagedef2d": _render_imagedef2d,
        "pixint_grid": _render_pixint_grid,
        "pixint_speck": _render_pixint_speck,
        "riley": _render_riley,
    }
    return renderers[backend](case, output_dir)


def feebee_scene(case: RenderConformanceCase) -> render.Scene3D:
    """Build a case for validation before the Feebee backend lands."""
    mesh = _common_mesh(
        case,
        render.FeebeeColourShader(
            np.full((2, len(case.connectivity), 3), 0.65),
        ),
    )
    return render.Scene3D([mesh], [_camera_3d()])


def preview_range(backend: str) -> tuple[float, float]:
    """Return the fixed intensity range for an 8-bit preview."""
    if backend in {"blender", "imagedef2d"}:
        return 0.0, 255.0
    return 0.0, 1.0


def _translation_case() -> RenderConformanceCase:
    coords = np.array((
        (-1.05, -0.85),
        (0.85, -0.65),
        (-0.35, 1.05),
    ))
    displacement = np.tile((0.30, 0.20), (3, 1))
    return _case("tri3_deforming", coords, ((0, 1, 2),), displacement)


def _shared_edge_case() -> RenderConformanceCase:
    coords = np.array((
        (-0.95, -0.95),
        (0.95, -0.95),
        (0.95, 0.95),
        (-0.95, 0.95),
    ))
    displacement = np.column_stack((
        0.18 + 0.12 * coords[:, 0],
        0.10 - 0.08 * coords[:, 0],
    ))
    return _case(
        "tri3_shared_edge",
        coords,
        ((0, 1, 2), (0, 2, 3)),
        displacement,
    )


def _clipping_case() -> RenderConformanceCase:
    coords = np.array((
        (0.35, -0.75),
        (1.35, -0.55),
        (0.80, 0.75),
    ))
    displacement = np.tile((0.55, 0.45), (3, 1))
    return _case("tri3_clipping", coords, ((0, 1, 2),), displacement)


def _case(
    name: str,
    coords: np.ndarray,
    connectivity: tuple[tuple[int, ...], ...],
    displacement: np.ndarray,
) -> RenderConformanceCase:
    displacements = np.zeros((2, len(coords), 2), dtype=np.float64)
    displacements[1] = displacement
    return RenderConformanceCase(
        name,
        np.ascontiguousarray(coords, dtype=np.float64),
        np.asarray(connectivity, dtype=np.intp),
        displacements,
    )


def _camera_2d() -> render.Camera2D:
    return render.Camera2D(
        pixels_count=np.array((IMAGE_SIZE, IMAGE_SIZE)),
        leng_per_px=0.1,
        roi_cent_world=np.zeros(3),
        background=0.15,
        subsample=1,
    )


def _camera_3d() -> render.Camera:
    return render.Camera(
        pixels_num=np.array((IMAGE_SIZE, IMAGE_SIZE)),
        pixels_size=np.array((0.1, 0.1)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=2.0,
    )


def _common_mesh(case: RenderConformanceCase, shader: object) -> render.Mesh3D:
    return render.Mesh3D(
        render.EElementType.TRI3,
        np.pad(case.coords, ((0, 0), (0, 1))),
        case.connectivity,
        shader,
        np.pad(case.displacements, ((0, 0), (0, 0), (0, 1))),
    )


def _mesh_2d(case: RenderConformanceCase) -> render.Mesh2D:
    return render.Mesh2D(
        render.EElementType.TRI3,
        case.coords,
        case.connectivity,
        case.displacements,
    )


def _render_blender(
    case: RenderConformanceCase,
    output_dir: Path,
) -> np.ndarray:
    texture = np.linspace(80, 220, 64, dtype=np.uint8).reshape((8, 8))
    shader = render.BlenderImageShader(texture, 0.4)
    mesh = _common_mesh(case, shader)
    light = render.Light(
        render.ELightType.POINT,
        np.array((-0.8, 0.7, 2.5)),
        np.array((0.0, 0.0, -1.0)),
        1.0,
    )
    config = render.BlenderConfig(
        output_dir,
        device=render.EBlenderDevice.CPU,
        samples=32,
        max_bounces=2,
        threads=1,
        render_deformed=True,
        seed=0,
        use_denoising=False,
        use_adaptive_sampling=False,
    )
    result = render.Blender(config).render(
        render.Scene3D([mesh], [_camera_3d()], [light]),
    )
    assert result.images is not None
    return result.images


def _render_riley(
    case: RenderConformanceCase,
    output_dir: Path,
) -> np.ndarray:
    del output_dir
    params = riley.FuncShaderParams(
        linear_coeffs=(0.55, 0.12, -0.08),
    )
    mesh = riley.Mesh(
        riley.MeshType.tri3,
        np.pad(case.coords, ((0, 0), (0, 1))),
        case.connectivity,
        disp=np.pad(case.displacements, ((0, 0), (0, 0), (0, 1))),
        shader_type=riley.ShaderType.func,
        func_shader_builtin=riley.FuncShaderBuiltin.linear,
        func_shader_coord_mode=riley.FuncCoordMode.world_reference,
        func_shader_params=params,
    )
    config = riley.create_raster_config(
        2,
        total_threads=1,
        save_strategy=riley.SaveStrategy.memory,
    )
    config.background_value = 0.15
    result = render.Riley(config).render(
        render.Scene3D([mesh], [_camera_3d()]),
    )
    assert result.images is not None
    return result.images


def _render_imagedef2d(
    case: RenderConformanceCase,
    output_dir: Path,
) -> np.ndarray:
    output_dir.mkdir(parents=True, exist_ok=True)
    axis = np.linspace(0.0, 1.0, IMAGE_SIZE)
    source_image = 255.0 * (
        0.25 + 0.5 * axis[None, :] + 0.25 * axis[:, None]
    )
    scene = render.Scene2D(_mesh_2d(case), _camera_2d(), source_image)
    return render.ImageDef2D(
        render.ImageDefOpts(save_path=output_dir),
    ).render(scene).images


def _render_pixint_grid(
    case: RenderConformanceCase,
    output_dir: Path,
) -> np.ndarray:
    del output_dir
    scene = render.Scene2D(_mesh_2d(case), _camera_2d())
    renderer = render.PixIntGrid2D(
        render.Eggbox(period=(0.8, 0.7), phase=(0.2, -0.3)),
        render.PxInt2DOpts(integration=render.RectRule(2)),
    )
    return renderer.render(scene).images


def _render_pixint_speck(
    case: RenderConformanceCase,
    output_dir: Path,
) -> np.ndarray:
    del output_dir
    pattern = render.AdditiveSpeckles.jittered_lattice(
        kind="gaussian",
        speckle_diameter=0.32,
        black_area_fraction=0.45,
        jitter_pdf="uniform",
        jitter=0.15,
        seed=7,
        bounds=(-2.2, 2.2, -2.2, 2.2),
    )
    renderer = render.PixIntSpeck2D(
        pattern,
        render.PxInt2DOpts(integration=render.RectRule(2)),
    )
    return renderer.render(
        render.Scene2D(_mesh_2d(case), _camera_2d()),
    ).images


__all__ = [
    "CASE_NAMES",
    "IMAGE_SIZE",
    "RenderConformanceCase",
    "conformance_cases",
    "feebee_scene",
    "preview_range",
    "render_backend_case",
]

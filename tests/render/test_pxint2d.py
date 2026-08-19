# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Regression tests for the PixInt2D renderers and Newton maps."""

from pathlib import Path

import numpy as np
import pytest

import pyvale.render as render
from pyvale.render.pxint2d.elements import shape_functions
from render_checks import assert_render_allclose


GOLD = Path(__file__).parent / "gold_pxint2d"


def make_camera() -> render.Camera2D:
    """Create the RCC-compatible 32 by 32 orthographic test camera."""
    return render.Camera2D(
        pixels_count=np.array((32, 32)), leng_per_px=1.0,
        roi_cent_world=np.zeros(3), subsample=1,
    )


def make_mesh(element_type: render.EElementType) -> render.Mesh2D:
    """Create one Riley-ordered element that fully covers the camera view."""
    if element_type is render.EElementType.TRI3:
        coords = ((-100, -100), (200, -100), (-100, 200))
    elif element_type is render.EElementType.TRI6:
        coords = ((-100, -100), (200, -100), (-100, 200), (50, -100),
                  (50, 50), (-100, 50))
    elif element_type is render.EElementType.QUAD4:
        coords = ((-20, -20), (20, -20), (20, 20), (-20, 20))
    elif element_type is render.EElementType.QUAD8:
        coords = ((-20, -20), (20, -20), (20, 20), (-20, 20), (0, -20),
                  (20, 0), (0, 20), (-20, 0))
    else:
        coords = ((-20, -20), (20, -20), (20, 20), (-20, 20), (0, -20),
                  (20, 0), (0, 20), (-20, 0), (0, 0))
    coords_array = np.asarray(coords, dtype=np.float64)
    return render.Mesh2D(
        element_type, coords_array, np.arange(len(coords_array))[None, :],
    )


def make_mesh_multi(element_type: render.EElementType) -> render.Mesh2D:
    """Create four quads or eight triangles covering the camera field of view."""
    nodes: list[tuple[float, float]] = []
    connectivity: list[list[int]] = []

    def add_element(points: tuple[tuple[float, float], ...]) -> None:
        start = len(nodes)
        nodes.extend(points)
        connectivity.append(list(range(start, start + len(points))))

    for x_low, x_high in ((-20.0, 0.0), (0.0, 20.0)):
        for y_low, y_high in ((-20.0, 0.0), (0.0, 20.0)):
            if element_type.name.startswith("TRI"):
                triangles = (
                    ((x_low, y_low), (x_high, y_low), (x_high, y_high)),
                    ((x_low, y_low), (x_high, y_high), (x_low, y_high)),
                )
                for triangle in triangles:
                    if element_type is render.EElementType.TRI3:
                        add_element(triangle)
                    else:
                        first, second, third = triangle
                        add_element((
                            first, second, third,
                            ((first[0] + second[0]) / 2.0,
                             (first[1] + second[1]) / 2.0),
                            ((second[0] + third[0]) / 2.0,
                             (second[1] + third[1]) / 2.0),
                            ((third[0] + first[0]) / 2.0,
                             (third[1] + first[1]) / 2.0),
                        ))
                continue

            quad = ((x_low, y_low), (x_high, y_low), (x_high, y_high),
                    (x_low, y_high))
            if element_type is render.EElementType.QUAD4:
                add_element(quad)
            elif element_type is render.EElementType.QUAD8:
                add_element(quad + (
                    ((x_low + x_high) / 2.0, y_low),
                    (x_high, (y_low + y_high) / 2.0),
                    ((x_low + x_high) / 2.0, y_high),
                    (x_low, (y_low + y_high) / 2.0),
                ))
            else:
                add_element(quad + (
                    ((x_low + x_high) / 2.0, y_low),
                    (x_high, (y_low + y_high) / 2.0),
                    ((x_low + x_high) / 2.0, y_high),
                    (x_low, (y_low + y_high) / 2.0),
                    ((x_low + x_high) / 2.0, (y_low + y_high) / 2.0),
                ))
    return render.Mesh2D(
        element_type, np.asarray(nodes), np.asarray(connectivity, dtype=np.intp),
    )


def affine_displacements(mesh: render.Mesh2D) -> render.DisplacementSeries2D:
    """Create a zero and globally affine displacement frame."""
    x_coord, y_coord = mesh.coords[:, 0], mesh.coords[:, 1]
    affine = np.column_stack((0.03*x_coord + 0.01*y_coord + 0.2,
                              -0.02*x_coord + 0.04*y_coord - 0.1))
    return render.DisplacementSeries2D(np.stack((np.zeros_like(affine), affine)))


def rcc_rigid_displacements(mesh: render.Mesh2D) -> render.DisplacementSeries2D:
    """Apply the frame-three rigid displacement from the copied RCC fixture."""
    directory = (Path("src/pyvale/render/pxint2d/data/single_elem")
                 / "plate42_cam32_quad9_rigid")
    displacement_x = np.loadtxt(directory / "field_disp_x.csv", delimiter=",")
    displacement_y = np.loadtxt(directory / "field_disp_y.csv", delimiter=",")
    translation = np.array((displacement_x[0, 3], displacement_y[0, 3]))
    frame_zero = np.zeros((mesh.coords.shape[0], 2))
    frame_three = np.broadcast_to(translation, frame_zero.shape).copy()
    return render.DisplacementSeries2D(np.stack((frame_zero, frame_three)))


def rcc_affine_displacements(mesh: render.Mesh2D) -> render.DisplacementSeries2D:
    """Apply the copied RCC frame-three globally affine displacement field."""
    directory = (Path("src/pyvale/render/pxint2d/data/single_elem")
                 / "plate42_cam32_quad9_affine")
    source_coords = np.loadtxt(directory / "coords.csv", delimiter=",")[:, :2]
    source_x = np.loadtxt(directory / "field_disp_x.csv", delimiter=",")[:, 3]
    source_y = np.loadtxt(directory / "field_disp_y.csv", delimiter=",")[:, 3]
    design = np.column_stack((source_coords, np.ones(len(source_coords))))
    coefficients, _, _, _ = np.linalg.lstsq(
        design, np.column_stack((source_x, source_y)), rcond=None,
    )
    displacement = np.column_stack((mesh.coords, np.ones(len(mesh.coords))))
    displacement = displacement @ coefficients
    return render.DisplacementSeries2D(
        np.stack((np.zeros_like(displacement), displacement)),
    )


def make_speckles(kind: str) -> render.AdditiveSpeckles:
    """Create one deterministic RCC-equivalent disk or Gaussian pattern."""
    if kind == "disk":
        return render.AdditiveSpeckles.jittered_lattice(
            kind="disk", speckle_diameter=5.0, black_area_fraction=0.6,
            jitter_pdf="uniform", jitter=0.25, seed=3,
            bounds=(-20.0, 20.0, -20.0, 20.0),
        )
    return render.AdditiveSpeckles.jittered_lattice(
        kind="gaussian", speckle_diameter=5.0, black_area_fraction=0.6,
        jitter_pdf="gaussian", jitter=0.12, seed=3,
        bounds=(-20.0, 20.0, -20.0, 20.0),
        gaussian_edge_fraction=0.4, tail_sigmas=8.0,
    )


@pytest.mark.parametrize("element_type", list(render.EElementType))
def test_shape_functions_reproduce_partition_of_unity(
    element_type: render.EElementType,
) -> None:
    """Every Riley-ordered element has consistent shape functions."""
    xi, eta = ((0.2, 0.3) if element_type.name.startswith("TRI") else (0.2, -0.3))
    values, d_xi, d_eta = shape_functions(element_type, xi, eta)
    assert np.isclose(values.sum(), 1.0)
    assert np.isclose(d_xi.sum(), 0.0)
    assert np.isclose(d_eta.sum(), 0.0)


@pytest.mark.parametrize("element_type", list(render.EElementType))
@pytest.mark.parametrize("samples", (1, 2, 4))
def test_newton_maps_match_affine_for_every_element(
    element_type: render.EElementType,
    samples: int,
) -> None:
    """Both Newton maps reproduce globally affine renders for all topologies."""
    mesh = make_mesh(element_type)
    camera = make_camera()
    displacements = affine_displacements(mesh)
    quad_mesh = make_mesh(render.EElementType.QUAD9)
    baseline = render.PixIntGrid2D(
        options=render.PxInt2DOpts(
            mapping=render.EPxIntMapping.AFFINE,
            integration=render.RectRule(samples),
        ),
    ).render(quad_mesh, camera, affine_displacements(quad_mesh)).images
    for mode in (render.EPxIntMapping.NEWTON_ONE_ELEM,
                 render.EPxIntMapping.NEWTON_MESH_UNSTRUCT,
                 render.EPxIntMapping.NEWTON_MESH_STRUCT,
                 render.EPxIntMapping.VTK):
        actual = render.PixIntGrid2D(
            options=render.PxInt2DOpts(
                mapping=mode, integration=render.RectRule(samples),
            ),
        ).render(mesh, camera, displacements).images
        assert_render_allclose(
            actual, baseline, f"grid_{element_type.value}_{samples}_{mode}",
        )


@pytest.mark.parametrize("samples", (1, 2, 4))
def test_rcc_quad9_subpixel_gold(samples: int) -> None:
    """The copied RCC affine Quad9 fixture matches each subpixel level."""
    directory = (Path("src/pyvale/render/pxint2d/data/single_elem")
                 / "plate42_cam32_quad9_affine")
    coords = np.loadtxt(directory / "coords.csv", delimiter=",")[:, :2]
    connect = np.loadtxt(directory / "connectivity.csv", delimiter=",", dtype=int)
    displacement_x = np.loadtxt(directory / "field_disp_x.csv", delimiter=",")
    displacement_y = np.loadtxt(directory / "field_disp_y.csv", delimiter=",")
    values = np.stack((displacement_x, displacement_y), axis=2).transpose(1, 0, 2)
    mesh = render.Mesh2D(render.EElementType.QUAD9, coords, connect[None, :])
    camera = render.Camera2D(
        pixels_count=np.array((32, 32)), leng_per_px=1.0,
        roi_cent_world=np.zeros(3), subsample=1,
    )
    actual = render.PixIntGrid2D(
        options=render.PxInt2DOpts(
            mapping=render.EPxIntMapping.AFFINE,
            integration=render.RectRule(samples),
        ),
    ).render(mesh, camera, render.DisplacementSeries2D(values)).images[3, 0, :, :, 0]
    expected = np.load(GOLD / f"affine_grid_rect{samples}.npy")
    assert_render_allclose(actual, expected, f"quad9_affine_grid_{samples}")


@pytest.mark.parametrize("element_type", list(render.EElementType))
@pytest.mark.parametrize("samples", (1, 2, 4))
@pytest.mark.parametrize("mesh_factory, mapping", (
    (make_mesh, render.EPxIntMapping.NEWTON_ONE_ELEM),
    (make_mesh_multi, render.EPxIntMapping.NEWTON_MESH_UNSTRUCT),
    (make_mesh_multi, render.EPxIntMapping.NEWTON_MESH_STRUCT),
))
def test_grid_element_types_match_affine_gold(
    element_type: render.EElementType,
    samples: int,
    mesh_factory,
    mapping: render.EPxIntMapping,
) -> None:
    """Single and multi-element Newton renders match the affine gold image."""
    mesh = mesh_factory(element_type)
    actual = render.PixIntGrid2D(
        options=render.PxInt2DOpts(
            mapping=mapping,
            integration=render.RectRule(samples),
        ),
    ).render(mesh, make_camera(), rcc_affine_displacements(mesh)).images[1, 0, :, :, 0]
    expected = np.load(GOLD / f"affine_grid_rect{samples}.npy")
    assert_render_allclose(
        actual, expected,
        f"grid_{element_type.value}_{mapping}_{samples}",
    )


@pytest.mark.parametrize("element_type", list(render.EElementType))
@pytest.mark.parametrize("kind", ("disk", "gaussian"))
@pytest.mark.parametrize("samples", (1, 2, 4))
@pytest.mark.parametrize("mesh_factory, mapping", (
    (make_mesh, render.EPxIntMapping.NEWTON_ONE_ELEM),
    (make_mesh_multi, render.EPxIntMapping.NEWTON_MESH_UNSTRUCT),
    (make_mesh_multi, render.EPxIntMapping.NEWTON_MESH_STRUCT),
))
def test_speck_element_types_match_affine_gold(
    element_type: render.EElementType,
    kind: str,
    samples: int,
    mesh_factory,
    mapping: render.EPxIntMapping,
) -> None:
    """Single and multi-element Newton renders match affine Speck2D gold."""
    mesh = mesh_factory(element_type)
    actual = render.PixIntSpeck2D(
        make_speckles(kind),
        options=render.PxInt2DOpts(
            mapping=mapping,
            integration=render.RectRule(samples),
        ),
    ).render(mesh, make_camera(), rcc_affine_displacements(mesh)).images[1, 0, :, :, 0]
    expected = np.load(GOLD / f"affine_speck_{kind}_rect{samples}.npy")
    assert_render_allclose(
        actual, expected,
        f"speck_{kind}_{element_type.value}_{mapping}_{samples}",
    )


def test_copied_rcc_analytic_gold_is_preserved() -> None:
    """The original RCC 32-pixel analytic reference remains reproducible."""
    directory = (Path("src/pyvale/render/pxint2d/data/single_elem")
                 / "plate42_cam32_quad9_rigid")
    coords = np.loadtxt(directory / "coords.csv", delimiter=",")[:, :2]
    connect = np.loadtxt(directory / "connectivity.csv", delimiter=",", dtype=int)
    displacement_x = np.loadtxt(directory / "field_disp_x.csv", delimiter=",")
    displacement_y = np.loadtxt(directory / "field_disp_y.csv", delimiter=",")
    values = np.stack((displacement_x, displacement_y), axis=2).transpose(1, 0, 2)
    mesh = render.Mesh2D(render.EElementType.QUAD9, coords, connect[None, :])
    camera = render.Camera2D(
        pixels_count=np.array((32, 32)), leng_per_px=1.0,
        roi_cent_world=np.zeros(3), subsample=1,
    )
    actual = render.PixIntGrid2D(
        options=render.PxInt2DOpts(
            mapping=render.EPxIntMapping.AFFINE,
            integration=render.AnalyticRule(),
        ),
    ).render(mesh, camera, render.DisplacementSeries2D(values)).images[0, 0, :, :, 0]
    expected = np.load(GOLD / "rcc_reference/grid2d_eggbox/rigid_f00.npy")
    assert_render_allclose(actual, expected, "quad9_rcc_analytic")


def test_speck_renderer_uses_the_shared_newton_map() -> None:
    """Speck2D returns canonical images and masks through Newton mapping."""
    mesh = make_mesh(render.EElementType.QUAD9)
    pattern = render.AdditiveSpeckles.jittered_lattice(
        kind="disk", speckle_diameter=2.0, black_area_fraction=0.5,
        jitter_pdf="uniform", jitter=0.1, seed=3,
        bounds=(-20.0, 20.0, -20.0, 20.0),
    )
    result = render.PixIntSpeck2D(
        pattern, render.PxInt2DOpts(integration=render.RectRule(2)),
    ).render(mesh, make_camera(), affine_displacements(mesh))
    assert result.images.shape == (2, 1, 32, 32, 1)
    assert result.masks is not None and result.masks.shape == result.images.shape


def test_newton_one_element_rejects_a_multi_element_request() -> None:
    """Validation rejects an incompatible one-element mapping request."""
    mesh = make_mesh(render.EElementType.QUAD4)
    mesh.connectivity = np.vstack((mesh.connectivity, mesh.connectivity))
    with pytest.raises(ValueError, match="one element"):
        render.PixIntGrid2D(
            options=render.PxInt2DOpts(
                mapping=render.EPxIntMapping.NEWTON_ONE_ELEM,
            ),
        ).render(mesh, make_camera(), affine_displacements(mesh))

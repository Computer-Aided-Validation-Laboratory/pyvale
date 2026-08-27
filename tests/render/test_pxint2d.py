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
import pyvale.verif.renderverif as renderverif
from pyvale.render.pxint2d.elements import shape_functions
from pyvale.verif.renderverif import assert_render_allclose


GOLD = Path(__file__).parent / "gold_pxint2d"


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


def test_grid_rejects_meshes_outside_the_shared_convention() -> None:
    """Planar renderers reject clockwise connectivity before rendering."""
    mesh = render.Mesh2D(
        render.EElementType.TRI3,
        np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.array(((0, 2, 1),)),
    )
    scene = render.Scene2D(mesh=mesh, camera=renderverif.pxint2d_camera())

    with pytest.raises(ValueError, match="shared Riley/VTK convention"):
        render.PixIntGrid2D().verify_input(scene)


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


def affine_displacements(mesh: render.Mesh2D) -> np.ndarray:
    """Create a zero and globally affine displacement frame."""
    x_coord, y_coord = mesh.coords[:, 0], mesh.coords[:, 1]
    affine = np.column_stack((0.03*x_coord + 0.01*y_coord + 0.2,
                              -0.02*x_coord + 0.04*y_coord - 0.1))
    return np.stack((np.zeros_like(affine), affine))


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
    mesh.displacement = affine_displacements(mesh)
    camera = renderverif.pxint2d_camera()
    quad_mesh = make_mesh(render.EElementType.QUAD9)
    quad_mesh.displacement = affine_displacements(quad_mesh)
    baseline = render.PixIntGrid2D(
        options=render.PxInt2DOpts(
            mapping=render.EPxIntMapping.AFFINE,
            integration=render.RectRule(samples),
        ),
    ).render(render.Scene2D(mesh=quad_mesh, camera=camera)).images
    for mode in (render.EPxIntMapping.NEWTON_ONE_ELEM,
                 render.EPxIntMapping.NEWTON_MESH_UNSTRUCT,
                 render.EPxIntMapping.NEWTON_MESH_STRUCT,
                 render.EPxIntMapping.VTK):
        actual = render.PixIntGrid2D(
            options=render.PxInt2DOpts(
                mapping=mode, integration=render.RectRule(samples),
            ),
        ).render(render.Scene2D(mesh=mesh, camera=camera)).images
        assert_render_allclose(
            actual, baseline, f"grid_{element_type.value}_{samples}_{mode}",
        )


@pytest.mark.parametrize("samples", (1, 2, 4))
def test_rcc_quad9_subpixel_gold(samples: int) -> None:
    """The copied RCC affine Quad9 fixture matches each subpixel level."""
    actual = renderverif.pxint2d_affine_reference(samples)
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
    mesh.displacement = renderverif.rcc_affine_displacements(mesh)
    actual = render.PixIntGrid2D(
        options=render.PxInt2DOpts(
            mapping=mapping,
            integration=render.RectRule(samples),
        ),
    ).render(render.Scene2D(
        mesh=mesh, camera=renderverif.pxint2d_camera(),
    )).images[1, 0, :, :, 0]
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
    mesh.displacement = renderverif.rcc_affine_displacements(mesh)
    actual = render.PixIntSpeck2D(
        renderverif.speckle_pattern(kind),
        options=render.PxInt2DOpts(
            mapping=mapping,
            integration=render.RectRule(samples),
        ),
    ).render(render.Scene2D(
        mesh=mesh, camera=renderverif.pxint2d_camera(),
    )).images[1, 0, :, :, 0]
    expected = np.load(GOLD / f"affine_speck_{kind}_rect{samples}.npy")
    assert_render_allclose(
        actual, expected,
        f"speck_{kind}_{element_type.value}_{mapping}_{samples}",
    )


def test_copied_rcc_analytic_gold_is_preserved() -> None:
    """The original RCC 32-pixel analytic reference remains reproducible."""
    mesh = renderverif.rcc_quad9_mesh("plate42_cam32_quad9_rigid")
    actual = render.PixIntGrid2D(
        options=render.PxInt2DOpts(
            mapping=render.EPxIntMapping.AFFINE,
            integration=render.AnalyticRule(),
        ),
    ).render(render.Scene2D(
        mesh=mesh, camera=renderverif.pxint2d_camera(),
    )).images[0, 0, :, :, 0]
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
    ).render(
        render.Scene2D(
            mesh=render.Mesh2D(
                mesh.element_type,
                mesh.coords,
                mesh.connectivity,
                affine_displacements(mesh),
            ),
            camera=renderverif.pxint2d_camera(),
        ),
    )
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
        ).render(render.Scene2D(mesh=mesh, camera=renderverif.pxint2d_camera()))

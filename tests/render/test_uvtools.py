# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ============================================================================

"""Tests for renderer-independent UV generation and transforms."""

from __future__ import annotations

import numpy as np
import pytest
import riley

from pyvale import render


RECTANGLE = np.array(
    ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
     (2.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
)


def test_pixel_uv_round_trip_for_both_origins() -> None:
    pixels = np.array(((0.0, 0.0), (200.0, 100.0), (81.5, 22.25)))
    for origin in render.EUVOrigin:
        uvs = render.uv_from_pixels(pixels, (101, 201), origin)
        actual = render.uv_to_pixels(uvs, (101, 201), origin)
        assert np.allclose(actual, pixels)
        assert actual.flags.c_contiguous
        assert actual.dtype == np.float64


@pytest.mark.parametrize(
    ("plane", "coords"),
    (
        (render.EUVPlane.XY, RECTANGLE),
        (render.EUVPlane.YZ, RECTANGLE[:, (2, 0, 1)]),
        (render.EUVPlane.XZ, RECTANGLE[:, (0, 2, 1)]),
    ),
)
def test_axis_aligned_projection(plane: render.EUVPlane,
                                 coords: np.ndarray) -> None:
    uvs = render.uv_project_planar_centered(
        coords, (101, 201), span=0.8, plane=plane,
    )
    assert uvs.shape == (4, 2)
    assert np.all((uvs >= 0.0) & (uvs <= 1.0))
    assert np.isclose(np.ptp(uvs[:, 0]), 0.8)
    assert np.isclose(np.ptp(uvs[:, 1]), 0.8)


def test_stretch_fills_requested_uv_bounds() -> None:
    uvs = render.uv_project_planar(
        RECTANGLE,
        uv_bounds=(0.2, 0.1, 0.7, 0.9),
        fit=render.EUVFit.STRETCH,
        texture_shape=(101, 201),
    )
    assert np.allclose(np.min(uvs, axis=0), (0.2, 0.1))
    assert np.allclose(np.max(uvs, axis=0), (0.7, 0.9))


def test_custom_plane_with_up_controls_orientation() -> None:
    angle = np.deg2rad(30.0)
    axis_u = np.array((np.cos(angle), 0.0, -np.sin(angle)))
    axis_v = np.array((0.0, 1.0, 0.0))
    normal = np.cross(axis_u, axis_v)
    coords = RECTANGLE[:, [0]] * axis_u + RECTANGLE[:, [1]] * axis_v
    plane = render.UVPlane(normal, np.zeros(3), up=axis_v)
    uvs = render.uv_project_planar_centered(
        coords, (101, 201), plane=plane,
    )
    assert np.allclose(np.ptp(uvs, axis=0), (1.0, 1.0))


def test_uv_transform_uses_scale_rotate_translate_order() -> None:
    uvs = np.array(((0.5, 0.0), (1.0, 0.5)))
    transformed = render.uv_transform(
        uvs,
        render.UVTransform(
            translation=(0.1, -0.2),
            rotation_degrees=90.0,
            scale=(2.0, 1.0),
            pivot=(0.5, 0.5),
        ),
    )
    assert np.allclose(transformed, ((1.1, 0.3), (0.6, 1.3)))


def test_feature_scale_conversions_match_analytic_reference() -> None:
    """Texture, physical, and image feature scales remain consistent."""
    image_leng_per_px = 0.2
    image_px_per_feature = 5.0
    texture_px_per_feature = 10.0

    feature_leng = render.uv_calc_feature_leng(
        image_px_per_feature,
        image_leng_per_px,
    )
    texture_px_per_leng = render.uv_calc_texture_px_per_leng_from_image(
        texture_px_per_feature,
        image_px_per_feature,
        image_leng_per_px,
    )

    assert feature_leng == pytest.approx(1.0)
    assert texture_px_per_leng == pytest.approx(10.0)
    assert render.uv_calc_image_px_per_feature(
        feature_leng,
        image_leng_per_px,
    ) == pytest.approx(image_px_per_feature)


def test_scaled_planar_mapping_has_requested_texture_scale() -> None:
    """A unit world displacement maps to the requested texture pixels."""
    texture = np.zeros((101, 201), dtype=np.uint8)
    mapping = render.uv_map_planar_scaled(
        RECTANGLE,
        texture,
        texture_px_per_leng=20.0,
    )
    pixels = render.uv_to_pixels(mapping.uvs, texture.shape)

    assert np.ptp(pixels[:, 0]) == pytest.approx(40.0)
    assert np.ptp(pixels[:, 1]) == pytest.approx(20.0)
    assert mapping.texture is texture
    assert mapping.tile_counts == (1, 1)


def test_scaled_planar_mapping_saturates_with_warning() -> None:
    """The default bounds policy clips a mapping that exceeds its texture."""
    texture = np.zeros((11, 11), dtype=np.uint8)
    with pytest.warns(UserWarning, match="saturated"):
        mapping = render.uv_map_planar_scaled(
            RECTANGLE,
            texture,
            texture_px_per_leng=20.0,
        )

    assert np.all((mapping.uvs >= 0.0) & (mapping.uvs <= 1.0))
    assert np.any((mapping.uvs == 0.0) | (mapping.uvs == 1.0))


def test_scaled_planar_mapping_tiles_without_changing_scale() -> None:
    """Tiling expands the texture and retains texture pixels per length."""
    texture = np.arange(121, dtype=np.uint8).reshape(11, 11)
    mapping = render.uv_map_planar_scaled(
        RECTANGLE,
        texture,
        texture_px_per_leng=20.0,
        bounds=render.EUVBounds.TILED,
    )
    pixels = render.uv_to_pixels(mapping.uvs, mapping.texture.shape[:2])

    assert mapping.tile_counts == (5, 3)
    assert mapping.texture.shape == (33, 55)
    assert np.ptp(pixels[:, 0]) == pytest.approx(40.0)
    assert np.ptp(pixels[:, 1]) == pytest.approx(20.0)
    assert np.array_equal(mapping.texture[:11, :11], texture)


def test_riley_centered_projection_parity() -> None:
    expected = riley.project_uvs_planar_centered(
        RECTANGLE,
        (201, 101),
        uv_span_max=0.8,
        proj_plane=riley.EProjPlane.XY,
    )
    actual = render.uv_project_planar_centered(
        RECTANGLE,
        (101, 201),
        span=0.8,
        plane=render.EUVPlane.XY,
    )
    assert np.allclose(actual, expected)


def test_riley_pixel_bounds_projection_parity() -> None:
    expected = riley.project_uvs_planar_bbox(
        RECTANGLE,
        (201, 101),
        (20.0, 10.0, 180.0, 90.0),
        riley.EProjPlane.XY,
        riley.EPlanarProjMode.FIT_X,
    )
    actual = render.uv_project_planar_pixels(
        RECTANGLE,
        (101, 201),
        (20.0, 10.0, 180.0, 90.0),
        plane=render.EUVPlane.XY,
        fit=render.EUVFit.FIT_U,
    )
    assert np.allclose(actual, expected)


def test_riley_arbitrary_plane_projection_parity() -> None:
    normal = np.array((0.2, -0.4, 1.0))
    origin = np.array((0.5, -0.25, 0.75))
    axis_u = np.cross(np.array((0.0, 0.0, 1.0)), normal)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal / np.linalg.norm(normal), axis_u)
    coords = origin + RECTANGLE[:, [0]] * axis_u + RECTANGLE[:, [1]] * axis_v
    expected = riley.project_uvs_planar_centered(
        coords,
        (201, 101),
        uv_span_max=0.75,
        proj_plane=(normal, origin),
    )
    actual = render.uv_project_planar_centered(
        coords,
        (101, 201),
        span=0.75,
        plane=render.UVPlane(normal, origin),
    )
    assert np.allclose(actual, expected)


def test_lower_left_origin_reverses_v_axis() -> None:
    upper = render.uv_project_planar_centered(
        RECTANGLE, (101, 201), origin=render.EUVOrigin.UPPER_LEFT,
    )
    lower = render.uv_project_planar_centered(
        RECTANGLE, (101, 201), origin=render.EUVOrigin.LOWER_LEFT,
    )
    assert np.allclose(lower[:, 0], upper[:, 0])
    assert np.allclose(lower[:, 1], 1.0 - upper[:, 1])


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (lambda: render.uv_project_planar_centered(
            RECTANGLE, (1, 10)), "at least 2"),
        (lambda: render.uv_project_planar_centered(
            RECTANGLE[:2], (10, 10)), "zero area"),
        (lambda: render.uv_project_planar_centered(
            RECTANGLE, (10, 10), span=0.0), "span"),
        (lambda: render.uv_project_planar_centered(
            RECTANGLE, (10, 10), plane=render.UVPlane(
                np.zeros(3), np.zeros(3))), "nonzero"),
        (lambda: render.uv_project_planar_centered(
            RECTANGLE, (10, 10), plane=render.UVPlane(
                np.array((0.0, 0.0, 1.0)), np.zeros(3),
                np.array((0.0, 0.0, 2.0)))), "parallel"),
    ),
)
def test_invalid_inputs(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()

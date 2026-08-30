# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ============================================================================
"""Renderer independent tools for generating and transforming nodal UVs.

Texture shapes follow NumPy convention, ``(height, width)``. Pixel coordinates
refer to pixel centres, so the final pixel centres are at ``width - 1`` and
``height - 1``. UV arrays have shape ``(node_count, 2)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import warnings

import numpy as np


class EUVPlane(Enum):
    """Axis aligned plane used for planar UV projection."""

    XY = "xy"
    YZ = "yz"
    XZ = "xz"


class EUVFit(Enum):
    """Rule used to fit projected coordinates into texture bounds."""

    CONTAIN = "contain"
    FIT_U = "fit_u"
    FIT_V = "fit_v"
    STRETCH = "stretch"


class EUVOrigin(Enum):
    """Location of the texture-space V origin."""

    UPPER_LEFT = "upper_left"
    LOWER_LEFT = "lower_left"


class EUVBounds(Enum):
    """Handling for physically scaled UVs outside the source texture."""

    SATURATE = "saturate"
    TILED = "tiled"


@dataclass(frozen=True, slots=True)
class UVPlane:
    """Arbitrary projection plane.

    Parameters
    ----------
    normal : numpy.ndarray
        Nonzero three-component plane normal.
    origin : numpy.ndarray
        Three-component point on the plane.
    up : numpy.ndarray or None, optional
        Preferred positive V direction. Its component normal to the plane is
        removed. When omitted, a deterministic basis is constructed.
    """

    normal: np.ndarray
    origin: np.ndarray
    up: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class UVTransform:
    """Affine UV transform applied about a pivot before translation."""

    translation: tuple[float, float] = (0.0, 0.0)
    rotation_degrees: float = 0.0
    scale: tuple[float, float] = (1.0, 1.0)
    pivot: tuple[float, float] = (0.5, 0.5)


@dataclass(frozen=True, slots=True)
class UVMapping:
    """UV coordinates and the texture image they address.

    ``TILED`` mappings may contain an expanded texture assembled from the
    supplied source image. ``tile_counts`` is in ``(U, V)`` order.
    """

    uvs: np.ndarray
    texture: np.ndarray
    tile_counts: tuple[int, int] = (1, 1)


def _finite_array(
    values: object,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return finite contiguous float64 input with an optional exact shape."""
    array = np.ascontiguousarray(values, dtype=np.float64)

    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")

    return array


def _coords_array(coords: np.ndarray) -> np.ndarray:
    """Validate three-dimensional nodal coordinates."""
    array = _finite_array(coords, "coords")

    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 2:
        raise ValueError("coords must have shape (node_count, 3).")

    return array


def _uv_array(uvs: np.ndarray) -> np.ndarray:
    """Validate nodal UV coordinates."""
    array = _finite_array(uvs, "uvs")

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("uvs must have shape (node_count, 2).")

    return array


def _texture_size(texture_shape: tuple[int, int]) -> tuple[float, float]:
    """Return validated texture width and height from NumPy shape order."""
    shape = _finite_array(texture_shape, "texture_shape", (2,))

    if np.any(shape < 2.0):
        raise ValueError("Texture height and width must both be at least 2.")

    return float(shape[1]), float(shape[0])


def _plane_basis(
    plane: EUVPlane | UVPlane,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve a projection plane to an origin and orthonormal U/V basis."""
    zero = np.zeros(3, dtype=np.float64)

    if plane is EUVPlane.XY:
        return zero, np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0))

    if plane is EUVPlane.YZ:
        return zero, np.array((0.0, 1.0, 0.0)), np.array((0.0, 0.0, 1.0))

    if plane is EUVPlane.XZ:
        return zero, np.array((1.0, 0.0, 0.0)), np.array((0.0, 0.0, 1.0))

    if not isinstance(plane, UVPlane):
        raise ValueError(f"Unsupported UV projection plane: {plane!r}.")

    normal = _finite_array(plane.normal, "plane.normal", (3,))
    origin = _finite_array(plane.origin, "plane.origin", (3,))
    normal_norm = float(np.linalg.norm(normal))

    if normal_norm == 0.0:
        raise ValueError("plane.normal must be nonzero.")

    normal = normal / normal_norm

    if plane.up is not None:
        up = _finite_array(plane.up, "plane.up", (3,))
        v_axis = up - np.dot(up, normal) * normal
        v_norm = float(np.linalg.norm(v_axis))

        if v_norm <= np.finfo(np.float64).eps:
            raise ValueError("plane.up must not be parallel to plane.normal.")

        v_axis /= v_norm
        u_axis = np.cross(v_axis, normal)
        u_axis /= np.linalg.norm(u_axis)

        return origin, u_axis, v_axis

    if abs(normal[2]) < 0.999:
        u_axis = np.cross(np.array((0.0, 0.0, 1.0)), normal)
    else:
        u_axis = np.cross(normal, np.array((0.0, 1.0, 0.0)))

    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)

    return origin, u_axis, v_axis


def _project(coords: np.ndarray, plane: EUVPlane | UVPlane) -> np.ndarray:
    """Project coordinates onto the selected two-dimensional plane."""

    if plane is EUVPlane.XY:
        return coords[:, :2]

    if plane is EUVPlane.YZ:
        return coords[:, 1:3]

    if plane is EUVPlane.XZ:
        return coords[:, (0, 2)]

    origin, u_axis, v_axis = _plane_basis(plane)
    difference = coords - origin

    return np.column_stack((difference @ u_axis, difference @ v_axis))


def _bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return projected bounds and reject a zero-area projection."""
    lower = np.min(values, axis=0)
    upper = np.max(values, axis=0)
    extent = upper - lower

    if np.any(extent <= 0.0):
        raise ValueError("Projected mesh has zero area in the chosen plane.")

    return lower, upper, extent



def _fit_projected(
    projected: np.ndarray,
    pixel_bounds: tuple[float, float, float, float],
    fit: EUVFit,
) -> np.ndarray:
    """Fit projected coordinates into pixel bounds."""

    lower, upper, extent = _bounds(projected)
    bounds = _finite_array(pixel_bounds, "pixel_bounds", (4,))
    target_lower = bounds[:2]
    target_upper = bounds[2:]
    target_extent = target_upper - target_lower

    if np.any(target_extent <= 0.0):
        raise ValueError("pixel_bounds upper bounds must exceed lower bounds.")

    scale_axes = target_extent / extent

    if fit is EUVFit.CONTAIN:
        scale = np.repeat(np.min(scale_axes), 2)
    elif fit is EUVFit.FIT_U:
        scale = np.repeat(scale_axes[0], 2)
    elif fit is EUVFit.FIT_V:
        scale = np.repeat(scale_axes[1], 2)
    elif fit is EUVFit.STRETCH:
        scale = scale_axes
    else:
        raise ValueError(f"Unsupported UV fit mode: {fit!r}.")

    source_center = 0.5 * (lower + upper)
    target_center = 0.5 * (target_lower + target_upper)

    return target_center + (projected - source_center) * scale


def uv_from_pixels(
    pixel_coords: np.ndarray,
    texture_shape: tuple[int, int],
    origin: EUVOrigin = EUVOrigin.UPPER_LEFT,
) -> np.ndarray:
    """Convert pixel-centre coordinates to normalized UV coordinates."""

    pixels = _uv_array(pixel_coords)
    width, height = _texture_size(texture_shape)
    uvs = np.empty_like(pixels)
    uvs[:, 0] = pixels[:, 0] / (width - 1.0)

    if origin is EUVOrigin.UPPER_LEFT:
        uvs[:, 1] = 1.0 - pixels[:, 1] / (height - 1.0)
    elif origin is EUVOrigin.LOWER_LEFT:
        uvs[:, 1] = pixels[:, 1] / (height - 1.0)
    else:
        raise ValueError(f"Unsupported UV origin: {origin!r}.")

    return np.ascontiguousarray(uvs)


def uv_to_pixels(
    uvs: np.ndarray,
    texture_shape: tuple[int, int],
    origin: EUVOrigin = EUVOrigin.UPPER_LEFT,
) -> np.ndarray:
    """Convert normalized UV coordinates to pixel-centre coordinates."""

    uv_coords = _uv_array(uvs)
    width, height = _texture_size(texture_shape)
    pixels = np.empty_like(uv_coords)
    pixels[:, 0] = uv_coords[:, 0] * (width - 1.0)

    if origin is EUVOrigin.UPPER_LEFT:
        pixels[:, 1] = (1.0 - uv_coords[:, 1]) * (height - 1.0)
    elif origin is EUVOrigin.LOWER_LEFT:
        pixels[:, 1] = uv_coords[:, 1] * (height - 1.0)
    else:
        raise ValueError(f"Unsupported UV origin: {origin!r}.")

    return np.ascontiguousarray(pixels)


def uv_project_planar_pixels(
    coords: np.ndarray,
    texture_shape: tuple[int, int],
    pixel_bounds: tuple[float, float, float, float],
    plane: EUVPlane | UVPlane = EUVPlane.XY,
    fit: EUVFit = EUVFit.CONTAIN,
    origin: EUVOrigin = EUVOrigin.UPPER_LEFT,
) -> np.ndarray:
    """Project mesh coordinates into a texture-space pixel rectangle."""

    coords_in = _coords_array(coords)
    _texture_size(texture_shape)
    projected = _project(coords_in, plane)
    pixels = _fit_projected(projected, pixel_bounds, fit)

    return uv_from_pixels(pixels, texture_shape, origin)


def uv_project_planar(
    coords: np.ndarray,
    plane: EUVPlane | UVPlane = EUVPlane.XY,
    uv_bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    fit: EUVFit = EUVFit.CONTAIN,
    texture_shape: tuple[int, int] | None = None,
    origin: EUVOrigin = EUVOrigin.UPPER_LEFT,
) -> np.ndarray:
    """Project mesh coordinates into normalized UV bounds.

    When ``texture_shape`` is omitted, a square two-pixel texture is assumed
    for aspect fitting. Supply the actual image shape when its aspect ratio
    should influence ``CONTAIN``, ``FIT_U``, or ``FIT_V``.
    """
    if texture_shape is None:
        texture_shape = (2, 2)

    width, height = _texture_size(texture_shape)
    bounds = _finite_array(uv_bounds, "uv_bounds", (4,))

    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("uv_bounds upper bounds must exceed lower bounds.")

    x_bounds = bounds[[0, 2]] * (width - 1.0)
    if origin is EUVOrigin.UPPER_LEFT:
        y_bounds = (1.0 - bounds[[3, 1]]) * (height - 1.0)
    elif origin is EUVOrigin.LOWER_LEFT:
        y_bounds = bounds[[1, 3]] * (height - 1.0)
    else:
        raise ValueError(f"Unsupported UV origin: {origin!r}.")

    pixel_bounds = (x_bounds[0], y_bounds[0], x_bounds[1], y_bounds[1])

    return uv_project_planar_pixels(
        coords,
        texture_shape,
        pixel_bounds,
        plane=plane,
        fit=fit,
        origin=origin,
    )


def uv_project_planar_centered(
    coords: np.ndarray,
    texture_shape: tuple[int, int],
    *,
    span: float = 1.0,
    plane: EUVPlane | UVPlane = EUVPlane.XY,
    origin: EUVOrigin = EUVOrigin.UPPER_LEFT,
) -> np.ndarray:
    """Project coordinates into a centred aspect-preserving UV region."""

    if not np.isfinite(span) or not 0.0 < span <= 1.0:
        raise ValueError("span must be finite and in the interval (0, 1].")

    margin = 0.5 * (1.0 - span)

    return uv_project_planar(
        coords,
        plane,
        uv_bounds=(margin, margin, 1.0 - margin, 1.0 - margin),
        fit=EUVFit.CONTAIN,
        texture_shape=texture_shape,
        origin=origin,
    )


def uv_calc_feature_leng(
    image_px_per_feature: float,
    image_leng_per_px: float,
) -> float:
    """Calculate physical feature size or pitch from its rendered size."""
    values = _finite_array(
        (image_px_per_feature, image_leng_per_px),
        "feature scale inputs",
        (2,),
    )
    if np.any(values <= 0.0):
        raise ValueError("Feature scale inputs must be positive.")
    return float(values[0] * values[1])


def uv_calc_image_px_per_feature(
    feature_leng: float,
    image_leng_per_px: float,
) -> float:
    """Calculate rendered pixels per feature size or pitch."""
    values = _finite_array(
        (feature_leng, image_leng_per_px),
        "feature scale inputs",
        (2,),
    )
    if np.any(values <= 0.0):
        raise ValueError("Feature scale inputs must be positive.")
    return float(values[0] / values[1])


def uv_calc_texture_px_per_leng(
    texture_px_per_feature: float,
    feature_leng: float,
) -> float:
    """Calculate texture pixels per simulation length unit."""
    values = _finite_array(
        (texture_px_per_feature, feature_leng),
        "feature scale inputs",
        (2,),
    )
    if np.any(values <= 0.0):
        raise ValueError("Feature scale inputs must be positive.")
    return float(values[0] / values[1])


def uv_calc_texture_px_per_leng_from_image(
    texture_px_per_feature: float,
    image_px_per_feature: float,
    image_leng_per_px: float,
) -> float:
    """Calculate texture scale for a desired rendered feature size."""
    feature_leng = uv_calc_feature_leng(
        image_px_per_feature,
        image_leng_per_px,
    )
    return uv_calc_texture_px_per_leng(
        texture_px_per_feature,
        feature_leng,
    )


def uv_map_planar_scaled(
    coords: np.ndarray,
    texture: np.ndarray,
    texture_px_per_leng: float | np.ndarray,
    *,
    plane: EUVPlane | UVPlane = EUVPlane.XY,
    texture_center_px: np.ndarray | None = None,
    origin: EUVOrigin = EUVOrigin.UPPER_LEFT,
    bounds: EUVBounds = EUVBounds.SATURATE,
) -> UVMapping:
    """Map a planar surface using a fixed physical texture scale.

    The projected specimen centre is placed at ``texture_center_px``. When no
    centre is supplied, the centre of the source texture is used. Texture
    scale may be one isotropic value or independent ``(U, V)`` values.
    """
    coords_in = _coords_array(coords)
    texture_in = np.asarray(texture)

    if texture_in.ndim not in (2, 3):
        raise ValueError("texture must have shape (height, width[, channels]).")

    texture_shape = (int(texture_in.shape[0]), int(texture_in.shape[1]))
    width, height = _texture_size(texture_shape)
    scale = np.asarray(texture_px_per_leng, dtype=np.float64)

    if scale.ndim == 0:
        scale = np.repeat(scale, 2)

    if scale.shape != (2,) or not np.isfinite(scale).all():
        raise ValueError(
            "texture_px_per_leng must be finite and scalar or shape (2,)."
        )

    if np.any(scale <= 0.0):
        raise ValueError("texture_px_per_leng must be positive.")

    projected = _project(coords_in, plane)
    projected_center = 0.5 * (
        np.min(projected, axis=0) + np.max(projected, axis=0)
    )

    if texture_center_px is None:
        texture_center = np.array((0.5 * (width - 1.0), 0.5 * (height - 1.0)))
    else:
        texture_center = _finite_array(
            texture_center_px,
            "texture_center_px",
            (2,),
        )

    pixels = texture_center + (projected - projected_center) * scale
    raw_uvs = uv_from_pixels(pixels, texture_shape, origin)
    outside = np.any((raw_uvs < 0.0) | (raw_uvs > 1.0))

    if bounds is EUVBounds.SATURATE:
        if outside:
            warnings.warn(
                "Physically scaled UVs exceed the texture bounds and were "
                "saturated to [0, 1]. Use EUVBounds.TILED to preserve scale.",
                UserWarning,
                stacklevel=2,
            )
        return UVMapping(
            np.ascontiguousarray(np.clip(raw_uvs, 0.0, 1.0)),
            texture_in,
        )

    if bounds is not EUVBounds.TILED:
        raise ValueError(f"Unsupported UV bounds mode: {bounds!r}.")
    if not outside:
        return UVMapping(np.ascontiguousarray(raw_uvs), texture_in)

    source_size = np.array((texture_in.shape[1], texture_in.shape[0]))
    tile_lower = np.floor(np.min(pixels, axis=0) / source_size).astype(np.int64)
    tile_upper = np.floor(np.max(pixels, axis=0) / source_size).astype(np.int64)
    tile_counts_array = tile_upper - tile_lower + 1

    tile_u, tile_v = (int(value) for value in tile_counts_array)
    repetitions = (tile_v, tile_u) + (1,) * (texture_in.ndim - 2)

    tiled_texture = np.tile(texture_in, repetitions)
    tiled_pixels = pixels - tile_lower * source_size

    tiled_uvs = uv_from_pixels(
        tiled_pixels,
        (int(tiled_texture.shape[0]), int(tiled_texture.shape[1])),
        origin,
    )

    return UVMapping(
        np.ascontiguousarray(np.clip(tiled_uvs, 0.0, 1.0)),
        np.ascontiguousarray(tiled_texture),
        (tile_u, tile_v),
    )


def uv_transform(
    uvs: np.ndarray,
    transform: UVTransform,
) -> np.ndarray:
    """Scale and rotate UVs about a pivot, then apply translation."""

    uv_coords = _uv_array(uvs)
    translation = _finite_array(
        transform.translation, "transform.translation", (2,),
    )

    scale = _finite_array(transform.scale, "transform.scale", (2,))
    pivot = _finite_array(transform.pivot, "transform.pivot", (2,))
    angle = float(transform.rotation_degrees)

    if not np.isfinite(angle):
        raise ValueError("transform.rotation_degrees must be finite.")

    radians = np.deg2rad(angle)
    rotation = np.array(
        ((np.cos(radians), -np.sin(radians)),
         (np.sin(radians), np.cos(radians))),
    )

    transformed = (uv_coords - pivot) * scale
    transformed = transformed @ rotation.T
    transformed += pivot + translation

    return np.ascontiguousarray(transformed, dtype=np.float64)


__all__ = [
    "EUVBounds",
    "EUVFit",
    "EUVOrigin",
    "EUVPlane",
    "UVPlane",
    "UVMapping",
    "UVTransform",
    "uv_calc_feature_leng",
    "uv_calc_image_px_per_feature",
    "uv_calc_texture_px_per_leng",
    "uv_calc_texture_px_per_leng_from_image",
    "uv_from_pixels",
    "uv_map_planar_scaled",
    "uv_project_planar",
    "uv_project_planar_centered",
    "uv_project_planar_pixels",
    "uv_to_pixels",
    "uv_transform",
]

from __future__ import annotations

import os
import math
from dataclasses import dataclass
import json
from pathlib import Path, PureWindowsPath
import re
import tempfile
from typing import Literal

from matplotlib.path import Path as MatplotlibPath
import numpy as np
from PIL import Image, ImageDraw
import yaml


VfmInputRoiSourceKind = Literal["auto", "matchid", "yaml", "old-yaml", "mask-image", "mask-npy", "mask-text"]


@dataclass(frozen=True)
class VfmInputRoiArtifacts:
    source_kind: str
    roi_yaml: Path
    metadata_json: Path
    mask_tiff: Path
    overlay_image: Path | None
    roi_definition: RoiDefinition
    mask_shape: tuple[int, int]
    mask_pixel_count: int


@dataclass(frozen=True)
class RoiShape:
    shape_type: Literal["rectangle", "circle", "polygon"]
    index: int
    is_cutting: bool
    vertices: tuple[tuple[float, float], ...] = ()
    center: tuple[float, float] | None = None
    radius: float | None = None
    rectangle: tuple[float, float, float, float] | None = None
    initial_subset: tuple[float, float] | None = None
    local_subset_size: int | None = None
    local_step_size: int | None = None
    local_shape_function: int | None = None

    def rasterise(self, image_shape: tuple[int, int]) -> np.ndarray:
        height, width = _normalise_image_shape(image_shape)
        image = Image.new("1", (width, height), 0)
        drawer = ImageDraw.Draw(image)

        if self.shape_type == "rectangle":
            if self.rectangle is None:
                raise ValueError("Rectangle ROI shape is missing rectangle coordinates.")
            x_origin, y_origin, box_width, box_height = self.rectangle
            drawer.rectangle((x_origin, y_origin, x_origin + box_width, y_origin + box_height), fill=1)
        elif self.shape_type == "circle":
            if self.center is None or self.radius is None:
                raise ValueError("Circle ROI shape is missing its center or radius.")
            x_centre, y_centre = self.center
            radius = self.radius
            drawer.ellipse(
                (
                    x_centre - radius,
                    y_centre - radius,
                    x_centre + radius,
                    y_centre + radius,
                ),
                fill=1,
            )
        else:
            drawer.polygon(self.vertices, fill=1)
        return np.asarray(image, dtype=bool)


@dataclass(frozen=True)
class RoiDefinition:
    shapes: tuple[RoiShape, ...] = ()
    pixel_to_mm: float | None = None
    source_path: str | None = None
    mask_image_path: str | None = None
    mask_threshold: int = 0
    source_image_path: str | None = None
    generation: dict[str, object] = None  # type: ignore[assignment]
    metrics: dict[str, object] = None  # type: ignore[assignment]
    mask_array: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", {} if self.generation is None else dict(self.generation))
        object.__setattr__(self, "metrics", {} if self.metrics is None else dict(self.metrics))

    def rasterise_mask(self, image_shape: tuple[int, int] | None = None) -> np.ndarray:
        mask_from_shapes: np.ndarray | None = None
        if self.shapes:
            if image_shape is None and self.mask_image_path is None:
                raise ValueError("image_shape is required when rasterising a shape-only ROI.")
            if image_shape is None:
                image_shape = _load_mask_array(self.resolve_mask_image_path()).shape
            height, width = _normalise_image_shape(image_shape)
            mask_from_shapes = np.zeros((height, width), dtype=bool)
            for shape in self.shapes:
                shape_mask = shape.rasterise((height, width))
                if shape.is_cutting:
                    mask_from_shapes &= ~shape_mask
                else:
                    mask_from_shapes |= shape_mask

        mask_from_file: np.ndarray | None = None
        if self.mask_image_path is not None:
            resolved_mask_path = self.resolve_mask_image_path()
            mask_from_file = _load_mask_array(resolved_mask_path) > self.mask_threshold
        elif self.mask_array is not None:
            mask_from_file = np.asarray(self.mask_array, dtype=bool)

        if mask_from_shapes is not None and mask_from_file is not None:
            return mask_from_shapes & mask_from_file
        if mask_from_file is not None:
            return mask_from_file
        if mask_from_shapes is not None:
            return mask_from_shapes
        raise ValueError("ROI definition does not contain shapes or a mask image.")

    def resolve_mask_image_path(self) -> Path:
        if self.mask_image_path is None:
            raise ValueError("ROI definition does not reference a mask image.")
        candidate = Path(self.mask_image_path)
        if candidate.exists():
            return candidate
        if self.source_path is not None:
            source_parent = Path(self.source_path).parent
            sibling_names = {candidate.name, PureWindowsPath(self.mask_image_path).name}
            for sibling_name in sibling_names:
                sibling = source_parent / sibling_name
                if sibling.exists():
                    return sibling
        return candidate


@dataclass(frozen=True)
class VfmRegionOfInterest:
    roi_definition: RoiDefinition

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VfmRegionOfInterest":
        return cls(load_roi_yaml(path))

    @classmethod
    def from_definition(cls, roi_definition: RoiDefinition) -> "VfmRegionOfInterest":
        return cls(roi_definition)

    def save_yaml(self, path: str | Path) -> Path:
        return write_roi_yaml(self.roi_definition, path)

    def sample_specimen_mask(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        return sample_roi_definition_at_coordinates(self.roi_definition, x, y)


def load_roi_definition(
    input_path: str | Path,
    *,
    source_kind: VfmInputRoiSourceKind = "auto",
    reference_image: str | Path | None = None,
    image_shape: tuple[int, int] | None = None,
    simplification_pixels: float = 1.0,
) -> RoiDefinition:
    """Load a ROI source and normalise it to the canonical VFM ROI definition.

    Supported inputs currently include:

    - reference-image ROI definitions such as MatchID ``.m2inp``/``.m3inp``
    - pyvale ROI ``.yaml``/``.yml`` files
    - logical specimen masks stored as images, ``.npy`` arrays, or text grids
      such as ``.csv`` data

    Any supported source can be used, but more accurate ROI definitions usually
    lead to better downstream specimen metrics.

    FE mesh inputs are intended for a future implementation and are not yet
    supported here.
    """

    resolved_input = Path(input_path)
    resolved_reference_image = Path(reference_image) if reference_image is not None else None
    resolved_source_kind = infer_vfm_input_roi_source_kind(resolved_input, explicit_kind=source_kind)
    roi_definition, _ = _load_source_roi(
        resolved_input,
        source_kind=resolved_source_kind,
        reference_image=resolved_reference_image,
        explicit_image_shape=image_shape,
        simplification_pixels=simplification_pixels,
    )
    return roi_definition


def load_roi_yaml(path: str | Path) -> RoiDefinition:
    """Load a DIC-compatible ROI YAML file."""

    return _parse_roi_yaml(Path(path))


def rasterise_roi_definition(
    roi_definition: RoiDefinition,
    *,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Rasterise a ROI definition to a boolean mask."""

    return np.asarray(roi_definition.rasterise_mask(image_shape=image_shape), dtype=bool)


def sample_roi_mask_at_pixel_coordinates(
    roi_mask: np.ndarray,
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Sample a full-resolution ROI mask at pixel coordinates on a DIC grid."""

    if roi_mask.shape == x_pixels.shape:
        return roi_mask.astype(bool), {"mode": "direct-grid-match", "out_of_bounds_count": 0}

    valid_coords = np.isfinite(x_pixels) & np.isfinite(y_pixels)
    y_index = np.rint(y_pixels[valid_coords]).astype(np.int64)
    x_index = np.rint(x_pixels[valid_coords]).astype(np.int64)
    in_bounds = (
        (y_index >= 0)
        & (y_index < roi_mask.shape[0])
        & (x_index >= 0)
        & (x_index < roi_mask.shape[1])
    )
    if not np.any(in_bounds):
        raise ValueError(
            "The ROI mask shape does not match the DIC grid, and the finite x/y coordinates do not look like "
            "pixel locations inside the ROI mask image. The ROI could not be aligned to the measurement grid."
        )

    specimen_mask = np.zeros_like(valid_coords, dtype=bool)
    specimen_mask_indices = np.flatnonzero(valid_coords)
    specimen_mask.ravel()[specimen_mask_indices[in_bounds]] = roi_mask[y_index[in_bounds], x_index[in_bounds]]
    return specimen_mask, {
        "mode": "sample-full-resolution-mask-at-coordinate-pixels",
        "out_of_bounds_count": int(np.count_nonzero(~in_bounds)),
    }


def sample_roi_definition_at_pixel_coordinates(
    roi_definition: RoiDefinition,
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
    *,
    image_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Rasterise a ROI definition and sample it at pixel coordinates on a DIC grid."""

    roi_mask = rasterise_roi_definition(roi_definition, image_shape=image_shape)
    return sample_roi_mask_at_pixel_coordinates(roi_mask, x_pixels, y_pixels)


def sample_roi_definition_at_coordinates(
    roi_definition: RoiDefinition,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Evaluate a ROI definition directly on a physical coordinate grid."""

    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}.")

    valid_coords = np.isfinite(x) & np.isfinite(y)
    specimen_mask = np.zeros(x.shape, dtype=bool)
    if not np.any(valid_coords):
        return specimen_mask

    points_xy = np.column_stack((x[valid_coords], y[valid_coords]))
    boundary_tolerance = _estimate_coordinate_boundary_tolerance(x, y)
    evaluated_mask = np.zeros(points_xy.shape[0], dtype=bool)
    for shape in roi_definition.shapes:
        shape_mask = _evaluate_shape_at_points(shape, points_xy, boundary_tolerance=boundary_tolerance)
        if shape.is_cutting:
            evaluated_mask &= ~shape_mask
        else:
            evaluated_mask |= shape_mask

    specimen_mask[valid_coords] = evaluated_mask
    return specimen_mask


def convert_roi_definition_to_physical_coordinates(
    roi_definition: RoiDefinition,
    *,
    x: np.ndarray,
    y: np.ndarray,
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
) -> RoiDefinition:
    """Map a pixel-space ROI definition onto the physical DIC coordinates."""

    mapped_shapes = tuple(
        _map_shape_to_physical_coordinates(
            shape,
            x=x,
            y=y,
            x_pixels=x_pixels,
            y_pixels=y_pixels,
        )
        for shape in roi_definition.shapes
    )

    generation = dict(roi_definition.generation)
    generation["coordinate_space"] = "physical"
    generation["physical_mapping"] = {
        "method": "bilinear-interpolation-on-dic-grid",
    }

    return RoiDefinition(
        shapes=mapped_shapes,
        pixel_to_mm=None,
        source_path=roi_definition.source_path,
        mask_image_path=None,
        mask_threshold=0,
        source_image_path=roi_definition.source_image_path,
        generation=generation,
        metrics=dict(roi_definition.metrics),
    )


def convert_mask_to_physical_roi(
    mask: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    simplification_pixels: float = 0.0,
) -> RoiDefinition:
    """Convert a DIC-grid specimen mask into a physical-coordinate ROI definition."""

    grid_mask = np.asarray(mask, dtype=bool)
    if grid_mask.shape != x.shape or grid_mask.shape != y.shape:
        raise ValueError("mask, x, and y must all have the same shape.")

    index_center_x = np.broadcast_to(np.arange(grid_mask.shape[1], dtype=np.float64)[None, :] + 0.5, grid_mask.shape)
    index_center_y = np.broadcast_to(np.arange(grid_mask.shape[0], dtype=np.float64)[:, None] + 0.5, grid_mask.shape)
    index_space_definition = _mask_polygonised_roi_definition(
        grid_mask,
        pixel_to_mm=None,
        source_image_path=None,
        simplification_pixels=simplification_pixels,
    )
    return convert_roi_definition_to_physical_coordinates(
        index_space_definition,
        x=x,
        y=y,
        x_pixels=index_center_x,
        y_pixels=index_center_y,
    )


def generate_vfm_input_roi(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    source_kind: VfmInputRoiSourceKind = "auto",
    reference_image: str | Path | None = None,
    image_shape: tuple[int, int] | None = None,
    simplification_pixels: float = 1.0,
) -> VfmInputRoiArtifacts:
    """Generate a canonical ROI for VFM identification.

    Valid input files currently include:
      - MatchID input files: `.m2inp`, `.m3inp`
      - pyvale or legacy ROI YAML files: `.yaml`, `.yml`
      - binary mask images: `.tif`, `.tiff`, `.png`, `.bmp`, `.jpg`, `.jpeg`
      - NumPy masks: `.npy`
      - text masks: `.txt`, `.csv`, `.dat`

    Any supported source can be normalised to the canonical VFM ROI format.
    Reference-image ROI definitions are typically the most accurate. Logical
    masks are also supported, but the resulting geometry is limited by the
    source mask resolution.

    FE mesh files are not implemented yet, but they are an intended future ROI
    source for nodal specimen geometry.

    The output directory receives:
      - `<stem>_vfm_roi.yaml`
      - `<stem>_vfm_roi.metadata.json`
      - `<stem>_vfm_roi_mask.tiff`
      - optional `<stem>_vfm_roi_overlay.png` when `reference_image` is supplied
    """

    resolved_input = Path(input_path)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_reference_image = Path(reference_image) if reference_image is not None else None

    resolved_source_kind = infer_vfm_input_roi_source_kind(resolved_input, explicit_kind=source_kind)
    roi_definition, final_mask = _load_source_roi(
        resolved_input,
        source_kind=resolved_source_kind,
        reference_image=resolved_reference_image,
        explicit_image_shape=image_shape,
        simplification_pixels=simplification_pixels,
    )

    output_stem = f"{resolved_input.stem}_vfm_roi"
    roi_yaml = resolved_output_dir / f"{output_stem}.yaml"
    metadata_json = resolved_output_dir / f"{output_stem}.metadata.json"
    mask_tiff = resolved_output_dir / f"{output_stem}_mask.tiff"
    overlay_image = resolved_output_dir / f"{output_stem}_overlay.png" if resolved_reference_image is not None else None

    write_mask_tiff(final_mask, mask_tiff)
    _write_roi_yaml(
        roi_definition,
        roi_yaml,
    )
    stored_mask_path = _relative_or_absolute_path(mask_tiff, roi_yaml.parent)
    output_roi_definition = RoiDefinition(
        shapes=roi_definition.shapes,
        pixel_to_mm=roi_definition.pixel_to_mm,
        source_path=str(roi_yaml),
        mask_image_path=stored_mask_path,
        mask_threshold=0,
        source_image_path=(
            str(resolved_reference_image)
            if resolved_reference_image is not None
            else roi_definition.source_image_path
        ),
        generation=dict(roi_definition.generation),
        metrics=dict(roi_definition.metrics),
    )
    _write_vfm_roi_metadata_json(
        metadata_json,
        source_kind=resolved_source_kind,
        roi_yaml=roi_yaml,
        mask_tiff=mask_tiff,
        overlay_image=overlay_image,
        roi_definition=output_roi_definition,
        mask_shape=tuple(final_mask.shape),
        mask_pixel_count=int(np.count_nonzero(final_mask)),
    )

    if overlay_image is not None:
        reference_shape = load_grayscale_image(resolved_reference_image).shape
        if reference_shape != final_mask.shape:
            raise ValueError(
                f"Reference image '{resolved_reference_image}' has shape {reference_shape}, "
                f"but the ROI mask shape is {final_mask.shape}."
            )
        save_roi_overlay_plot(resolved_reference_image, final_mask, overlay_image)

    return VfmInputRoiArtifacts(
        source_kind=resolved_source_kind,
        roi_yaml=roi_yaml,
        metadata_json=metadata_json,
        mask_tiff=mask_tiff,
        overlay_image=overlay_image,
        roi_definition=output_roi_definition,
        mask_shape=tuple(final_mask.shape),
        mask_pixel_count=int(np.count_nonzero(final_mask)),
    )


def write_roi_yaml(
    roi_definition: RoiDefinition,
    path: str | Path,
) -> Path:
    """Write a ROI definition to the DIC-compatible YAML format."""

    return _write_roi_yaml(roi_definition, Path(path))


def infer_vfm_input_roi_source_kind(
    path: str | Path,
    *,
    explicit_kind: VfmInputRoiSourceKind = "auto",
) -> VfmInputRoiSourceKind:
    if explicit_kind == "old-yaml":
        return "yaml"
    if explicit_kind != "auto":
        return explicit_kind

    suffix = Path(path).suffix.lower()
    if suffix in {".m2inp", ".m3inp"}:
        return "matchid"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".npy":
        return "mask-npy"
    if suffix in {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}:
        return "mask-image"
    if suffix in {".txt", ".csv", ".dat"}:
        return "mask-text"
    raise ValueError(f"Could not infer source kind from '{path}'. Please set it explicitly.")


def _load_source_roi(
    input_path: Path,
    *,
    source_kind: ExternallyResolvedSourceKind,
    reference_image: Path | None,
    explicit_image_shape: tuple[int, int] | None,
    simplification_pixels: float,
) -> tuple[RoiDefinition, np.ndarray]:
    if source_kind == "matchid":
        source_roi_definition = parse_matchid_input_file(input_path)
        resolved_image_shape = _resolve_image_shape(
            roi_definition=source_roi_definition,
            reference_image=reference_image,
            explicit_image_shape=explicit_image_shape,
        )
        if source_roi_definition.shapes:
            geometry_roi_definition = _with_source_metadata(
                _shape_only_final_roi_definition(
                    source_roi_definition,
                    image_shape=resolved_image_shape,
                    simplification_pixels=simplification_pixels,
                ),
                origin="matchid_input",
                source_path=input_path,
            )
            return geometry_roi_definition, geometry_roi_definition.rasterise_mask(image_shape=resolved_image_shape)

        final_mask = np.asarray(source_roi_definition.rasterise_mask(image_shape=resolved_image_shape), dtype=bool)
        return (
            _with_source_metadata(
                _mask_polygonised_roi_definition(
                    final_mask,
                    pixel_to_mm=source_roi_definition.pixel_to_mm,
                    source_image_path=(
                        str(reference_image) if reference_image is not None else source_roi_definition.source_image_path
                    ),
                    simplification_pixels=simplification_pixels,
                ),
                origin="matchid_input",
                source_path=input_path,
            ),
            final_mask,
        )

    if source_kind == "yaml":
        source_roi_definition = _parse_roi_yaml(input_path)
        resolved_image_shape = explicit_image_shape
        if resolved_image_shape is None and reference_image is not None:
            resolved_image_shape = load_grayscale_image(reference_image).shape
        if resolved_image_shape is None:
            resolved_image_shape = _derive_shape_only_image_shape(source_roi_definition.shapes)
        geometry_roi_definition = _with_source_metadata(
            _shape_only_final_roi_definition(
                source_roi_definition,
                image_shape=resolved_image_shape,
                simplification_pixels=simplification_pixels,
            ),
            origin="roi_yaml",
            source_path=input_path,
        )
        return geometry_roi_definition, geometry_roi_definition.rasterise_mask(image_shape=resolved_image_shape)

    if source_kind == "mask-image":
        final_mask = _load_mask_image(input_path)
    elif source_kind == "mask-npy":
        final_mask = np.asarray(np.load(input_path), dtype=bool)
    elif source_kind == "mask-text":
        final_mask = np.asarray(np.loadtxt(input_path), dtype=bool)
    else:
        raise ValueError(f"Unsupported source kind '{source_kind}'.")

    final_mask = np.asarray(final_mask, dtype=bool)
    return (
        _with_source_metadata(
            _mask_polygonised_roi_definition(
                final_mask,
                pixel_to_mm=None,
                source_image_path=str(reference_image) if reference_image is not None else None,
                simplification_pixels=simplification_pixels,
            ),
            origin=source_kind,
            source_path=input_path,
        ),
        final_mask,
    )


ExternallyResolvedSourceKind = Literal["matchid", "yaml", "mask-image", "mask-npy", "mask-text"]


def _resolve_image_shape(
    *,
    roi_definition: RoiDefinition,
    reference_image: Path | None,
    explicit_image_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    if explicit_image_shape is not None:
        return explicit_image_shape
    if reference_image is not None:
        return load_grayscale_image(reference_image).shape
    if roi_definition.mask_image_path is not None:
        return _load_mask_image(roi_definition.resolve_mask_image_path()).shape
    return _derive_shape_only_image_shape(roi_definition.shapes)


def _shape_only_final_roi_definition(
    roi_definition: RoiDefinition,
    *,
    image_shape: tuple[int, int],
    simplification_pixels: float,
) -> RoiDefinition:
    final_mask = roi_definition.rasterise_mask(image_shape=image_shape)
    return _mask_polygonised_roi_definition(
        final_mask,
        pixel_to_mm=roi_definition.pixel_to_mm,
        source_image_path=roi_definition.source_image_path,
        simplification_pixels=simplification_pixels,
    )


def _mask_polygonised_roi_definition(
    mask: np.ndarray,
    *,
    pixel_to_mm: float | None,
    source_image_path: str | None,
    simplification_pixels: float,
) -> RoiDefinition:
    return RoiDefinition(
        shapes=_mask_to_polygon_roi_shapes(mask, simplification_pixels=simplification_pixels),
        pixel_to_mm=pixel_to_mm,
        source_image_path=source_image_path,
        generation={
            "finalisation": {
                "enabled": True,
                "method": "mask-boundary-polygonisation",
                "simplification_pixels": float(max(0.0, simplification_pixels)),
            }
        },
        metrics={},
    )


def _with_source_metadata(
    roi_definition: RoiDefinition,
    *,
    origin: str,
    source_path: Path,
) -> RoiDefinition:
    generation = dict(roi_definition.generation)
    generation["origin"] = origin
    generation["source_path"] = str(source_path)
    return RoiDefinition(
        shapes=roi_definition.shapes,
        pixel_to_mm=roi_definition.pixel_to_mm,
        source_path=roi_definition.source_path,
        mask_image_path=roi_definition.mask_image_path,
        mask_threshold=roi_definition.mask_threshold,
        source_image_path=roi_definition.source_image_path,
        generation=generation,
        metrics=dict(roi_definition.metrics),
    )


def _parse_roi_yaml(path: Path) -> RoiDefinition:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"ROI YAML '{path}' must contain a list of ROI entries.")

    shapes: list[RoiShape] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError(f"ROI YAML entry must be a mapping, got {type(entry)!r}.")
        if entry.get("type") == "SeedROI":
            continue
        shapes.append(_yaml_entry_to_shape(entry, index=len(shapes)))
    return RoiDefinition(shapes=tuple(shapes))


def _yaml_entry_to_shape(entry: dict[str, object], *, index: int) -> RoiShape:
    roi_type = str(entry.get("type", ""))
    is_cutting = not bool(entry.get("add", True))

    if roi_type == "RectROI":
        x_pos, y_pos = map(float, entry["pos"])
        box_width, box_height = map(float, entry["size"])
        return RoiShape(
            shape_type="polygon",
            index=index,
            is_cutting=is_cutting,
            vertices=(
                (x_pos, y_pos),
                (x_pos + box_width, y_pos),
                (x_pos + box_width, y_pos + box_height),
                (x_pos, y_pos + box_height),
            ),
        )
    if roi_type == "CircleROI":
        centre_x, centre_y = map(float, entry["pos"])
        box_width, box_height = map(float, entry["size"])
        return RoiShape(
            shape_type="polygon",
            index=index,
            is_cutting=is_cutting,
            vertices=_ellipse_vertices(
                centre_x=centre_x,
                centre_y=centre_y,
                radius_x=0.5 * box_width,
                radius_y=0.5 * box_height,
            ),
        )
    if roi_type == "PolyLineROI":
        return RoiShape(
            shape_type="polygon",
            index=index,
            is_cutting=is_cutting,
            vertices=tuple(tuple(map(float, point)) for point in entry["points"]),
        )
    raise ValueError(f"Unsupported legacy ROI entry type '{roi_type}'.")


def _evaluate_shape_at_points(
    shape: RoiShape,
    points_xy: np.ndarray,
    *,
    boundary_tolerance: float = 0.0,
) -> np.ndarray:
    if shape.shape_type == "polygon":
        if len(shape.vertices) < 3:
            return np.zeros(points_xy.shape[0], dtype=bool)
        path = MatplotlibPath(np.asarray(shape.vertices, dtype=np.float64), closed=True)
        return path.contains_points(points_xy, radius=max(1.0e-12, float(boundary_tolerance)))

    if shape.shape_type == "rectangle":
        if shape.rectangle is None:
            raise ValueError("Rectangle ROI shape is missing rectangle coordinates.")
        x_origin, y_origin, width, height = map(float, shape.rectangle)
        return (
            (points_xy[:, 0] >= x_origin - boundary_tolerance)
            & (points_xy[:, 0] <= x_origin + width + boundary_tolerance)
            & (points_xy[:, 1] >= y_origin - boundary_tolerance)
            & (points_xy[:, 1] <= y_origin + height + boundary_tolerance)
        )

    if shape.center is None or shape.radius is None:
        raise ValueError("Circle ROI shape is missing its center or radius.")
    return (
        (points_xy[:, 0] - float(shape.center[0])) ** 2
        + (points_xy[:, 1] - float(shape.center[1])) ** 2
        <= (float(shape.radius) + boundary_tolerance) ** 2
    )


def _map_shape_to_physical_coordinates(
    shape: RoiShape,
    *,
    x: np.ndarray,
    y: np.ndarray,
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
) -> RoiShape:
    if shape.shape_type == "polygon":
        mapped_vertices = _map_pixel_points_to_physical_coordinates(
            np.asarray(shape.vertices, dtype=np.float64),
            x=x,
            y=y,
            x_pixels=x_pixels,
            y_pixels=y_pixels,
        )
        return RoiShape(
            shape_type="polygon",
            index=shape.index,
            is_cutting=shape.is_cutting,
            vertices=tuple((float(x_coord), float(y_coord)) for x_coord, y_coord in mapped_vertices),
        )

    if shape.shape_type == "rectangle":
        if shape.rectangle is None:
            raise ValueError("Rectangle ROI shape is missing rectangle coordinates.")
        x_origin, y_origin, width, height = map(float, shape.rectangle)
        rectangle_vertices = np.asarray(
            (
                (x_origin, y_origin),
                (x_origin + width, y_origin),
                (x_origin + width, y_origin + height),
                (x_origin, y_origin + height),
            ),
            dtype=np.float64,
        )
        mapped_vertices = _map_pixel_points_to_physical_coordinates(
            rectangle_vertices,
            x=x,
            y=y,
            x_pixels=x_pixels,
            y_pixels=y_pixels,
        )
        return RoiShape(
            shape_type="polygon",
            index=shape.index,
            is_cutting=shape.is_cutting,
            vertices=tuple((float(x_coord), float(y_coord)) for x_coord, y_coord in mapped_vertices),
        )

    if shape.center is None or shape.radius is None:
        raise ValueError("Circle ROI shape is missing its center or radius.")
    circle_vertices = np.asarray(
        _ellipse_vertices(
            centre_x=float(shape.center[0]),
            centre_y=float(shape.center[1]),
            radius_x=float(shape.radius),
            radius_y=float(shape.radius),
        ),
        dtype=np.float64,
    )
    mapped_vertices = _map_pixel_points_to_physical_coordinates(
        circle_vertices,
        x=x,
        y=y,
        x_pixels=x_pixels,
        y_pixels=y_pixels,
    )
    return RoiShape(
        shape_type="polygon",
        index=shape.index,
        is_cutting=shape.is_cutting,
        vertices=tuple((float(x_coord), float(y_coord)) for x_coord, y_coord in mapped_vertices),
    )


def _ellipse_vertices(
    *,
    centre_x: float,
    centre_y: float,
    radius_x: float,
    radius_y: float,
    point_count: int = 64,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            centre_x + radius_x * math.cos(angle),
            centre_y + radius_y * math.sin(angle),
        )
        for angle in np.linspace(0.0, 2.0 * math.pi, point_count, endpoint=False)
    )


def _estimate_coordinate_boundary_tolerance(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x_diffs = np.abs(np.diff(np.asarray(x, dtype=np.float64), axis=1))
    y_diffs = np.abs(np.diff(np.asarray(y, dtype=np.float64), axis=0))
    finite_x_diffs = x_diffs[np.isfinite(x_diffs) & (x_diffs > 0.0)]
    finite_y_diffs = y_diffs[np.isfinite(y_diffs) & (y_diffs > 0.0)]

    spacings: list[float] = []
    if finite_x_diffs.size > 0:
        spacings.append(float(np.median(finite_x_diffs)))
    if finite_y_diffs.size > 0:
        spacings.append(float(np.median(finite_y_diffs)))
    if not spacings:
        return 1.0e-12
    return max(1.0e-12, 0.05 * min(spacings))


def _map_pixel_points_to_physical_coordinates(
    points_xy: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
) -> np.ndarray:
    if x.shape != y.shape or x.shape != x_pixels.shape or x.shape != y_pixels.shape:
        raise ValueError("x, y, x_pixels, and y_pixels must all have the same shape.")

    col_axis = _fill_missing_axis_values(_grid_axis_median(x_pixels, axis=0))
    row_axis = _fill_missing_axis_values(_grid_axis_median(y_pixels, axis=1))
    col_indices = _interpolate_axis_to_fractional_indices(col_axis, np.asarray(points_xy[:, 0], dtype=np.float64))
    row_indices = _interpolate_axis_to_fractional_indices(row_axis, np.asarray(points_xy[:, 1], dtype=np.float64))

    mapped_x = _bilinear_interpolate_grid(x, row_indices, col_indices)
    mapped_y = _bilinear_interpolate_grid(y, row_indices, col_indices)
    if np.any(~np.isfinite(mapped_x)) or np.any(~np.isfinite(mapped_y)):
        raise ValueError("Could not map all ROI vertices from pixel coordinates to physical coordinates.")

    return np.column_stack((mapped_x, mapped_y))


def _fill_missing_axis_values(axis_values: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis_values, dtype=np.float64).copy()
    finite_mask = np.isfinite(axis)
    if not np.any(finite_mask):
        raise ValueError("Could not determine interpolation axis from all-NaN coordinate data.")
    if np.all(finite_mask):
        return axis

    indices = np.arange(axis.size, dtype=np.float64)
    finite_indices = indices[finite_mask]
    finite_values = axis[finite_mask]
    axis[~finite_mask] = np.interp(indices[~finite_mask], finite_indices, finite_values)

    first_finite_index = int(finite_indices[0])
    last_finite_index = int(finite_indices[-1])
    if first_finite_index > 0 and finite_indices.size >= 2:
        leading_slope = (finite_values[1] - finite_values[0]) / (finite_indices[1] - finite_indices[0])
        for index in range(first_finite_index - 1, -1, -1):
            axis[index] = axis[index + 1] - leading_slope
    if last_finite_index < axis.size - 1 and finite_indices.size >= 2:
        trailing_slope = (finite_values[-1] - finite_values[-2]) / (finite_indices[-1] - finite_indices[-2])
        for index in range(last_finite_index + 1, axis.size):
            axis[index] = axis[index - 1] + trailing_slope
    return axis


def _grid_axis_median(grid: np.ndarray, *, axis: int) -> np.ndarray:
    if axis == 0:
        slices = [grid[:, index] for index in range(grid.shape[1])]
    elif axis == 1:
        slices = [grid[index, :] for index in range(grid.shape[0])]
    else:
        raise ValueError("axis must be 0 or 1")

    medians = np.empty(len(slices), dtype=np.float64)
    for index, values in enumerate(slices):
        finite_values = np.asarray(values, dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        medians[index] = float(np.median(finite_values)) if finite_values.size > 0 else np.nan
    return medians


def _interpolate_axis_to_fractional_indices(
    axis_values: np.ndarray,
    target_values: np.ndarray,
) -> np.ndarray:
    axis = np.asarray(axis_values, dtype=np.float64)
    targets = np.asarray(target_values, dtype=np.float64)
    if axis.ndim != 1:
        raise ValueError("Interpolation axis must be one-dimensional.")
    if axis.size < 2:
        raise ValueError("Interpolation axis must contain at least two values.")
    if np.any(~np.isfinite(axis)):
        raise ValueError("Interpolation axis contains non-finite values.")

    increasing = axis[-1] >= axis[0]
    if increasing:
        interp_axis = axis
        interp_indices = np.arange(axis.size, dtype=np.float64)
        low_slope = 1.0 / (axis[1] - axis[0])
        high_slope = 1.0 / (axis[-1] - axis[-2])
    else:
        interp_axis = axis[::-1]
        interp_indices = np.arange(axis.size, dtype=np.float64)[::-1]
        low_slope = 1.0 / (interp_axis[1] - interp_axis[0])
        high_slope = 1.0 / (interp_axis[-1] - interp_axis[-2])

    fractional_indices = np.interp(targets, interp_axis, interp_indices)
    below = targets < interp_axis[0]
    above = targets > interp_axis[-1]
    if np.any(below):
        fractional_indices[below] = interp_indices[0] + (targets[below] - interp_axis[0]) * low_slope
    if np.any(above):
        fractional_indices[above] = interp_indices[-1] + (targets[above] - interp_axis[-1]) * high_slope
    return fractional_indices


def _bilinear_interpolate_grid(
    grid: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
) -> np.ndarray:
    n_rows, n_cols = grid.shape
    clamped_rows = np.clip(row_indices, 0.0, n_rows - 1.0)
    clamped_cols = np.clip(col_indices, 0.0, n_cols - 1.0)

    row0 = np.floor(clamped_rows).astype(np.int64)
    col0 = np.floor(clamped_cols).astype(np.int64)
    row1 = np.clip(row0 + 1, 0, n_rows - 1)
    col1 = np.clip(col0 + 1, 0, n_cols - 1)

    row_weight = clamped_rows - row0
    col_weight = clamped_cols - col0

    sampled_values = np.empty(clamped_rows.shape[0], dtype=np.float64)
    for point_index in range(sampled_values.shape[0]):
        corner_values = np.asarray(
            (
                grid[row0[point_index], col0[point_index]],
                grid[row0[point_index], col1[point_index]],
                grid[row1[point_index], col0[point_index]],
                grid[row1[point_index], col1[point_index]],
            ),
            dtype=np.float64,
        )
        corner_weights = np.asarray(
            (
                (1.0 - row_weight[point_index]) * (1.0 - col_weight[point_index]),
                (1.0 - row_weight[point_index]) * col_weight[point_index],
                row_weight[point_index] * (1.0 - col_weight[point_index]),
                row_weight[point_index] * col_weight[point_index],
            ),
            dtype=np.float64,
        )
        finite_mask = np.isfinite(corner_values)
        if not np.any(finite_mask):
            sampled_values[point_index] = _nearest_finite_grid_value(
                grid,
                row0[point_index],
                col0[point_index],
            )
            continue
        sampled_values[point_index] = float(
            np.sum(corner_values[finite_mask] * corner_weights[finite_mask])
            / np.sum(corner_weights[finite_mask])
        )
    return sampled_values


def _nearest_finite_grid_value(
    grid: np.ndarray,
    row_index: int,
    col_index: int,
) -> float:
    max_radius = max(grid.shape)
    for radius in range(max_radius):
        row_min = max(0, row_index - radius)
        row_max = min(grid.shape[0], row_index + radius + 1)
        col_min = max(0, col_index - radius)
        col_max = min(grid.shape[1], col_index + radius + 1)
        window = np.asarray(grid[row_min:row_max, col_min:col_max], dtype=np.float64)
        finite_values = window[np.isfinite(window)]
        if finite_values.size > 0:
            return float(finite_values[0])
    return float("nan")


def _derive_shape_only_image_shape(shapes: tuple[RoiShape, ...]) -> tuple[int, int]:
    if not shapes:
        raise ValueError("Could not determine image shape from an empty shape-only ROI.")

    max_x = 0.0
    max_y = 0.0
    for shape in shapes:
        if shape.shape_type == "polygon":
            for x_coord, y_coord in shape.vertices:
                max_x = max(max_x, float(x_coord))
                max_y = max(max_y, float(y_coord))
        elif shape.shape_type == "rectangle" and shape.rectangle is not None:
            x_origin, y_origin, width, height = shape.rectangle
            max_x = max(max_x, float(x_origin + width))
            max_y = max(max_y, float(y_origin + height))
        elif shape.shape_type == "circle" and shape.center is not None and shape.radius is not None:
            centre_x, centre_y = shape.center
            max_x = max(max_x, float(centre_x + shape.radius))
            max_y = max(max_y, float(centre_y + shape.radius))

    return int(math.ceil(max_y)) + 2, int(math.ceil(max_x)) + 2


def _load_mask_image(path: Path) -> np.ndarray:
    image = np.asarray(load_grayscale_image(path))
    return np.asarray(image > 0, dtype=bool)


def _relative_or_absolute_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def _write_roi_yaml(
    roi_definition: RoiDefinition,
    path: Path,
) -> Path:
    serialised = [_yaml_entry_from_shape(shape) for shape in roi_definition.shapes]
    path.write_text(yaml.safe_dump(serialised, sort_keys=False), encoding="utf-8")
    return path


def _yaml_entry_from_shape(shape: RoiShape) -> dict[str, object]:
    add_flag = not shape.is_cutting
    if shape.shape_type == "polygon":
        return {
            "type": "PolyLineROI",
            "points": [[float(x_coord), float(y_coord)] for x_coord, y_coord in shape.vertices],
            "add": add_flag,
        }
    if shape.shape_type == "rectangle":
        if shape.rectangle is None:
            raise ValueError("Rectangle ROI shape is missing rectangle coordinates.")
        x_origin, y_origin, width, height = shape.rectangle
        return {
            "type": "RectROI",
            "pos": [float(x_origin), float(y_origin)],
            "size": [float(width), float(height)],
            "add": add_flag,
        }
    if shape.center is None or shape.radius is None:
        raise ValueError("Circle ROI shape is missing its centre or radius.")
    diameter = 2.0 * float(shape.radius)
    return {
        "type": "CircleROI",
        "pos": [float(shape.center[0]), float(shape.center[1])],
        "size": [diameter, diameter],
        "add": add_flag,
    }


def _write_vfm_roi_metadata_json(
    path: Path,
    *,
    source_kind: str,
    roi_yaml: Path,
    mask_tiff: Path,
    overlay_image: Path | None,
    roi_definition: RoiDefinition,
    mask_shape: tuple[int, int],
    mask_pixel_count: int,
) -> Path:
    payload = {
        "source_kind": source_kind,
        "roi_yaml": str(roi_yaml),
        "mask_tiff": str(mask_tiff),
        "overlay_image": str(overlay_image) if overlay_image is not None else None,
        "mask_shape": list(mask_shape),
        "mask_pixel_count": int(mask_pixel_count),
        "pixel_to_mm": roi_definition.pixel_to_mm,
        "source_image_path": roi_definition.source_image_path,
        "generation": roi_definition.generation,
        "metrics": roi_definition.metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def parse_matchid_input_file(path: str | Path) -> RoiDefinition:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    conversion_match = re.search(r"<Conversion>=<([^>]+)>", text)
    pixel_to_mm = float(conversion_match.group(1)) if conversion_match else None

    shapes: list[RoiShape] = []
    mask_image_path: str | None = None
    mask_threshold = 0
    generation: dict[str, object] = {"origin": "matchid_input"}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("<Shape>=<"):
            shapes.append(_parse_shape_line(line))
            continue
        if line.startswith("<AutoMask>=<"):
            auto_mask_info = _parse_automask_line(line)
            generation["automask"] = auto_mask_info
            if auto_mask_info["type"] == 5 and auto_mask_info.get("mask_image_path") is not None:
                mask_image_path = str(auto_mask_info["mask_image_path"])
            elif auto_mask_info["type"] == 3:
                generation["automask_threshold"] = auto_mask_info.get("threshold")

    return RoiDefinition(
        shapes=tuple(shapes),
        pixel_to_mm=pixel_to_mm,
        source_path=str(source),
        mask_image_path=mask_image_path,
        mask_threshold=mask_threshold,
        generation=generation,
    )


def _parse_shape_line(line: str) -> RoiShape:
    extracted = re.search(r"<Shape>=<(.+)>", line)
    if extracted is None:
        raise ValueError(f"Could not parse ROI shape line: {line}")
    tokens = [token.strip() for token in extracted.group(1).split(";") if token.strip()]
    if len(tokens) < 3:
        raise ValueError(f"ROI shape line is incomplete: {line}")

    form = int(tokens[0])
    index = int(tokens[1])
    is_cutting = tokens[2].lower() == "true"
    numbers = [float(token) for token in tokens[3:]]

    if form == 0:
        if len(numbers) < 4:
            raise ValueError(f"Rectangle ROI line is incomplete: {line}")
        rectangle = tuple(numbers[:4])
        trailing = numbers[4:]
        initial_subset, local_subset_size, local_step_size, local_shape_function = _parse_trailing_numbers(
            trailing,
            is_cutting,
        )
        return RoiShape(
            shape_type="rectangle",
            index=index,
            is_cutting=is_cutting,
            rectangle=rectangle,  # type: ignore[arg-type]
            initial_subset=initial_subset,
            local_subset_size=local_subset_size,
            local_step_size=local_step_size,
            local_shape_function=local_shape_function,
        )

    if form == 1:
        if len(numbers) < 3:
            raise ValueError(f"Circle ROI line is incomplete: {line}")
        centre = (numbers[0], numbers[1])
        radius = numbers[2]
        trailing = numbers[3:]
        if len(trailing) >= 6:
            trailing = trailing[6:]
        initial_subset, local_subset_size, local_step_size, local_shape_function = _parse_trailing_numbers(
            trailing,
            is_cutting,
        )
        return RoiShape(
            shape_type="circle",
            index=index,
            is_cutting=is_cutting,
            center=centre,
            radius=radius,
            initial_subset=initial_subset,
            local_subset_size=local_subset_size,
            local_step_size=local_step_size,
            local_shape_function=local_shape_function,
        )

    if form == 2:
        if len(numbers) < 1:
            raise ValueError(f"Polygon ROI line is incomplete: {line}")
        point_count = int(numbers[0])
        coord_count = point_count * 2
        coords = numbers[1 : 1 + coord_count]
        if len(coords) != coord_count:
            raise ValueError(f"Polygon ROI line is missing vertices: {line}")
        vertices = tuple((coords[i], coords[i + 1]) for i in range(0, len(coords), 2))
        trailing = numbers[1 + coord_count :]
        initial_subset, local_subset_size, local_step_size, local_shape_function = _parse_trailing_numbers(
            trailing,
            is_cutting,
        )
        return RoiShape(
            shape_type="polygon",
            index=index,
            is_cutting=is_cutting,
            vertices=vertices,
            initial_subset=initial_subset,
            local_subset_size=local_subset_size,
            local_step_size=local_step_size,
            local_shape_function=local_shape_function,
        )

    raise ValueError(f"Unsupported ROI shape form '{form}' in line: {line}")


def _parse_automask_line(line: str) -> dict[str, object]:
    extracted = re.search(r"<AutoMask>=<(.+)>", line)
    if extracted is None:
        raise ValueError(f"Could not parse automask line: {line}")
    tokens = [token.strip() for token in extracted.group(1).split(";") if token.strip()]
    if len(tokens) != 3:
        raise ValueError(f"AutoMask line is incomplete: {line}")

    roi_index = int(tokens[0])
    automask_type = int(tokens[1])
    if automask_type == 3:
        return {
            "roi_index": roi_index,
            "type": automask_type,
            "threshold": float(tokens[2]),
        }
    return {
        "roi_index": roi_index,
        "type": automask_type,
        "mask_image_path": tokens[2],
    }


def _parse_trailing_numbers(
    trailing: list[float],
    is_cutting: bool,
) -> tuple[tuple[float, float] | None, int | None, int | None, int | None]:
    initial_subset: tuple[float, float] | None = None
    remaining = list(trailing)
    if not is_cutting and len(remaining) >= 2:
        initial_subset = (remaining[0], remaining[1])
        remaining = remaining[2:]

    local_subset_size = _optional_int(remaining[0]) if len(remaining) >= 1 else None
    local_step_size = _optional_int(remaining[1]) if len(remaining) >= 2 else None
    local_shape_function = _optional_int(remaining[2]) if len(remaining) >= 3 else None
    return initial_subset, local_subset_size, local_step_size, local_shape_function


def _mask_to_polygon_roi_shapes(
    mask: np.ndarray,
    *,
    simplification_pixels: float,
) -> tuple[RoiShape, ...]:
    contours = _mask_boundary_contours(mask)
    shapes: list[RoiShape] = []
    for contour in contours:
        if contour.shape[0] < 4:
            continue
        simplified = _simplify_closed_polyline(contour, tolerance=max(0.0, float(simplification_pixels)))
        if simplified.shape[0] < 3:
            continue
        area = _signed_polygon_area(simplified)
        if abs(area) < 1.0:
            continue
        vertices = tuple((float(x_coord), float(y_coord)) for x_coord, y_coord in simplified)
        if area > 0.0:
            shapes.append(RoiShape(shape_type="polygon", index=len(shapes), is_cutting=False, vertices=vertices))
        else:
            shapes.append(
                RoiShape(shape_type="polygon", index=len(shapes), is_cutting=True, vertices=tuple(reversed(vertices)))
            )
    return tuple(shapes)


def _mask_boundary_contours(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    boolean_mask = np.asarray(mask, dtype=bool)
    if not boolean_mask.any():
        return ()
    height, width = boolean_mask.shape[:2]
    edge_successors: dict[tuple[float, float], list[tuple[float, float]]] = {}

    def _add_edge(start: tuple[float, float], stop: tuple[float, float]) -> None:
        edge_successors.setdefault(start, []).append(stop)

    for y_coord, x_coord in zip(*np.nonzero(boolean_mask)):
        x = float(x_coord)
        y = float(y_coord)
        if y_coord == 0 or not boolean_mask[y_coord - 1, x_coord]:
            _add_edge((x, y), (x + 1.0, y))
        if x_coord == width - 1 or not boolean_mask[y_coord, x_coord + 1]:
            _add_edge((x + 1.0, y), (x + 1.0, y + 1.0))
        if y_coord == height - 1 or not boolean_mask[y_coord + 1, x_coord]:
            _add_edge((x + 1.0, y + 1.0), (x, y + 1.0))
        if x_coord == 0 or not boolean_mask[y_coord, x_coord - 1]:
            _add_edge((x, y + 1.0), (x, y))

    contours: list[np.ndarray] = []
    while edge_successors:
        start = next(iter(edge_successors))
        contour = [start]
        current = start
        previous_direction: tuple[float, float] | None = None
        while True:
            successors = edge_successors.get(current)
            if not successors:
                break
            next_point = _pop_next_boundary_point(edge_successors, current, previous_direction)
            if next_point is None:
                break
            direction = (next_point[0] - current[0], next_point[1] - current[1])
            current = next_point
            if current == start:
                break
            contour.append(current)
            previous_direction = direction
        if len(contour) >= 3:
            contours.append(np.asarray(contour, dtype=np.float64))
    contours.sort(key=lambda polygon: abs(_signed_polygon_area(polygon)), reverse=True)
    return tuple(contours)


def _pop_next_boundary_point(
    successors_by_start: dict[tuple[float, float], list[tuple[float, float]]],
    current: tuple[float, float],
    previous_direction: tuple[float, float] | None,
) -> tuple[float, float] | None:
    successors = successors_by_start.get(current)
    if not successors:
        return None
    if len(successors) == 1 or previous_direction is None:
        next_index = 0
    else:
        direction_rank = {
            (previous_direction[1], -previous_direction[0]): 0,
            previous_direction: 1,
            (-previous_direction[1], previous_direction[0]): 2,
            (-previous_direction[0], -previous_direction[1]): 3,
        }
        next_index = min(
            range(len(successors)),
            key=lambda index: direction_rank.get(
                (successors[index][0] - current[0], successors[index][1] - current[1]),
                4,
            ),
        )
    next_point = successors.pop(next_index)
    if successors:
        successors_by_start[current] = successors
    else:
        del successors_by_start[current]
    return next_point


def _signed_polygon_area(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape[0] < 3:
        return 0.0
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return 0.5 * float(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1)))


def _simplify_closed_polyline(points_xy: np.ndarray, *, tolerance: float) -> np.ndarray:
    points = _remove_collinear_closed_vertices(np.asarray(points_xy, dtype=np.float64))
    if tolerance <= 0.0 or points.shape[0] <= 3:
        return points
    open_points = np.vstack([points, points[0]])
    keep = _douglas_peucker_keep_mask(open_points, tolerance=tolerance)
    simplified = open_points[keep][:-1]
    simplified = _remove_collinear_closed_vertices(simplified)
    return simplified if simplified.shape[0] >= 3 else points


def _douglas_peucker_keep_mask(points_xy: np.ndarray, *, tolerance: float) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    keep = np.zeros(points.shape[0], dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack: list[tuple[int, int]] = [(0, points.shape[0] - 1)]
    while stack:
        start_index, stop_index = stack.pop()
        if stop_index <= start_index + 1:
            continue
        start = points[start_index]
        stop = points[stop_index]
        segment = stop - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length < 1.0e-9:
            distances = np.linalg.norm(points[start_index + 1 : stop_index] - start, axis=1)
        else:
            offsets = points[start_index + 1 : stop_index] - start
            distances = np.abs(segment[0] * offsets[:, 1] - segment[1] * offsets[:, 0]) / segment_length
        if distances.size == 0:
            continue
        relative_index = int(np.argmax(distances))
        max_distance = float(distances[relative_index])
        if max_distance <= tolerance:
            continue
        split_index = start_index + 1 + relative_index
        keep[split_index] = True
        stack.append((start_index, split_index))
        stack.append((split_index, stop_index))
    return keep


def _remove_collinear_closed_vertices(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape[0] <= 3:
        return points.copy()
    keep: list[np.ndarray] = []
    for index in range(points.shape[0]):
        previous = points[index - 1]
        current = points[index]
        next_point = points[(index + 1) % points.shape[0]]
        vec_a = current - previous
        vec_b = next_point - current
        cross = float(vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0])
        if abs(cross) > 1.0e-9:
            keep.append(current)
    if len(keep) < 3:
        return points.copy()
    return np.asarray(keep, dtype=np.float64)


def write_mask_tiff(mask: np.ndarray, path: str | Path) -> Path:
    destination = Path(path)
    image_array = np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)
    Image.fromarray(image_array).save(destination)
    return destination


def load_grayscale_image(path: str | Path) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim == 3:
        image = image[..., 0]
    return image


def save_roi_overlay_plot(
    image: str | Path | np.ndarray,
    roi_mask: np.ndarray,
    output_path: str | Path,
    *,
    alpha: float = 0.35,
) -> Path:
    pyplot = _load_pyplot()
    image_array = load_grayscale_image(image) if isinstance(image, (str, Path)) else np.asarray(image)
    mask = np.asarray(roi_mask, dtype=bool)
    masked_overlay = np.ma.masked_where(~mask, mask.astype(np.float64))
    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.imshow(image_array, cmap="gray")
    ax.imshow(masked_overlay, cmap="summer", alpha=alpha, interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.contour(mask.astype(np.float64), levels=[0.5], colors="red", linewidths=1.0)
    ax.set_title("ROI overlay")
    ax.set_axis_off()
    fig.tight_layout()
    destination = Path(output_path)
    fig.savefig(destination, dpi=200)
    pyplot.close(fig)
    return destination


def _load_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as pyplot

    return pyplot


def _load_mask_array(path: str | Path) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim == 3:
        image = image[..., 0]
    return image


def _normalise_image_shape(image_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int]:
    if len(image_shape) < 2:
        raise ValueError("Image shape must contain at least height and width.")
    return int(image_shape[0]), int(image_shape[1])


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(float(value))

"""
Prepare DIC or interpolated FE data for VFM and normalise specimen ROI inputs.

The ``region_of_interest_input_file`` can currently be one of:
- a ROI defined in the reference-image pixel space during DIC processing,
  such as MatchID ``.m2inp``/``.m3inp`` files or a pyvale ROI
  ``.yaml``/``.yml`` file
- a logical specimen mask derived from DIC outputs, such as arrays or text
  grids whose finite values represent specimen pixels and whose ``NaN`` values
  represent non-specimen pixels

For FE centroid exports, this script can also interpolate element-centroid
fields onto a regular grid. When a Gmsh ``.msh`` file is provided as the
``region_of_interest_input_file``, the specimen mask is sampled from the mesh
geometry so holes and other cut-outs are preserved.

Any supported ROI source can be used. More accurate ROI definitions generally
improve inverse identification (as virtual field metrics depend on the specimen
geometry). Prepared outputs are normalised to include:
- ``region_of_interest.yaml`` in physical coordinates
- ``specimen_mask.npy`` as the confirmed specimen definition
- filled ``x.npy`` and ``y.npy`` coordinate grids once the user confirms the
  ROI-derived specimen mask against the original DIC ``NaN`` mask

Run this module as a script (``python -m pyvale.vfm.inputdataprocessor``) after
editing ``CONFIG`` and ``BOUNDARY_CONDITIONS`` for the dataset to prepare.
"""

import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import interpfedata
from . import roi


@dataclass(slots=True)
class InputDataConfig:
    """Inputs and options for preparing one VFM dataset.

    ``region_of_interest_input_file`` accepts a reference-image ROI definition
    or a logical specimen mask. MatchID ``.m2inp``/``.m3inp`` files and pyvale
    ROI ``.yaml``/``.yml`` files preserve the reference-image ROI resolution.
    Mask-like inputs can also be used, including DIC-style arrays/text grids
    that distinguish specimen and non-specimen pixels with finite values and
    ``NaN`` values respectively, but the resulting geometry is only as accurate
    as that source mask.

    If ``fe_element_data_file`` is provided, FE centroid data is interpolated
    onto a regular grid before the rest of the preparation flow runs.
    """

    input_folder: Path
    output_folder: Path
    x_coordinates_input_file: str
    y_coordinates_input_file: str
    strain_input_file: str
    force_input_file: str
    time_input_file: str
    x_coordinates_pixel_input_file: str | None = None
    y_coordinates_pixel_input_file: str | None = None
    region_of_interest_input_file: str | None = None
    reference_image_file: str | None = None
    metadata_file: str | None = "metadata.json"
    selected_spatial_indices_file: str | None = "selected_spatial_indices.txt"
    strain_h5_dataset_names: tuple[str, ...] = ("exx", "eyy", "exy")
    strain_component_order: tuple[str, ...] = ("exx", "eyy", "exy")
    fe_element_data_file: str | None = None
    fe_strain_component_columns: tuple[str, ...] | None = None
    fe_grid_upsample_factor: float = 2.0
    fe_grid_spacing: float | None = None
    csv_preview_rows: int = 5


BoundaryDofState = Literal["free", "fixed", "traction"]
ForceComponent = Literal["x", "y"]
ForceDirectionSign = Literal["+", "-"]


@dataclass(slots=True, frozen=True)
class EdgeBoundaryConfig:
    x: BoundaryDofState = "free"
    y: BoundaryDofState = "free"


@dataclass(slots=True, frozen=True)
class BoundaryConditionConfig:
    min_x_edge: EdgeBoundaryConfig = field(default_factory=EdgeBoundaryConfig)
    max_x_edge: EdgeBoundaryConfig = field(default_factory=EdgeBoundaryConfig)
    min_y_edge: EdgeBoundaryConfig = field(default_factory=EdgeBoundaryConfig)
    max_y_edge: EdgeBoundaryConfig = field(default_factory=EdgeBoundaryConfig)
    force_component: ForceComponent = "x"
    force_positive_direction: ForceDirectionSign = "+"
    arrow_scale_fraction: float = 0.18
    arrow_label: str = "Applied force"


@dataclass(slots=True, frozen=True)
class CoordinateConventionTransform:
    reflect_x: bool = False
    reflect_y: bool = False

    @property
    def applied(self) -> bool:
        return self.reflect_x or self.reflect_y


# Edit this section for a new dataset.
CONFIG = InputDataConfig(
    input_folder=Path(
        "/Users/chris/work/example_input_data/single-element-plane-stress/fe-data"
    ),
    output_folder=Path(
        "/Users/chris/work/example_input_data/single-element-plane-stress/vfm-input-data"
    ),
    x_coordinates_input_file="x_coordinates.txt",
    y_coordinates_input_file="y_coordinates.txt",
    strain_input_file="element_data.csv",
    force_input_file="reaction_history.csv",
    time_input_file="time_values.txt",
    region_of_interest_input_file="single_element_square.msh",
    strain_component_order=("exx", "eyy", "exy"),
    fe_element_data_file="element_data.csv",
    fe_strain_component_columns=("eps_xx", "eps_yy", "eps_xy"),
    fe_grid_upsample_factor=2.0,
)

# # DIC reference (no fe_element_data_file: gridded x/y and strain are loaded directly)
# CONFIG = InputDataConfig(
#     input_folder=Path("/path/to/dic-processed-data"),
#     output_folder=Path("/path/to/vfm-input-data"),
#     x_coordinates_input_file="x_ref.csv",
#     y_coordinates_input_file="y_ref.csv",
#     strain_input_file="strain_data.h5",
#     force_input_file="force_history.csv",
#     time_input_file="force_history.csv",
#     x_coordinates_pixel_input_file="x_ref_pixel.csv",
#     y_coordinates_pixel_input_file="y_ref_pixel.csv",
#     region_of_interest_input_file="correlation.m3inp",
#     reference_image_file="Image_0000_0.tiff",
#     strain_h5_dataset_names=("exx", "eyy", "exy"),
#     strain_component_order=("exx", "eyy", "exy"),
# )

# Edit this section so the prepared-data diagnostics use the same intended
# boundary-condition convention as the later VFM identification setup.
BOUNDARY_CONDITIONS = BoundaryConditionConfig(
    min_x_edge=EdgeBoundaryConfig(x="free", y="free"),
    max_x_edge=EdgeBoundaryConfig(x="free", y="free"),
    min_y_edge=EdgeBoundaryConfig(x="fixed", y="fixed"),
    max_y_edge=EdgeBoundaryConfig(x="free", y="traction"),
    force_component="y",
    force_positive_direction="+",
    arrow_scale_fraction=0.14,
    arrow_label="Applied load",
)


@dataclass(slots=True)
class CsvTable:
    path: Path
    header: list[str]
    rows: list[list[str]]
    numeric_columns: list[bool]

    @property
    def column_count(self) -> int:
        return len(self.header)


@dataclass(slots=True)
class LoadedFieldData:
    """Field data loaded from either a DIC export or interpolated FE data.

    Both loading paths produce the same normalised set of coordinate grids,
    strain array, and diagnostic info dicts. The FE path additionally supplies
    the specimen mask and ROI summary directly (sampled from the mesh), whereas
    the DIC path leaves ``specimen_mask`` as ``None`` so it can be derived from
    the ``region_of_interest_input_file`` later in :func:`main`.
    """

    strain: np.ndarray
    x_raw: np.ndarray
    y_raw: np.ndarray
    roi_alignment_x: np.ndarray
    roi_alignment_y: np.ndarray
    coordinate_load_info: dict[str, Any]
    roi_alignment_info: dict[str, Any]
    spatial_selection_info: dict[str, Any] | None
    specimen_mask: np.ndarray | None
    roi_summary: dict[str, Any] | None
    total_area_override: float | None
    fe_interpolation: dict[str, Any] | None
    warnings: list[str]

    @property
    def original_coordinate_valid_mask(self) -> np.ndarray:
        return np.isfinite(self.x_raw) & np.isfinite(self.y_raw)


def main() -> None:
    config = CONFIG
    boundary_conditions = BOUNDARY_CONDITIONS
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    output_folder = config.output_folder.parent / f"{config.output_folder.name}-{timestamp}"
    generated_outputs_folder = output_folder / "generated-outputs"
    output_folder.mkdir(parents=True, exist_ok=True)
    generated_outputs_folder.mkdir(parents=True, exist_ok=True)

    print(f"Input folder:  {config.input_folder}")
    print(f"Output folder: {output_folder}")

    data = _load_field_data(config)
    strain = data.strain
    x_raw = data.x_raw
    y_raw = data.y_raw
    roi_alignment_x = data.roi_alignment_x
    roi_alignment_y = data.roi_alignment_y
    original_coordinate_valid_mask = data.original_coordinate_valid_mask

    validation_warnings: list[str] = []
    validation_warnings.extend(data.coordinate_load_info["warnings"])
    validation_warnings.extend(data.roi_alignment_info["warnings"])
    validation_warnings.extend(data.warnings)
    validation_warnings.extend(
        _check_main_and_pixel_coordinate_grids(x_raw, y_raw, roi_alignment_x, roi_alignment_y)
    )

    csv_cache: dict[Path, CsvTable] = {}
    previewed_paths: set[Path] = set()

    force, force_info = _load_signal_vector_from_file(
        config=config,
        file_name=config.force_input_file,
        signal_name="force",
        default_name_hints=("force", "load"),
        csv_cache=csv_cache,
        previewed_paths=previewed_paths,
    )
    force, force_sign_flip_applied = _maybe_flip_force_sign(force)
    time, time_info = _load_signal_vector_from_file(
        config=config,
        file_name=config.time_input_file,
        signal_name="time",
        default_name_hints=("time", "timestamp", "timestep"),
        csv_cache=csv_cache,
        previewed_paths=previewed_paths,
    )
    time, time_offset_correction = _maybe_zero_time_offset(time, time_info["unit"])

    validation_warnings.extend(_validate_force_and_time(force, time, strain.shape[0]))
    raw_coordinate_convention_warnings = _check_coordinate_conventions(x_raw, y_raw)

    if data.specimen_mask is None:
        specimen_mask, roi_summary, roi_warnings = _prepare_specimen_mask(
            config,
            x_raw,
            y_raw,
            roi_alignment_x,
            roi_alignment_y,
            output_folder,
            generated_outputs_folder,
            spatial_selection_info=data.spatial_selection_info,
        )
        validation_warnings.extend(roi_warnings)
    else:
        specimen_mask = data.specimen_mask
        roi_summary = data.roi_summary

    roi_confirmation = _confirm_specimen_mask(
        specimen_mask=specimen_mask,
        original_coordinate_valid_mask=original_coordinate_valid_mask,
        x_raw=x_raw,
        y_raw=y_raw,
        output_folder=generated_outputs_folder,
        roi_summary=roi_summary,
    )

    x, y, coordinate_fill_info = _fill_coordinate_grids_after_confirmation(
        x_raw=x_raw,
        y_raw=y_raw,
        specimen_mask=specimen_mask,
        roi_summary=roi_summary,
        roi_confirmed=roi_confirmation["confirmed"],
    )

    orientation_transform = _determine_coordinate_convention_transform(x, y)
    x_raw_display = _apply_coordinate_convention_to_axis_grid(x_raw, reflect_axis=orientation_transform.reflect_x)
    y_raw_display = _apply_coordinate_convention_to_axis_grid(y_raw, reflect_axis=orientation_transform.reflect_y)
    x = _apply_coordinate_convention_to_axis_grid(x, reflect_axis=orientation_transform.reflect_x)
    y = _apply_coordinate_convention_to_axis_grid(y, reflect_axis=orientation_transform.reflect_y)
    strain = _apply_coordinate_convention_to_strain(
        strain,
        orientation_transform,
        component_names=config.strain_component_order,
    )
    force = _apply_coordinate_convention_to_force(
        force,
        orientation_transform,
        boundary_conditions=boundary_conditions,
    )

    final_coordinate_convention_warnings = _check_coordinate_conventions(x, y)
    validation_warnings.extend(final_coordinate_convention_warnings)

    specimen_mask, roi_summary, roi_final_warnings = _finalise_region_of_interest_outputs(
        specimen_mask=specimen_mask,
        x=x,
        y=y,
        roi_alignment_x=roi_alignment_x,
        roi_alignment_y=roi_alignment_y,
        output_folder=output_folder,
        generated_outputs_folder=generated_outputs_folder,
        roi_summary=roi_summary,
        reference_image_path=_resolve_input_path(config, config.reference_image_file),
        spatial_selection_info=data.spatial_selection_info,
        orientation_transform=orientation_transform,
    )
    validation_warnings.extend(roi_final_warnings)

    pixel_area = _estimate_point_area(
        x,
        y,
        total_area_override=data.total_area_override,
    )
    area_checks, area_warnings = _check_specimen_area(
        point_area=pixel_area,
        original_coordinate_valid_mask=original_coordinate_valid_mask,
        specimen_mask=specimen_mask,
    )
    validation_warnings.extend(area_warnings)
    if roi_summary is not None:
        roi_summary["area_checks"] = area_checks
        roi_summary["saved_paths"] = _save_region_of_interest_summary(
            output_folder,
            generated_outputs_folder,
            roi_summary,
        )

    _save_outputs(
        output_folder=output_folder,
        generated_outputs_folder=generated_outputs_folder,
        x=x,
        y=y,
        strain=strain,
        force=force,
        time=time,
        specimen_mask=specimen_mask,
        pixel_area=pixel_area,
    )

    plot_paths = _create_diagnostic_plots(
        output_folder=generated_outputs_folder,
        x_main=x_raw_display,
        y_main=y_raw_display,
        x_main_prepared=x,
        y_main_prepared=y,
        x_pixel=roi_alignment_x,
        y_pixel=roi_alignment_y,
        strain=strain,
        component_names=config.strain_component_order,
        force=force,
        time=time,
        specimen_mask=specimen_mask,
        original_coordinate_valid_mask=original_coordinate_valid_mask,
        roi_summary=roi_summary,
        boundary_conditions=boundary_conditions,
    )

    summary = {
        "input_folder": str(config.input_folder),
        "output_folder": str(output_folder),
        "generated_outputs_folder": str(generated_outputs_folder),
        "files": {
            "x_coordinates_input_file": config.x_coordinates_input_file,
            "y_coordinates_input_file": config.y_coordinates_input_file,
            "x_coordinates_pixel_input_file": config.x_coordinates_pixel_input_file,
            "y_coordinates_pixel_input_file": config.y_coordinates_pixel_input_file,
            "strain_input_file": config.strain_input_file,
            "fe_element_data_file": config.fe_element_data_file,
            "fe_strain_component_columns": list(config.fe_strain_component_columns)
            if config.fe_strain_component_columns is not None
            else None,
            "force_input_file": config.force_input_file,
            "time_input_file": config.time_input_file,
            "region_of_interest_input_file": config.region_of_interest_input_file,
            "reference_image_file": config.reference_image_file,
        },
        "shapes": {
            "x": list(x.shape),
            "y": list(y.shape),
            "strain": list(strain.shape),
            "force": list(force.shape),
            "time": list(time.shape),
            "specimen_mask": list(specimen_mask.shape),
            "pixel_area": list(pixel_area.shape),
        },
        "strain_component_order": list(config.strain_component_order),
        "boundary_conditions": _serialise_boundary_conditions(boundary_conditions),
        "spatial_selection": data.spatial_selection_info,
        "coordinate_load_info": data.coordinate_load_info,
        "fe_interpolation": data.fe_interpolation,
        "coordinate_fill": coordinate_fill_info,
        "roi_alignment_coordinate_info": data.roi_alignment_info,
        "coordinate_convention_check_before_transform": {
            "warnings": raw_coordinate_convention_warnings,
        },
        "coordinate_convention_transform": _serialise_coordinate_convention_transform(orientation_transform),
        "coordinate_convention_check_after_transform": {
            "warnings": final_coordinate_convention_warnings,
        },
        "force_selection": force_info,
        "force_sign_flip_applied": force_sign_flip_applied,
        "time_selection": {
            **time_info,
            "time_offset_corrected": time_offset_correction,
        },
        "roi_summary": roi_summary,
        "roi_confirmation": roi_confirmation,
        "plots": {name: str(path) for name, path in plot_paths.items()},
        "warnings": validation_warnings,
    }
    summary_path = generated_outputs_folder / "inputdataprocessor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nSaved arrays:")
    for name in (
        "x.npy",
        "y.npy",
        "strain.npy",
        "force.npy",
        "time.npy",
        "pixel_area.npy",
        "specimen_mask.npy",
    ):
        print(f"  - {output_folder / name}")
    print(f"  - {generated_outputs_folder / 'specimen_mask.npy'}")

    print("\nSaved plots:")
    for name, path in plot_paths.items():
        print(f"  - {name}: {path}")

    print(f"\nSaved summary: {summary_path}")

    transform_description = _serialise_coordinate_convention_transform(orientation_transform)
    print("\nCoordinate convention transform:")
    print(
        "  "
        f"reflect_x={transform_description['reflect_x']}, "
        f"reflect_y={transform_description['reflect_y']}"
    )

    if validation_warnings:
        print("\nWarnings:")
        for warning in validation_warnings:
            print(f"  - {warning}")
    else:
        print("\nValidation checks passed with no warnings.")


def _load_field_data(config: InputDataConfig) -> LoadedFieldData:
    """Load field data from interpolated FE centroids or a gridded DIC export."""
    if config.fe_element_data_file is not None:
        return _load_fe_field_data(config)
    return _load_dic_field_data(config)


def _load_dic_field_data(config: InputDataConfig) -> LoadedFieldData:
    strain = _load_strain_data(config)
    x_raw, y_raw, coordinate_load_info = _load_main_coordinate_grids(config, strain.shape[2:])
    _validate_coordinate_grids(x_raw, y_raw)
    _validate_strain_shape(strain, x_raw.shape)
    spatial_selection_info = _load_spatial_selection_info(config, x_raw.shape)

    roi_alignment_x, roi_alignment_y, roi_alignment_info = _load_roi_alignment_coordinate_grids(
        config=config,
        fallback_x=x_raw,
        fallback_y=y_raw,
        target_shape=strain.shape[2:],
    )

    return LoadedFieldData(
        strain=strain,
        x_raw=x_raw,
        y_raw=y_raw,
        roi_alignment_x=roi_alignment_x,
        roi_alignment_y=roi_alignment_y,
        coordinate_load_info=coordinate_load_info,
        roi_alignment_info=roi_alignment_info,
        spatial_selection_info=spatial_selection_info,
        specimen_mask=None,
        roi_summary=None,
        total_area_override=None,
        fe_interpolation=None,
        warnings=[],
    )


def _load_fe_field_data(config: InputDataConfig) -> LoadedFieldData:
    element_data_path = _resolve_input_path(config, config.fe_element_data_file)
    mesh_path = _resolve_input_path(config, config.region_of_interest_input_file)
    component_columns = _resolve_fe_component_columns(config)
    interpolated = interpfedata.interpolate_fe_data_to_grid(
        element_data_path=element_data_path,
        component_columns=component_columns,
        mesh_path=mesh_path,
        upsample_factor=config.fe_grid_upsample_factor,
        target_spacing=config.fe_grid_spacing,
    )

    x_raw = np.asarray(interpolated.x_grid, dtype=np.float64)
    y_raw = np.asarray(interpolated.y_grid, dtype=np.float64)
    strain = np.asarray(interpolated.strain, dtype=np.float64)
    _validate_coordinate_grids(x_raw, y_raw)
    _validate_strain_shape(strain, x_raw.shape)

    warnings = list(interpolated.metadata.get("warnings", []))
    if x_raw.shape[0] < 2 or x_raw.shape[1] < 2:
        warnings.append(
            "The interpolated FE grid is smaller than 2x2, so some spatial diagnostics and downstream VFM steps "
            "will be limited."
        )

    grid_shape = [int(value) for value in x_raw.shape]
    coordinate_load_info = {
        "path_x": str(element_data_path),
        "path_y": str(element_data_path),
        "original_shape": grid_shape,
        "final_shape": grid_shape,
        "assumed_unit": "mm",
        "warnings": [
            "Main x/y coordinate grids were generated by interpolating FE centroid coordinates onto a regular grid.",
        ],
    }
    roi_alignment_info = {
        "path_x": None,
        "path_y": None,
        "final_shape": grid_shape,
        "used_fallback_main_coordinates": True,
        "warnings": [
            "Using the interpolated FE regular grid for ROI alignment coordinates.",
        ],
    }
    roi_summary = {
        "source_kind": "fe-mesh" if mesh_path is not None else "fe-point-cloud",
        "input_path": str(mesh_path) if mesh_path is not None else str(element_data_path),
        "source_mask_shape": grid_shape,
        "sampled_mask_shape": grid_shape,
        "mask_pixel_count": int(np.count_nonzero(interpolated.specimen_mask)),
        "roi_yaml": None,
        "intermediate_pixel_roi_yaml": None,
        "sampled_pixel_roi_yaml": None,
        "sampled_pixel_mask_tiff": None,
        "metadata_json": None,
        "source_mask_tiff": None,
        "source_overlay_image": None,
        "coordinate_space": "physical-grid-from-fe",
        "alignment": {
            "mode": "fe-interpolated-physical-grid",
            "grid_spacing": interpolated.metadata.get("grid_spacing"),
        },
        "spatial_selection": None,
        "mismatch_count_vs_coordinate_nan_mask": 0,
        "notes": [
            "The FE centroid fields were interpolated onto a regular physical grid before VFM preparation.",
            "The specimen mask was sampled from the FE mesh geometry when a .msh file was provided.",
        ],
    }

    return LoadedFieldData(
        strain=strain,
        x_raw=x_raw,
        y_raw=y_raw,
        roi_alignment_x=x_raw,
        roi_alignment_y=y_raw,
        coordinate_load_info=coordinate_load_info,
        roi_alignment_info=roi_alignment_info,
        spatial_selection_info=None,
        specimen_mask=np.asarray(interpolated.specimen_mask, dtype=bool),
        roi_summary=roi_summary,
        total_area_override=interpolated.total_specimen_area,
        fe_interpolation=dict(interpolated.metadata),
        warnings=warnings,
    )


def _resolve_input_path(config: InputDataConfig, file_name: str | None) -> Path | None:
    if file_name is None:
        return None
    path = Path(file_name)
    if not path.is_absolute():
        path = config.input_folder / path
    return path


def _load_numeric_grid(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix in {".csv", ".txt", ".dat"}:
        array = np.genfromtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported grid file type for '{path}'.")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grid in '{path}', got shape {array.shape}.")
    return array


def _validate_coordinate_grids(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y coordinates must both be 2D arrays.")
    if x.shape != y.shape:
        raise ValueError(f"x and y coordinate grids must have the same shape, got {x.shape} and {y.shape}.")


def _load_main_coordinate_grids(
    config: InputDataConfig,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_path = _resolve_input_path(config, config.x_coordinates_input_file)
    y_path = _resolve_input_path(config, config.y_coordinates_input_file)
    if x_path is None or y_path is None:
        raise ValueError("Both main x and y coordinate files must be provided.")

    x_raw = _load_numeric_grid(x_path)
    y_raw = _load_numeric_grid(y_path)
    _validate_coordinate_grids(x_raw, y_raw)

    original_shape = x_raw.shape
    if x_raw.shape != target_shape:
        raise ValueError(
            "Main coordinate files must match the strain grid shape exactly. "
            f"Got {x_raw.shape}, expected {target_shape}."
        )

    return x_raw, y_raw, {
        "path_x": str(x_path),
        "path_y": str(y_path),
        "original_shape": list(original_shape),
        "final_shape": list(x_raw.shape),
        "assumed_unit": "mm",
        "warnings": [],
    }


def _load_roi_alignment_coordinate_grids(
    *,
    config: InputDataConfig,
    fallback_x: np.ndarray,
    fallback_y: np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_path = _resolve_input_path(config, config.x_coordinates_pixel_input_file)
    y_path = _resolve_input_path(config, config.y_coordinates_pixel_input_file)
    warnings: list[str] = []

    if x_path is None or y_path is None:
        warnings.append(
            "No dedicated pixel-coordinate files were provided for ROI alignment; using the main x/y coordinates."
        )
        return fallback_x, fallback_y, {
            "path_x": None,
            "path_y": None,
            "final_shape": list(fallback_x.shape),
            "used_fallback_main_coordinates": True,
            "warnings": warnings,
        }

    x = _load_numeric_grid(x_path)
    y = _load_numeric_grid(y_path)
    _validate_coordinate_grids(x, y)
    if x.shape != target_shape:
        raise ValueError(
            "Pixel-coordinate files used for ROI alignment must match the strain grid shape exactly. "
            f"Got {x.shape}, expected {target_shape}."
        )

    return x, y, {
        "path_x": str(x_path),
        "path_y": str(y_path),
        "final_shape": list(x.shape),
        "used_fallback_main_coordinates": False,
        "warnings": warnings,
    }


def _load_spatial_selection_info(
    config: InputDataConfig,
    target_shape: tuple[int, int],
) -> dict[str, Any] | None:
    metadata_path = _resolve_input_path(config, config.metadata_file)
    indices_path = _resolve_input_path(config, config.selected_spatial_indices_file)

    metadata_payload: dict[str, Any] | None = None
    if metadata_path is not None and metadata_path.exists():
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    selection_payload = metadata_payload.get("selection") if isinstance(metadata_payload, dict) else None
    row_indices = _coerce_index_list(selection_payload.get("selected_spatial_row_indices")) if isinstance(selection_payload, dict) else None
    col_indices = _coerce_index_list(selection_payload.get("selected_spatial_column_indices")) if isinstance(selection_payload, dict) else None

    if (row_indices is None or col_indices is None) and indices_path is not None and indices_path.exists():
        parsed_rows, parsed_cols = _parse_selected_spatial_indices_file(indices_path)
        row_indices = parsed_rows if row_indices is None else row_indices
        col_indices = parsed_cols if col_indices is None else col_indices

    if row_indices is None and col_indices is None and metadata_payload is None:
        return None

    info: dict[str, Any] = {
        "metadata_path": str(metadata_path) if metadata_path is not None and metadata_path.exists() else None,
        "selected_spatial_indices_path": (
            str(indices_path) if indices_path is not None and indices_path.exists() else None
        ),
        "row_count": len(row_indices) if row_indices is not None else None,
        "column_count": len(col_indices) if col_indices is not None else None,
        "row_range": [row_indices[0], row_indices[-1]] if row_indices else None,
        "column_range": [col_indices[0], col_indices[-1]] if col_indices else None,
        "matches_prepared_grid_shape": (
            len(row_indices) == target_shape[0] and len(col_indices) == target_shape[1]
            if row_indices is not None and col_indices is not None
            else None
        ),
        "note": (
            "These selected spatial indices describe the crop in the full DIC grid. "
            "ROI alignment still uses the pixel-coordinate grids because the uncropped ROI source lives in the "
            "reference-image pixel space."
        ),
    }
    return info


def _coerce_index_list(values: Any) -> list[int] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        return None
    return [int(value) for value in values]


def _parse_selected_spatial_indices_file(path: Path) -> tuple[list[int] | None, list[int] | None]:
    text = path.read_text(encoding="utf-8")
    row_match = re.search(r"selected_spatial_row_indices\s*=\s*\[([^\]]*)\]", text, flags=re.MULTILINE)
    col_match = re.search(r"selected_spatial_column_indices\s*=\s*\[([^\]]*)\]", text, flags=re.MULTILINE)

    def _parse_match(match: re.Match[str] | None) -> list[int] | None:
        if match is None:
            return None
        body = match.group(1).strip()
        if not body:
            return []
        return [int(value.strip()) for value in body.split(",") if value.strip()]

    return _parse_match(row_match), _parse_match(col_match)


def _resolve_fe_component_columns(config: InputDataConfig) -> tuple[str, ...]:
    if config.fe_strain_component_columns is not None:
        if len(config.fe_strain_component_columns) != len(config.strain_component_order):
            raise ValueError(
                "fe_strain_component_columns must have the same length as strain_component_order."
            )
        return config.fe_strain_component_columns

    return tuple(_default_fe_component_column_name(name) for name in config.strain_component_order)


def _default_fe_component_column_name(component_name: str) -> str:
    normalised = component_name.strip().lower()
    if normalised in {"exx", "e_xx", "strain_xx"}:
        return "eps_xx"
    if normalised in {"eyy", "e_yy", "strain_yy"}:
        return "eps_yy"
    if normalised in {"exy", "eyx", "e_xy", "e_yx", "e12", "e21", "c12", "c21", "strain_xy"}:
        return "eps_xy"
    return component_name


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Loading '.h5' strain data requires h5py. "
            "A simple way to run this script is:\n"
            "  uv run --with h5py python src/pyvale/vfm/prepareinputdata.py"
        ) from exc
    return h5py


def _load_strain_data(config: InputDataConfig) -> np.ndarray:
    path = _resolve_input_path(config, config.strain_input_file)
    if path is None:
        raise ValueError("A strain input file must be provided.")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        strain = np.asarray(np.load(path), dtype=np.float64)
    elif suffix in {".h5", ".hdf5"}:
        h5py = _import_h5py()
        with h5py.File(path, "r") as handle:
            dataset_names = config.strain_h5_dataset_names
            if len(dataset_names) == 1:
                strain = np.asarray(handle[dataset_names[0]][...], dtype=np.float64)
            else:
                components = [np.asarray(handle[name][...], dtype=np.float64) for name in dataset_names]
                _validate_component_shapes(components, dataset_names)
                strain = np.stack(components, axis=1)
    else:
        raise ValueError(f"Unsupported strain file type for '{path}'.")

    if strain.ndim != 4:
        raise ValueError(
            "Strain data must be a 4D array with shape (timestep, component, y, x). "
            f"Got shape {strain.shape}."
        )
    return strain


def _validate_component_shapes(components: list[np.ndarray], names: tuple[str, ...]) -> None:
    reference_shape = components[0].shape
    for name, component in zip(names, components):
        if component.shape != reference_shape:
            raise ValueError(
                f"All strain component datasets must have the same shape. "
                f"Dataset '{name}' has shape {component.shape}, expected {reference_shape}."
            )
        if component.ndim != 3:
            raise ValueError(
                f"Each HDF5 strain component dataset must be 3D with shape (timestep, y, x). "
                f"Dataset '{name}' has shape {component.shape}."
            )


def _validate_strain_shape(strain: np.ndarray, coordinate_shape: tuple[int, int]) -> None:
    if strain.shape[2:] != coordinate_shape:
        raise ValueError(
            "Strain y/x dimensions must match the coordinate grid shape. "
            f"Got strain shape {strain.shape} and coordinate shape {coordinate_shape}."
        )


def _load_signal_vector_from_file(
    *,
    config: InputDataConfig,
    file_name: str,
    signal_name: str,
    default_name_hints: tuple[str, ...],
    csv_cache: dict[Path, CsvTable],
    previewed_paths: set[Path],
) -> tuple[np.ndarray, dict[str, Any]]:
    path = _resolve_input_path(config, file_name)
    if path is None:
        raise ValueError(f"A {signal_name} input file must be provided.")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
        return data, {"path": str(path), "column_index": None, "column_name": None, "unit": None}

    if suffix in {".csv", ".txt", ".dat"}:
        table = csv_cache.get(path)
        if table is None:
            table = _load_csv_table(path)
            csv_cache[path] = table

        numeric_column_indices = [index for index, is_numeric in enumerate(table.numeric_columns) if is_numeric]
        if not numeric_column_indices:
            raise ValueError(f"No numeric columns were found in '{path}' for {signal_name}.")

        default_index = _guess_default_column_index(table.header, numeric_column_indices, default_name_hints)

        if path not in previewed_paths:
            _print_csv_preview(table, config.csv_preview_rows)
            previewed_paths.add(path)

        column_index = _prompt_for_column_index(
            table=table,
            signal_name=signal_name,
            numeric_column_indices=numeric_column_indices,
            default_index=default_index,
        )
        column_name = table.header[column_index]
        unit_guess = _guess_unit_from_header(column_name)
        unit = _prompt_for_unit(signal_name, unit_guess)
        data = _extract_numeric_column(table, column_index)
        return data, {
            "path": str(path),
            "column_index": int(column_index),
            "column_name": column_name,
            "unit": unit,
        }

    raise ValueError(f"Unsupported signal file type for '{path}'.")


def _load_csv_table(path: Path) -> CsvTable:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        raw_rows = [row for row in csv.reader(handle)]

    if not raw_rows:
        raise ValueError(f"CSV file '{path}' is empty.")

    width = max(len(row) for row in raw_rows)
    rows = [row + [""] * (width - len(row)) for row in raw_rows]

    first_row = rows[0]
    header_present = not all(_is_float_like(cell) or cell == "" for cell in first_row)
    header = first_row if header_present else [f"column_{index}" for index in range(width)]
    data_rows = rows[1:] if header_present else rows

    numeric_columns: list[bool] = []
    for column_index in range(width):
        column_values = [row[column_index].strip() for row in data_rows if row[column_index].strip()]
        numeric_columns.append(bool(column_values) and all(_is_float_like(value) for value in column_values))

    return CsvTable(path=path, header=header, rows=data_rows, numeric_columns=numeric_columns)


def _is_float_like(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _guess_default_column_index(
    header: list[str],
    numeric_column_indices: list[int],
    name_hints: tuple[str, ...],
) -> int:
    for hint in name_hints:
        for index in numeric_column_indices:
            if hint.lower() in header[index].lower():
                return index
    return numeric_column_indices[0]


def _print_csv_preview(table: CsvTable, n_rows: int) -> None:
    columns_per_batch = 5
    max_cell_width = 24

    print(f"\nPreview of {table.path}:")
    preview_rows = table.rows[:n_rows]
    for batch_start in range(0, table.column_count, columns_per_batch):
        batch_end = min(table.column_count, batch_start + columns_per_batch)
        print(f"\nColumns {batch_start}-{batch_end - 1}:")

        header_cells = [f"[{index}] {table.header[index]}" for index in range(batch_start, batch_end)]
        row_cells = [
            [row[index].strip() for index in range(batch_start, batch_end)]
            for row in preview_rows
        ]
        column_widths = []
        for offset in range(batch_end - batch_start):
            cells = [header_cells[offset], *(row[offset] for row in row_cells)]
            width = max(len(_truncate(cell, max_cell_width)) for cell in cells)
            column_widths.append(width)

        print(_format_preview_line(header_cells, column_widths, max_cell_width=max_cell_width))
        for row in row_cells:
            print(_format_preview_line(row, column_widths, max_cell_width=max_cell_width))


def _format_preview_line(
    cells: list[str],
    column_widths: list[int],
    *,
    max_cell_width: int,
) -> str:
    formatted_cells = [
        _truncate(cell, max_cell_width).ljust(column_widths[index])
        for index, cell in enumerate(cells)
    ]
    return " | ".join(formatted_cells)


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else f"{text[: width - 3]}..."


def _prompt_for_column_index(
    *,
    table: CsvTable,
    signal_name: str,
    numeric_column_indices: list[int],
    default_index: int,
) -> int:
    prompt = (
        f"Select the column index to use for {signal_name} "
        f"(numeric columns: {numeric_column_indices}, default: {default_index}): "
    )
    while True:
        response = input(prompt).strip()
        if not response:
            return default_index
        try:
            selected = int(response)
        except ValueError:
            print("Please enter an integer column index.")
            continue
        if selected not in numeric_column_indices:
            print(f"Column {selected} is not a numeric column in this file.")
            continue
        return selected


def _guess_unit_from_header(header: str) -> str | None:
    bracket_match = re.search(r"\(([^()]+)\)|\[([^\[\]]+)\]", header)
    if bracket_match:
        return next(group for group in bracket_match.groups() if group is not None).strip()

    underscore_match = re.search(r"_([A-Za-z%]+)$", header.strip())
    if underscore_match:
        return underscore_match.group(1)

    return None


def _prompt_for_unit(signal_name: str, unit_guess: str | None) -> str:
    if unit_guess is None:
        return input(f"Enter the unit for {signal_name}: ").strip() or "unknown"
    response = input(f"Confirm the unit for {signal_name} [{unit_guess}]: ").strip()
    return response or unit_guess


def _extract_numeric_column(table: CsvTable, column_index: int) -> np.ndarray:
    values = []
    for row_index, row in enumerate(table.rows, start=1):
        cell = row[column_index].strip()
        if not cell:
            raise ValueError(
                f"Blank value found in column {column_index} ('{table.header[column_index]}') "
                f"at data row {row_index} in '{table.path}'."
            )
        values.append(float(cell))
    return np.asarray(values, dtype=np.float64)


def _maybe_zero_time_offset(time: np.ndarray, unit: str | None) -> tuple[np.ndarray, bool]:
    if time.size == 0 or np.isclose(time[0], 0.0):
        return time, False

    unit_text = f" {unit}" if unit and unit != "unknown" else ""
    response = input(
        f"The first timestep is {time[0]:.6g}{unit_text}. "
        "Subtract this offset so the first timestep becomes 0? [Y/n]: "
    ).strip()
    should_shift = response.lower() not in {"n", "no"}
    return (time - time[0], True) if should_shift else (time, False)


def _maybe_flip_force_sign(force: np.ndarray) -> tuple[np.ndarray, bool]:
    response = input("Multiply the selected force signal by -1 before saving? [y/N]: ").strip().lower()
    should_flip = response in {"y", "yes"}
    if not should_flip:
        return force, False
    return -np.asarray(force, dtype=np.float64), True


def _validate_force_and_time(force: np.ndarray, time: np.ndarray, expected_timesteps: int) -> list[str]:
    warnings: list[str] = []
    if force.ndim != 1:
        raise ValueError(f"Force data must be 1D, got shape {force.shape}.")
    if time.ndim != 1:
        raise ValueError(f"Time data must be 1D, got shape {time.shape}.")
    if force.shape[0] != expected_timesteps:
        raise ValueError(
            f"Force length must match the number of strain timesteps. "
            f"Got force length {force.shape[0]} and strain timesteps {expected_timesteps}."
        )
    if time.shape[0] != expected_timesteps:
        raise ValueError(
            f"Time length must match the number of strain timesteps. "
            f"Got time length {time.shape[0]} and strain timesteps {expected_timesteps}."
        )
    if np.any(~np.isfinite(force)):
        raise ValueError("Force data contains NaN or inf values.")
    if np.any(~np.isfinite(time)):
        raise ValueError("Time data contains NaN or inf values.")
    if np.any(np.diff(time) < 0.0):
        warnings.append("Time values are not monotonically increasing.")
    return warnings


def _check_coordinate_conventions(x: np.ndarray, y: np.ndarray) -> list[str]:
    warnings: list[str] = []
    mean_dx_columns = _safe_nanmedian(np.diff(x, axis=1))
    mean_dy_rows = _safe_nanmedian(np.diff(y, axis=0))
    mean_dx_rows = _safe_nanmedian(np.diff(x, axis=0))
    mean_dy_columns = _safe_nanmedian(np.diff(y, axis=1))

    if mean_dx_columns is not None and mean_dx_columns <= 0.0:
        warnings.append("x does not increase left-to-right on average. A horizontal flip may be needed.")
    if mean_dy_rows is not None and mean_dy_rows <= 0.0:
        warnings.append("y does not increase top-to-bottom on average. A vertical flip may be needed.")
    if mean_dx_rows is not None and abs(mean_dx_rows) > 1.0:
        warnings.append("x changes noticeably from row to row. Check for rotation or transposition.")
    if mean_dy_columns is not None and abs(mean_dy_columns) > 1.0:
        warnings.append("y changes noticeably from column to column. Check for rotation or transposition.")
    return warnings


def _check_main_and_pixel_coordinate_grids(
    x_main: np.ndarray,
    y_main: np.ndarray,
    x_pixel: np.ndarray,
    y_pixel: np.ndarray,
) -> list[str]:
    warnings: list[str] = []
    finite_mask = np.isfinite(x_main) & np.isfinite(y_main) & np.isfinite(x_pixel) & np.isfinite(y_pixel)
    if not np.any(finite_mask):
        return warnings

    same_x = np.allclose(x_main[finite_mask], x_pixel[finite_mask], rtol=0.0, atol=1.0e-9)
    same_y = np.allclose(y_main[finite_mask], y_pixel[finite_mask], rtol=0.0, atol=1.0e-9)
    if same_x and same_y:
        warnings.append(
            "The main x/y coordinate grids are numerically identical to the pixel-coordinate grids. "
            "No physical-unit conversion has been applied, so the 'coordinate_fields_mm' diagnostics will still "
            "show pixel-valued coordinates unless different physical x/y inputs are provided."
        )
    return warnings


def _safe_nanmedian(array: np.ndarray) -> float | None:
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _prepare_specimen_mask(
    config: InputDataConfig,
    x: np.ndarray,
    y: np.ndarray,
    roi_alignment_x: np.ndarray,
    roi_alignment_y: np.ndarray,
    output_folder: Path,
    generated_outputs_folder: Path,
    spatial_selection_info: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    coordinate_valid_mask = np.isfinite(x) & np.isfinite(y)

    roi_input_path = _resolve_input_path(config, config.region_of_interest_input_file)
    if roi_input_path is None:
        warnings.append(
            "No region_of_interest_input_file was provided. Using the coordinate NaN mask as specimen_mask."
        )
        return coordinate_valid_mask, None, warnings

    output_dir = generated_outputs_folder / "roi_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_kwargs: dict[str, Any] = {}
    reference_image_path = _resolve_input_path(config, config.reference_image_file)
    if reference_image_path is not None:
        roi_kwargs["reference_image"] = reference_image_path

    artifacts = roi.generate_vfm_input_roi(roi_input_path, output_dir, **roi_kwargs)
    roi_mask = roi.rasterise_roi_definition(artifacts.roi_definition)
    sampled_mask, alignment_summary = roi.sample_roi_mask_at_pixel_coordinates(
        roi_mask,
        roi_alignment_x,
        roi_alignment_y,
    )

    sampled_pixel_roi_yaml = generated_outputs_folder / "region_of_interest_pixel_source-aligned.yaml"
    sampled_pixel_mask_tiff = generated_outputs_folder / "region_of_interest_pixel_source-aligned_mask.tiff"
    sampled_pixel_roi_definition = roi.convert_mask_to_physical_roi(
        sampled_mask,
        x=roi_alignment_x,
        y=roi_alignment_y,
    )
    roi.write_roi_yaml(sampled_pixel_roi_definition, sampled_pixel_roi_yaml)
    roi.write_mask_tiff(sampled_mask, sampled_pixel_mask_tiff)
    specimen_mask = sampled_mask.copy()

    mismatch_mask = specimen_mask ^ coordinate_valid_mask
    mismatch_count = int(np.count_nonzero(mismatch_mask))
    if mismatch_count > 10:
        warnings.append(
            f"ROI-derived specimen mask differs from the coordinate finite-value mask at {mismatch_count} points."
        )

    if artifacts.source_kind in {"mask-image", "mask-npy", "mask-text"}:
        warnings.append(
            "The region_of_interest_input_file was a logical mask. The saved ROI YAML was derived from that mask, "
            "so downstream geometry accuracy is limited by the source mask resolution."
        )

    summary = {
        "source_kind": artifacts.source_kind,
        "input_path": str(roi_input_path),
        "source_mask_shape": list(roi_mask.shape),
        "sampled_mask_shape": list(specimen_mask.shape),
        "mask_pixel_count": int(artifacts.mask_pixel_count),
        "roi_yaml": None,
        "intermediate_pixel_roi_yaml": str(artifacts.roi_yaml),
        "sampled_pixel_roi_yaml": str(sampled_pixel_roi_yaml),
        "sampled_pixel_mask_tiff": str(sampled_pixel_mask_tiff),
        "metadata_json": str(artifacts.metadata_json),
        "source_mask_tiff": str(artifacts.mask_tiff),
        "source_overlay_image": str(artifacts.overlay_image) if artifacts.overlay_image is not None else None,
        "coordinate_space": "pixel-sampled-then-physical",
        "alignment": alignment_summary,
        "spatial_selection": spatial_selection_info,
        "mismatch_count_vs_coordinate_nan_mask": mismatch_count,
        "notes": [
            "The ROI source artifacts remain in the uncropped source-image space.",
            "The sampled pixel-space ROI artifacts represent the effective cropped ROI on the prepared DIC grid.",
        ],
    }
    return specimen_mask, summary, warnings


def _confirm_specimen_mask(
    *,
    specimen_mask: np.ndarray,
    original_coordinate_valid_mask: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    output_folder: Path,
    roi_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    mismatch_mask = specimen_mask ^ original_coordinate_valid_mask
    roi_only_mask = specimen_mask & ~original_coordinate_valid_mask
    coordinate_only_mask = original_coordinate_valid_mask & ~specimen_mask
    mismatch_count = int(np.count_nonzero(mismatch_mask))
    roi_only_count = int(np.count_nonzero(roi_only_mask))
    coordinate_only_count = int(np.count_nonzero(coordinate_only_mask))

    plot_path = output_folder / "specimen_mask_confirmation.png"
    mismatch_report_path = output_folder / "specimen_mask_mismatches.txt"
    _save_mask_comparison_plot(
        output_path=plot_path,
        specimen_mask=specimen_mask,
        coordinate_mask=original_coordinate_valid_mask,
    )
    mismatch_preview = _write_mismatch_report(
        output_path=mismatch_report_path,
        specimen_mask=specimen_mask,
        coordinate_mask=original_coordinate_valid_mask,
        x_raw=x_raw,
        y_raw=y_raw,
    )

    if roi_summary is None:
        print(
            "\nNo ROI-derived specimen mask was created, so the original coordinate NaN mask "
            "will remain the specimen definition."
        )
        return {
            "confirmed": False,
            "confirmation_required": False,
            "mismatch_count": mismatch_count,
            "roi_only_count": roi_only_count,
            "coordinate_only_count": coordinate_only_count,
            "comparison_plot": str(plot_path),
            "mismatch_report": str(mismatch_report_path),
            "mismatch_preview": mismatch_preview,
        }

    if mismatch_count == 0:
        print("\nSpecimen mask review: ROI-derived specimen mask matches the coordinate mask exactly.")
        return {
            "confirmed": True,
            "confirmation_required": False,
            "mismatch_count": mismatch_count,
            "roi_only_count": roi_only_count,
            "coordinate_only_count": coordinate_only_count,
            "comparison_plot": str(plot_path),
            "mismatch_report": str(mismatch_report_path),
            "mismatch_preview": mismatch_preview,
        }

    print("\nSpecimen mask review:")
    print(f"  ROI source: {roi_summary['input_path']}")
    print(f"  Comparison plot: {plot_path}")
    print(f"  Mismatch report: {mismatch_report_path}")
    print(f"  Total ROI vs coordinate-mask differences: {mismatch_count}")
    print(f"  ROI-only points (inside ROI, NaN in original x/y): {roi_only_count}")
    print(f"  Coordinate-only points (finite in original x/y, outside ROI): {coordinate_only_count}")
    _print_mismatch_preview(mismatch_preview)

    response = input(
        "Confirm this specimen mask and fill the NaN holes in x/y for the prepared outputs? [y/N]: "
    ).strip()
    confirmed = response.lower() in {"y", "yes"}
    if not confirmed:
        raise SystemExit(
            "Preparation cancelled so the ROI or mask can be reviewed before filling coordinate NaNs."
        )

    return {
        "confirmed": True,
        "confirmation_required": True,
        "mismatch_count": mismatch_count,
        "roi_only_count": roi_only_count,
        "coordinate_only_count": coordinate_only_count,
        "comparison_plot": str(plot_path),
        "mismatch_report": str(mismatch_report_path),
        "mismatch_preview": mismatch_preview,
    }


def _fill_coordinate_grids_after_confirmation(
    *,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    specimen_mask: np.ndarray,
    roi_summary: dict[str, Any] | None,
    roi_confirmed: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_missing_mask = ~np.isfinite(x_raw)
    y_missing_mask = ~np.isfinite(y_raw)

    if roi_summary is None:
        return x_raw, y_raw, {
            "performed": False,
            "reason": "No ROI-derived specimen mask was available for confirmation.",
            "x_missing_before": int(np.count_nonzero(x_missing_mask)),
            "y_missing_before": int(np.count_nonzero(y_missing_mask)),
        }

    if not roi_confirmed:
        return x_raw, y_raw, {
            "performed": False,
            "reason": "ROI-derived specimen mask was not confirmed.",
            "x_missing_before": int(np.count_nonzero(x_missing_mask)),
            "y_missing_before": int(np.count_nonzero(y_missing_mask)),
        }

    x_filled = _reconstruct_uniform_x_grid(x_raw)
    y_filled = _reconstruct_uniform_y_grid(y_raw)
    return x_filled, y_filled, {
        "performed": True,
        "reason": "Filled after the ROI-derived specimen mask was confirmed by the user.",
        "method": "column-wise x reconstruction and row-wise y reconstruction",
        "x_missing_before": int(np.count_nonzero(x_missing_mask)),
        "y_missing_before": int(np.count_nonzero(y_missing_mask)),
        "x_missing_inside_specimen_before": int(np.count_nonzero(x_missing_mask & specimen_mask)),
        "y_missing_inside_specimen_before": int(np.count_nonzero(y_missing_mask & specimen_mask)),
        "x_missing_after": int(np.count_nonzero(~np.isfinite(x_filled))),
        "y_missing_after": int(np.count_nonzero(~np.isfinite(y_filled))),
    }


def _reconstruct_uniform_x_grid(x_raw: np.ndarray) -> np.ndarray:
    x_axis = _representative_axis_from_grid(x_raw, axis=0)
    return np.broadcast_to(x_axis[None, :], x_raw.shape).copy()


def _reconstruct_uniform_y_grid(y_raw: np.ndarray) -> np.ndarray:
    y_axis = _representative_axis_from_grid(y_raw, axis=1)
    return np.broadcast_to(y_axis[:, None], y_raw.shape).copy()


def _representative_axis_from_grid(grid: np.ndarray, *, axis: int) -> np.ndarray:
    representative = np.empty(grid.shape[1] if axis == 0 else grid.shape[0], dtype=np.float64)
    if axis == 0:
        for col_index in range(grid.shape[1]):
            median = _safe_nanmedian(grid[:, col_index])
            representative[col_index] = np.nan if median is None else median
    elif axis == 1:
        for row_index in range(grid.shape[0]):
            median = _safe_nanmedian(grid[row_index, :])
            representative[row_index] = np.nan if median is None else median
    else:
        raise ValueError("axis must be 0 for columns or 1 for rows.")

    return _fill_missing_axis_values(representative)


def _fill_missing_axis_values(axis_values: np.ndarray) -> np.ndarray:
    filled = np.asarray(axis_values, dtype=np.float64).copy()
    finite_mask = np.isfinite(filled)
    if not np.any(finite_mask):
        raise ValueError("Could not reconstruct a coordinate axis from all-NaN values.")
    if np.all(finite_mask):
        return filled

    indices = np.arange(filled.size, dtype=np.float64)
    finite_indices = indices[finite_mask]
    finite_values = filled[finite_mask]
    filled[~finite_mask] = np.interp(indices[~finite_mask], finite_indices, finite_values)

    if finite_indices.size >= 2:
        first_finite_index = int(finite_indices[0])
        last_finite_index = int(finite_indices[-1])
        leading_slope = (finite_values[1] - finite_values[0]) / (finite_indices[1] - finite_indices[0])
        trailing_slope = (finite_values[-1] - finite_values[-2]) / (finite_indices[-1] - finite_indices[-2])
        for index in range(first_finite_index - 1, -1, -1):
            filled[index] = filled[index + 1] - leading_slope
        for index in range(last_finite_index + 1, filled.size):
            filled[index] = filled[index - 1] + trailing_slope
    return filled


def _write_mismatch_report(
    *,
    output_path: Path,
    specimen_mask: np.ndarray,
    coordinate_mask: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
) -> list[str]:
    mismatch_indices = np.argwhere(specimen_mask ^ coordinate_mask)
    lines = [
        "# specimen mask mismatch report",
        "# columns: row_index, col_index, mismatch_type, roi_mask, coordinate_mask, x_raw, y_raw",
    ]
    preview: list[str] = []
    for row_index, col_index in mismatch_indices:
        roi_value = bool(specimen_mask[row_index, col_index])
        coordinate_value = bool(coordinate_mask[row_index, col_index])
        mismatch_type = "roi_only" if roi_value and not coordinate_value else "coordinate_only"
        x_value = x_raw[row_index, col_index]
        y_value = y_raw[row_index, col_index]
        line = (
            f"{row_index}, {col_index}, {mismatch_type}, "
            f"{int(roi_value)}, {int(coordinate_value)}, {x_value:.9g}, {y_value:.9g}"
        )
        lines.append(line)
        if len(preview) < 10:
            preview.append(line)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return preview


def _print_mismatch_preview(preview_lines: list[str]) -> None:
    if not preview_lines:
        print("  Mismatch preview: no mismatched points")
        return
    print("  Mismatch preview:")
    for line in preview_lines:
        print(f"    {line}")


def _determine_coordinate_convention_transform(
    x: np.ndarray,
    y: np.ndarray,
) -> CoordinateConventionTransform:
    mean_dx_columns = _safe_nanmedian(np.diff(x, axis=1))
    mean_dy_rows = _safe_nanmedian(np.diff(y, axis=0))
    return CoordinateConventionTransform(
        reflect_x=bool(mean_dx_columns is not None and mean_dx_columns < 0.0),
        reflect_y=bool(mean_dy_rows is not None and mean_dy_rows < 0.0),
    )


def _apply_coordinate_convention_to_axis_grid(
    grid: np.ndarray,
    *,
    reflect_axis: bool,
) -> np.ndarray:
    transformed = np.asarray(grid, dtype=np.float64).copy()
    if not reflect_axis:
        return transformed

    finite_mask = np.isfinite(transformed)
    if not np.any(finite_mask):
        return transformed

    min_value = float(np.min(transformed[finite_mask]))
    max_value = float(np.max(transformed[finite_mask]))
    transformed[finite_mask] = min_value + max_value - transformed[finite_mask]
    return transformed


def _apply_coordinate_convention_to_strain(
    strain: np.ndarray,
    transform: CoordinateConventionTransform,
    *,
    component_names: tuple[str, ...],
) -> np.ndarray:
    transformed = np.asarray(strain, dtype=np.float64).copy()
    should_flip_shear_sign = transform.reflect_x ^ transform.reflect_y
    if not should_flip_shear_sign:
        return transformed

    for component_index in _find_shear_component_indices(component_names, strain.shape[1]):
        transformed[:, component_index, :, :] *= -1.0
    return transformed.copy()


def _find_shear_component_indices(
    component_names: tuple[str, ...],
    component_count: int,
) -> list[int]:
    indices: list[int] = []
    for component_index in range(min(component_count, len(component_names))):
        component_name = component_names[component_index].strip().lower()
        if (
            "xy" in component_name
            or "yx" in component_name
            or component_name in {"exy", "eyx", "c12", "c21", "e12", "e21"}
            or "12" in component_name
            or "21" in component_name
            or "shear" in component_name
        ):
            indices.append(component_index)
    return indices


def _apply_coordinate_convention_to_force(
    force: np.ndarray,
    transform: CoordinateConventionTransform,
    *,
    boundary_conditions: BoundaryConditionConfig,
) -> np.ndarray:
    transformed = np.asarray(force, dtype=np.float64).copy()
    reflected_axis = (
        transform.reflect_x if boundary_conditions.force_component == "x" else transform.reflect_y
    )
    if not reflected_axis:
        return transformed

    if transformed.ndim == 1:
        transformed *= -1.0
        return transformed

    if transformed.ndim != 2:
        raise ValueError(f"Unsupported force array shape {transformed.shape}.")

    component_index = 0 if boundary_conditions.force_component == "x" else 1
    if transformed.shape[1] <= component_index:
        raise ValueError(
            f"Force array shape {transformed.shape} does not contain the required "
            f"{boundary_conditions.force_component}-component."
        )
    transformed[:, component_index] *= -1.0
    return transformed


def _serialise_coordinate_convention_transform(
    transform: CoordinateConventionTransform,
) -> dict[str, Any]:
    return {
        "applied": transform.applied,
        "reflect_x": transform.reflect_x,
        "reflect_y": transform.reflect_y,
        "method": "coordinate-axis reflection in place",
        "notes": [
            "The physical coordinate fields are reflected, when needed, so x increases left-to-right and y increases top-to-bottom.",
            "If exactly one axis is reflected, shear strain components are negated.",
            "If the configured force component lies on a reflected axis, that force sign is negated too.",
        ],
    }


def _finalise_region_of_interest_outputs(
    *,
    specimen_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    roi_alignment_x: np.ndarray,
    roi_alignment_y: np.ndarray,
    output_folder: Path,
    generated_outputs_folder: Path,
    roi_summary: dict[str, Any] | None,
    reference_image_path: Path | None,
    spatial_selection_info: dict[str, Any] | None,
    orientation_transform: CoordinateConventionTransform,
) -> tuple[np.ndarray, dict[str, Any] | None, list[str]]:
    if roi_summary is None:
        return specimen_mask, roi_summary, []

    warnings: list[str] = []

    physical_roi_definition = roi.convert_mask_to_physical_roi(
        specimen_mask,
        x=x,
        y=y,
    )
    region_of_interest_yaml = output_folder / "region_of_interest.yaml"
    roi.write_roi_yaml(physical_roi_definition, region_of_interest_yaml)
    specimen_mask = roi.sample_roi_definition_at_coordinates(physical_roi_definition, x, y)

    sampled_pixel_roi_definition = roi.convert_mask_to_physical_roi(
        specimen_mask,
        x=roi_alignment_x,
        y=roi_alignment_y,
    )
    sampled_pixel_roi_yaml = generated_outputs_folder / "region_of_interest_pixel.yaml"
    sampled_pixel_mask_tiff = generated_outputs_folder / "region_of_interest_pixel_mask.tiff"
    roi.write_roi_yaml(sampled_pixel_roi_definition, sampled_pixel_roi_yaml)
    roi.write_mask_tiff(specimen_mask, sampled_pixel_mask_tiff)

    cropped_overlay_path = None
    source_mask_tiff = roi_summary.get("source_mask_tiff")
    if (
        reference_image_path is not None
        and reference_image_path.exists()
        and source_mask_tiff is not None
        and Path(str(source_mask_tiff)).exists()
    ):
        cropped_overlay_path = generated_outputs_folder / "region_of_interest_pixel_overlay.png"
        _save_cropped_source_space_roi_overlay(
            reference_image_path=reference_image_path,
            source_mask_path=Path(str(source_mask_tiff)),
            x_pixels=roi_alignment_x,
            y_pixels=roi_alignment_y,
            output_path=cropped_overlay_path,
        )

    roi_summary["roi_yaml"] = str(region_of_interest_yaml)
    roi_summary["sampled_pixel_roi_yaml"] = str(sampled_pixel_roi_yaml)
    roi_summary["sampled_pixel_mask_tiff"] = str(sampled_pixel_mask_tiff)
    roi_summary["pixel_overlay_image"] = str(cropped_overlay_path) if cropped_overlay_path is not None else None
    roi_summary["spatial_selection"] = spatial_selection_info
    roi_summary["coordinate_convention_transform"] = _serialise_coordinate_convention_transform(orientation_transform)

    consistency_mask = roi.sample_roi_definition_at_coordinates(physical_roi_definition, x, y)
    consistency_mismatch_count = int(np.count_nonzero(consistency_mask ^ specimen_mask))
    roi_summary["final_consistency_mismatch_count"] = consistency_mismatch_count
    if consistency_mismatch_count > 0:
        warnings.append(
            f"The final physical ROI YAML differs from the saved specimen mask at {consistency_mismatch_count} points."
        )

    return specimen_mask, roi_summary, warnings


def _save_cropped_source_space_roi_overlay(
    *,
    reference_image_path: Path,
    source_mask_path: Path,
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
    output_path: Path,
) -> Path:
    reference_image = roi.load_grayscale_image(reference_image_path)
    source_mask = roi.load_grayscale_image(source_mask_path) > 0
    valid = np.isfinite(x_pixels) & np.isfinite(y_pixels)
    if not np.any(valid):
        raise ValueError("Could not create a pixel-space ROI overlay because the pixel-coordinate grids are all NaN.")

    min_col = max(0, int(np.floor(np.nanmin(x_pixels[valid]))))
    max_col = min(reference_image.shape[1], int(np.ceil(np.nanmax(x_pixels[valid]))) + 1)
    min_row = max(0, int(np.floor(np.nanmin(y_pixels[valid]))))
    max_row = min(reference_image.shape[0], int(np.ceil(np.nanmax(y_pixels[valid]))) + 1)

    image_crop = reference_image[min_row:max_row, min_col:max_col]
    overlay_mask = source_mask[min_row:max_row, min_col:max_col]

    roi.save_roi_overlay_plot(image_crop, overlay_mask, output_path)
    return output_path


def _estimate_point_area(
    x: np.ndarray,
    y: np.ndarray,
    *,
    total_area_override: float | None = None,
) -> np.ndarray:
    point_area = np.full(x.shape, np.nan, dtype=np.float64)
    valid_mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid_mask):
        return point_area

    if x.shape[0] < 2 or x.shape[1] < 2:
        if total_area_override is not None:
            point_area[valid_mask] = float(total_area_override) / int(np.count_nonzero(valid_mask))
        return point_area

    dx = np.abs(np.gradient(x, axis=1))
    dy = np.abs(np.gradient(y, axis=0))
    point_area = dx * dy
    point_area[~valid_mask] = np.nan
    return point_area


def _check_specimen_area(
    *,
    point_area: np.ndarray,
    original_coordinate_valid_mask: np.ndarray,
    specimen_mask: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    area_from_coordinate_mask = float(np.nansum(np.where(original_coordinate_valid_mask, point_area, 0.0)))
    area_from_specimen_mask = float(np.nansum(np.where(specimen_mask, point_area, 0.0)))
    point_count = int(np.count_nonzero(specimen_mask))

    if point_count > 0:
        median_point_area = float(np.nanmedian(point_area[specimen_mask]))
        representative_area = float(median_point_area * point_count)
    else:
        median_point_area = float("nan")
        representative_area = float("nan")

    relative_difference = None
    if area_from_coordinate_mask > 0.0:
        relative_difference = abs(area_from_specimen_mask - area_from_coordinate_mask) / area_from_coordinate_mask
        if relative_difference > 0.05:
            warnings.append(
                "Estimated area from the final specimen mask differs noticeably from the area implied by all finite "
                f"coordinate points ({100.0 * relative_difference:.1f}% difference)."
            )

    return {
        "area_from_coordinate_valid_mask_mm2": area_from_coordinate_mask,
        "area_from_final_specimen_mask_mm2": area_from_specimen_mask,
        "representative_point_area_times_point_count_mm2": representative_area,
        "final_specimen_point_count": point_count,
        "median_point_area_mm2": median_point_area,
        "relative_difference_vs_coordinate_valid_mask": relative_difference,
    }, warnings


def _save_region_of_interest_summary(
    output_folder: Path,
    generated_outputs_folder: Path,
    roi_summary: dict[str, Any] | None,
) -> dict[str, str] | None:
    if roi_summary is None:
        return None

    saved_paths: dict[str, str] = {}
    roi_yaml = roi_summary.get("roi_yaml")
    if roi_yaml is not None:
        saved_paths["region_of_interest_yaml"] = str(Path(str(roi_yaml)))

    metadata_json = roi_summary.get("metadata_json")
    if metadata_json is not None:
        metadata_source = Path(str(metadata_json))
        metadata_destination = generated_outputs_folder / "region_of_interest.metadata.json"
        metadata_payload = json.loads(metadata_source.read_text(encoding="utf-8"))
        metadata_payload["roi_yaml"] = str(saved_paths.get("region_of_interest_yaml", roi_summary.get("roi_yaml")))
        metadata_payload["coordinate_space"] = str(roi_summary.get("coordinate_space", "physical"))
        intermediate_pixel_roi_yaml = roi_summary.get("intermediate_pixel_roi_yaml")
        if intermediate_pixel_roi_yaml is not None:
            metadata_payload["intermediate_pixel_roi_yaml"] = str(intermediate_pixel_roi_yaml)
        sampled_pixel_roi_yaml = roi_summary.get("sampled_pixel_roi_yaml")
        if sampled_pixel_roi_yaml is not None:
            metadata_payload["sampled_pixel_roi_yaml"] = str(sampled_pixel_roi_yaml)
            saved_paths["sampled_pixel_roi_yaml"] = str(sampled_pixel_roi_yaml)
        sampled_pixel_mask_tiff = roi_summary.get("sampled_pixel_mask_tiff")
        if sampled_pixel_mask_tiff is not None:
            metadata_payload["sampled_pixel_mask_tiff"] = str(sampled_pixel_mask_tiff)
            saved_paths["sampled_pixel_mask_tiff"] = str(sampled_pixel_mask_tiff)
        pixel_overlay_image = roi_summary.get("pixel_overlay_image")
        if pixel_overlay_image is not None:
            metadata_payload["pixel_overlay_image"] = str(pixel_overlay_image)
            saved_paths["pixel_overlay_image"] = str(pixel_overlay_image)
        source_mask_tiff = roi_summary.get("source_mask_tiff")
        if source_mask_tiff is not None:
            metadata_payload["source_mask_tiff"] = str(source_mask_tiff)
        source_overlay_image = roi_summary.get("source_overlay_image")
        if source_overlay_image is not None:
            metadata_payload["source_overlay_image"] = str(source_overlay_image)
        metadata_payload["alignment"] = roi_summary.get("alignment")
        metadata_payload["spatial_selection"] = roi_summary.get("spatial_selection")
        metadata_payload["coordinate_convention_transform"] = roi_summary.get("coordinate_convention_transform")
        metadata_payload["notes"] = roi_summary.get("notes", [])
        metadata_destination.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
        saved_paths["region_of_interest_metadata_json"] = str(metadata_destination)

    return saved_paths


def _save_outputs(
    *,
    output_folder: Path,
    generated_outputs_folder: Path,
    x: np.ndarray,
    y: np.ndarray,
    strain: np.ndarray,
    force: np.ndarray,
    time: np.ndarray,
    specimen_mask: np.ndarray,
    pixel_area: np.ndarray,
) -> None:
    np.save(output_folder / "x.npy", x)
    np.save(output_folder / "y.npy", y)
    np.save(output_folder / "strain.npy", strain)
    np.save(output_folder / "force.npy", force)
    np.save(output_folder / "time.npy", time)
    np.save(output_folder / "pixel_area.npy", pixel_area)
    np.save(output_folder / "specimen_mask.npy", specimen_mask.astype(bool))
    np.save(generated_outputs_folder / "specimen_mask.npy", specimen_mask.astype(bool))


def _create_diagnostic_plots(
    *,
    output_folder: Path,
    x_main: np.ndarray,
    y_main: np.ndarray,
    x_main_prepared: np.ndarray,
    y_main_prepared: np.ndarray,
    x_pixel: np.ndarray,
    y_pixel: np.ndarray,
    strain: np.ndarray,
    component_names: tuple[str, ...],
    force: np.ndarray,
    time: np.ndarray,
    specimen_mask: np.ndarray,
    original_coordinate_valid_mask: np.ndarray,
    roi_summary: dict[str, Any] | None,
    boundary_conditions: BoundaryConditionConfig,
) -> dict[str, Path]:
    pyplot = _load_pyplot()
    plot_paths: dict[str, Path] = {}

    coordinate_mask = np.asarray(original_coordinate_valid_mask, dtype=bool)

    coordinate_plot_path = output_folder / "coordinate_fields_mm.png"
    fig, axes = pyplot.subplots(1, 3, figsize=(15, 4.5))
    _scatter_with_colorbar(pyplot, fig, axes[0], x_main, y_main, x_main, "Configured main x coordinates")
    _scatter_with_colorbar(pyplot, fig, axes[1], x_main, y_main, y_main, "Configured main y coordinates")
    axes[2].imshow(coordinate_mask, cmap="gray", interpolation="nearest")
    axes[2].set_title("Coordinate valid mask")
    axes[2].set_xlabel("x index")
    axes[2].set_ylabel("y index")
    fig.tight_layout()
    fig.savefig(coordinate_plot_path, dpi=200)
    pyplot.close(fig)
    plot_paths["coordinate_fields_mm"] = coordinate_plot_path

    coordinate_plot_pixel_path = output_folder / "coordinate_fields_pixel.png"
    fig, axes = pyplot.subplots(1, 3, figsize=(15, 4.5))
    _scatter_with_colorbar(pyplot, fig, axes[0], x_pixel, y_pixel, x_pixel, "x pixel coordinates")
    _scatter_with_colorbar(pyplot, fig, axes[1], x_pixel, y_pixel, y_pixel, "y pixel coordinates")
    axes[2].imshow(coordinate_mask, cmap="gray", interpolation="nearest")
    axes[2].set_title("Coordinate valid mask")
    axes[2].set_xlabel("x index")
    axes[2].set_ylabel("y index")
    fig.tight_layout()
    fig.savefig(coordinate_plot_pixel_path, dpi=200)
    pyplot.close(fig)
    plot_paths["coordinate_fields_pixel"] = coordinate_plot_pixel_path

    load_plot_path = output_folder / "force_time_checks.png"
    fig, axes = pyplot.subplots(1, 3, figsize=(15, 4.5))
    indices = np.arange(time.size)
    axes[0].plot(indices, time, marker="o")
    axes[0].set_title("Time by timestep")
    axes[0].set_xlabel("Timestep index")
    axes[0].set_ylabel("Time")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(indices, force, marker="o")
    axes[1].set_title("Force by timestep")
    axes[1].set_xlabel("Timestep index")
    axes[1].set_ylabel("Force")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(time, force, marker="o")
    axes[2].set_title("Force vs time")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Force")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(load_plot_path, dpi=200)
    pyplot.close(fig)
    plot_paths["force_time_checks"] = load_plot_path

    strain_plot_path = output_folder / "strain_component_checks_mm.png"
    component_count = min(strain.shape[1], len(component_names))
    fig, axes = pyplot.subplots(2, component_count, figsize=(5 * component_count, 8))
    axes_array = np.atleast_2d(axes)
    for component_index in range(component_count):
        component_name = component_names[component_index]
        _scatter_with_colorbar(
            pyplot,
            fig,
            axes_array[0, component_index],
            x_main_prepared,
            y_main_prepared,
            strain[0, component_index],
            f"{component_name} at first timestep (main coords)",
            valid_mask=specimen_mask,
        )
        _scatter_with_colorbar(
            pyplot,
            fig,
            axes_array[1, component_index],
            x_main_prepared,
            y_main_prepared,
            strain[-1, component_index],
            f"{component_name} at last timestep (main coords)",
            valid_mask=specimen_mask,
        )
    fig.tight_layout()
    fig.savefig(strain_plot_path, dpi=200)
    pyplot.close(fig)
    plot_paths["strain_component_checks_mm"] = strain_plot_path

    strain_plot_pixel_path = output_folder / "strain_component_checks_px.png"
    fig, axes = pyplot.subplots(2, component_count, figsize=(5 * component_count, 8))
    axes_array = np.atleast_2d(axes)
    for component_index in range(component_count):
        component_name = component_names[component_index]
        _scatter_with_colorbar(
            pyplot,
            fig,
            axes_array[0, component_index],
            x_pixel,
            y_pixel,
            strain[0, component_index],
            f"{component_name} at first timestep (pixel coords)",
            valid_mask=specimen_mask,
        )
        _scatter_with_colorbar(
            pyplot,
            fig,
            axes_array[1, component_index],
            x_pixel,
            y_pixel,
            strain[-1, component_index],
            f"{component_name} at last timestep (pixel coords)",
            valid_mask=specimen_mask,
        )
    fig.tight_layout()
    fig.savefig(strain_plot_pixel_path, dpi=200)
    pyplot.close(fig)
    plot_paths["strain_component_checks_px"] = strain_plot_pixel_path

    mask_plot_path = output_folder / "mask_checks.png"
    _save_mask_comparison_plot(
        output_path=mask_plot_path,
        specimen_mask=specimen_mask,
        coordinate_mask=coordinate_mask,
    )
    plot_paths["mask_checks"] = mask_plot_path

    boundary_plot_path = output_folder / "boundary_conditions.png"
    _save_boundary_condition_plot(
        output_path=boundary_plot_path,
        x=x_main_prepared,
        y=y_main_prepared,
        specimen_mask=specimen_mask,
        force=force,
        boundary_conditions=boundary_conditions,
    )
    plot_paths["boundary_conditions"] = boundary_plot_path

    if roi_summary is not None and roi_summary.get("pixel_overlay_image") is not None:
        plot_paths["roi_pixel_overlay"] = Path(str(roi_summary["pixel_overlay_image"]))

    return plot_paths


def _scatter_with_colorbar(
    pyplot,
    fig,
    ax,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    values: np.ndarray,
    title: str,
    valid_mask: np.ndarray | None = None,
) -> None:
    valid = np.isfinite(x_coords) & np.isfinite(y_coords) & np.isfinite(values)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return

    scatter = ax.scatter(
        x_coords[valid],
        y_coords[valid],
        c=values[valid],
        s=6,
        linewidths=0.0,
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    unique_y = np.unique(y_coords[valid])
    if unique_y.size >= 2 and np.nanmedian(np.diff(unique_y)) > 0.0:
        ax.invert_yaxis()
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)


def _save_mask_comparison_plot(
    *,
    output_path: Path,
    specimen_mask: np.ndarray,
    coordinate_mask: np.ndarray,
) -> None:
    pyplot = _load_pyplot()
    fig, axes = pyplot.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(specimen_mask, cmap="gray", interpolation="nearest")
    axes[0].set_title("Final specimen mask")
    axes[0].set_xlabel("x index")
    axes[0].set_ylabel("y index")
    axes[1].imshow(coordinate_mask, cmap="gray", interpolation="nearest")
    axes[1].set_title("Coordinate finite-value mask")
    axes[1].set_xlabel("x index")
    axes[1].set_ylabel("y index")
    mismatch = specimen_mask ^ coordinate_mask
    axes[2].imshow(mismatch, cmap="magma", interpolation="nearest")
    axes[2].set_title("Mask mismatch")
    axes[2].set_xlabel("x index")
    axes[2].set_ylabel("y index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    pyplot.close(fig)


def _save_boundary_condition_plot(
    *,
    output_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    specimen_mask: np.ndarray,
    force: np.ndarray,
    boundary_conditions: BoundaryConditionConfig,
) -> None:
    pyplot = _load_pyplot()
    valid = specimen_mask & np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        raise ValueError("Could not plot boundary conditions because the specimen mask is empty.")

    x_valid = x[valid]
    y_valid = y[valid]
    x_min = float(np.min(x_valid))
    x_max = float(np.max(x_valid))
    y_min = float(np.min(y_valid))
    y_max = float(np.max(y_valid))
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)
    edge_line_width = 3.0

    fig, ax = pyplot.subplots(figsize=(7, 6))
    ax.scatter(x_valid, y_valid, s=3, c="#c7d2da", alpha=0.6, linewidths=0.0, label="Specimen points")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("Physical x")
    ax.set_ylabel("Physical y")
    ax.set_title("Boundary-condition diagnostic")

    edge_segments = {
        "min_x_edge": ((x_min, y_min), (x_min, y_max), boundary_conditions.min_x_edge),
        "max_x_edge": ((x_max, y_min), (x_max, y_max), boundary_conditions.max_x_edge),
        "min_y_edge": ((x_min, y_min), (x_max, y_min), boundary_conditions.min_y_edge),
        "max_y_edge": ((x_min, y_max), (x_max, y_max), boundary_conditions.max_y_edge),
    }

    for edge_name, (start, end, edge_config) in edge_segments.items():
        color = _edge_boundary_colour(edge_config)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=edge_line_width,
            solid_capstyle="round",
        )
        midpoint_x = 0.5 * (start[0] + end[0])
        midpoint_y = 0.5 * (start[1] + end[1])
        ax.text(
            midpoint_x,
            midpoint_y,
            _edge_boundary_label(edge_name, edge_config),
            fontsize=8,
            color=color,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.0},
        )

    arrow_vector = _resolve_force_arrow_vector(force, boundary_conditions)
    traction_edge_name = _find_traction_edge_name(boundary_conditions)
    if traction_edge_name is not None:
        arrow_length = boundary_conditions.arrow_scale_fraction * max(width, height)
        if traction_edge_name == "min_x_edge":
            arrow_origin = (x_min, 0.5 * (y_min + y_max))
        elif traction_edge_name == "max_x_edge":
            arrow_origin = (x_max, 0.5 * (y_min + y_max))
        elif traction_edge_name == "min_y_edge":
            arrow_origin = (0.5 * (x_min + x_max), y_min)
        else:
            arrow_origin = (0.5 * (x_min + x_max), y_max)

        ax.arrow(
            arrow_origin[0],
            arrow_origin[1],
            arrow_length * arrow_vector[0],
            arrow_length * arrow_vector[1],
            color="tab:orange",
            width=0.01 * max(width, height),
            length_includes_head=True,
            head_width=0.04 * max(width, height),
            head_length=0.06 * max(width, height),
        )
        ax.text(
            arrow_origin[0] + 0.55 * arrow_length * arrow_vector[0],
            arrow_origin[1] + 0.55 * arrow_length * arrow_vector[1],
            boundary_conditions.arrow_label,
            color="tab:orange",
            fontsize=9,
            ha="left" if arrow_vector[0] >= 0.0 else "right",
            va="bottom" if arrow_vector[1] <= 0.0 else "top",
        )

    margin_x = 0.08 * width
    margin_y = 0.08 * height
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_max + margin_y, y_min - margin_y)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    pyplot.close(fig)


def _edge_boundary_colour(edge_config: EdgeBoundaryConfig) -> str:
    states = {edge_config.x, edge_config.y}
    if "traction" in states:
        return "tab:orange"
    if "fixed" in states:
        return "tab:red"
    return "tab:green"


def _edge_boundary_label(edge_name: str, edge_config: EdgeBoundaryConfig) -> str:
    return (
        f"{edge_name}\n"
        f"x={edge_config.x}, y={edge_config.y}"
    )


def _find_traction_edge_name(boundary_conditions: BoundaryConditionConfig) -> str | None:
    ordered_edges = (
        ("min_x_edge", boundary_conditions.min_x_edge),
        ("max_x_edge", boundary_conditions.max_x_edge),
        ("min_y_edge", boundary_conditions.min_y_edge),
        ("max_y_edge", boundary_conditions.max_y_edge),
    )
    for edge_name, edge_config in ordered_edges:
        if edge_config.x == "traction" or edge_config.y == "traction":
            return edge_name
    return None


def _resolve_force_arrow_vector(
    force: np.ndarray,
    boundary_conditions: BoundaryConditionConfig,
) -> tuple[float, float]:
    representative_sign = 1.0
    nonzero_force = force[np.isfinite(force) & ~np.isclose(force, 0.0)]
    if nonzero_force.size > 0:
        representative_sign = float(np.sign(nonzero_force[-1]))

    configured_sign = 1.0 if boundary_conditions.force_positive_direction == "+" else -1.0
    direction_sign = representative_sign * configured_sign
    if boundary_conditions.force_component == "x":
        return direction_sign, 0.0
    return 0.0, direction_sign


def _serialise_boundary_conditions(
    boundary_conditions: BoundaryConditionConfig,
) -> dict[str, Any]:
    return {
        "min_x_edge": {"x": boundary_conditions.min_x_edge.x, "y": boundary_conditions.min_x_edge.y},
        "max_x_edge": {"x": boundary_conditions.max_x_edge.x, "y": boundary_conditions.max_x_edge.y},
        "min_y_edge": {"x": boundary_conditions.min_y_edge.x, "y": boundary_conditions.min_y_edge.y},
        "max_y_edge": {"x": boundary_conditions.max_y_edge.x, "y": boundary_conditions.max_y_edge.y},
        "force_component": boundary_conditions.force_component,
        "force_positive_direction": boundary_conditions.force_positive_direction,
        "arrow_scale_fraction": boundary_conditions.arrow_scale_fraction,
        "arrow_label": boundary_conditions.arrow_label,
    }


def _load_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as pyplot

    return pyplot


if __name__ == "__main__":
    main()

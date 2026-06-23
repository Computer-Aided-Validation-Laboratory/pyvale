from __future__ import annotations

import csv
from datetime import datetime
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class PreparationConfig:
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
    strain_h5_dataset_names: tuple[str, ...] = ("exx", "eyy", "exy")
    strain_component_order: tuple[str, ...] = ("exx", "eyy", "exy")
    csv_preview_rows: int = 5


@dataclass(slots=True)
class CsvTable:
    path: Path
    header: list[str]
    rows: list[list[str]]
    numeric_columns: list[bool]

    @property
    def column_count(self) -> int:
        return len(self.header)


# Edit this section for a new dataset.
CONFIG = PreparationConfig(
    input_folder=Path(
        "/home/robh/1_Projects/pyvale/dev/vfm/rob-data/wdbn4-temporally-processed-data-260622-1404"
    ),
    output_folder=Path(
        "/home/robh/1_Projects/pyvale/dev/vfm/rob-data/wdbn4-temporally-processed-data-260622-1404/prepared-vfm-inputs"
    ),
    x_coordinates_input_file="Image_0000_0.tiff_x.csv",
    y_coordinates_input_file="Image_0000_0.tiff_y.csv",
    strain_input_file="strain_data.h5",
    force_input_file="force_history.csv",
    time_input_file="force_history.csv",
    x_coordinates_pixel_input_file="Image_0000_0.tiff_x_pic.csv",
    y_coordinates_pixel_input_file="Image_0000_0.tiff_y_pic.csv",
    region_of_interest_input_file="WDBN4_correlation_cam0_SS49_ST3_SFaffine_SW3_Q4.m3inp",
    reference_image_file="Image_0000_0.tiff",
    strain_h5_dataset_names=("exx", "eyy", "exy"),
    strain_component_order=("exx", "eyy", "exy"),
)


def main() -> None:
    config = CONFIG
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    output_folder = config.output_folder.parent / f"{config.output_folder.name}-{timestamp}"
    generated_outputs_folder = output_folder / "generated-outputs"
    output_folder.mkdir(parents=True, exist_ok=True)
    generated_outputs_folder.mkdir(parents=True, exist_ok=True)

    print(f"Input folder:  {config.input_folder}")
    print(f"Output folder: {output_folder}")

    strain = _load_strain_data(config)
    validation_warnings: list[str] = []

    x, y, coordinate_load_info = _load_main_coordinate_grids(config, strain.shape[2:])
    _validate_coordinate_grids(x, y)
    _validate_strain_shape(strain, x.shape)
    validation_warnings.extend(coordinate_load_info["warnings"])

    roi_alignment_x, roi_alignment_y, roi_alignment_info = _load_roi_alignment_coordinate_grids(
        config=config,
        fallback_x=x,
        fallback_y=y,
        target_shape=strain.shape[2:],
    )
    validation_warnings.extend(roi_alignment_info["warnings"])

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
    validation_warnings.extend(_check_coordinate_conventions(x, y))

    specimen_mask, roi_summary, roi_warnings = _prepare_specimen_mask(
        config,
        x,
        y,
        roi_alignment_x,
        roi_alignment_y,
        output_folder,
        generated_outputs_folder,
    )
    validation_warnings.extend(roi_warnings)

    pixel_area = _estimate_point_area(x, y)
    area_checks, area_warnings = _check_specimen_area(
        point_area=pixel_area,
        x=x,
        y=y,
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
        x=x,
        y=y,
        strain=strain,
        component_names=config.strain_component_order,
        force=force,
        time=time,
        specimen_mask=specimen_mask,
        roi_summary=roi_summary,
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
        "coordinate_load_info": coordinate_load_info,
        "roi_alignment_coordinate_info": roi_alignment_info,
        "force_selection": force_info,
        "time_selection": {
            **time_info,
            "time_offset_corrected": time_offset_correction,
        },
        "roi_summary": roi_summary,
        "plots": {name: str(path) for name, path in plot_paths.items()},
        "warnings": validation_warnings,
    }
    summary_path = generated_outputs_folder / "prepareinputdata_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nSaved arrays:")
    for name in ("x.npy", "y.npy", "strain.npy", "force.npy", "time.npy", "pixel_area.npy"):
        print(f"  - {output_folder / name}")
    print(f"  - {generated_outputs_folder / 'specimen_mask.npy'}")

    print("\nSaved plots:")
    for name, path in plot_paths.items():
        print(f"  - {name}: {path}")

    print(f"\nSaved summary: {summary_path}")

    if validation_warnings:
        print("\nWarnings:")
        for warning in validation_warnings:
            print(f"  - {warning}")
    else:
        print("\nValidation checks passed with no warnings.")


def _resolve_input_path(config: PreparationConfig, file_name: str | None) -> Path | None:
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
    config: PreparationConfig,
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
    config: PreparationConfig,
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


def _load_strain_data(config: PreparationConfig) -> np.ndarray:
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
    config: PreparationConfig,
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

        if 2 <= table.column_count <= 10 and path not in previewed_paths:
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
    print(f"\nPreview of {table.path}:")
    header_cells = [f"[{index}] {name}" for index, name in enumerate(table.header)]
    print(" | ".join(_truncate(cell, 28) for cell in header_cells))
    for row in table.rows[:n_rows]:
        print(" | ".join(_truncate(cell.strip(), 28) for cell in row))


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


def _safe_nanmedian(array: np.ndarray) -> float | None:
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _import_roi_helper():
    try:
        from . import vfmregionofinterest as roi_helper
        return roi_helper
    except Exception:
        module_dir = Path(__file__).resolve().parent
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        import vfmregionofinterest as roi_helper

        return roi_helper


def _prepare_specimen_mask(
    config: PreparationConfig,
    x: np.ndarray,
    y: np.ndarray,
    roi_alignment_x: np.ndarray,
    roi_alignment_y: np.ndarray,
    output_folder: Path,
    generated_outputs_folder: Path,
) -> tuple[np.ndarray, dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    coordinate_valid_mask = np.isfinite(x) & np.isfinite(y)

    roi_input_path = _resolve_input_path(config, config.region_of_interest_input_file)
    if roi_input_path is None:
        warnings.append(
            "No region_of_interest_input_file was provided. Using the coordinate NaN mask as specimen_mask."
        )
        return coordinate_valid_mask, None, warnings

    roi_helper = _import_roi_helper()
    output_dir = generated_outputs_folder / "roi_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_kwargs: dict[str, Any] = {}
    reference_image_path = _resolve_input_path(config, config.reference_image_file)
    if reference_image_path is not None:
        roi_kwargs["reference_image"] = reference_image_path

    artifacts = roi_helper.generate_vfm_input_roi(roi_input_path, output_dir, **roi_kwargs)
    roi_mask = roi_helper.rasterise_roi_definition(artifacts.roi_definition)
    specimen_mask, alignment_summary = roi_helper.sample_roi_mask_at_pixel_coordinates(
        roi_mask,
        roi_alignment_x,
        roi_alignment_y,
    )
    physical_roi_definition = roi_helper.convert_mask_to_physical_roi(
        specimen_mask,
        x=x,
        y=y,
    )
    region_of_interest_yaml = output_folder / "region_of_interest.yaml"
    roi_helper.write_roi_yaml(physical_roi_definition, region_of_interest_yaml)
    specimen_mask = roi_helper.sample_roi_definition_at_coordinates(physical_roi_definition, x, y)

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
        "mask_shape": list(roi_mask.shape),
        "sampled_mask_shape": list(specimen_mask.shape),
        "mask_pixel_count": int(artifacts.mask_pixel_count),
        "roi_yaml": str(region_of_interest_yaml),
        "intermediate_pixel_roi_yaml": str(artifacts.roi_yaml),
        "metadata_json": str(artifacts.metadata_json),
        "mask_tiff": str(artifacts.mask_tiff),
        "overlay_image": str(artifacts.overlay_image) if artifacts.overlay_image is not None else None,
        "coordinate_space": "physical",
        "alignment": alignment_summary,
        "mismatch_count_vs_coordinate_nan_mask": mismatch_count,
    }
    return specimen_mask, summary, warnings


def _estimate_point_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.abs(np.gradient(x, axis=1))
    dy = np.abs(np.gradient(y, axis=0))
    point_area = dx * dy
    point_area[~np.isfinite(x) | ~np.isfinite(y)] = np.nan
    return point_area


def _check_specimen_area(
    *,
    point_area: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    specimen_mask: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    coordinate_valid_mask = np.isfinite(x) & np.isfinite(y)
    area_from_coordinate_mask = float(np.nansum(np.where(coordinate_valid_mask, point_area, 0.0)))
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
    np.save(generated_outputs_folder / "specimen_mask.npy", specimen_mask.astype(bool))


def _create_diagnostic_plots(
    *,
    output_folder: Path,
    x: np.ndarray,
    y: np.ndarray,
    strain: np.ndarray,
    component_names: tuple[str, ...],
    force: np.ndarray,
    time: np.ndarray,
    specimen_mask: np.ndarray,
    roi_summary: dict[str, Any] | None,
) -> dict[str, Path]:
    pyplot = _load_pyplot()
    plot_paths: dict[str, Path] = {}

    coordinate_mask = np.isfinite(x) & np.isfinite(y)

    coordinate_plot_path = output_folder / "coordinate_fields.png"
    fig, axes = pyplot.subplots(1, 3, figsize=(15, 4.5))
    _imshow_with_colorbar(pyplot, fig, axes[0], x, "x coordinates")
    _imshow_with_colorbar(pyplot, fig, axes[1], y, "y coordinates")
    axes[2].imshow(coordinate_mask, cmap="gray", interpolation="nearest")
    axes[2].set_title("Coordinate valid mask")
    axes[2].set_xlabel("x index")
    axes[2].set_ylabel("y index")
    fig.tight_layout()
    fig.savefig(coordinate_plot_path, dpi=200)
    pyplot.close(fig)
    plot_paths["coordinate_fields"] = coordinate_plot_path

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

    strain_plot_path = output_folder / "strain_component_checks.png"
    component_count = min(strain.shape[1], len(component_names))
    fig, axes = pyplot.subplots(2, component_count, figsize=(5 * component_count, 8))
    axes_array = np.atleast_2d(axes)
    for component_index in range(component_count):
        component_name = component_names[component_index]
        _imshow_with_colorbar(
            pyplot,
            fig,
            axes_array[0, component_index],
            strain[0, component_index],
            f"{component_name} at first timestep",
        )
        _imshow_with_colorbar(
            pyplot,
            fig,
            axes_array[1, component_index],
            strain[-1, component_index],
            f"{component_name} at last timestep",
        )
    fig.tight_layout()
    fig.savefig(strain_plot_path, dpi=200)
    pyplot.close(fig)
    plot_paths["strain_component_checks"] = strain_plot_path

    mask_plot_path = output_folder / "mask_checks.png"
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
    fig.savefig(mask_plot_path, dpi=200)
    pyplot.close(fig)
    plot_paths["mask_checks"] = mask_plot_path

    if roi_summary is not None and roi_summary.get("overlay_image") is not None:
        plot_paths["roi_overlay"] = Path(str(roi_summary["overlay_image"]))

    return plot_paths


def _imshow_with_colorbar(pyplot, fig, ax, image: np.ndarray, title: str) -> None:
    im = ax.imshow(image, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _load_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as pyplot

    return pyplot


if __name__ == "__main__":
    main()

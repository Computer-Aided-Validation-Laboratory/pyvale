#!/usr/bin/env python3
"""
Assemble DIC CSV exports into a single toolkit-style `.npz` archive.

Edit the USER CONFIG section below, then run:
    python callers/assemble_dic_test_data_npz.py

The saved archive matches the field layout used by the Python-side VFM
tooling:
    x, y, specimen_mask, area, strain, force, time, thickness,
    boundary_conditions, source_path

The strain array is saved with shape:
    (timestep, component, y, x)

where the component order is:
    0 -> exx
    1 -> eyy
    2 -> exy

example use:
    define user config options
    run:
        python callers/assemble_dic_test_data_npz.py
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------------

INPUT_PARENT_DIR = Path("data/wdbn1-results")
OUTPUT_ROOT_DIR = INPUT_PARENT_DIR / "npz-results"
OUTPUT_BASENAME = "test_data"

SPECIMEN_THICKNESS = 0.8

BOUNDARY_CONDITIONS = {
    "left": {"x": "FREE", "y": "FREE"},
    "upper": {"x": "FREE", "y": "TRACTION"},
    "right": {"x": "FREE", "y": "FREE"},
    "lower": {"x": "FIXED", "y": "FIXED"},
}

FLIP_UD = False
ZERO_TIME_AT_FIRST_SELECTED = True

STRAIN_COMPONENTS = {
    "exx": {"folder": "exx", "suffix": ".strain_exx.csv"},
    "eyy": {"folder": "eyy", "suffix": ".strain_eyy.csv"},
    "exy": {"folder": "exy", "suffix": ".strain_exy.csv"},
}

REFERENCE_COORDINATES = {
    "x": {"folder": "x", "filename": "Image_0000_0.averaged.tif_x.csv"},
    "y": {"folder": "y", "filename": "Image_0000_0.averaged.tif_y.csv"},
}

# If `AREA.filename` is None, the area map is estimated from the reference
# x/y spacing using a constant point area over the valid specimen mask.
AREA = {"folder": None, "filename": None}

LOAD_HISTORY_MODE = "image_csv"  # "image_csv" or "separate_series"

IMAGE_CSV = {
    "path": INPUT_PARENT_DIR / "Image.csv",
    "delimiter": ";",
    "file_column": "File",
    "time_column": "TimeStamp",
    "force_column": "Analog signal 0 [N]",
    "time_scale": 1.0,
    "force_scale": 1.0,
}

TIME_SERIES = {
    "path": None,
    "delimiter": ",",
    "frame_column": None,
    "value_column": None,
    "scale": 1.0,
}

FORCE_SERIES = {
    "path": None,
    "delimiter": ",",
    "frame_column": None,
    "value_column": None,
    "scale": 1.0,
}


CSV_DELIMITER = ","
CSV_FMT = "%.6E"


FRAME_KEY_PATTERN = re.compile(r"(Image_(\d+)_(\d+))", flags=re.IGNORECASE)
STRAIN_COMPONENT_ORDER = ("exx", "eyy", "exy")


@dataclass(frozen=True)
class FolderSpec:
    folder: str | None
    filename: str | None = None
    suffix: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.folder) and (bool(self.filename) or bool(self.suffix))


def _natural_sort_key(value: str) -> list[int | str]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", value)]


def _clean_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned.lower() in {"", "none", "null"}:
        return None
    return cleaned


def _normalise_folder_spec(raw_spec: dict[str, str | None]) -> FolderSpec:
    return FolderSpec(
        folder=_clean_optional_string(raw_spec.get("folder")),
        filename=_clean_optional_string(raw_spec.get("filename")),
        suffix=_clean_optional_string(raw_spec.get("suffix")),
    )


def _extract_frame_key(text: str) -> str:
    match = FRAME_KEY_PATTERN.search(text)
    if not match:
        raise ValueError(f"Could not extract an Image_<number>_<camera> frame key from: {text}")
    return match.group(1)


def _extract_frame_number(frame_key: str) -> int:
    match = FRAME_KEY_PATTERN.search(frame_key)
    if not match:
        raise ValueError(f"Could not extract image number from frame key: {frame_key}")
    return int(match.group(2))


def _normalise_header_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def _load_matrix_csv(path: str | Path, *, delimiter: str = CSV_DELIMITER) -> np.ndarray:
    return np.genfromtxt(Path(path), delimiter=delimiter, dtype=np.float64)


def _save_npz(
    output_path: str | Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    specimen_mask: np.ndarray,
    area: np.ndarray,
    strain: np.ndarray,
    force: np.ndarray,
    time: np.ndarray,
    thickness: float,
    boundary_conditions: np.ndarray,
    source_path: str,
) -> Path:
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")

    np.savez_compressed(
        output_path,
        x=x,
        y=y,
        specimen_mask=specimen_mask,
        area=area,
        strain=strain,
        force=force,
        time=time,
        thickness=np.array(thickness, dtype=np.float64),
        boundary_conditions=boundary_conditions,
        source_path=np.array(source_path, dtype="<U512"),
    )
    return output_path


def _boundary_conditions_to_array(boundary_conditions: dict[str, dict[str, str]]) -> np.ndarray:
    return np.array(
        [
            [
                boundary_conditions["left"]["x"],
                boundary_conditions["left"]["y"],
                boundary_conditions["upper"]["x"],
                boundary_conditions["upper"]["y"],
                boundary_conditions["right"]["x"],
                boundary_conditions["right"]["y"],
                boundary_conditions["lower"]["x"],
                boundary_conditions["lower"]["y"],
            ]
        ],
        dtype="<U16",
    )


def _collect_component_file_map(parent_dir: Path, component_name: str, spec: FolderSpec) -> dict[str, Path]:
    if not spec.folder or not spec.suffix:
        raise ValueError(f"Component {component_name} requires both folder and suffix.")

    component_dir = parent_dir / spec.folder
    if not component_dir.is_dir():
        raise FileNotFoundError(f"Component folder does not exist for {component_name}: {component_dir}")

    matches = sorted(component_dir.glob(f"*{spec.suffix}"))
    if not matches:
        raise FileNotFoundError(
            f"No CSV files ending with {spec.suffix!r} were found for {component_name} in {component_dir}"
        )

    file_map: dict[str, Path] = {}
    for path in matches:
        frame_key = path.name[: -len(spec.suffix)]
        if frame_key in file_map:
            raise ValueError(f"Duplicate frame key {frame_key!r} for {component_name}.")
        file_map[frame_key] = path
    return file_map


def collect_strain_frame_paths(
    parent_dir: str | Path,
    component_specs: dict[str, dict[str, str | None]],
) -> dict[str, dict[str, Path]]:
    parent_dir = Path(parent_dir)
    specs = {name: _normalise_folder_spec(spec) for name, spec in component_specs.items()}

    missing = [name for name in STRAIN_COMPONENT_ORDER if not specs.get(name, FolderSpec(None)).enabled]
    if missing:
        raise ValueError(f"Missing required strain component specs: {missing}")

    component_maps = {
        name: _collect_component_file_map(parent_dir, name, specs[name])
        for name in STRAIN_COMPONENT_ORDER
    }

    common_frame_keys = set(component_maps[STRAIN_COMPONENT_ORDER[0]])
    for component_name, component_map in component_maps.items():
        common_frame_keys &= set(component_map)
        if not common_frame_keys:
            raise ValueError(f"No common frame keys remain after checking {component_name}.")

    return {
        frame_key: {component_name: component_map[frame_key] for component_name, component_map in component_maps.items()}
        for frame_key in sorted(common_frame_keys, key=_natural_sort_key)
    }


def _resolve_reference_file(parent_dir: Path, name: str, spec: FolderSpec) -> Path:
    if not spec.folder:
        raise ValueError(f"Reference spec for {name} must include a folder.")

    folder = parent_dir / spec.folder
    if not folder.is_dir():
        raise FileNotFoundError(f"Reference folder does not exist for {name}: {folder}")

    if spec.filename:
        path = folder / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"Reference file does not exist for {name}: {path}")
        return path

    candidates = sorted(folder.glob("*averaged*.csv"))
    if not candidates:
        candidates = sorted(folder.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files were found in reference folder {folder} for {name}.")
    return candidates[0]


def load_reference_fields(
    parent_dir: str | Path,
    reference_specs: dict[str, dict[str, str | None]],
    *,
    delimiter: str = CSV_DELIMITER,
    flip_ud: bool = False,
) -> dict[str, np.ndarray]:
    parent_dir = Path(parent_dir)
    fields: dict[str, np.ndarray] = {}
    for name, raw_spec in reference_specs.items():
        spec = _normalise_folder_spec(raw_spec)
        path = _resolve_reference_file(parent_dir, name, spec)
        values = _load_matrix_csv(path, delimiter=delimiter)
        if flip_ud:
            values = np.flip(values, axis=0)
        fields[name] = values
    return fields


def estimate_area_from_reference_grid(
    x: np.ndarray,
    y: np.ndarray,
    specimen_mask: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    specimen_mask = np.asarray(specimen_mask, dtype=bool)

    dx_candidates = np.abs(np.diff(x, axis=1))
    dy_candidates = np.abs(np.diff(y, axis=0))

    dx = float(np.nanmedian(dx_candidates[np.isfinite(dx_candidates)]))
    dy = float(np.nanmedian(dy_candidates[np.isfinite(dy_candidates)]))
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        raise ValueError("Could not estimate a positive point area from the reference x/y grids.")

    area = np.zeros_like(x, dtype=np.float64)
    area[specimen_mask] = dx * dy
    return area


def load_area_field(
    parent_dir: str | Path,
    area_spec: dict[str, str | None],
    *,
    x: np.ndarray,
    y: np.ndarray,
    specimen_mask: np.ndarray,
    delimiter: str = CSV_DELIMITER,
    flip_ud: bool = False,
) -> np.ndarray:
    spec = _normalise_folder_spec(area_spec)
    if not spec.enabled:
        return estimate_area_from_reference_grid(x, y, specimen_mask)

    path = _resolve_reference_file(Path(parent_dir), "area", spec)
    area = _load_matrix_csv(path, delimiter=delimiter)
    if flip_ud:
        area = np.flip(area, axis=0)
    return np.asarray(area, dtype=np.float64)


def load_strain_stack(
    frame_paths: dict[str, dict[str, Path]],
    *,
    delimiter: str = CSV_DELIMITER,
    flip_ud: bool = False,
) -> np.ndarray:
    strain_by_time: list[np.ndarray] = []
    for frame_key in frame_paths:
        component_arrays = []
        for component_name in STRAIN_COMPONENT_ORDER:
            values = _load_matrix_csv(frame_paths[frame_key][component_name], delimiter=delimiter)
            if flip_ud:
                values = np.flip(values, axis=0)
            component_arrays.append(values)
        strain_by_time.append(np.stack(component_arrays, axis=0))
    return np.stack(strain_by_time, axis=0)


def _read_delimited_rows(path: Path, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def load_force_time_from_image_csv(
    frame_keys: list[str],
    image_csv_config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(image_csv_config["path"])
    headers, rows = _read_delimited_rows(path, str(image_csv_config.get("delimiter", ";")))
    header_lookup = {_normalise_header_name(name): name for name in headers}

    def resolve_header(config_name: str) -> str:
        target = _normalise_header_name(str(image_csv_config[config_name]))
        if target not in header_lookup:
            raise KeyError(
                f"Column {image_csv_config[config_name]!r} was not found in {path}. "
                f"Available columns: {headers}"
            )
        return header_lookup[target]

    file_header = resolve_header("file_column")
    time_header = resolve_header("time_column")
    force_header = resolve_header("force_column")
    time_scale = float(image_csv_config.get("time_scale", 1.0))
    force_scale = float(image_csv_config.get("force_scale", 1.0))

    frame_map: dict[str, tuple[float, float]] = {}
    for row in rows:
        frame_key = _extract_frame_key(Path(row[file_header].replace("\\", "/")).name)
        frame_map[frame_key] = (
            float(row[time_header]) * time_scale,
            float(row[force_header]) * force_scale,
        )

    missing = [frame_key for frame_key in frame_keys if frame_key not in frame_map]
    if missing:
        raise KeyError(f"Image.csv is missing load-history rows for: {missing[:5]}")

    time = np.array([frame_map[frame_key][0] for frame_key in frame_keys], dtype=np.float64)
    force = np.array([frame_map[frame_key][1] for frame_key in frame_keys], dtype=np.float64)
    return time, force


def _load_series_values(
    config: dict[str, object],
    frame_keys: list[str],
) -> np.ndarray:
    path_value = config.get("path")
    if path_value is None:
        raise ValueError("Separate-series mode requires both TIME_SERIES.path and FORCE_SERIES.path.")

    path = Path(path_value)
    delimiter = str(config.get("delimiter", ","))
    scale = float(config.get("scale", 1.0))
    frame_column = _clean_optional_string(config.get("frame_column"))  # type: ignore[arg-type]
    value_column = _clean_optional_string(config.get("value_column"))  # type: ignore[arg-type]

    if frame_column or value_column:
        headers, rows = _read_delimited_rows(path, delimiter)
        header_lookup = {_normalise_header_name(name): name for name in headers}
        resolved_value = None if value_column is None else header_lookup.get(_normalise_header_name(value_column))
        resolved_frame = None if frame_column is None else header_lookup.get(_normalise_header_name(frame_column))

        if value_column is not None and resolved_value is None:
            raise KeyError(f"Value column {value_column!r} was not found in {path}. Available: {headers}")
        if frame_column is not None and resolved_frame is None:
            raise KeyError(f"Frame column {frame_column!r} was not found in {path}. Available: {headers}")

        if resolved_value is None:
            candidate_headers = [name for name in headers if _normalise_header_name(name) != _normalise_header_name(resolved_frame or "")]
            if not candidate_headers:
                raise ValueError(f"Could not infer a value column from {path}.")
            resolved_value = candidate_headers[0]

        if resolved_frame is not None:
            frame_map: dict[str, float] = {}
            for row in rows:
                frame_key = _extract_frame_key(str(row[resolved_frame]))
                frame_map[frame_key] = float(row[resolved_value]) * scale
            missing = [frame_key for frame_key in frame_keys if frame_key not in frame_map]
            if missing:
                raise KeyError(f"{path} is missing rows for: {missing[:5]}")
            return np.array([frame_map[frame_key] for frame_key in frame_keys], dtype=np.float64)

        return np.array([float(row[resolved_value]) * scale for row in rows], dtype=np.float64)

    values = np.genfromtxt(path, delimiter=delimiter, dtype=np.float64)
    values = np.atleast_1d(values).astype(np.float64).reshape(-1) * scale
    return values


def load_force_time_from_separate_series(
    frame_keys: list[str],
    time_config: dict[str, object],
    force_config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    time = _load_series_values(time_config, frame_keys)
    force = _load_series_values(force_config, frame_keys)
    if time.shape[0] != len(frame_keys):
        raise ValueError(
            f"Time series length {time.shape[0]} does not match the number of selected frames {len(frame_keys)}."
        )
    if force.shape[0] != len(frame_keys):
        raise ValueError(
            f"Force series length {force.shape[0]} does not match the number of selected frames {len(frame_keys)}."
        )
    return time, force


def load_force_time(
    frame_keys: list[str],
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "image_csv":
        return load_force_time_from_image_csv(frame_keys, IMAGE_CSV)
    if mode == "separate_series":
        return load_force_time_from_separate_series(frame_keys, TIME_SERIES, FORCE_SERIES)
    raise ValueError(f"Unsupported LOAD_HISTORY_MODE {mode!r}. Use 'image_csv' or 'separate_series'.")


def create_run_output_dir(output_root_dir: str | Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root_dir) / f"assembled_test_data_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_run_metadata(
    output_dir: Path,
    *,
    frame_keys: list[str],
    npz_path: Path,
    source_path: Path,
    thickness: float,
    flip_ud: bool,
) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_parent_dir": str(source_path),
        "npz_path": str(npz_path),
        "frame_count": len(frame_keys),
        "first_frame_key": frame_keys[0] if frame_keys else None,
        "last_frame_key": frame_keys[-1] if frame_keys else None,
        "image_numbers": [_extract_frame_number(frame_key) for frame_key in frame_keys],
        "thickness": float(thickness),
        "flip_ud": bool(flip_ud),
        "load_history_mode": LOAD_HISTORY_MODE,
        "strain_components": STRAIN_COMPONENTS,
        "reference_coordinates": REFERENCE_COORDINATES,
        "area": AREA,
        "boundary_conditions": BOUNDARY_CONDITIONS,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def assemble_test_data(
    input_parent_dir: str | Path,
    output_root_dir: str | Path,
    *,
    output_basename: str = OUTPUT_BASENAME,
    thickness: float = SPECIMEN_THICKNESS,
    flip_ud: bool = FLIP_UD,
    zero_time_at_first_selected: bool = ZERO_TIME_AT_FIRST_SELECTED,
) -> Path:
    input_parent_dir = Path(input_parent_dir)
    frame_paths = collect_strain_frame_paths(input_parent_dir, STRAIN_COMPONENTS)
    frame_keys = list(frame_paths)

    reference_fields = load_reference_fields(
        input_parent_dir,
        REFERENCE_COORDINATES,
        delimiter=CSV_DELIMITER,
        flip_ud=flip_ud,
    )
    x = np.asarray(reference_fields["x"], dtype=np.float64)
    y = np.asarray(reference_fields["y"], dtype=np.float64)
    specimen_mask = np.isfinite(x) & np.isfinite(y)
    area = load_area_field(
        input_parent_dir,
        AREA,
        x=x,
        y=y,
        specimen_mask=specimen_mask,
        delimiter=CSV_DELIMITER,
        flip_ud=flip_ud,
    )
    strain = load_strain_stack(frame_paths, delimiter=CSV_DELIMITER, flip_ud=flip_ud)
    time, force = load_force_time(frame_keys, mode=LOAD_HISTORY_MODE)

    if zero_time_at_first_selected and time.size > 0:
        time = time - time[0]

    output_dir = create_run_output_dir(output_root_dir)
    npz_path = _save_npz(
        output_dir / f"{output_basename}.npz",
        x=x,
        y=y,
        specimen_mask=specimen_mask,
        area=np.asarray(area, dtype=np.float64),
        strain=np.asarray(strain, dtype=np.float64),
        force=np.asarray(force, dtype=np.float64),
        time=np.asarray(time, dtype=np.float64),
        thickness=float(thickness),
        boundary_conditions=_boundary_conditions_to_array(BOUNDARY_CONDITIONS),
        source_path=str(input_parent_dir),
    )
    write_run_metadata(
        output_dir,
        frame_keys=frame_keys,
        npz_path=npz_path,
        source_path=input_parent_dir,
        thickness=float(thickness),
        flip_ud=flip_ud,
    )
    return npz_path


def main() -> int:
    npz_path = assemble_test_data(
        INPUT_PARENT_DIR,
        OUTPUT_ROOT_DIR,
        output_basename=OUTPUT_BASENAME,
        thickness=SPECIMEN_THICKNESS,
        flip_ud=FLIP_UD,
        zero_time_at_first_selected=ZERO_TIME_AT_FIRST_SELECTED,
    )
    print(f"Saved assembled test data to: {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import (
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.hardening import HardeningLinear
from pyvale.vfm.inputdata import process_input_data
from pyvale.vfm.inputdataconfig import AnsysConfig
from pyvale.vfm.slicewise_utils import (
    SliceConfig,
    resolve_cell_aligned_slice_boundaries,
)


DEFAULT_VFM_TEST_DATA_ROOT = Path(
    os.environ.get(
        "PYVALE_VFM_TEST_DATA_DIR",
        "/home/robh/1_Projects/pyvale-vfm-test-data",
    )
)
RECTANGLE_SLICEWISE_YIELD_CASE = "rectangle-slicewise-yield-strength"
VALIDATION_SLICE_AXIS = "x"
PARAMETER_MAP_NAMES = (
    "elastic_modulus",
    "poissons_ratio",
    "yield_strength",
    "hardening_modulus",
)


@dataclass(slots=True, frozen=True)
class SlicewiseValidationMaterial:
    """Material values used by the deterministic slice-wise yield fixture."""

    elastic_modulus: float = 210_000.0
    poissons_ratio: float = 0.3
    yield_strengths: tuple[float, ...] = (180.0, 230.0, 280.0, 330.0)
    hardening_modulus: float = 7_000.0


@dataclass(slots=True, frozen=True)
class SlicewiseValidationRawExport:
    """Paths and metadata for the raw Ansys-style text files."""

    raw_dir: Path
    x_file: Path
    y_file: Path
    strain_xx_file: Path
    strain_yy_file: Path
    strain_xy_file: Path
    force_file: Path
    time_file: Path
    element_ids_file: Path
    thickness: float
    target_stress: npt.NDArray[np.float64]
    material: SlicewiseValidationMaterial


@dataclass(slots=True, frozen=True)
class SlicewiseValidationProcessedCase:
    """Stable paths for the raw and prepared slice-wise validation data."""

    experiment_data_file: Path
    raw_export: SlicewiseValidationRawExport
    known_parameter_maps_file: Path
    case_root: Path
    prepared_dir: Path


def default_slicewise_edge_conditions() -> EdgeConditions:
    """Return edge conditions for a rectangular x-direction tensile test."""

    return EdgeConditions(
        min_x_edge=Edge(EEdgeCondition.Fixed, EEdgeCondition.Free),
        max_x_edge=Edge(EEdgeCondition.Traction, EEdgeCondition.Free),
        min_y_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Free),
        max_y_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Free),
    )


def ensure_rectangle_slicewise_yield_case(
    data_root: str | Path = DEFAULT_VFM_TEST_DATA_ROOT,
    *,
    overwrite: bool = False,
    material: SlicewiseValidationMaterial = SlicewiseValidationMaterial(),
) -> SlicewiseValidationProcessedCase:
    """Return the canonical external fixture, generating it when required.

    The fixture lives outside the source tree so the raw and prepared data can
    grow into a reusable VFM validation dataset. The current raw data are a
    deterministic Python stand-in for the intended Ansys export while the
    licence server is unavailable.
    """

    case_root = Path(data_root) / RECTANGLE_SLICEWISE_YIELD_CASE
    prepared_experiment = case_root / "prepared" / "experiment_data.yaml"
    known_maps = case_root / "prepared" / "known_parameter_maps.npz"
    if (
        not overwrite
        and prepared_experiment.exists()
        and _known_maps_are_current(known_maps)
    ):
        return SlicewiseValidationProcessedCase(
            experiment_data_file=prepared_experiment,
            raw_export=_raw_export_from_existing_files(case_root / "raw", material),
            known_parameter_maps_file=known_maps,
            case_root=case_root,
            prepared_dir=case_root / "prepared",
        )

    return process_slicewise_ansys_validation_case(
        case_root,
        raw_dir_name="raw",
        processed_dir_name="prepared",
        material=material,
        overwrite=True,
    )


def write_slicewise_ansys_raw_export(
    raw_dir: str | Path,
    *,
    num_rows: int = 5,
    num_cols: int = 16,
    length: float = 16.0,
    height: float = 4.0,
    thickness: float = 0.8,
    target_stress: npt.ArrayLike | None = None,
    material: SlicewiseValidationMaterial = SlicewiseValidationMaterial(),
) -> SlicewiseValidationRawExport:
    """Write a compact deterministic Ansys-style point-cloud export.

    The synthetic specimen is a rectangle split into contiguous groups of
    datapoint cells along the x-axis. Each group has a different yield
    strength. For each load step, the helper solves the uniaxial strain needed
    for that group's material law to reconstruct the same target longitudinal
    stress. With the matching cell-aligned slice support, the reconstructed
    slice force is therefore exactly the applied force up to numerical
    precision.
    """

    if num_cols % len(material.yield_strengths) != 0:
        raise ValueError("num_cols must be divisible by the number of material slices.")
    if num_rows < 2 or num_cols < 2:
        raise ValueError("num_rows and num_cols must both be at least 2.")

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_stress_array = (
        np.asarray(target_stress, dtype=np.float64)
        if target_stress is not None
        else np.linspace(40.0, 420.0, 9, dtype=np.float64)
    )
    if target_stress_array.ndim != 1 or target_stress_array.size < 2:
        raise ValueError("target_stress must be a 1D array with at least two entries.")
    if np.any(np.diff(target_stress_array) <= 0.0):
        raise ValueError("target_stress must be strictly increasing.")

    x_axis = np.linspace(0.0, length, num_cols, dtype=np.float64)
    y_axis = np.linspace(0.0, height, num_rows, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    # Components follow the pyvale convention [xx, yy, xy]. The transverse
    # and shear strains are zero by design so that the target stress is driven
    # only by the solved longitudinal strain history.
    strain = np.zeros((target_stress_array.size, 3, num_rows, num_cols), dtype=np.float64)
    columns_per_slice = num_cols // len(material.yield_strengths)
    for slice_index, yield_strength in enumerate(material.yield_strengths):
        epsilon_xx = _solve_uniaxial_strain_history(
            target_stress_array,
            yield_strength=yield_strength,
            material=material,
        )
        col_slice = slice(
            slice_index * columns_per_slice,
            (slice_index + 1) * columns_per_slice,
        )
        strain[:, 0, :, col_slice] = epsilon_xx[:, np.newaxis, np.newaxis]

    # Force is consistent with a uniform x-stress over the rectangular
    # cross-section. Units are MPa * mm^2 = N.
    force_x = target_stress_array * thickness * height
    force = np.column_stack((force_x, np.zeros_like(force_x)))
    timesteps = np.linspace(0.1, 1.0, target_stress_array.size, dtype=np.float64)

    point_count = num_rows * num_cols
    x_file = raw_dir / "x.txt"
    y_file = raw_dir / "y.txt"
    strain_xx_file = raw_dir / "strain_xx.txt"
    strain_yy_file = raw_dir / "strain_yy.txt"
    strain_xy_file = raw_dir / "strain_xy.txt"
    force_file = raw_dir / "force_reaction.csv"
    time_file = raw_dir / "time.txt"
    element_ids_file = raw_dir / "element_ids.txt"

    np.savetxt(x_file, x_grid.ravel())
    np.savetxt(y_file, y_grid.ravel())
    np.savetxt(element_ids_file, np.arange(1, point_count + 1, dtype=np.int64), fmt="%d")
    np.savetxt(time_file, timesteps)
    np.savetxt(strain_xx_file, strain[:, 0].reshape(target_stress_array.size, -1).T)
    np.savetxt(strain_yy_file, strain[:, 1].reshape(target_stress_array.size, -1).T)
    np.savetxt(strain_xy_file, strain[:, 2].reshape(target_stress_array.size, -1).T)
    np.savetxt(
        force_file,
        force,
        delimiter=",",
        header="reaction_fx,reaction_fy",
        comments="",
    )

    raw_export = SlicewiseValidationRawExport(
        raw_dir=raw_dir,
        x_file=x_file,
        y_file=y_file,
        strain_xx_file=strain_xx_file,
        strain_yy_file=strain_yy_file,
        strain_xy_file=strain_xy_file,
        force_file=force_file,
        time_file=time_file,
        element_ids_file=element_ids_file,
        thickness=thickness,
        target_stress=target_stress_array,
        material=material,
    )
    _write_raw_readme(raw_export, length=length, height=height, num_rows=num_rows, num_cols=num_cols)
    return raw_export


def process_slicewise_ansys_validation_case(
    output_root: str | Path,
    *,
    raw_dir_name: str = "raw",
    processed_dir_name: str = "prepared",
    material: SlicewiseValidationMaterial = SlicewiseValidationMaterial(),
    overwrite: bool = True,
) -> SlicewiseValidationProcessedCase:
    """Generate raw files, run `process_input_data`, and save known maps.

    `process_input_data` writes timestamped run folders. This helper preserves
    that code path, then copies the resulting standard files into a stable
    `prepared/` folder for tests and diagnostic scripts.
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_dir = output_root / raw_dir_name
    prepared_dir = output_root / processed_dir_name
    staging_dir = output_root / "_process_input_data_staging"

    if overwrite:
        for path in (prepared_dir, staging_dir):
            if path.exists():
                shutil.rmtree(path)
    elif (
        (prepared_dir / "experiment_data.yaml").exists()
        and _known_maps_are_current(prepared_dir / "known_parameter_maps.npz")
    ):
        return SlicewiseValidationProcessedCase(
            experiment_data_file=prepared_dir / "experiment_data.yaml",
            raw_export=_raw_export_from_existing_files(raw_dir, material),
            known_parameter_maps_file=prepared_dir / "known_parameter_maps.npz",
            case_root=output_root,
            prepared_dir=prepared_dir,
        )

    raw_export = write_slicewise_ansys_raw_export(
        raw_dir,
        material=material,
    )

    experiment_data_file = process_input_data(
        AnsysConfig(
            x_file=raw_export.x_file,
            y_file=raw_export.y_file,
            strain_xx_file=raw_export.strain_xx_file,
            strain_yy_file=raw_export.strain_yy_file,
            strain_xy_file=raw_export.strain_xy_file,
            force_file=raw_export.force_file,
            time_file=raw_export.time_file,
            thickness=raw_export.thickness,
            edge_conditions=default_slicewise_edge_conditions(),
            element_ids_file=raw_export.element_ids_file,
            target_spacing=16.0 / 15.0,
        ),
        staging_dir,
    )

    shutil.copytree(experiment_data_file.parent, prepared_dir)
    shutil.rmtree(staging_dir)

    experiment_data = ExperimentData.load_from_file(prepared_dir / "experiment_data.yaml")
    slice_boundaries = build_validation_slice_boundaries(
        experiment_data.specimen_geometry,
        material=material,
    )
    known_parameter_maps = build_known_parameter_maps(
        experiment_data.specimen_geometry.x,
        material=material,
        boundaries=slice_boundaries,
    )
    known_parameter_maps_file = prepared_dir / "known_parameter_maps.npz"
    np.savez(
        known_parameter_maps_file,
        **known_parameter_maps,
        slice_boundaries=slice_boundaries,
        slice_axis=np.asarray(VALIDATION_SLICE_AXIS),
    )
    _write_prepared_readme(
        prepared_dir,
        raw_export,
        known_parameter_maps_file,
        slice_boundaries=slice_boundaries,
    )

    return SlicewiseValidationProcessedCase(
        experiment_data_file=prepared_dir / "experiment_data.yaml",
        raw_export=raw_export,
        known_parameter_maps_file=known_parameter_maps_file,
        case_root=output_root,
        prepared_dir=prepared_dir,
    )


def build_known_parameter_maps(
    x: npt.NDArray[np.float64],
    *,
    material: SlicewiseValidationMaterial = SlicewiseValidationMaterial(),
    boundaries: npt.NDArray[np.float64] | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Build material maps matching the validation slice distribution."""

    shape = x.shape
    yield_strength = np.empty(shape, dtype=np.float64)
    resolved_boundaries = (
        np.asarray(boundaries, dtype=np.float64)
        if boundaries is not None
        else np.linspace(
            float(np.nanmin(x)),
            float(np.nanmax(x)),
            len(material.yield_strengths) + 1,
            dtype=np.float64,
        )
    )
    if resolved_boundaries.size != len(material.yield_strengths) + 1:
        raise ValueError(
            "Known-map boundaries must contain one more value than the "
            "number of validation yield strengths."
        )
    slice_indices = np.searchsorted(resolved_boundaries, x, side="right") - 1
    slice_indices = np.clip(slice_indices, 0, len(material.yield_strengths) - 1)
    for slice_index, value in enumerate(material.yield_strengths):
        yield_strength[slice_indices == slice_index] = value

    return {
        "elastic_modulus": np.full(shape, material.elastic_modulus, dtype=np.float64),
        "poissons_ratio": np.full(shape, material.poissons_ratio, dtype=np.float64),
        "yield_strength": yield_strength,
        "hardening_modulus": np.full(shape, material.hardening_modulus, dtype=np.float64),
    }


def build_validation_slice_boundaries(
    specimen_geometry: SpecimenGeometry,
    *,
    material: SlicewiseValidationMaterial = SlicewiseValidationMaterial(),
) -> npt.NDArray[np.float64]:
    """Return the cell-aligned material-slice boundaries for this fixture."""

    return resolve_cell_aligned_slice_boundaries(
        specimen_geometry,
        SliceConfig(
            axis=VALIDATION_SLICE_AXIS,
            num_slices=len(material.yield_strengths),
        ),
    )


def make_identification_parameters(
    map_size: npt.NDArray[np.uint32],
    *,
    material: SlicewiseValidationMaterial = SlicewiseValidationMaterial(),
    initial_yield_strength: float = 250.0,
) -> dict[str, ConstitutiveParameter]:
    """Build bounded constitutive parameters for validation identifications."""

    return {
        "elastic_modulus": ConstitutiveParameter(
            material.elastic_modulus,
            100_000.0,
            300_000.0,
            map_size,
        ),
        "poissons_ratio": ConstitutiveParameter(
            material.poissons_ratio,
            0.2,
            0.4,
            map_size,
        ),
        "yield_strength": ConstitutiveParameter(
            initial_yield_strength,
            100.0,
            500.0,
            map_size,
        ),
        "hardening_modulus": ConstitutiveParameter(
            material.hardening_modulus,
            1_000.0,
            20_000.0,
            map_size,
        ),
    }


def _solve_uniaxial_strain_history(
    target_stress: npt.NDArray[np.float64],
    *,
    yield_strength: float,
    material: SlicewiseValidationMaterial,
) -> npt.NDArray[np.float64]:
    """Invert the local radial-return law by monotone bisection."""

    strain_history: list[float] = []

    for target in target_stress:
        lower = strain_history[-1] if strain_history else 0.0
        upper = max(lower + 1.0e-6, 1.0e-3)

        while (
            _calculate_uniaxial_stress(
                [*strain_history, upper],
                yield_strength=yield_strength,
                material=material,
            )
            < target
        ):
            upper *= 2.0

        for _ in range(60):
            midpoint = 0.5 * (lower + upper)
            if (
                _calculate_uniaxial_stress(
                    [*strain_history, midpoint],
                    yield_strength=yield_strength,
                    material=material,
                )
                < target
            ):
                lower = midpoint
            else:
                upper = midpoint

        strain_history.append(0.5 * (lower + upper))

    return np.asarray(strain_history, dtype=np.float64)


def _calculate_uniaxial_stress(
    strain_xx_history: list[float],
    *,
    yield_strength: float,
    material: SlicewiseValidationMaterial,
) -> float:
    constitutive_law = IsotropicVonMisesElastoplasticity(HardeningLinear())
    parameter_maps = {
        "elastic_modulus": np.full((1, 1), material.elastic_modulus, dtype=np.float64),
        "poissons_ratio": np.full((1, 1), material.poissons_ratio, dtype=np.float64),
        "yield_strength": np.full((1, 1), yield_strength, dtype=np.float64),
        "hardening_modulus": np.full((1, 1), material.hardening_modulus, dtype=np.float64),
    }
    strain = np.zeros((len(strain_xx_history), 3, 1, 1), dtype=np.float64)
    strain[:, 0, 0, 0] = np.asarray(strain_xx_history, dtype=np.float64)
    stress = constitutive_law.calculate_stress(strain, parameter_maps)
    return float(stress[-1, 0, 0, 0])


def _raw_export_from_existing_files(
    raw_dir: Path,
    material: SlicewiseValidationMaterial,
) -> SlicewiseValidationRawExport:
    metadata_file = raw_dir / "generation_metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
    target_stress = np.asarray(metadata.get("target_stress_mpa", np.linspace(40.0, 420.0, 9)), dtype=np.float64)
    thickness = float(metadata.get("thickness_mm", 0.8))
    return SlicewiseValidationRawExport(
        raw_dir=raw_dir,
        x_file=raw_dir / "x.txt",
        y_file=raw_dir / "y.txt",
        strain_xx_file=raw_dir / "strain_xx.txt",
        strain_yy_file=raw_dir / "strain_yy.txt",
        strain_xy_file=raw_dir / "strain_xy.txt",
        force_file=raw_dir / "force_reaction.csv",
        time_file=raw_dir / "time.txt",
        element_ids_file=raw_dir / "element_ids.txt",
        thickness=thickness,
        target_stress=target_stress,
        material=material,
    )


def _known_maps_are_current(
    known_maps_file: Path,
) -> bool:
    if not known_maps_file.exists():
        return False
    try:
        with np.load(known_maps_file) as loaded:
            return all(name in loaded.files for name in PARAMETER_MAP_NAMES) and (
                "slice_boundaries" in loaded.files
            )
    except Exception:
        return False


def _write_raw_readme(
    raw_export: SlicewiseValidationRawExport,
    *,
    length: float,
    height: float,
    num_rows: int,
    num_cols: int,
) -> None:
    metadata = {
        "generator": "tests/vfm/slicewise_validation_case.py",
        "generation_mode": "deterministic-python-ansys-stand-in",
        "intended_future_source": "Ansys plane-stress/3D FE export when the licence server is available",
        "length_mm": length,
        "height_mm": height,
        "thickness_mm": raw_export.thickness,
        "grid_rows": num_rows,
        "grid_columns": num_cols,
        "target_stress_mpa": raw_export.target_stress.tolist(),
        "material": _material_to_dict(raw_export.material),
    }
    (raw_export.raw_dir / "generation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (raw_export.raw_dir / "README.md").write_text(
        "\n".join(
            [
                "# Rectangle Slicewise Yield Strength: Raw Data",
                "",
                "This folder contains deterministic Ansys-style text files for a rectangular tensile specimen.",
                "The data are a temporary Python-generated stand-in for an Ansys export while the licence server is unavailable.",
                "",
                "The rectangle is split into four contiguous groups of datapoint cells along x. The groups have yield strengths of "
                + ", ".join(f"{value:g} MPa" for value in raw_export.material.yield_strengths)
                + ".",
                "",
                "The prepared fixture resolves the intended equal geometric slices and then snaps the internal boundaries to support-cell edges. This keeps each datapoint cell as the smallest material representative volume.",
                "",
                "For each load step, the generator solves the uniaxial strain history required for each slice's material law to reconstruct the same target longitudinal stress. The applied force is Fx = sigma_xx * height * thickness.",
                "",
                "Files are laid out in the raw Ansys loader convention used by `AnsysConfig`: `x.txt`, `y.txt`, `strain_xx.txt`, `strain_yy.txt`, `strain_xy.txt`, `time.txt`, `element_ids.txt`, and `force_reaction.csv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_prepared_readme(
    prepared_dir: Path,
    raw_export: SlicewiseValidationRawExport,
    known_parameter_maps_file: Path,
    slice_boundaries: npt.NDArray[np.float64],
) -> None:
    (prepared_dir / "README.md").write_text(
        "\n".join(
            [
                "# Rectangle Slicewise Yield Strength: Prepared Data",
                "",
                "This folder is the stable pyvale VFM input-data fixture generated from the sibling `raw/` folder with `process_input_data` and `AnsysConfig`.",
                "",
                "`experiment_data.yaml` references the prepared NumPy arrays in this folder. `known_parameter_maps.npz` contains the exact elastic modulus, Poisson ratio, yield strength, and hardening modulus maps used to generate the synthetic strain history, plus the x-direction cell-aligned slice boundaries.",
                "",
                "Cell-aligned x-boundaries [mm]: "
                + ", ".join(f"{value:.12g}" for value in slice_boundaries)
                + ".",
                "",
                "Intended use:",
                "",
                "- Validate that the slice-wise force reconstruction error is zero when stress is reconstructed from the known maps.",
                "- Validate that slice-wise identification recovers the four yield-strength slices when the correct support is used.",
                "- Validate point/column-wise slice supports and merge/split refinement behaviour on noiseless data.",
                "",
                "The validation support uses cell-aligned slice boundaries because datapoint cells are treated as the smallest material representative volumes. With this convention, noiseless slice-wise identification and force reconstruction should be exact up to numerical tolerance.",
                "",
                f"Known maps file: `{known_parameter_maps_file.name}`.",
                "Raw source folder: `../raw`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _material_to_dict(material: SlicewiseValidationMaterial) -> dict[str, object]:
    return {
        "elastic_modulus_mpa": material.elastic_modulus,
        "poissons_ratio": material.poissons_ratio,
        "yield_strengths_mpa": list(material.yield_strengths),
        "hardening_modulus_mpa": material.hardening_modulus,
    }

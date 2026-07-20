from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from scipy.interpolate import LinearNDInterpolator

from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import (
    BoundaryConditions,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.hardening import LinearHardening
from pyvale.vfm.identification import Identification, IdentificationPhase
from pyvale.vfm.metricsbvf import SensitivityBasedVirtualFieldsMetric
from pyvale.vfm.objectivefuncvector import VectorFirstResultPassthrough
from pyvale.vfm.optimiserleastsquares import LeastSquares
from pyvale.vfm.spatialparamhomogeneous import HomogeneousSpatialParameterisation
from pyvale.vfm.vfm import run_identification

PYVALE_ROOT = Path(__file__).resolve().parents[2]
ANSYS_DATA_ROOT = PYVALE_ROOT / "dev" / "vfm" / "rob-data"

PLATE_WITH_HOLE_ROOT = ANSYS_DATA_ROOT / "plate-with-hole-hom-lin-hard"
PLATE_WITH_HOLE_FE_ROOT = PLATE_WITH_HOLE_ROOT / "fe-data"
SINGLE_ELEMENT_ROOT = ANSYS_DATA_ROOT / "single-element-plane-stress"
SINGLE_ELEMENT_FE_ROOT = ANSYS_DATA_ROOT / "single-element-plane-stress" / "fe-data"
RECTANGLE_ROOT = ANSYS_DATA_ROOT / "rectangle-hom-lin-hard"


PLATE_THICKNESS = 1e-3  # m

KNOWN_PARAMETERS = {
    "elastic_modulus": 200_000.0,  # MPa
    "poissons_ratio": 0.3,
    "yield_strength": 200.0,       # MPa
    "hardening_modulus": 1_000.0,  # MPa
}

# Keep plotting opt-in for routine test runs; turn these on when debugging.
PLOT_STRESS_RECON_ABS_DIFF = True
PLOT_STRESS_RECON_ABS_PERC_DIFF = True
PLOT_METRIC_IDENTIFIED_DIFF = True
PLOT_IDENTIFICATION_DIFF = True

# Identification is much heavier than stress reconstruction, so this single
# flag controls both pytest-style runs and direct "Debug File" execution.
RUN_IDENTIFICATION = True

# Defaults used when you press "Debug File" in VS Code and do not pass any CLI
# arguments.
#
# `grid`:
# Uses the prepared regular grid and supports identification. This expects a
# dataset root (e.g. `PLATE_WITH_HOLE_ROOT`) that contains both `fe-data/` or
# `fe_data/` and
# a `vfm-input-data-*` folder.
#
# `raw-centroids`:
# Uses FE centroids directly for the most accurate stress reconstruction and
# automatically skips identification. This can use either a dataset root or a
# `fe-data/` / `fe_data/` folder.
DEBUG_DATA_SOURCE = "raw-centroids" #"grid" #"raw-centroids" #"grid"

# Manual dataset selection for "Debug File". Uncomment one line at a time.
# MANUAL_RUN_DATASET = PLATE_WITH_HOLE_ROOT
# MANUAL_RUN_DATASET = PLATE_WITH_HOLE_FE_ROOT
# MANUAL_RUN_DATASET = SINGLE_ELEMENT_FE_ROOT
MANUAL_RUN_DATASET = RECTANGLE_ROOT

STRESS_COMPONENT_LABELS = ("xx", "yy", "xy")


@dataclass(frozen=True, slots=True)
class AnsysGridDataset:
    x_grid_mm: npt.NDArray[np.float64]
    y_grid_mm: npt.NDArray[np.float64]
    specimen_mask: npt.NDArray[np.bool_]
    pixel_area_mm2: npt.NDArray[np.float64]
    strain: npt.NDArray[np.float64]
    stress_fe: npt.NDArray[np.float64]
    force: npt.NDArray[np.float64]
    time: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AnsysPointDataset:
    strain: npt.NDArray[np.float64]
    stress_fe: npt.NDArray[np.float64]
    point_x_mm: npt.NDArray[np.float64]
    point_y_mm: npt.NDArray[np.float64]
    time: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AnsysBoundarySetup:
    edge_conditions: EdgeConditions
    force_component: str
    raw_force_sign: float


def _rms(array: npt.NDArray[np.float64]) -> float:
    return float(np.sqrt(np.nanmean(np.square(array))))


def _root_mean_square_percentage_error(
    predicted: npt.NDArray[np.float64],
    known: npt.NDArray[np.float64],
) -> float:
    percentage_error = (predicted - known) / known * 100.0
    return float(np.sqrt(np.nanmean(np.square(percentage_error))))


def _plot_stress_abs_diff(
    x_grid: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    abs_diff: npt.NDArray[np.float64],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, label, component in zip(
        axes, STRESS_COMPONENT_LABELS, range(3), strict=True
    ):
        field = abs_diff[component, :, :]
        image = ax.pcolormesh(x_grid, y_grid, field)
        fig.colorbar(image, ax=ax, label="|calc - FE| [MPa]")
        ax.set_title(f"stress_{label} abs diff")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.invert_yaxis()
    plt.show()


def _plot_stress_abs_perc_diff(
    x_grid: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    abs_perc_diff: npt.NDArray[np.float64],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, label, component in zip(
        axes, STRESS_COMPONENT_LABELS, range(3), strict=True
    ):
        field = abs_perc_diff[component, :, :]
        image = ax.pcolormesh(x_grid, y_grid, field)
        fig.colorbar(image, ax=ax, label="|calc - FE| / |FE| [%]")
        image.set_clim(np.nanpercentile(field, 5), np.nanpercentile(field, 95))
        ax.set_title(f"stress_{label} abs % diff")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.invert_yaxis()
    plt.show()


def _plot_centroid_stress_abs_diff(
    point_x_m: npt.NDArray[np.float64],
    point_y_m: npt.NDArray[np.float64],
    abs_diff: npt.NDArray[np.float64],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, label, component in zip(
        axes, STRESS_COMPONENT_LABELS, range(3), strict=True
    ):
        image = ax.scatter(
            point_x_m,
            point_y_m,
            c=abs_diff[component, :],
            s=8,
            cmap="viridis",
        )
        fig.colorbar(image, ax=ax, label="|calc - FE| [MPa]")
        ax.set_title(f"stress_{label} abs diff")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.invert_yaxis()
    plt.show()


def _plot_centroid_stress_abs_perc_diff(
    point_x_m: npt.NDArray[np.float64],
    point_y_m: npt.NDArray[np.float64],
    abs_perc_diff: npt.NDArray[np.float64],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, label, component in zip(
        axes, STRESS_COMPONENT_LABELS, range(3), strict=True
    ):
        field = abs_perc_diff[component, :]
        image = ax.scatter(
            point_x_m,
            point_y_m,
            c=field,
            s=8,
            cmap="viridis",
        )
        fig.colorbar(image, ax=ax, label="|calc - FE| / |FE| [%]")
        finite = np.isfinite(field)
        if np.any(finite):
            image.set_clim(
                np.nanpercentile(field[finite], 5),
                np.nanpercentile(field[finite], 95),
            )
        ax.set_title(f"stress_{label} abs % diff")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.invert_yaxis()
    plt.show()


def _plot_metric_virtual_work(
    internal_virtual_work_a: npt.NDArray[np.float64],
    external_virtual_work_a: npt.NDArray[np.float64],
    internal_virtual_work_b: npt.NDArray[np.float64],
    external_virtual_work_b: npt.NDArray[np.float64],
    label_a: str,
    label_b: str,
    sbvf_labels: tuple[str, ...],
) -> None:
    num_virtual_fields = internal_virtual_work_a.shape[0]

    fig_work, axes = plt.subplots(
        num_virtual_fields,
        4,
        figsize=(18, 3.5 * num_virtual_fields),
        constrained_layout=True,
        squeeze=False,
    )

    fig_residual, residual_axes = plt.subplots(
        num_virtual_fields,
        3,
        figsize=(13.5, 3.5 * num_virtual_fields),
        constrained_layout=True,
        squeeze=False,
    )

    for vf in range(num_virtual_fields):
        ivw_a = internal_virtual_work_a[vf]
        ivw_b = internal_virtual_work_b[vf]
        evw_a = external_virtual_work_a[vf]
        evw_b = external_virtual_work_b[vf]

        ivw_abs_diff = np.abs(ivw_b - ivw_a)
        evw_abs_diff = np.abs(evw_b - evw_a)
        ivw_percentage_diff = np.divide(
            ivw_abs_diff * 100.0,
            np.abs(ivw_a),
            out=np.full_like(ivw_abs_diff, np.nan),
            where=ivw_a != 0.0,
        )
        evw_percentage_diff = np.divide(
            evw_abs_diff * 100.0,
            np.abs(evw_a),
            out=np.full_like(evw_abs_diff, np.nan),
            where=evw_a != 0.0,
        )

        ivw_evw_diff_a = np.abs(ivw_a - evw_a)
        ivw_evw_diff_b = np.abs(ivw_b - evw_b)
        ivw_evw_percentage_diff_a = np.divide(
            ivw_evw_diff_a * 100.0,
            np.abs(evw_a),
            out=np.full_like(evw_a, np.nan),
            where=evw_a != 0.0,
        )
        ivw_evw_percentage_diff_b = np.divide(
            ivw_evw_diff_b * 100.0,
            np.abs(evw_b),
            out=np.full_like(evw_b, np.nan),
            where=evw_b != 0.0,
        )

        sbvf_label = sbvf_labels[vf]

        axes[vf, 0].plot(ivw_a, marker=".", label=label_a)
        axes[vf, 0].plot(ivw_b, marker=".", label=label_b)
        axes[vf, 0].set_title(f"{sbvf_label} IVW")
        axes[vf, 0].set_ylabel("internal virtual work")
        axes[vf, 0].legend()

        axes[vf, 1].plot(evw_a, marker=".", label=label_a)
        axes[vf, 1].plot(evw_b, marker=".", label=label_b)
        axes[vf, 1].set_title(f"{sbvf_label} EVW")
        axes[vf, 1].set_ylabel("external virtual work")
        axes[vf, 1].legend()

        axes[vf, 2].plot(ivw_abs_diff, marker=".", label="IVW")
        axes[vf, 2].plot(evw_abs_diff, marker=".", label="EVW")
        axes[vf, 2].set_title(f"{sbvf_label} abs diff")
        axes[vf, 2].set_ylabel(f"|{label_b} - {label_a}|")
        axes[vf, 2].legend()

        axes[vf, 3].plot(ivw_percentage_diff, marker=".", label="IVW")
        axes[vf, 3].plot(evw_percentage_diff, marker=".", label="EVW")
        axes[vf, 3].set_title(f"{sbvf_label} percentage diff")
        axes[vf, 3].set_ylabel("% diff")
        axes[vf, 3].legend()

        residual_axes[vf, 0].plot(
            ivw_a, marker=".", color="blue", linestyle="-", label=f"IVW {label_a}"
        )
        residual_axes[vf, 0].plot(
            evw_a, marker=".", color="blue", linestyle="--", label=f"EVW {label_a}"
        )
        residual_axes[vf, 0].plot(
            ivw_b, marker=".", color="orange", linestyle="-", label=f"IVW {label_b}"
        )
        residual_axes[vf, 0].plot(
            evw_b, marker=".", color="orange", linestyle="--", label=f"EVW {label_b}"
        )
        residual_axes[vf, 0].set_title(f"{sbvf_label} IVW & EVW")
        residual_axes[vf, 0].set_ylabel("virtual work")
        residual_axes[vf, 0].legend()

        residual_axes[vf, 1].plot(ivw_evw_diff_a, marker=".", label=label_a)
        residual_axes[vf, 1].plot(ivw_evw_diff_b, marker=".", label=label_b)
        residual_axes[vf, 1].set_title(f"{sbvf_label} |IVW - EVW|")
        residual_axes[vf, 1].set_ylabel("|IVW - EVW|")
        residual_axes[vf, 1].legend()

        residual_axes[vf, 2].plot(ivw_evw_percentage_diff_a, marker=".", label=label_a)
        residual_axes[vf, 2].plot(ivw_evw_percentage_diff_b, marker=".", label=label_b)
        residual_axes[vf, 2].set_title(f"{sbvf_label} |IVW - EVW| percentage diff")
        residual_axes[vf, 2].set_ylabel("% diff (IVW vs EVW)")
        residual_axes[vf, 2].legend()

        for column in range(4):
            axes[vf, column].set_xlabel("timestep")
        for column in range(3):
            residual_axes[vf, column].set_xlabel("timestep")

    plt.show()


def _plot_identification_diff(
    x_grid: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    identified_maps: dict[str, npt.NDArray[np.float64]],
    known_maps: dict[str, npt.NDArray[np.float64]],
) -> None:
    fig, axes = plt.subplots(1, len(known_maps), figsize=(16, 4), constrained_layout=True)
    for ax, param_name in zip(axes, known_maps, strict=True):
        field = identified_maps[param_name] - known_maps[param_name]
        image = ax.pcolormesh(x_grid, y_grid, field)
        fig.colorbar(image, ax=ax, label="identified - known")
        ax.set_title(param_name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.invert_yaxis()
    plt.show()


def _load_material_properties(path: Path) -> dict[str, float]:
    with path.open(newline="") as file_handle:
        row = next(csv.DictReader(file_handle))

    return {
        "elastic_modulus": float(row["youngs_modulus"]),
        "poissons_ratio": float(row["poisson_ratio"]),
        "yield_strength": float(row["yield_strength"]),
        "hardening_modulus": float(row["hardening_modulus"]),
    }


def _load_point_history(path: Path, num_points: int) -> npt.NDArray[np.float64]:
    values = np.asarray(np.loadtxt(path), dtype=np.float64)
    if values.ndim == 0:
        return values.reshape(num_points, 1)
    if values.ndim == 1:
        if num_points == 1:
            return values.reshape(1, -1)
        if values.shape[0] == num_points:
            return values.reshape(num_points, 1)
        raise ValueError(
            f"Cannot infer point-history shape for {path} with {num_points} points."
        )
    return values


def _find_latest_prepared_grid(dataset_root: Path) -> Path | None:
    prepared_dirs = sorted(path for path in dataset_root.glob("vfm-input-data-*") if path.is_dir())
    if prepared_dirs:
        return prepared_dirs[-1]
    return None


def _find_raw_data_root(dataset_path: Path) -> Path:
    if dataset_path.name in {"fe-data", "fe_data"}:
        return dataset_path

    for candidate_name in ("fe-data", "fe_data"):
        candidate = dataset_path / candidate_name
        if candidate.is_dir():
            return candidate

    return dataset_path / "fe-data"


def _normalise_grid_dataset_root(dataset_path: Path) -> Path:
    if dataset_path.name in {"fe-data", "fe_data"}:
        return dataset_path.parent
    return dataset_path


def _normalise_raw_dataset_root(dataset_path: Path) -> Path:
    return _find_raw_data_root(dataset_path)


def _get_boundary_setup(dataset_root: Path) -> AnsysBoundarySetup:
    dataset_name = dataset_root.name.lower()

    if "rectangle" in dataset_name:
        return AnsysBoundarySetup(
            edge_conditions=EdgeConditions(
                min_x_edge=Edge(x=EEdgeCondition.Fixed, y=EEdgeCondition.Free),
                max_x_edge=Edge(x=EEdgeCondition.Traction, y=EEdgeCondition.Free),
                min_y_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Free),
                max_y_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Free),
            ),
            force_component="x",
            raw_force_sign=-1.0,
        )

    return AnsysBoundarySetup(
        edge_conditions=EdgeConditions(
            min_x_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Free),
            max_x_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Free),
            min_y_edge=Edge(x=EEdgeCondition.Fixed, y=EEdgeCondition.Fixed),
            max_y_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Traction),
        ),
        force_component="y",
        raw_force_sign=-1.0,
    )


def _load_force_time_from_raw(
    raw_root: Path,
    boundary_setup: AnsysBoundarySetup,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    reaction_path = raw_root / "reaction_history.csv"
    with reaction_path.open(newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))

    component_suffix = boundary_setup.force_component
    column_name = f"reaction_f{component_suffix}"
    force = boundary_setup.raw_force_sign * np.asarray(
        [float(row[column_name]) for row in rows],
        dtype=np.float64,
    )

    time_path = raw_root / "time_values.txt"
    if time_path.exists():
        time = np.atleast_1d(np.loadtxt(time_path)).astype(np.float64)
    else:
        time = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)

    return force, time


def _reshape_regular_grid_field(
    point_x_mm: npt.NDArray[np.float64],
    point_y_mm: npt.NDArray[np.float64],
    values_by_timestep: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    x_values_mm, x_inverse = np.unique(point_x_mm, return_inverse=True)
    y_values_mm, y_inverse = np.unique(point_y_mm, return_inverse=True)
    num_x = x_values_mm.size
    num_y = y_values_mm.size
    num_points = point_x_mm.size

    if num_x * num_y != num_points:
        raise ValueError("Raw FE centroids do not form a complete regular grid.")

    linear_indices = y_inverse * num_x + x_inverse
    order = np.argsort(linear_indices)
    if not np.array_equal(linear_indices[order], np.arange(num_points)):
        raise ValueError("Raw FE centroids do not map uniquely onto a regular grid.")

    x_grid_mm, y_grid_mm = np.meshgrid(x_values_mm, y_values_mm)
    field = values_by_timestep[order].reshape(num_y, num_x, values_by_timestep.shape[1])
    field = np.transpose(field, (2, 0, 1))
    return x_grid_mm, y_grid_mm, field


def _build_regular_grid_dataset_from_raw(
    dataset_root: Path,
    raw_root: Path,
) -> AnsysGridDataset:
    point_x_mm = np.atleast_1d(np.loadtxt(raw_root / "x_coordinates.txt")).astype(np.float64)
    point_y_mm = np.atleast_1d(np.loadtxt(raw_root / "y_coordinates.txt")).astype(np.float64)
    boundary_setup = _get_boundary_setup(dataset_root)
    force, time = _load_force_time_from_raw(raw_root, boundary_setup)

    raw_fields = [
        np.asarray(np.loadtxt(raw_root / file_name), dtype=np.float64)
        for file_name in (
            "eps_xx.txt",
            "eps_yy.txt",
            "eps_xy.txt",
            "sig_xx.txt",
            "sig_yy.txt",
            "sig_xy.txt",
        )
    ]

    reshaped_fields = []
    x_grid_mm: npt.NDArray[np.float64] | None = None
    y_grid_mm: npt.NDArray[np.float64] | None = None
    for values in raw_fields:
        x_grid_mm_candidate, y_grid_mm_candidate, field = _reshape_regular_grid_field(
            point_x_mm,
            point_y_mm,
            values,
        )
        if x_grid_mm is None or y_grid_mm is None:
            x_grid_mm = x_grid_mm_candidate
            y_grid_mm = y_grid_mm_candidate
        reshaped_fields.append(field)

    assert x_grid_mm is not None
    assert y_grid_mm is not None

    specimen_mask = np.ones_like(x_grid_mm, dtype=bool)
    x_spacing_mm = float(np.median(np.diff(np.unique(x_grid_mm))))
    y_spacing_mm = float(np.median(np.diff(np.unique(y_grid_mm))))
    pixel_area_mm2 = np.full_like(x_grid_mm, x_spacing_mm * y_spacing_mm, dtype=np.float64)

    strain = np.stack(reshaped_fields[:3], axis=1)
    strain[:, 2, :, :] *= 0.5
    stress_fe = np.stack(reshaped_fields[3:], axis=1)

    print(
        "No prepared ANSYS VFM input folder found; using the raw FE regular grid "
        f"from {raw_root} directly."
    )

    return AnsysGridDataset(
        x_grid_mm=x_grid_mm,
        y_grid_mm=y_grid_mm,
        specimen_mask=specimen_mask,
        pixel_area_mm2=pixel_area_mm2,
        strain=strain,
        stress_fe=stress_fe,
        force=force,
        time=time,
    )


def _interpolate_raw_field_to_grid(
    points_xy_mm: npt.NDArray[np.float64],
    values_by_timestep: npt.NDArray[np.float64],
    x_grid_mm: npt.NDArray[np.float64],
    y_grid_mm: npt.NDArray[np.float64],
    specimen_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    grid_values = np.empty((values_by_timestep.shape[1],) + x_grid_mm.shape, dtype=np.float64)

    for timestep in range(values_by_timestep.shape[1]):
        interpolator = LinearNDInterpolator(points_xy_mm, values_by_timestep[:, timestep])
        field = np.asarray(interpolator(x_grid_mm, y_grid_mm), dtype=np.float64)
        field[~specimen_mask] = np.nan
        grid_values[timestep] = field

    return grid_values


def _load_ansys_data_to_grid(dataset_root: Path) -> AnsysGridDataset:
    dataset_root = _normalise_grid_dataset_root(dataset_root)
    if not dataset_root.is_dir():
        pytest.skip(f"ANSYS dataset not found: {dataset_root}")

    prepared_root = _find_latest_prepared_grid(dataset_root)
    raw_root = _find_raw_data_root(dataset_root)

    if prepared_root is None:
        return _build_regular_grid_dataset_from_raw(dataset_root, raw_root)

    x_grid_mm = np.load(prepared_root / "x.npy")
    y_grid_mm = np.load(prepared_root / "y.npy")
    specimen_mask = np.load(prepared_root / "specimen_mask.npy")
    pixel_area_mm2 = np.load(prepared_root / "pixel_area.npy")
    force = np.load(prepared_root / "force.npy")
    time = np.load(prepared_root / "time.npy")

    point_x_mm = np.atleast_1d(np.loadtxt(raw_root / "x_coordinates.txt")).astype(np.float64)
    point_y_mm = np.atleast_1d(np.loadtxt(raw_root / "y_coordinates.txt")).astype(np.float64)
    points_xy_mm = np.column_stack((point_x_mm, point_y_mm))

    raw_fields = [
        np.asarray(np.loadtxt(raw_root / file_name), dtype=np.float64)
        for file_name in (
            "eps_xx.txt",
            "eps_yy.txt",
            "eps_xy.txt",
            "sig_xx.txt",
            "sig_yy.txt",
            "sig_xy.txt",
        )
    ]

    interpolated_fields = [
        _interpolate_raw_field_to_grid(
            points_xy_mm,
            values,
            x_grid_mm,
            y_grid_mm,
            specimen_mask,
        )
        for values in raw_fields
    ]

    strain = np.stack(interpolated_fields[:3], axis=1)
    # ANSYS exports engineering shear strain gamma_xy; radial_return expects
    # tensorial shear epsilon_xy and doubles it internally.
    strain[:, 2, :, :] *= 0.5
    stress_fe = np.stack(interpolated_fields[3:], axis=1)

    return AnsysGridDataset(
        x_grid_mm=x_grid_mm,
        y_grid_mm=y_grid_mm,
        specimen_mask=specimen_mask,
        pixel_area_mm2=pixel_area_mm2,
        strain=strain,
        stress_fe=stress_fe,
        force=force,
        time=time,
    )


def _build_force_history(
    force: npt.NDArray[np.float64],
    boundary_setup: AnsysBoundarySetup,
) -> npt.NDArray[np.float64]:
    if boundary_setup.force_component == "x":
        return np.column_stack((force, np.zeros_like(force)))
    return np.column_stack((np.zeros_like(force), force))


def _print_virtual_work_residual_summary(
    label: str,
    internal_virtual_work: npt.NDArray[np.float64],
    external_virtual_work: npt.NDArray[np.float64],
) -> None:
    residual = internal_virtual_work - external_virtual_work
    residual_abs = np.abs(residual)
    residual_rms = _rms(residual)
    residual_mean_abs = float(np.nanmean(residual_abs))
    residual_max_abs = float(np.nanmax(residual_abs))
    scale = _rms(external_virtual_work)
    relative_rms = residual_rms / scale if scale != 0.0 else np.nan

    print(
        f"{label} |IVW - EVW| mean = {residual_mean_abs:.6f}, "
        f"max = {residual_max_abs:.6f}, rms = {residual_rms:.6f}, "
        f"relative rms = {relative_rms:.6f}"
    )


def _load_ansys_point_data(raw_root: Path) -> AnsysPointDataset:
    raw_root = _normalise_raw_dataset_root(raw_root)
    point_x_mm = np.atleast_1d(np.loadtxt(raw_root / "x_coordinates.txt")).astype(np.float64)
    point_y_mm = np.atleast_1d(np.loadtxt(raw_root / "y_coordinates.txt")).astype(np.float64)
    num_points = point_x_mm.size
    time = np.atleast_1d(np.loadtxt(raw_root / "time_values.txt")).astype(np.float64)

    strain = np.stack(
        (
            _load_point_history(raw_root / "eps_xx.txt", num_points),
            _load_point_history(raw_root / "eps_yy.txt", num_points),
            0.5 * _load_point_history(raw_root / "eps_xy.txt", num_points),
        ),
        axis=1,
    )
    stress_fe = np.stack(
        (
            _load_point_history(raw_root / "sig_xx.txt", num_points),
            _load_point_history(raw_root / "sig_yy.txt", num_points),
            _load_point_history(raw_root / "sig_xy.txt", num_points),
        ),
        axis=1,
    )

    return AnsysPointDataset(
        strain=np.transpose(strain, (2, 1, 0))[:, :, np.newaxis, :],
        stress_fe=np.transpose(stress_fe, (2, 1, 0))[:, :, np.newaxis, :],
        point_x_mm=point_x_mm,
        point_y_mm=point_y_mm,
        time=time,
    )


def _build_known_parameter_maps(shape: tuple[int, int]) -> dict[str, npt.NDArray[np.float64]]:
    return {
        name: np.full(shape, value, dtype=np.float64)
        for name, value in KNOWN_PARAMETERS.items()
    }


def _build_known_parameter_maps_for_points(num_points: int) -> dict[str, npt.NDArray[np.float64]]:
    return {
        name: np.full((1, num_points), value, dtype=np.float64)
        for name, value in KNOWN_PARAMETERS.items()
    }


def _build_identification_parameters(
    dataset_root: Path,
    parameter_map_size: npt.NDArray[np.uint32],
) -> dict[str, ConstitutiveParameter]:
    if "rectangle" in dataset_root.name.lower():
        return {
            "elastic_modulus": ConstitutiveParameter(
                210_000.0, 100_000.0, 500_000.0, parameter_map_size
            ),
            "poissons_ratio": ConstitutiveParameter(
                0.30, 0.1, 0.49, parameter_map_size
            ),
            "yield_strength": ConstitutiveParameter(
                220.0, 100.0, 400.0, parameter_map_size
            ),
            "hardening_modulus": ConstitutiveParameter(
                1_200.0, 100.0, 5_000.0, parameter_map_size
            ),
        }

    return {
        "elastic_modulus": ConstitutiveParameter(
            450_000, 100_000, 500_000, parameter_map_size
        ),
        "poissons_ratio": ConstitutiveParameter(
            0.45, 0.1, 0.5, parameter_map_size
        ),
        "yield_strength": ConstitutiveParameter(
            800, 100, 1000, parameter_map_size
        ),
        "hardening_modulus": ConstitutiveParameter(
            7000, 500, 10_000, parameter_map_size
        ),
    }


def _print_stress_reconstruction_summary(
    stress_calc: npt.NDArray[np.float64],
    stress_fe: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    stress_abs_diff = np.abs(stress_calc[-1] - stress_fe[-1])
    stress_abs_diff_mean = float(np.nanmean(stress_abs_diff))
    stress_abs_diff_max = float(np.nanmax(stress_abs_diff))
    stress_abs_diff_rms = _rms(stress_abs_diff)

    print(f"stress recon abs diff mean [MPa] = {stress_abs_diff_mean:.6f}")
    print(f"stress recon abs diff max  [MPa] = {stress_abs_diff_max:.6f}")
    print(f"stress recon abs diff rms  [MPa] = {stress_abs_diff_rms:.6f}")

    return stress_abs_diff_mean, stress_abs_diff_max, stress_abs_diff_rms


def run_end_to_end_ansys_grid(
    dataset_path: Path = PLATE_WITH_HOLE_ROOT,
    *,
    should_run_identification: bool = False,
) -> None:
    dataset_root = _normalise_grid_dataset_root(dataset_path)
    print(f"Loading ANSYS data on interpolated grid from {dataset_root}...")
    dataset = _load_ansys_data_to_grid(dataset_root)
    boundary_setup = _get_boundary_setup(dataset_root)

    x_grid = dataset.x_grid_mm * 1e-3
    y_grid = dataset.y_grid_mm * 1e-3
    strain = dataset.strain
    stress_fe = dataset.stress_fe
    specimen_mask = dataset.specimen_mask
    pixel_area = dataset.pixel_area_mm2 * 1e-6
    force = dataset.force
    time = dataset.time

    specimen_geometry = SpecimenGeometry(
        x_grid,
        y_grid,
        specimen_mask,
        PLATE_THICKNESS,
        pixel_area,
    )

    boundary_conditions = BoundaryConditions(
        boundary_setup.edge_conditions,
        _build_force_history(force, boundary_setup),
    )

    experiment_data = ExperimentData(
        strain,
        specimen_geometry,
        boundary_conditions,
        time,
    )

    constitutive_law = IsotropicVonMisesElastoplasticity(LinearHardening())

    parameter_map_size = np.array(x_grid.shape, dtype=np.uint32)

    parameters = _build_identification_parameters(dataset_root, parameter_map_size)

    metric = SensitivityBasedVirtualFieldsMetric(
        experiment_data.specimen_geometry.x,
        experiment_data.specimen_geometry.y,
        experiment_data.specimen_geometry.region_of_interest,
        experiment_data.boundary_conditions.edge_conditions,
        np.array([15, 15]),
    )

    phases = [
        IdentificationPhase(
            {
                "elastic_modulus": HomogeneousSpatialParameterisation(),
                "poissons_ratio": HomogeneousSpatialParameterisation(),
                "yield_strength": HomogeneousSpatialParameterisation(),
                "hardening_modulus": HomogeneousSpatialParameterisation(),
            },
            [metric],
            VectorFirstResultPassthrough(),
            LeastSquares(),
        )
    ]

    identification = Identification(constitutive_law, parameters, phases)

    known_parameter_maps = _build_known_parameter_maps(x_grid.shape)

    print("Reconstructing stress...")
    stress_calc = constitutive_law.calculate_stress(strain, known_parameter_maps)

    stress_abs_diff = np.abs(stress_calc[-1] - stress_fe[-1])

    if PLOT_STRESS_RECON_ABS_DIFF:
        _plot_stress_abs_diff(x_grid, y_grid, stress_abs_diff)

    stress_abs_perc_diff = np.full_like(stress_fe[-1], np.nan, dtype=np.float64)
    valid = np.abs(stress_fe[-1]) > 0.01
    stress_abs_perc_diff[valid] = (
        np.abs(stress_calc[-1][valid] - stress_fe[-1][valid])
        / np.abs(stress_fe[-1][valid])
    ) * 100.0

    if PLOT_STRESS_RECON_ABS_PERC_DIFF:
        _plot_stress_abs_perc_diff(x_grid, y_grid, stress_abs_perc_diff)

    stress_abs_diff_mean, stress_abs_diff_max, stress_abs_diff_rms = (
        _print_stress_reconstruction_summary(stress_calc, stress_fe)
    )

    assert stress_abs_diff_mean < 0.2
    assert stress_abs_diff_max < 30.0
    assert stress_abs_diff_rms < 0.5

    if not should_run_identification:
        print("Skipping identification. Re-run with --run-identification to continue.")
        return

    print("Running identification...")
    identified_parameters = run_identification(experiment_data, identification)

    ivw_identified = metric._internal_virtual_work.copy()
    evw_identified = metric._external_virtual_work.copy()

    identified_maps = {
        name: param.value for name, param in identified_parameters.items()
    }

    for name, param in identified_parameters.items():
        print(f"{name} = {np.nanmean(param.value):.6f}")

    print("Evaluating metric...")
    metric_spatial_parameterisations = {
        name: HomogeneousSpatialParameterisation()
        for name in KNOWN_PARAMETERS
    }
    for name, spatial_parameterisation in metric_spatial_parameterisations.items():
        spatial_parameterisation.update_from_constitutive_parameter(
            ConstitutiveParameter(
                known_parameter_maps[name],
                parameters[name].lower_bound,
                parameters[name].upper_bound,
            )
        )

    metric.evaluate(
        stress_fe,
        constitutive_law,
        parameter_map_size,
        metric_spatial_parameterisations,
        experiment_data,
    )
    ivw_known = metric._internal_virtual_work.copy()
    evw_known = metric._external_virtual_work.copy()
    _print_virtual_work_residual_summary("known", ivw_known, evw_known)
    _print_virtual_work_residual_summary("identified", ivw_identified, evw_identified)

    sbvf_labels = tuple(name.replace("_", " ") for name in KNOWN_PARAMETERS)

    if PLOT_METRIC_IDENTIFIED_DIFF:
        _plot_metric_virtual_work(
            ivw_known, evw_known, ivw_identified, evw_identified,
            "known", "identified", sbvf_labels,
        )

    ivw_relative_diff = _rms(ivw_identified - ivw_known) / _rms(ivw_known)
    evw_relative_diff = _rms(evw_identified - evw_known) / _rms(evw_known)

    print(f"metric IVW relative diff (known vs identified) = {ivw_relative_diff:.6f}")
    print(f"metric EVW relative diff (known vs identified) = {evw_relative_diff:.6f}")

    assert ivw_relative_diff < 0.1
    assert evw_relative_diff < 0.1

    if PLOT_IDENTIFICATION_DIFF:
        _plot_identification_diff(
            x_grid, y_grid, identified_maps, known_parameter_maps
        )

    abs_diff_rms_tolerances = {
        "elastic_modulus": 500.0,
        "poissons_ratio": 2e-3,
        "yield_strength": 2.0,
        "hardening_modulus": 300.0,
    }

    for name in KNOWN_PARAMETERS:
        abs_diff = np.abs(identified_maps[name] - known_parameter_maps[name])
        abs_diff_rms = _rms(abs_diff)
        rmspe = _root_mean_square_percentage_error(
            identified_maps[name], known_parameter_maps[name]
        )
        print(
            f"{name}: abs diff rms = {abs_diff_rms:.6f}, rmspe = {rmspe:.6f} %"
        )

        assert abs_diff_rms < abs_diff_rms_tolerances[name]
        assert rmspe < 20.0


def run_stress_reconstruction_ansys_raw_centroids(
    dataset_path: Path = PLATE_WITH_HOLE_FE_ROOT,
) -> None:
    raw_root = _normalise_raw_dataset_root(dataset_path)
    print(f"Loading ANSYS raw centroid data from {raw_root}...")
    dataset = _load_ansys_point_data(raw_root)
    constitutive_law = IsotropicVonMisesElastoplasticity(LinearHardening())
    known_parameter_maps = _build_known_parameter_maps_for_points(dataset.strain.shape[-1])

    print("Reconstructing stress from raw centroids...")
    stress_calc = constitutive_law.calculate_stress(dataset.strain, known_parameter_maps)

    stress_abs_diff = np.abs(stress_calc[-1] - dataset.stress_fe[-1])
    point_x_m = dataset.point_x_mm * 1e-3
    point_y_m = dataset.point_y_mm * 1e-3

    if PLOT_STRESS_RECON_ABS_DIFF:
        _plot_centroid_stress_abs_diff(point_x_m, point_y_m, stress_abs_diff)

    stress_abs_perc_diff = np.full_like(dataset.stress_fe[-1], np.nan, dtype=np.float64)
    valid = np.abs(dataset.stress_fe[-1]) > 0.01
    stress_abs_perc_diff[valid] = (
        np.abs(stress_calc[-1][valid] - dataset.stress_fe[-1][valid])
        / np.abs(dataset.stress_fe[-1][valid])
    ) * 100.0

    if PLOT_STRESS_RECON_ABS_PERC_DIFF:
        _plot_centroid_stress_abs_perc_diff(
            point_x_m,
            point_y_m,
            stress_abs_perc_diff,
        )

    stress_abs_diff_mean, stress_abs_diff_max, stress_abs_diff_rms = (
        _print_stress_reconstruction_summary(stress_calc, dataset.stress_fe)
    )

    assert stress_abs_diff_mean < 0.1
    assert stress_abs_diff_max < 20.0
    assert stress_abs_diff_rms < 0.5


def test_end_to_end_ansys() -> None:
    run_end_to_end_ansys_grid(
        PLATE_WITH_HOLE_ROOT,
        should_run_identification=RUN_IDENTIFICATION,
    )


def test_single_element_ansys_stress_reconstruction() -> None:
    if not SINGLE_ELEMENT_FE_ROOT.is_dir():
        pytest.skip(f"ANSYS dataset not found: {SINGLE_ELEMENT_FE_ROOT}")

    num_points = int(np.atleast_1d(np.loadtxt(SINGLE_ELEMENT_FE_ROOT / "x_coordinates.txt")).size)

    strain = np.stack(
        (
            _load_point_history(SINGLE_ELEMENT_FE_ROOT / "eps_xx.txt", num_points),
            _load_point_history(SINGLE_ELEMENT_FE_ROOT / "eps_yy.txt", num_points),
            0.5 * _load_point_history(SINGLE_ELEMENT_FE_ROOT / "eps_xy.txt", num_points),
        ),
        axis=1,
    )
    strain = np.transpose(strain, (2, 1, 0))[:, :, np.newaxis, :]

    stress_fe = np.stack(
        (
            _load_point_history(SINGLE_ELEMENT_FE_ROOT / "sig_xx.txt", num_points),
            _load_point_history(SINGLE_ELEMENT_FE_ROOT / "sig_yy.txt", num_points),
            _load_point_history(SINGLE_ELEMENT_FE_ROOT / "sig_xy.txt", num_points),
        ),
        axis=1,
    )
    stress_fe = np.transpose(stress_fe, (2, 1, 0))[:, :, np.newaxis, :]

    parameter_maps = {
        name: np.full((1, num_points), value, dtype=np.float64)
        for name, value in KNOWN_PARAMETERS.items()
    }

    constitutive_law = IsotropicVonMisesElastoplasticity(LinearHardening())
    stress_calc = constitutive_law.calculate_stress(strain, parameter_maps)
    stress_abs_diff = np.abs(stress_calc - stress_fe)

    assert float(np.nanmax(stress_abs_diff)) < 1e-5


def main() -> None:
    if len(sys.argv) == 1:
        data_source = DEBUG_DATA_SOURCE
        run_identification = RUN_IDENTIFICATION
        dataset_path = MANUAL_RUN_DATASET
        print(
            "No CLI args supplied; using debug defaults: "
            f"data_source={data_source!r}, "
            f"run_identification={run_identification}, "
            f"dataset={dataset_path}."
        )
    else:
        parser = argparse.ArgumentParser(
            description=(
                "Run the ANSYS VFM end-to-end debug workflow either on the regular "
                "interpolated grid or directly on raw FE centroids."
            )
        )
        parser.add_argument(
            "--data-source",
            choices=("grid", "raw-centroids"),
            default="grid",
            help=(
                "`grid` uses the prepared regular grid and supports identification; "
                "`raw-centroids` uses FE centroids directly for more accurate "
                "stress reconstruction only."
            ),
        )
        parser.add_argument(
            "--run-identification",
            action="store_true",
            help="Continue past stress reconstruction and run the identification workflow.",
        )
        parser.add_argument(
            "--dataset",
            type=Path,
            default=MANUAL_RUN_DATASET,
            help=(
                "Dataset root or fe-data folder to run. For grid mode, prefer the "
                "dataset root containing both fe-data/ and vfm-input-data-*."
            ),
        )
        args = parser.parse_args()
        data_source = args.data_source
        run_identification = args.run_identification
        dataset_path = args.dataset

    if data_source == "raw-centroids" and run_identification:
        print("Raw centroid mode does not run identification; skipping identification.")
        run_identification = False

    if data_source == "grid":
        run_end_to_end_ansys_grid(
            dataset_path,
            should_run_identification=run_identification,
        )
        return

    run_stress_reconstruction_ansys_raw_centroids(dataset_path)


if __name__ == "__main__":
    main()

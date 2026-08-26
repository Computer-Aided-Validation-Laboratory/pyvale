"""Measure how well bivariate Gaussian bases represent the EBW truth map.

This is a map-only diagnostic: it does not evaluate VFM metrics or reconstruct
stress.  The fitted representation is the phase-1 form used by the EBW caller,
namely a homogeneous base value plus additive bivariate Gaussian functions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pyvale.vfm import ExperimentData, SpatialParameterisationBasisFunction
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm import spatialparambasisfuncs
from pyvale.vfm.spatialparambasisfuncs import BasisFunctionKernelBivariate


DEFAULT_INPUT = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/"
    "synthetic-fe/wdbn1-idealised-yield/pyvale-vfm/prepared"
)
DEFAULT_OUTPUT = Path("dev/vfm/output/notched_ebw_parameterisation_capacity")
BASE_YIELD_MPA = 543.0
YIELD_RANGE_MPA = 1800.0
WELD_CENTRE_X_MM = 70.0


def main() -> None:
    args = _parse_args()
    input_dir = args.input.parent if args.input.is_file() else args.input
    experiment_file = args.input if args.input.is_file() else args.input / "experiment_data.yaml"
    experiment = ExperimentData.load_from_file(experiment_file)
    known = dict(np.load(input_dir / "known_parameter_maps.npz"))
    truth = np.asarray(known["yield_strength"], dtype=np.float64)
    x = experiment.specimen_geometry.x
    y = experiment.specimen_geometry.y
    roi = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(x, y)
    regions = _region_masks(x, truth, roi)
    spatialparambasisfuncs.MIN_FEATURE_SIZE_POINTS = args.min_feature_size_points

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    fitted_maps: list[np.ndarray] = []
    basis_rows: list[dict[str, Any]] = []

    all_configurations = (("data_bounds", 1.0), ("double_span_bounds", 2.0))
    configurations = tuple(
        item for item in all_configurations
        if args.bound_config in {"both", item[0]}
    )
    for configuration, centre_span_factor in configurations:
      for requested_count in range(args.max_bases + 1):
        if requested_count == 0:
            fitted = np.full_like(truth, BASE_YIELD_MPA)
            basis = None
        else:
            basis = SpatialParameterisationBasisFunction(
                x=x,
                y=y,
                kernel_type="bivariate",
                centre_bounds_span_factor=centre_span_factor,
            )
            basis.fit_to_map(
                truth - BASE_YIELD_MPA,
                parameter_range=YIELD_RANGE_MPA,
                max_basis_functions=requested_count,
                minimum_relative_improvement=0.0,
                fit_mask=roi if args.fit_domain == "roi" else None,
            )
            fitted = BASE_YIELD_MPA + basis.to_map(np.asarray(truth.shape, dtype=np.uint32))

        fitted_maps.append(fitted)
        row: dict[str, Any] = {
            "configuration": configuration,
            "centre_bounds_span_factor": centre_span_factor,
            "requested_bases": requested_count,
            "fitted_bases": 0 if basis is None else len(basis.kernels),
            **_errors(truth, fitted, roi, "roi"),
        }
        for name, mask in regions.items():
            row.update(_errors(truth, fitted, mask, name))
        previous_rmse = (
            None if requested_count == 0 else float(rows[-1]["roi_rmse_mpa"])
        )
        row["incremental_roi_rmse_improvement_fraction"] = (
            None if previous_rmse is None else
            (previous_rmse - float(row["roi_rmse_mpa"])) / previous_rmse
        )
        rows.append(row)
        if basis is not None:
            new_records = _basis_records(requested_count, basis)
            for record in new_records:
                record["configuration"] = configuration
            basis_rows.extend(new_records)
        print(
            f"{configuration}: {requested_count:2d} requested / {row['fitted_bases']:2d} fitted: "
            f"MAPE={row['roi_mape_percent']:.3f}%  "
            f"RMSE={row['roi_rmse_mpa']:.3f} MPa  "
            f"bias={row['roi_bias_mpa']:.3f} MPa"
        )
        _plot_map_fit(
            args.output / "maps" / configuration / f"fit_{requested_count:02d}_bases.png",
            x,
            y,
            truth,
            fitted,
            roi,
            configuration,
            requested_count,
        )

    _write_csv(args.output / "capacity_metrics.csv", rows)
    _write_csv(args.output / "basis_parameters.csv", basis_rows)
    np.savez_compressed(
        args.output / "fitted_maps.npz",
        requested_basis_counts=np.tile(np.arange(args.max_bases + 1), len(configurations)),
        configuration=np.repeat([item[0] for item in configurations], args.max_bases + 1),
        fitted_yield_strength=np.stack(fitted_maps),
        known_yield_strength=truth,
        roi_mask=roi,
        **{f"region_{name}": mask for name, mask in regions.items()},
    )
    _plot_summary(args.output / "capacity_summary.png", rows)
    summary = {
        "method": "residual-seeded sequential growth with joint refit of accepted basis DOFs",
        "fit_domain": args.fit_domain,
        "minimum_feature_size_points": args.min_feature_size_points,
        "centre_bound_configurations": {
            "data_bounds": "centres constrained to the measured coordinate domain",
            "double_span_bounds": "2x domain-centred span; half a domain of padding on each side",
        },
        "homogeneous_base_yield_mpa": BASE_YIELD_MPA,
        "input": str(experiment_file),
        "maximum_requested_bases": args.max_bases,
        "region_definitions": {
            "base_metal": "abs(truth - 543 MPa) <= 1 MPa",
            "soft_strip": "truth < 542 MPa",
            "weld": "abs(x - 70 mm) <= 1 mm",
            "haz": "truth > 544 MPa outside the weld region",
        },
        "results": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-bases", type=int, default=10)
    parser.add_argument(
        "--bound-config",
        choices=("both", "data_bounds", "double_span_bounds"),
        default="both",
        help="Centre-bound branch to run (use double_span_bounds to resume the padded branch).",
    )
    parser.add_argument(
        "--fit-domain",
        choices=("full_grid", "roi"),
        default="full_grid",
        help="Map-fitting support used by the basis objective.",
    )
    parser.add_argument("--min-feature-size-points", type=int, default=3)
    args = parser.parse_args()
    if args.max_bases < 0 or args.min_feature_size_points < 1:
        parser.error("--max-bases must be non-negative")
    return args


def _region_masks(
    x: np.ndarray,
    truth: np.ndarray,
    roi: np.ndarray,
) -> dict[str, np.ndarray]:
    weld = roi & (np.abs(x - WELD_CENTRE_X_MM) <= 1.0)
    return {
        "base_metal": roi & (np.abs(truth - BASE_YIELD_MPA) <= 1.0),
        "soft_strip": roi & (truth < BASE_YIELD_MPA - 1.0),
        "haz": roi & (truth > BASE_YIELD_MPA + 1.0) & ~weld,
        "weld": weld,
    }


def _errors(
    truth: np.ndarray,
    fitted: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, float | int]:
    error = fitted[mask] - truth[mask]
    return {
        f"{prefix}_point_count": int(np.count_nonzero(mask)),
        f"{prefix}_mape_percent": float(np.mean(np.abs(error) / np.abs(truth[mask])) * 100.0),
        f"{prefix}_rmse_mpa": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_bias_mpa": float(np.mean(error)),
    }


def _value(value: float | DegreeOfFreedom) -> float:
    return float(value.value if isinstance(value, DegreeOfFreedom) else value)


def _basis_records(
    requested_count: int,
    basis: SpatialParameterisationBasisFunction,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (kernel, height) in enumerate(zip(basis.kernels, basis.heights, strict=True)):
        assert isinstance(kernel, BasisFunctionKernelBivariate)
        variance_major, variance_minor, angle = kernel.canonical_values()
        records.append({
            "requested_bases": requested_count,
            "basis_index": index,
            "centre_x_mm": _value(kernel.x),
            "centre_y_mm": _value(kernel.y),
            "sigma_major_mm": float(np.sqrt(variance_major)),
            "sigma_minor_mm": float(np.sqrt(variance_minor)),
            "angle_radians": angle,
            "angle_degrees": float(np.degrees(angle)),
            "height_mpa": _value(height),
        })
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for configuration in dict.fromkeys(str(row["configuration"]) for row in rows):
        selected = [row for row in rows if row["configuration"] == configuration]
        counts = np.asarray([row["requested_bases"] for row in selected])
        axes[0].plot(counts, [row["roi_mape_percent"] for row in selected], marker="o", label=configuration)
        axes[1].plot(counts, [row["roi_rmse_mpa"] for row in selected], marker="o", label=configuration)
        axes[2].plot(counts, [row["roi_bias_mpa"] for row in selected], marker="o", label=configuration)
    axes[0].set_ylabel("ROI MAPE [%]")
    axes[1].set_ylabel("ROI RMSE [MPa]")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("ROI bias [MPa]")
    for axis in axes:
        axis.set_xlabel("Requested bivariate bases")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_map_fit(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    truth: np.ndarray,
    fitted: np.ndarray,
    roi: np.ndarray,
    configuration: str,
    basis_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    truth_roi = np.where(roi, truth, np.nan)
    fitted_roi = np.where(roi, fitted, np.nan)
    error_roi = np.where(roi, fitted - truth, np.nan)
    percent_error_roi = np.where(roi, 100.0 * (fitted - truth) / truth, np.nan)
    extent = (float(x.min()), float(x.max()), float(y.max()), float(y.min()))
    value_limits = (float(np.nanmin(truth_roi)), float(np.nanmax(truth_roi)))
    error_limit = float(np.nanmax(np.abs(error_roi)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    axes = axes.ravel()
    image = axes[0].imshow(truth_roi, extent=extent, vmin=value_limits[0], vmax=value_limits[1])
    axes[0].set_title("Known yield strength")
    axes[1].imshow(fitted_roi, extent=extent, vmin=value_limits[0], vmax=value_limits[1])
    axes[1].set_title(f"Fitted: {basis_count} bases")
    error_image = axes[2].imshow(error_roi, extent=extent, cmap="RdBu_r", vmin=-error_limit, vmax=error_limit)
    axes[2].set_title("Fitted - known [MPa]")
    percent_limit = float(np.nanmax(np.abs(percent_error_roi)))
    percent_image = axes[3].imshow(percent_error_roi, extent=extent, cmap="RdBu_r", vmin=-percent_limit, vmax=percent_limit)
    axes[3].set_title("Fitted - known [%]")
    fig.colorbar(image, ax=axes[:2], label="Yield strength [MPa]", shrink=0.85)
    fig.colorbar(error_image, ax=axes[2], label="Error [MPa]", shrink=0.85)
    fig.colorbar(percent_image, ax=axes[3], label="Error [%]", shrink=0.85)
    for axis in axes:
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
    fig.suptitle(configuration.replace("_", " "))
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

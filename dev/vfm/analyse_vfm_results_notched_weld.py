"""Post-process the univariate-Gaussian VFM identification of a notched weld."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cmcrameri.cm  # noqa: F401
except ImportError:
    CMCRAMERI_AVAILABLE = False
else:
    CMCRAMERI_AVAILABLE = True

from pyvale.vfm import ExperimentData, load_identification_result
from pyvale.vfm.postprocessing import (
    cache_equilibrium_gap_diagnostics,
    cache_force_reconstruction_diagnostics,
    cache_parameter_error_diagnostics,
    cache_plasticity_diagnostics,
    cache_stress,
    check_stress_against_saved,
    compute_equilibrium_gap_diagnostics,
    compute_force_reconstruction_diagnostics,
    compute_parameter_error_diagnostics,
    compute_plasticity_diagnostics,
    compute_stress_from_result,
    load_constitutive_law_from_result,
    load_known_parameter_maps,
    ParameterErrorDiagnostics,
    parameter_map_summary,
    plastic_parameter_names,
    plot_grid_map,
    plot_individual_maps,
    plot_map_collection,
    plot_stress_strain_tiled,
    plot_yielded_datapoints,
    write_summary_json,
)


# =============================================================================
# User inputs
# =============================================================================

DATASET_ROOT = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/notched-weld-data"
)
EXPERIMENT_DATA_PATH = DATASET_ROOT / "prepared"
KNOWN_PARAMETER_MAPS = DATASET_ROOT / "fe-data" / "raw" / "known_parameter_maps.npz"
RESULT_BUNDLE = (
    DATASET_ROOT
    / "identification"
    / "prepared"
    / "univariate_gaussian"
    / "identification_result.yaml"
)
POSTPROCESSING_OUTPUT_DIR = (
    DATASET_ROOT
    / "identification"
    / "prepared"
    / "univariate_gaussian_postprocessing_output"
)
DIAGNOSTIC_AXIS = "x"
DIAGNOSTIC_SLICES = 63
POINT_COUNT = 10
POINT_PLOT_COMPONENT = "vm"
POINT_PLOT_TIMESTEP = None
EGI_WINDOW_SIZE = 29
STRESS_RTOL = 1.0e-8
STRESS_ATOL = 1.0e-8
CACHE_FULL_EGI_HISTORY = False

SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "cmc.vik" if CMCRAMERI_AVAILABLE else "RdBu_r"


def main() -> None:
    args = _parse_args()
    experiment_data_file = _resolve_experiment_data_file(args.input)
    cache_dir = args.output / "cache"
    figure_dir = args.output / "figures"

    experiment_data = ExperimentData.load_from_file(experiment_data_file)
    result = load_identification_result(args.result)
    known_parameter_maps = load_known_parameter_maps(
        args.known_parameters,
        experiment_data_file.parent,
    )
    constitutive_law = load_constitutive_law_from_result(result)

    stress = compute_stress_from_result(experiment_data, result, constitutive_law)
    stress_check = check_stress_against_saved(
        stress,
        result.final_stress,
        rtol=args.stress_rtol,
        atol=args.stress_atol,
    )
    if (
        stress_check.saved_stress_available
        and stress_check.matches_saved_stress is False
        and not args.allow_stress_mismatch
    ):
        raise RuntimeError(stress_check.message)

    plasticity = compute_plasticity_diagnostics(
        experiment_data,
        constitutive_law,
        result.parameter_maps,
    )
    force_reconstruction = compute_force_reconstruction_diagnostics(
        experiment_data,
        stress,
        axis=args.axis,
        num_slices=args.diagnostic_slices,
    )
    equilibrium_gap = compute_equilibrium_gap_diagnostics(
        experiment_data,
        stress,
        window_size=args.egi_window_size,
    )
    parameter_errors = (
        None
        if known_parameter_maps is None
        else compute_parameter_error_diagnostics(
            result.parameter_maps,
            known_parameter_maps,
        )
    )

    cache_stress(cache_dir, stress)
    cache_plasticity_diagnostics(cache_dir, plasticity)
    cache_force_reconstruction_diagnostics(cache_dir, force_reconstruction)
    cache_equilibrium_gap_diagnostics(
        cache_dir,
        equilibrium_gap,
        cache_full_egi_history=args.cache_full_egi_history,
    )
    cache_parameter_error_diagnostics(cache_dir, parameter_errors)

    yielded_datapoints = (
        None if plasticity is None else plasticity.yielded_datapoints
    )
    transparent_parameter_names = plastic_parameter_names(constitutive_law)
    plot_map_collection(
        experiment_data,
        result.parameter_maps,
        figure_dir / "identified_parameter_maps.png",
        "Identified Parameter Maps",
        cmap=SEQUENTIAL_CMAP,
        yielded_datapoints=yielded_datapoints,
        transparent_names=transparent_parameter_names,
    )
    plot_individual_maps(
        experiment_data,
        result.parameter_maps,
        figure_dir,
        "identified_parameter_map",
        cmap=SEQUENTIAL_CMAP,
        yielded_datapoints=yielded_datapoints,
        transparent_names=transparent_parameter_names,
    )
    if plasticity is not None:
        plot_yielded_datapoints(
            plasticity.yielded_datapoints,
            figure_dir / "yielded_datapoints.png",
        )
    if parameter_errors is not None:
        _plot_parameter_errors(
            experiment_data,
            parameter_errors,
            figure_dir,
        )

    plot_grid_map(
        force_reconstruction.weighted_rms_newtons_map,
        figure_dir / "force_reconstruction_error_newtons.png",
        "FRE [N]",
        cmap=SEQUENTIAL_CMAP,
    )
    plot_grid_map(
        force_reconstruction.weighted_rms_percent_map,
        figure_dir / "force_reconstruction_error_percent.png",
        "FRE [% peak force]",
        cmap=SEQUENTIAL_CMAP,
    )
    if equilibrium_gap.weighted_temporal_rms_percent_map is not None:
        plot_grid_map(
            equilibrium_gap.weighted_temporal_rms_percent_map,
            figure_dir / "equilibrium_gap_indicator.png",
            "Equilibrium Gap Indicator [%]",
            cmap=SEQUENTIAL_CMAP,
        )

    point_rows, point_columns = _horizontal_plot_points(
        experiment_data,
        POINT_COUNT,
    )
    plot_stress_strain_tiled(
        experiment_data.strain,
        stress,
        POINT_PLOT_COMPONENT,
        point_rows,
        point_columns,
        timestep=POINT_PLOT_TIMESTEP,
        output_path=figure_dir / "stress_strain_tiled.png",
        cmap=SEQUENTIAL_CMAP,
    )

    summary = {
        "input": str(experiment_data_file),
        "result": str(args.result),
        "output_dir": str(args.output),
        "axis": args.axis,
        "diagnostic_slices": args.diagnostic_slices,
        "egi_window_size": list(equilibrium_gap.window_size),
        "point_plot_component": POINT_PLOT_COMPONENT,
        "point_plot_count": len(point_rows),
        **stress_check.to_summary(),
        **force_reconstruction.to_summary(),
        **equilibrium_gap.to_summary(),
        **({} if plasticity is None else plasticity.to_summary()),
        **parameter_map_summary(result.parameter_maps),
        **({} if parameter_errors is None else parameter_errors.summary),
    }
    write_summary_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Cached arrays in {cache_dir}")
    print(f"Saved figures in {figure_dir}")


def _plot_parameter_errors(
    experiment_data: ExperimentData,
    parameter_errors: ParameterErrorDiagnostics,
    figure_dir: Path,
) -> None:
    for maps, filename, title, prefix in (
        (
            parameter_errors.error_maps,
            "parameter_error_maps.png",
            "Identified Minus Known Parameter Maps",
            "parameter_error_map",
        ),
        (
            parameter_errors.percent_error_maps,
            "parameter_percent_error_maps.png",
            "Identified Minus Known Parameter Maps [%]",
            "parameter_percent_error_map",
        ),
    ):
        plot_map_collection(
            experiment_data,
            maps,
            figure_dir / filename,
            title,
            cmap=DIVERGING_CMAP,
            symmetric=True,
        )
        plot_individual_maps(
            experiment_data,
            maps,
            figure_dir,
            prefix,
            cmap=DIVERGING_CMAP,
            symmetric=True,
        )


def _horizontal_plot_points(
    experiment_data: ExperimentData,
    count: int,
) -> tuple[list[int], list[int]]:
    row = experiment_data.strain.shape[2] // 2
    roi = experiment_data.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment_data.specimen_geometry.x,
        experiment_data.specimen_geometry.y,
    )
    valid = roi[row, :] & np.isfinite(
        experiment_data.strain[:, :, row, :]
    ).all(axis=(0, 1))
    columns = np.flatnonzero(valid)
    if columns.size < count:
        raise ValueError(
            f"Only {columns.size} valid horizontal points; need {count}."
        )
    selected = np.rint(np.linspace(0, columns.size - 1, count)).astype(int)
    return [row] * count, columns[selected].tolist()


def _resolve_experiment_data_file(input_path: Path) -> Path:
    return input_path / "experiment_data.yaml" if input_path.is_dir() else input_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EXPERIMENT_DATA_PATH)
    parser.add_argument("--result", type=Path, default=RESULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=POSTPROCESSING_OUTPUT_DIR)
    parser.add_argument(
        "--known-parameters",
        type=Path,
        default=KNOWN_PARAMETER_MAPS,
        help=(
            "Known parameter-map file or directory for map-error comparisons."
        ),
    )
    parser.add_argument("--axis", choices=("x", "y"), default=DIAGNOSTIC_AXIS)
    parser.add_argument("--diagnostic-slices", type=int, default=DIAGNOSTIC_SLICES)
    parser.add_argument("--egi-window-size", type=int, default=EGI_WINDOW_SIZE)
    parser.add_argument("--stress-rtol", type=float, default=STRESS_RTOL)
    parser.add_argument("--stress-atol", type=float, default=STRESS_ATOL)
    parser.add_argument(
        "--cache-full-egi-history",
        action="store_true",
        default=CACHE_FULL_EGI_HISTORY,
    )
    parser.add_argument("--allow-stress-mismatch", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

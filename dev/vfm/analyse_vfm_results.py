from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import cmcrameri.cm  # noqa: F401
except ImportError:
    CMCRAMERI_AVAILABLE = False
else:
    CMCRAMERI_AVAILABLE = True

from pyvale.vfm import ExperimentData, load_identification_result
from pyvale.vfm.postprocessing import (
    StressCheckResult,
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
    parameter_map_summary,
    plastic_parameter_names,
    plot_grid_map,
    plot_individual_maps,
    plot_map_collection,
    plot_yielded_datapoints,
    write_summary_json,
)


# =============================================================================
# User inputs
# =============================================================================

EXPERIMENT_DATA_PATH = Path(
    "/media/data/3_Resources/gr91-weld-dic-results/wdbn1/pyvale-input/"
    "vfm-input-data_2026-08-12_15-43"
)
RESULT_BUNDLE = (
    Path(__file__).resolve().parent
    / "call_vfm_sw_refine_clean_output"
    / EXPERIMENT_DATA_PATH.name
    / "identification_result"
)
POSTPROCESSING_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "analyse_vfm_results_output"
    / EXPERIMENT_DATA_PATH.name
)
KNOWN_PARAMETER_MAPS = None

DIAGNOSTIC_AXIS = "y"
DIAGNOSTIC_SLICES = 30
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

    stress = compute_stress_from_result(
        experiment_data,
        result,
        constitutive_law,
    )
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
        None
        if plasticity is None
        else plasticity.yielded_datapoints
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
        plot_map_collection(
            experiment_data,
            parameter_errors.error_maps,
            figure_dir / "parameter_error_maps.png",
            "Identified Minus Known Parameter Maps",
            cmap=DIVERGING_CMAP,
            symmetric=True,
        )
        plot_individual_maps(
            experiment_data,
            parameter_errors.error_maps,
            figure_dir,
            "parameter_error_map",
            cmap=DIVERGING_CMAP,
            symmetric=True,
        )
        plot_map_collection(
            experiment_data,
            parameter_errors.percent_error_maps,
            figure_dir / "parameter_percent_error_maps.png",
            "Identified Minus Known Parameter Maps [%]",
            cmap=DIVERGING_CMAP,
            symmetric=True,
        )
        plot_individual_maps(
            experiment_data,
            parameter_errors.percent_error_maps,
            figure_dir,
            "parameter_percent_error_map",
            cmap=DIVERGING_CMAP,
            symmetric=True,
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

    summary = _build_summary(
        input_path=experiment_data_file,
        result_path=args.result,
        output_dir=args.output,
        axis=args.axis,
        diagnostic_slices=args.diagnostic_slices,
        egi_window_size=equilibrium_gap.window_size,
        stress_check=stress_check,
        parameter_maps=result.parameter_maps,
        plasticity_summary=(
            {} if plasticity is None else plasticity.to_summary()
        ),
        force_reconstruction_summary=force_reconstruction.to_summary(),
        equilibrium_gap_summary=equilibrium_gap.to_summary(),
        parameter_error_summary=(
            {} if parameter_errors is None else parameter_errors.summary
        ),
    )
    write_summary_json(args.output / "summary.json", summary)

    print(json.dumps(summary, indent=2))
    print(f"Cached arrays in {cache_dir}")
    print(f"Saved figures in {figure_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse a saved VFM identification result bundle."
    )
    parser.add_argument("--input", type=Path, default=EXPERIMENT_DATA_PATH)
    parser.add_argument("--result", type=Path, default=RESULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=POSTPROCESSING_OUTPUT_DIR)
    parser.add_argument(
        "--known-parameters",
        type=Path,
        default=KNOWN_PARAMETER_MAPS,
        help=(
            "Optional known_parameter_maps.npz file or folder containing it. "
            "When omitted, the script looks next to experiment_data.yaml."
        ),
    )
    parser.add_argument("--axis", choices=("x", "y"), default=DIAGNOSTIC_AXIS)
    parser.add_argument(
        "--diagnostic-slices",
        type=int,
        default=DIAGNOSTIC_SLICES,
    )
    parser.add_argument("--egi-window-size", type=int, default=EGI_WINDOW_SIZE)
    parser.add_argument("--stress-rtol", type=float, default=STRESS_RTOL)
    parser.add_argument("--stress-atol", type=float, default=STRESS_ATOL)
    parser.add_argument(
        "--cache-full-egi-history",
        action="store_true",
        default=CACHE_FULL_EGI_HISTORY,
        help="Also cache full raw/normalised EGI histories, which can be large.",
    )
    parser.add_argument(
        "--allow-stress-mismatch",
        action="store_true",
        help="Continue even if recomputed stress differs from saved stress.",
    )
    return parser.parse_args()


def _resolve_experiment_data_file(input_path: Path) -> Path:
    return (
        input_path / "experiment_data.yaml"
        if input_path.is_dir()
        else input_path
    )


def _build_summary(
    *,
    input_path: Path,
    result_path: Path,
    output_dir: Path,
    axis: str,
    diagnostic_slices: int,
    egi_window_size: tuple[int, int],
    stress_check: StressCheckResult,
    parameter_maps: dict[str, object],
    plasticity_summary: dict[str, object],
    force_reconstruction_summary: dict[str, object],
    equilibrium_gap_summary: dict[str, object],
    parameter_error_summary: dict[str, object],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "input": str(input_path),
        "result": str(result_path),
        "output_dir": str(output_dir),
        "axis": axis,
        "diagnostic_slices": int(diagnostic_slices),
        "egi_window_size": [int(value) for value in egi_window_size],
    }
    summary.update(stress_check.to_summary())
    summary.update(force_reconstruction_summary)
    summary.update(equilibrium_gap_summary)
    summary.update(plasticity_summary)
    summary.update(parameter_map_summary(parameter_maps))
    summary.update(parameter_error_summary)
    return summary


if __name__ == "__main__":
    main()

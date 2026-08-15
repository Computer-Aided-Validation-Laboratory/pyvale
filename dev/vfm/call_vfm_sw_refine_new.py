from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyvale.vfm import (
    ConstitutiveParameter,
    EquilibriumGapMetric,
    ExperimentData,
    HardeningLinear,
    IdentificationConfig,
    IdentificationPhase,
    IsotropicVonMisesElastoplasticity,
    SliceConfig,
    SliceMergeSplitRefinement,
    SliceWiseForceReconstructionMetric,
    SliceWiseIndependentLeastSquares,
    SliceWiseSpatialParameterisation,
    SpatialParameterisationKnown,
    SupportSlice,
    VectorWeightedObjective,
    run_identification,
)


DEFAULT_INPUTS_PATH = Path(
    "/media/data/3_Resources/gr91-weld-dic-results/wdbn1/pyvale-input/"
    "vfm-input-data_2026-08-12_15-43"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "call_vfm_sw_refine_new_output"


def main() -> None:
    args = _parse_args()
    input_path = args.input
    experiment_data_file = (
        input_path / "experiment_data.yaml"
        if input_path.is_dir()
        else input_path
    )
    experiment_data = ExperimentData.load_from_file(experiment_data_file)
    output_dir = args.output_root / experiment_data_file.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    known_parameter_maps = _load_known_parameter_maps(
        args.known_parameters,
        experiment_data_file.parent,
    )

    constitutive_law = IsotropicVonMisesElastoplasticity(HardeningLinear())
    parameter_map_size = np.asarray(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32,
    )
    shared_support = SupportSlice(
        slice_config=SliceConfig(axis=args.axis, num_slices=args.num_slices),
    )

    parameters = _build_parameters(parameter_map_size)
    phase = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [
                SliceWiseSpatialParameterisation(support=shared_support),
            ],
            "hardening_modulus": (
                [SpatialParameterisationKnown()]
                if args.fix_hardening
                else [SliceWiseSpatialParameterisation(support=shared_support)]
            ),
        },
        metrics=[SliceWiseForceReconstructionMetric(support=shared_support)],
        objective_function=VectorWeightedObjective(),
        optimiser=SliceWiseIndependentLeastSquares(),
        refinement_policy=SliceMergeSplitRefinement(
            target=shared_support,
            merge_parameter_tolerance=args.merge_parameter_tolerance,
            split_error_threshold=args.split_force_error_threshold,
            max_refinements=args.max_refinements,
        ),
    )

    result = run_identification(
        experiment_data,
        IdentificationConfig(
            constitutive_law=constitutive_law,
            parameters=parameters,
            phases=[phase],
        ),
    )
    result_file = result.save_to_yaml(output_dir / "identification_result")
    np.savez(output_dir / "identified_parameter_maps.npz", **result.parameter_maps)

    stress = constitutive_law.calculate_stress(
        experiment_data.strain,
        result.parameter_maps,
    )
    diagnostic_slices = (
        args.diagnostic_slices
        if args.diagnostic_slices is not None
        else args.num_slices
    )
    fre_metric = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis=args.axis, num_slices=diagnostic_slices),
    )
    fre_metric.initialise(experiment_data)
    fre_result = fre_metric.evaluate_force_recon_error(stress, experiment_data)

    egi_metric = EquilibriumGapMetric(
        window_size=_resolve_egi_window(
            experiment_data.specimen_geometry.x.shape,
            args.egi_window_size,
        ),
    )
    egi_metric.initialise(experiment_data)
    egi_result = egi_metric.evaluate_equilibrium_gap(stress)

    _plot_map_collection(
        experiment_data,
        result.parameter_maps,
        output_dir / "identified_parameter_maps.png",
        "Identified Parameter Maps",
        cmap="viridis",
    )
    _plot_individual_maps(
        experiment_data,
        result.parameter_maps,
        output_dir,
        "identified_parameter_map",
        cmap="viridis",
    )

    parameter_error_summary = {}
    if known_parameter_maps is not None:
        parameter_error_summary = _write_parameter_error_outputs(
            experiment_data,
            result.parameter_maps,
            known_parameter_maps,
            output_dir,
        )

    force_error_summary = _write_force_error_outputs(
        experiment_data,
        fre_metric,
        fre_result,
        output_dir,
    )
    egi_summary = _write_egi_outputs(
        experiment_data,
        egi_result.weighted_temporal_rms,
        output_dir,
    )

    summary = {
        "input": str(experiment_data_file),
        "result": str(result_file),
        "axis": args.axis,
        "initial_slices": args.num_slices,
        "diagnostic_slices": diagnostic_slices,
        "cell_aligned_slices": True,
        "weighted_force_reconstruction_error_percent": (
            100.0 * fre_result.weighted_spatiotemporal_rms
        ),
        "weighted_equilibrium_gap_percent": (
            None
            if egi_result.weighted_spatiotemporal_rms is None
            else 100.0 * egi_result.weighted_spatiotemporal_rms
        ),
        "yield_strength_min": float(np.nanmin(result.parameter_maps["yield_strength"])),
        "yield_strength_mean": float(np.nanmean(result.parameter_maps["yield_strength"])),
        "yield_strength_max": float(np.nanmax(result.parameter_maps["yield_strength"])),
        "hardening_modulus_mean": float(np.nanmean(result.parameter_maps["hardening_modulus"])),
        **force_error_summary,
        **egi_summary,
        **parameter_error_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved diagnostics to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUTS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    parser.add_argument("--num-slices", type=int, default=40)
    parser.add_argument("--diagnostic-slices", type=int, default=None)
    parser.add_argument("--max-refinements", type=int, default=1)
    parser.add_argument("--merge-parameter-tolerance", type=float, default=0.05)
    parser.add_argument("--split-force-error-threshold", type=float, default=0.1)
    parser.add_argument("--egi-window-size", type=int, default=29)
    parser.add_argument("--fix-hardening", action="store_true")
    parser.add_argument(
        "--known-parameters",
        type=Path,
        default=None,
        help=(
            "Optional known_parameter_maps.npz file or folder containing it. "
            "When omitted, the script looks next to experiment_data.yaml."
        ),
    )
    return parser.parse_args()


def _build_parameters(
    parameter_map_size: np.ndarray,
) -> dict[str, ConstitutiveParameter]:
    return {
        "elastic_modulus": ConstitutiveParameter(
            210_000.0,
            150_000.0,
            250_000.0,
            parameter_map_size,
        ),
        "poissons_ratio": ConstitutiveParameter(
            0.3,
            0.2,
            0.4,
            parameter_map_size,
        ),
        "yield_strength": ConstitutiveParameter(
            250.0,
            100.0,
            2_000.0,
            parameter_map_size,
        ),
        "hardening_modulus": ConstitutiveParameter(
            7_000.0,
            1_000.0,
            20_000.0,
            parameter_map_size,
        ),
    }


def _load_known_parameter_maps(
    known_parameters: Path | None,
    input_dir: Path,
) -> dict[str, np.ndarray] | None:
    if known_parameters is None:
        candidate = input_dir / "known_parameter_maps.npz"
        if not candidate.exists():
            return None
        known_parameters = candidate
    elif known_parameters.is_dir():
        known_parameters = known_parameters / "known_parameter_maps.npz"

    if not known_parameters.exists():
        raise FileNotFoundError(f"Known parameter maps file not found: {known_parameters}")
    parameter_names = {
        "elastic_modulus",
        "poissons_ratio",
        "yield_strength",
        "hardening_modulus",
    }
    with np.load(known_parameters) as loaded:
        return {
            name: np.asarray(loaded[name], dtype=np.float64)
            for name in loaded.files
            if name in parameter_names
        }


def _ordered_parameter_names(
    parameter_maps: dict[str, np.ndarray],
) -> list[str]:
    preferred = [
        "elastic_modulus",
        "poissons_ratio",
        "yield_strength",
        "hardening_modulus",
    ]
    ordered = [name for name in preferred if name in parameter_maps]
    ordered.extend(name for name in parameter_maps if name not in ordered)
    return ordered


def _parameter_label(name: str) -> str:
    labels = {
        "elastic_modulus": "Elastic Modulus [MPa]",
        "poissons_ratio": "Poisson Ratio [-]",
        "yield_strength": "Yield Strength [MPa]",
        "hardening_modulus": "Hardening Modulus [MPa]",
    }
    return labels.get(name, name.replace("_", " ").title())


def _specimen_mask(experiment_data: ExperimentData) -> np.ndarray:
    return experiment_data.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment_data.specimen_geometry.x,
        experiment_data.specimen_geometry.y,
    )


def _masked_map(
    experiment_data: ExperimentData,
    data: np.ndarray,
) -> np.ndarray:
    return np.where(_specimen_mask(experiment_data), data, np.nan)


def _plot_map_collection(
    experiment_data: ExperimentData,
    parameter_maps: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    *,
    cmap: str,
    symmetric: bool = False,
) -> None:
    names = _ordered_parameter_names(parameter_maps)
    if not names:
        return

    cols = min(2, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.2 * rows), constrained_layout=True)
    flat_axes = np.atleast_1d(axes).ravel()
    fig.suptitle(title)
    for ax, name in zip(flat_axes, names, strict=False):
        data = _masked_map(experiment_data, parameter_maps[name])
        vmin = vmax = None
        if symmetric:
            max_abs = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 0.0
            vmax = max_abs if max_abs > 0.0 else 1.0
            vmin = -vmax
        image = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(_parameter_label(name))
        fig.colorbar(image, ax=ax)
    for ax in flat_axes[len(names):]:
        ax.axis("off")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_individual_maps(
    experiment_data: ExperimentData,
    maps: dict[str, np.ndarray],
    output_dir: Path,
    prefix: str,
    *,
    cmap: str,
    symmetric: bool = False,
) -> None:
    for name in _ordered_parameter_names(maps):
        data = _masked_map(experiment_data, maps[name])
        vmin = vmax = None
        if symmetric:
            max_abs = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 0.0
            vmax = max_abs if max_abs > 0.0 else 1.0
            vmin = -vmax
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        image = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(_parameter_label(name))
        fig.colorbar(image, ax=ax)
        fig.savefig(output_dir / f"{prefix}_{name}.png", dpi=200)
        plt.close(fig)


def _write_parameter_error_outputs(
    experiment_data: ExperimentData,
    identified_maps: dict[str, np.ndarray],
    known_maps: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, float]:
    shared_names = [
        name
        for name in _ordered_parameter_names(identified_maps)
        if name in known_maps
    ]
    error_maps = {
        name: np.asarray(identified_maps[name], dtype=np.float64) - np.asarray(known_maps[name], dtype=np.float64)
        for name in shared_names
    }
    percent_error_maps = {
        name: np.divide(
            100.0 * error_maps[name],
            np.asarray(known_maps[name], dtype=np.float64),
            out=np.full_like(error_maps[name], np.nan, dtype=np.float64),
            where=np.abs(np.asarray(known_maps[name], dtype=np.float64)) > 1.0e-12,
        )
        for name in shared_names
    }

    np.savez(output_dir / "parameter_error_maps.npz", **error_maps)
    np.savez(output_dir / "parameter_percent_error_maps.npz", **percent_error_maps)

    _plot_map_collection(
        experiment_data,
        error_maps,
        output_dir / "parameter_error_maps.png",
        "Identified Minus Known Parameter Maps",
        cmap="coolwarm",
        symmetric=True,
    )
    _plot_individual_maps(
        experiment_data,
        error_maps,
        output_dir,
        "parameter_error_map",
        cmap="coolwarm",
        symmetric=True,
    )
    _plot_map_collection(
        experiment_data,
        percent_error_maps,
        output_dir / "parameter_percent_error_maps.png",
        "Identified Minus Known Parameter Maps [%]",
        cmap="coolwarm",
        symmetric=True,
    )
    _plot_individual_maps(
        experiment_data,
        percent_error_maps,
        output_dir,
        "parameter_percent_error_map",
        cmap="coolwarm",
        symmetric=True,
    )

    summary: dict[str, float] = {}
    for name in shared_names:
        summary[f"{name}_max_abs_error"] = float(np.nanmax(np.abs(error_maps[name])))
        summary[f"{name}_max_abs_percent_error"] = float(np.nanmax(np.abs(percent_error_maps[name])))
    return summary


def _write_force_error_outputs(
    experiment_data: ExperimentData,
    metric: SliceWiseForceReconstructionMetric,
    fre_result,
    output_dir: Path,
) -> dict[str, float]:
    if metric.slice_partition is None:
        raise RuntimeError("FRE metric partition has not been initialised.")

    raw_residual = fre_result.metric_result.additional_fields["raw_residual"]
    normalised_residual = fre_result.metric_result.additional_fields["normalised_residual"]
    temporal_weights = fre_result.metric_result.additional_fields["temporal_weights"]
    rms_newtons = np.sqrt(
        np.sum(temporal_weights[:, np.newaxis] * raw_residual**2, axis=0)
    )
    rms_percent = 100.0 * fre_result.weighted_temporal_rms
    newtons_map = _slice_values_to_grid(experiment_data, metric, rms_newtons)
    percent_map = _slice_values_to_grid(experiment_data, metric, rms_percent)

    np.savez(
        output_dir / "force_reconstruction_error.npz",
        raw_residual_newtons=raw_residual,
        normalised_residual_percent=100.0 * normalised_residual,
        weighted_rms_newtons_by_slice=rms_newtons,
        weighted_rms_percent_by_slice=rms_percent,
        weighted_rms_newtons_map=newtons_map,
        weighted_rms_percent_map=percent_map,
        reconstructed_force=fre_result.metric_result.additional_fields["reconstructed_force"],
        applied_longitudinal_force=fre_result.metric_result.additional_fields["applied_longitudinal_force"],
        slice_boundaries=metric.slice_partition.boundaries,
    )

    _plot_slice_values_on_grid(
        newtons_map,
        output_dir / "force_reconstruction_error_newtons.png",
        "FRE [N]",
        cmap="inferno",
    )
    _plot_slice_values_on_grid(
        percent_map,
        output_dir / "force_reconstruction_error_percent.png",
        "FRE [% peak force]",
        cmap="inferno",
    )
    return {
        "force_reconstruction_error_newtons_max": float(np.nanmax(rms_newtons)),
        "force_reconstruction_error_percent_max": float(np.nanmax(rms_percent)),
    }


def _slice_values_to_grid(
    experiment_data: ExperimentData,
    metric: SliceWiseForceReconstructionMetric,
    values: np.ndarray,
) -> np.ndarray:
    if metric.slice_partition is None:
        raise RuntimeError("Metric slice partition has not been initialised.")
    grid = np.full(experiment_data.specimen_geometry.x.shape, np.nan, dtype=np.float64)
    for slice_index, value in enumerate(values):
        grid[metric.slice_partition.get_slice_mask(slice_index)] = value
    return grid


def _plot_slice_values_on_grid(
    grid: np.ndarray,
    output_path: Path,
    label: str,
    *,
    cmap: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    image = ax.imshow(grid, origin="lower", cmap=cmap)
    ax.set_title(label)
    fig.colorbar(image, ax=ax)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_egi_outputs(
    experiment_data: ExperimentData,
    weighted_temporal_rms: np.ndarray | None,
    output_dir: Path,
) -> dict[str, float | None]:
    if weighted_temporal_rms is None:
        return {
            "equilibrium_gap_indicator_percent_max": None,
        }
    data = _masked_map(experiment_data, 100.0 * weighted_temporal_rms)
    np.save(
        output_dir / "equilibrium_gap_indicator_percent_map.npy",
        data,
    )
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    image = ax.imshow(data, origin="lower", cmap="inferno")
    ax.set_title("Equilibrium Gap Indicator [%]")
    fig.colorbar(image, ax=ax)
    fig.savefig(output_dir / "equilibrium_gap_indicator.png", dpi=200)
    plt.close(fig)
    return {
        "equilibrium_gap_indicator_percent_max": float(np.nanmax(data)),
    }


def _resolve_egi_window(
    shape: tuple[int, int],
    requested_size: int,
) -> tuple[int, int]:
    rows, cols = shape
    size = min(requested_size, rows if rows % 2 == 1 else rows - 1, cols if cols % 2 == 1 else cols - 1)
    size = max(3, size)
    if size % 2 == 0:
        size -= 1
    return (size, size)


if __name__ == "__main__":
    main()

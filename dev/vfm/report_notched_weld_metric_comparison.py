"""Compare notched-weld VFM formulations and generate a PDF report."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from pyvale.vfm import (
    ConstitutiveParameter,
    DegreeOfFreedom,
    EquilibriumGapMetric,
    ExperimentData,
    MetricSBVF,
    load_identification_result,
)
from pyvale.vfm.equilibriumgapaggregation import (
    aggregate_equilibrium_gap_results,
    calculate_nan_rms,
    extract_equilibrium_gap_temporal_rms,
)
from pyvale.vfm.postprocessing import (
    compute_force_reconstruction_diagnostics,
    load_constitutive_law_from_result,
)
from pyvale.vfm.spatialparam import ISpatialParameterisation


DATASET = Path("/home/robh/1_Projects/pyvale-vfm-test-data/notched-weld-data")
EXPERIMENT = DATASET / "prepared" / "experiment_data.yaml"
KNOWN_PARAMETERS = DATASET / "fe-data" / "raw" / "known_parameter_maps.npz"
DEFAULT_OUTPUT = Path("dev/vfm/output/notched_weld_metric_comparison")
EGI_WINDOWS = ((29, 29), (57, 57))
EGI_WEIGHTS = (29.0, 57.0)


@dataclass(slots=True)
class _MapParameterisation(ISpatialParameterisation):
    """Exact map with an optional dummy DOF for constitutive SBVF diagnostics."""

    parameter_map: np.ndarray
    active: bool = False

    def get_num_degrees_of_freedom(self) -> int:
        return int(self.active)

    def initialise_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter,
    ) -> None:
        self.parameter_map = np.asarray(constitutive_parameter.map, dtype=np.float64)

    def to_map(self, size: np.ndarray) -> np.ndarray:
        if tuple(size) != self.parameter_map.shape:
            raise ValueError("Requested map size does not match stored parameter map.")
        return self.parameter_map.copy()

    def collect_degrees_of_freedom(self) -> list[DegreeOfFreedom]:
        return [DegreeOfFreedom(0.0, -1.0, 1.0)] if self.active else []

    def update_from_degrees_of_freedom(self, degrees_of_freedom) -> None:
        if len(degrees_of_freedom) != self.get_num_degrees_of_freedom():
            raise ValueError("Unexpected number of diagnostic degrees of freedom.")


@dataclass(slots=True)
class MethodResult:
    name: str
    short_name: str
    result_path: Path
    runtime_seconds: float
    total_evaluations: int
    phase_2_accepted_evaluations: int
    convergence: str
    active_dofs: int
    basis_count: int
    sbvf_rms: float
    sbvf_residual_count: int
    identification_sbvf_rms: float | None
    fre_percent: float
    egi_values: tuple[float, float]
    combined_egi: float
    yield_rmse_mpa: float
    yield_mae_mpa: float
    yield_mean_abs_percent_error: float
    yield_max_abs_percent_error: float
    hardening_mpa: float
    hardening_percent_error: float
    yield_map: np.ndarray
    yield_percent_error_map: np.ndarray

    def table_row(self) -> dict[str, object]:
        return {
            "method": self.name,
            "runtime_seconds": self.runtime_seconds,
            "total_evaluations": self.total_evaluations,
            "phase_2_accepted_evaluations": self.phase_2_accepted_evaluations,
            "convergence": self.convergence,
            "active_dofs": self.active_dofs,
            "basis_count": self.basis_count,
            "final_sbvf_rms": self.sbvf_rms,
            "sbvf_residual_count": self.sbvf_residual_count,
            "identification_sbvf_rms": self.identification_sbvf_rms,
            "fre_percent": self.fre_percent,
            "egi_29": self.egi_values[0],
            "egi_57": self.egi_values[1],
            "combined_egi": self.combined_egi,
            "yield_rmse_mpa": self.yield_rmse_mpa,
            "yield_mae_mpa": self.yield_mae_mpa,
            "yield_mean_abs_percent_error": self.yield_mean_abs_percent_error,
            "yield_max_abs_percent_error": self.yield_max_abs_percent_error,
            "hardening_mpa": self.hardening_mpa,
            "hardening_percent_error": self.hardening_percent_error,
        }


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    experiment_data = ExperimentData.load_from_file(args.experiment)
    with np.load(args.known_parameters) as known_file:
        known_maps = {name: known_file[name] for name in known_file.files}
    mask = experiment_data.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment_data.specimen_geometry.x,
        experiment_data.specimen_geometry.y,
    )
    method_inputs = [
            ("FRE+EGI baseline", "FRE+EGI", args.baseline),
            (
                "Constitutive-parameter SBVF",
                "CP-SBVF",
                args.constitutive_sbvf,
            ),
            ("DOF SBVF (bounded smoke)", "DOF smoke", args.dof_sbvf),
    ]
    if args.scalar_sbvf is not None:
        method_inputs.append(
            ("Scalar pattern-search SBVF", "Scalar-SBVF", args.scalar_sbvf)
        )
    methods = [
        _analyse_method(name, short_name, path, experiment_data, known_maps, mask)
        for name, short_name, path in method_inputs
    ]
    rows = [method.table_row() for method in methods]
    _write_csv(args.output / "comparison.csv", rows)
    (args.output / "comparison.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    report_path = args.output / "notched_weld_metric_comparison.pdf"
    _write_report(report_path, methods, experiment_data, known_maps, mask)
    print(json.dumps({"report": str(report_path), "methods": rows}, indent=2))


def _analyse_method(
    name: str,
    short_name: str,
    result_path: Path,
    experiment_data: ExperimentData,
    known_maps: dict[str, np.ndarray],
    mask: np.ndarray,
) -> MethodResult:
    result = load_identification_result(result_path)
    constitutive_law = load_constitutive_law_from_result(result)
    stress = result.final_stress
    if stress is None:
        stress = constitutive_law.calculate_stress(
            experiment_data.strain,
            result.parameter_maps,
        )
    phase_2 = result.history.phases[-1]
    accepted_solves = [solve for solve in phase_2.solve_results if solve.accepted]
    if not accepted_solves:
        raise ValueError(f"{name} does not contain an accepted phase-2 solve.")
    accepted_solve = accepted_solves[-1]
    phase_2_attempts = "; ".join(
        f"{solve.message} ({'accepted' if solve.accepted else 'rejected'})"
        for solve in phase_2.solve_results
    )
    total_evaluations = sum(
        int(solve.num_evaluations or 0)
        for phase in result.history.phases
        for solve in phase.solve_results
    )
    basis_snapshot = accepted_solve.final_snapshot.spatial_parameterisations[
        "yield_strength"
    ][1]
    basis_count = int(basis_snapshot.summary.get("num_kernels", 0))
    yield_map = np.where(mask, result.parameter_maps["yield_strength"], np.nan)
    known_yield = np.where(mask, known_maps["yield_strength"], np.nan)
    yield_error = yield_map - known_yield
    yield_percent_error = 100.0 * yield_error / known_yield
    valid = np.isfinite(yield_error)

    sbvf_rms, sbvf_count = _compute_common_sbvf(
        experiment_data,
        constitutive_law,
        result.parameter_maps,
        stress,
    )
    fre = compute_force_reconstruction_diagnostics(
        experiment_data,
        stress,
        axis="x",
        num_slices=63,
    )
    phase_0_maps = _phase_0_parameter_maps(result, mask.shape)
    phase_0_stress = constitutive_law.calculate_stress(
        experiment_data.strain,
        phase_0_maps,
    )
    egi_values, combined_egi = _compute_egi(
        experiment_data,
        stress,
        phase_0_stress,
    )
    objective = accepted_solve.final_objective
    identification_sbvf_rms = None
    if "residual_norm" in objective and "residual_size" in objective:
        identification_sbvf_rms = float(objective["residual_norm"]) / np.sqrt(
            int(objective["residual_size"])
        )
    elif (
        name != "FRE+EGI baseline"
        and accepted_solve.optimiser.type_name == "OptimiserPatternSearch"
    ):
        identification_sbvf_rms = float(objective["cost"])

    hardening = float(np.nanmean(result.parameter_maps["hardening_modulus"][mask]))
    known_hardening = float(np.nanmean(known_maps["hardening_modulus"][mask]))
    return MethodResult(
        name=name,
        short_name=short_name,
        result_path=result_path,
        runtime_seconds=float(result.metadata.run.runtime_seconds or np.nan),
        total_evaluations=total_evaluations,
        phase_2_accepted_evaluations=int(accepted_solve.num_evaluations or 0),
        convergence=phase_2_attempts,
        active_dofs=len(accepted_solve.final_dofs),
        basis_count=basis_count,
        sbvf_rms=sbvf_rms,
        sbvf_residual_count=sbvf_count,
        identification_sbvf_rms=identification_sbvf_rms,
        fre_percent=float(100.0 * fre.metric_result.weighted_spatiotemporal_rms),
        egi_values=egi_values,
        combined_egi=combined_egi,
        yield_rmse_mpa=float(np.sqrt(np.mean(yield_error[valid] ** 2))),
        yield_mae_mpa=float(np.mean(np.abs(yield_error[valid]))),
        yield_mean_abs_percent_error=float(
            np.mean(np.abs(yield_percent_error[valid]))
        ),
        yield_max_abs_percent_error=float(
            np.max(np.abs(yield_percent_error[valid]))
        ),
        hardening_mpa=hardening,
        hardening_percent_error=100.0 * (hardening - known_hardening) / known_hardening,
        yield_map=yield_map,
        yield_percent_error_map=yield_percent_error,
    )


def _compute_common_sbvf(
    experiment_data: ExperimentData,
    constitutive_law,
    parameter_maps: dict[str, np.ndarray],
    stress: np.ndarray,
) -> tuple[float, int]:
    spatial_parameterisations = {
        name: [
            _MapParameterisation(
                np.asarray(parameter_map, dtype=np.float64),
                active=name in {"yield_strength", "hardening_modulus"},
            )
        ]
        for name, parameter_map in parameter_maps.items()
    }
    metric = MetricSBVF(
        mesh_size=np.asarray((15, 15), dtype=np.uint32),
        vf_scaling_fraction=0.3,
        perturbation_type="constitutive_parameter",
        perturbation_factor=0.15,
    )
    metric.initialise(experiment_data)
    metric_result = metric.evaluate(
        stress,
        constitutive_law,
        np.asarray(stress.shape[2:], dtype=np.uint32),
        spatial_parameterisations,
        experiment_data,
    )
    residual = np.asarray(metric_result.residual, dtype=np.float64)
    finite = residual[np.isfinite(residual)]
    return float(np.sqrt(np.mean(finite**2))), int(finite.size)


def _compute_egi(
    experiment_data: ExperimentData,
    stress: np.ndarray,
    baseline_stress: np.ndarray,
) -> tuple[tuple[float, float], float]:
    candidate_results = []
    baseline_values = []
    scalar_values = []
    for window in EGI_WINDOWS:
        metric = EquilibriumGapMetric(window_size=window)
        metric.initialise(experiment_data)
        candidate = metric.evaluate_equilibrium_gap(stress).metric_result
        baseline = metric.evaluate_equilibrium_gap(baseline_stress).metric_result
        candidate_results.append(candidate)
        scalar_values.append(
            float(candidate.additional_fields["weighted_spatiotemporal_rms"])
        )
        baseline_values.append(
            calculate_nan_rms(extract_equilibrium_gap_temporal_rms(baseline))
        )
    combined = aggregate_equilibrium_gap_results(
        candidate_results,
        egi_baseline_values=baseline_values,
        window_weights=EGI_WEIGHTS,
    )
    return (scalar_values[0], scalar_values[1]), combined.combined_egi_spatial_rms


def _phase_0_parameter_maps(result, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    snapshot = result.history.phases[0].final_snapshot
    maps = {}
    for name, parameter_map in result.parameter_maps.items():
        parameterisation = snapshot.spatial_parameterisations[name][0]
        if parameterisation.dof_values:
            maps[name] = np.full(shape, parameterisation.dof_values[0])
        else:
            maps[name] = np.asarray(parameter_map, dtype=np.float64)
    return maps


def _write_report(
    output_path: Path,
    methods: list[MethodResult],
    experiment_data: ExperimentData,
    known_maps: dict[str, np.ndarray],
    mask: np.ndarray,
) -> None:
    x = experiment_data.specimen_geometry.x
    y = experiment_data.specimen_geometry.y
    extent = (np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y))
    known_yield = np.where(mask, known_maps["yield_strength"], np.nan)
    with PdfPages(output_path) as pdf:
        _report_title_page(pdf, methods)
        _report_table_page(pdf, methods)
        _report_yield_maps(pdf, methods, known_yield, extent)
        _report_error_maps(pdf, methods, extent)
        _report_bar_charts(pdf, methods)
        _report_findings(pdf, methods)


def _report_title_page(pdf: PdfPages, methods: list[MethodResult]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.06, 0.88, "Notched-weld VFM metric comparison", fontsize=24, weight="bold")
    fig.text(
        0.06,
        0.80,
        "Completed and bounded identification runs available for comparison",
        fontsize=13,
    )
    completed = [method for method in methods if "smoke" not in method.name.lower()]
    fastest = min(completed, key=lambda item: item.runtime_seconds)
    most_accurate = min(completed, key=lambda item: item.yield_rmse_mpa)
    best_force = min(completed, key=lambda item: item.fre_percent)
    lines = [
        "Comparison basis",
        "• Same prepared notched-weld FE dataset and bivariate Gaussian yield parameterisation.",
        "• Known yield-strength and hardening fields provide accuracy references.",
        "• FRE, EGI and SBVF diagnostics are recomputed consistently from each final accepted map.",
        "• SBVF RMS is a common constitutive-parameter diagnostic (2 fields × 32 steps = 64 residuals).",
        "",
        f"Fastest completed run: {fastest.name} ({fastest.runtime_seconds / 60:.1f} min)",
        f"Lowest yield RMSE: {most_accurate.name} ({most_accurate.yield_rmse_mpa:.2f} MPa)",
        f"Lowest FRE: {best_force.name} ({best_force.fre_percent:.3f}%)",
    ]
    fig.text(0.08, 0.65, "\n".join(lines), fontsize=12, va="top", linespacing=1.45)
    fig.text(0.06, 0.06, "Generated from saved identification bundles and final stress fields.", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _report_table_page(pdf: PdfPages, methods: list[MethodResult]) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title("Identification metrics and accuracy", fontsize=18, pad=18)
    headers = [
        "Method", "Time\n[min]", "Evals", "DOFs", "Bases", "SBVF\nRMS",
        "FRE\n[%]", "EGI-29", "EGI-57", "Combined\nEGI", "Yield RMSE\n[MPa]",
        "Yield MAPE\n[%]", "H [MPa]", "H error\n[%]",
    ]
    rows = [
        [
            method.short_name,
            f"{method.runtime_seconds / 60:.1f}",
            str(method.total_evaluations),
            str(method.active_dofs),
            str(method.basis_count),
            f"{method.sbvf_rms:.3e}",
            f"{method.fre_percent:.3f}",
            f"{method.egi_values[0]:.3e}",
            f"{method.egi_values[1]:.3e}",
            f"{method.combined_egi:.3f}",
            f"{method.yield_rmse_mpa:.2f}",
            f"{method.yield_mean_abs_percent_error:.2f}",
            f"{method.hardening_mpa:.1f}",
            f"{method.hardening_percent_error:+.2f}",
        ]
        for method in methods
    ]
    table = ax.table(cellText=rows, colLabels=headers, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 2.0)
    for column in range(len(headers)):
        table[(0, column)].set_facecolor("#d9e8f5")
        table[(0, column)].set_text_props(weight="bold")
    notes = [
        f"{method.short_name}: {method.convergence}"
        for method in methods
    ]
    ax.text(0.01, 0.42, "Convergence/acceptance history", weight="bold", fontsize=11)
    ax.text(0.01, 0.38, "\n".join(notes), va="top", fontsize=9, linespacing=1.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _report_yield_maps(
    pdf: PdfPages,
    methods: list[MethodResult],
    known_yield: np.ndarray,
    extent,
) -> None:
    maps = [("Known", known_yield), *[(method.short_name, method.yield_map) for method in methods]]
    vmin = min(np.nanmin(values) for _, values in maps)
    vmax = max(np.nanmax(values) for _, values in maps)
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), constrained_layout=True)
    image = None
    for axis, (title, values) in zip(axes.flat, maps, strict=False):
        image = axis.imshow(values, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
    for axis in axes.flat[len(maps):]:
        axis.axis("off")
    fig.suptitle("Known and identified yield-strength fields", fontsize=18)
    fig.colorbar(image, ax=axes, label="Yield strength [MPa]", shrink=0.8)
    pdf.savefig(fig)
    plt.close(fig)


def _report_error_maps(pdf: PdfPages, methods: list[MethodResult], extent) -> None:
    limit = max(
        np.nanpercentile(np.abs(method.yield_percent_error_map), 99.5)
        for method in methods
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    image = None
    for axis, method in zip(axes.flat, methods):
        image = axis.imshow(
            method.yield_percent_error_map,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        axis.set_title(
            f"{method.short_name}: RMSE {method.yield_rmse_mpa:.2f} MPa, "
            f"MAPE {method.yield_mean_abs_percent_error:.2f}%"
        )
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
    for axis in axes.flat[len(methods):]:
        axis.axis("off")
    fig.suptitle("Yield-strength percentage error: 100 × (identified − known) / known", fontsize=16)
    fig.colorbar(image, ax=axes, label="Yield-strength error [%]", shrink=0.82)
    pdf.savefig(fig)
    plt.close(fig)


def _report_bar_charts(pdf: PdfPages, methods: list[MethodResult]) -> None:
    labels = [method.short_name for method in methods]
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    quantities = (
        ([method.runtime_seconds / 60 for method in methods], "Runtime [min]"),
        ([method.yield_rmse_mpa for method in methods], "Yield-strength RMSE [MPa]"),
        ([method.fre_percent for method in methods], "Force reconstruction RMS [%]"),
        ([abs(method.hardening_percent_error) for method in methods], "Absolute hardening error [%]"),
    )
    for axis, (values, title) in zip(axes.flat, quantities, strict=True):
        bars = axis.bar(labels, values, color=("#5276a7", "#5f9e6e", "#c17c54", "#8c6bb1"))
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    fig.suptitle("Timing and accuracy trade-offs", fontsize=18)
    pdf.savefig(fig)
    plt.close(fig)


def _report_findings(pdf: PdfPages, methods: list[MethodResult]) -> None:
    completed = [method for method in methods if "smoke" not in method.name.lower()]
    fastest = min(completed, key=lambda item: item.runtime_seconds)
    yield_rank = sorted(completed, key=lambda item: item.yield_rmse_mpa)
    force_rank = sorted(completed, key=lambda item: item.fre_percent)
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.06, 0.90, "Findings", fontsize=22, weight="bold")
    findings = [
        f"• {fastest.name} was the fastest completed run at {fastest.runtime_seconds / 60:.1f} min.",
        f"• Yield accuracy ranked: " + " < ".join(
            f"{method.short_name} ({method.yield_rmse_mpa:.2f} MPa)" for method in yield_rank
        ) + ".",
        f"• Force consistency ranked: " + " < ".join(
            f"{method.short_name} ({method.fre_percent:.3f}%)" for method in force_rank
        ) + ".",
        "• Constitutive-parameter SBVF keeps 64 residuals as bases grow; DOF-SBVF grows the residual vector with active DOFs.",
        "• EGI refinement can reject a basis even when the SBVF objective improves, preventing identification-objective overfitting from driving spatial complexity.",
        "• FRE was diagnostic-only for SBVF methods; its final value therefore provides an independent equilibrium check.",
        "• DOF-SBVF is a one-evaluation-per-solve smoke result because the production run was stopped after excessive runtime; it is not a converged accuracy result.",
        "• Scalar pattern-search SBVF was not run because the requested compute budget was curtailed.",
        "",
        "Interpretation caution",
        "The methods use different objective geometries and stopping conditions. Runtime and evaluation counts should be considered together; one evaluation does not have identical cost across methods.",
    ]
    fig.text(0.08, 0.82, "\n".join(findings), fontsize=12, va="top", linespacing=1.55)
    fig.text(0.06, 0.06, "Machine-readable values are saved beside this PDF in comparison.csv and comparison.json.", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--known-parameters", type=Path, default=KNOWN_PARAMETERS)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--constitutive-sbvf", type=Path, required=True)
    parser.add_argument("--dof-sbvf", type=Path, required=True)
    parser.add_argument("--scalar-sbvf", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main()

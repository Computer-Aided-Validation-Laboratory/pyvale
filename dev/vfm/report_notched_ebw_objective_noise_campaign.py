"""Create a concise PDF for the Notched-EBW objective/noise screen."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from pyvale.vfm import ExperimentData, load_identification_result
from pyvale.vfm.postprocessing import (
    evaluate_snapshot_parameter_maps,
    load_known_parameter_maps,
)


DISPLAY = {
    "current": "Current\n29/57 length",
    "multiscale_length": "Multiscale\nlength weighted",
    "multiscale_equal": "Multiscale\nequal",
    "fine_emphasis": "Fine-scale\nemphasis",
    "broad_fre_guard": "Broad/FRE\nguard",
    "sensitivity_equal": "Multiscale equal\n+ sensitivity",
}


def _load(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric = {
        "noise_scale", "force_weight", "basis_count", "optimised_cost",
        "common_clean_objective", "roi_rmse_mpa", "yielded_rmse_mpa",
        "high_plastic_rmse_mpa", "yielded_mape_percent",
        "yielded_above_10pct", "hardening_error_percent",
    }
    return [
        {key: float(value) if key in numeric else value for key, value in row.items()}
        for row in rows
    ]


def _page(pdf: PdfPages, title: str, subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    ax = fig.add_axes((0.065, 0.075, 0.87, 0.82))
    ax.axis("off")
    fig.text(0.065, 0.94, title, fontsize=22, weight="bold", color="#17324d")
    if subtitle:
        fig.text(0.065, 0.905, subtitle, fontsize=10.5, color="#536878")
    return fig, ax


def _save(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _title_page(pdf: PdfPages, rows: list[dict[str, object]]) -> None:
    fig, ax = _page(pdf, "Notched-EBW objective and noise screen")
    clean = [r for r in rows if r["condition"] == "clean"]
    noisy = [r for r in rows if r["condition"] == "noise"]
    best_clean = min(clean, key=lambda r: r["yielded_rmse_mpa"])
    best_noisy = min(noisy, key=lambda r: r["yielded_rmse_mpa"])
    ax.text(0, 0.88, "12 completed identifications: six objective variants × clean/artificial-noise data",
            fontsize=14, weight="bold", transform=ax.transAxes)
    ax.text(0, 0.74,
            "Purpose\nScreen whether additional EGI spatial scales and frozen parameter-sensitivity weighting "
            "improve recovery of the known synthetic property map.", fontsize=12, linespacing=1.55,
            transform=ax.transAxes)
    ax.text(0.03, 0.47,
            f"BEST CLEAN\n{DISPLAY[str(best_clean['objective'])].replace(chr(10), ' ')}\n"
            f"{best_clean['yielded_rmse_mpa']:.2f} MPa yielded RMSE",
            fontsize=15, weight="bold", color="#176b4d", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.8", fc="#e8f5ef", ec="#83b9a4"))
    ax.text(0.53, 0.47,
            f"BEST WITH NOISE\n{DISPLAY[str(best_noisy['objective'])].replace(chr(10), ' ')}\n"
            f"{best_noisy['yielded_rmse_mpa']:.2f} MPa yielded RMSE",
            fontsize=15, weight="bold", color="#174f7a", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.8", fc="#e9f2f8", ec="#8cb6d2"))
    ax.text(0, 0.22,
            "Headline: multiscale information helps. Equal 15/29/57 weighting performs best cleanly; "
            "the same scales with frozen sensitivity weighting are most robust to the imposed noise. "
            "This nominates two formulations for replicated validation—it does not yet justify a production change.",
            fontsize=12.5, linespacing=1.5, transform=ax.transAxes, wrap=True)
    ax.text(0, 0.03, f"Generated {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}",
            fontsize=9, color="#687680", transform=ax.transAxes)
    _save(pdf, fig)


def _comparison_page(pdf: PdfPages, rows: list[dict[str, object]]) -> None:
    order = list(DISPLAY)
    clean = {r["objective"]: r for r in rows if r["condition"] == "clean"}
    noisy = {r["objective"]: r for r in rows if r["condition"] == "noise"}
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.suptitle("Property-map accuracy: clean versus noisy data", x=0.06, ha="left",
                 fontsize=21, weight="bold", color="#17324d")
    axes = fig.subplots(1, 2, gridspec_kw={"left": 0.07, "right": 0.97, "bottom": 0.18,
                                           "top": 0.84, "wspace": 0.25})
    x = np.arange(len(order)); width = 0.36
    metrics = [("yielded_rmse_mpa", "Yielded-region RMSE [MPa]"),
               ("high_plastic_rmse_mpa", "High-plastic-region RMSE [MPa]")]
    for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
        a = [clean[name][metric] for name in order]
        b = [noisy[name][metric] for name in order]
        ax.bar(x - width/2, a, width, label="Clean", color="#3a7ca5")
        ax.bar(x + width/2, b, width, label="Artificial noise", color="#d9822b")
        ax.set_xticks(x, [DISPLAY[name] for name in order], fontsize=8)
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
        ax.legend(frameon=False)
    fig.text(0.07, 0.07,
             "Lower is better. The current objective suffers the largest yielded-region degradation "
             "(+29.30 MPa); sensitivity-equal has the smallest (+7.70 MPa).",
             fontsize=11, color="#334e60")
    _save(pdf, fig)


def _table_page(pdf: PdfPages, rows: list[dict[str, object]]) -> None:
    fig, ax = _page(pdf, "Numerical summary", "Errors are evaluated offline against the known synthetic property map.")
    order = list(DISPLAY)
    clean = {r["objective"]: r for r in rows if r["condition"] == "clean"}
    noisy = {r["objective"]: r for r in rows if r["condition"] == "noise"}
    columns = ["Objective", "Clean yield", "Noisy yield", "Noise Δ", "Clean high-P", "Noisy high-P", "Noisy >10%"]
    cells = []
    for name in order:
        c, n = clean[name], noisy[name]
        cells.append([
            DISPLAY[name].replace("\n", " "), f"{c['yielded_rmse_mpa']:.2f}",
            f"{n['yielded_rmse_mpa']:.2f}", f"{n['yielded_rmse_mpa']-c['yielded_rmse_mpa']:+.2f}",
            f"{c['high_plastic_rmse_mpa']:.2f}", f"{n['high_plastic_rmse_mpa']:.2f}",
            f"{100*n['yielded_above_10pct']:.1f}%",
        ])
    table = ax.table(cellText=cells, colLabels=columns, cellLoc="center", colLoc="center",
                     bbox=(0, 0.36, 1, 0.52), colWidths=[0.24, 0.12, 0.12, 0.11, 0.13, 0.13, 0.12])
    table.auto_set_font_size(False); table.set_fontsize(9.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#c7d2d9")
        if row == 0:
            cell.set_facecolor("#17324d"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f5f7")
    ax.text(0, 0.26, "All values except the final column are MPa. Every solve reached the seven-BF cap.",
            fontsize=10.5, transform=ax.transAxes)
    ax.text(0, 0.14,
            "Optimised costs are not compared across formulations because their definitions differ. "
            "The reported map errors and recomputed common clean objective are comparable.",
            fontsize=11, linespacing=1.45, transform=ax.transAxes, wrap=True)
    _save(pdf, fig)


def _interpretation_page(pdf: PdfPages) -> None:
    fig, ax = _page(pdf, "Interpretation and next decision")
    sections = [
        ("What the results support", [
            "Adding the 15-pixel scale is beneficial: both leading formulations use 15/29/57 windows.",
            "Equal scale weighting is a strong clean-data candidate (30.34 MPa yielded RMSE).",
            "Frozen sensitivity weighting materially improves noise robustness (40.16 MPa noisy yielded RMSE; +7.70 MPa penalty).",
            "Heavy broad/FRE weighting is not supported in this form; it is poorest cleanly and remains poor with noise.",
        ]),
        ("What remains unresolved", [
            "One optimiser seed cannot separate a genuinely better objective from stochastic path dependence.",
            "High-plastic errors remain 69–70 MPa even for the best noisy candidates.",
            "The artificial-noise model is representative of WDBN1 statistics but does not cover all experimental artefacts.",
            "This campaign changes spatial-scale weighting; it does not yet test late-stage temporal or residual-projection components directly.",
        ]),
        ("Smallest useful next study", [
            "Replicate current, multiscale-equal and sensitivity-equal over several matched seeds, clean and noisy.",
            "Retain seven or eight BFs and assess BF5–8 trajectories, not only final states.",
            "Use yielded/high-plastic map error only offline; compare objective ranking and pairwise discrimination at late stages.",
            "If sensitivity-equal remains robust, test a bounded sensitivity contribution rather than replacing EGI/FRE.",
        ]),
    ]
    y = 0.88
    for heading, bullets in sections:
        ax.text(0, y, heading, fontsize=14, weight="bold", color="#17324d", transform=ax.transAxes)
        y -= 0.055
        for bullet in bullets:
            ax.text(0.025, y, f"• {bullet}", fontsize=11.2, transform=ax.transAxes, va="top", wrap=True)
            y -= 0.073
        y -= 0.025
    ax.text(0, 0.015,
            "Recommendation: carry multiscale-equal and sensitivity-equal forward; retain the current objective as the control.",
            fontsize=12.5, weight="bold", color="#176b4d", transform=ax.transAxes)
    _save(pdf, fig)


def _map_data(
    analysis: Path, rows: list[dict[str, object]], dataset: Path
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, list[float]]:
    experiment = ExperimentData.load_from_file(dataset / "prepared/experiment_data.yaml")
    known = load_known_parameter_maps(dataset / "prepared/known_parameter_maps.npz")
    if known is None:
        raise RuntimeError("Known synthetic parameter maps are required for map pages.")
    geometry = experiment.specimen_geometry
    mask = geometry.region_of_interest.sample_specimen_mask(geometry.x, geometry.y)
    truth = np.asarray(known["yield_strength"], dtype=float)
    campaign = analysis.parent
    selected = [
        ("Current — clean", "current", "clean"),
        ("Multiscale equal — clean", "multiscale_equal", "clean"),
        ("Current — noisy", "current", "noise"),
        ("Sensitivity equal — noisy", "sensitivity_equal", "noise"),
    ]
    cases = []
    for title, objective, condition in selected:
        row = next(
            item for item in rows
            if item["objective"] == objective and item["condition"] == condition
        )
        suffix = "clean" if condition == "clean" else "noise"
        path = campaign / f"obj_{objective}_{suffix}_seed00/final_parameter_maps.npz"
        with np.load(path) as source:
            identified = np.asarray(source["yield_strength"], dtype=float)
        cases.append({
            "title": title,
            "map": identified,
            "row": row,
            "result_path": campaign / f"obj_{objective}_{suffix}_seed00/identification_result.yaml",
        })
    extent = [
        float(np.nanmin(geometry.x)), float(np.nanmax(geometry.x)),
        float(np.nanmin(geometry.y)), float(np.nanmax(geometry.y)),
    ]
    return cases, truth, np.asarray(mask, dtype=bool), extent


def _yield_map_page(
    pdf: PdfPages, cases: list[dict[str, object]], truth: np.ndarray,
    mask: np.ndarray, extent: list[float]
) -> None:
    values = [truth, *[np.asarray(case["map"]) for case in cases]]
    vmin = min(float(np.nanmin(value[mask])) for value in values)
    vmax = max(float(np.nanmax(value[mask])) for value in values)
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), constrained_layout=True)
    fig.suptitle("Known and identified yield-strength maps", fontsize=20, weight="bold", color="#17324d")
    panels = [("Known synthetic map", truth), *[(str(case["title"]), case["map"]) for case in cases]]
    image = None
    for ax, (title, values_i) in zip(axes.flat, panels, strict=False):
        image = ax.imshow(np.where(mask, values_i, np.nan), origin="lower", extent=extent,
                          aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11); ax.set(xlabel="x [mm]", ylabel="y [mm]")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.colorbar(image, ax=axes, label="Yield strength [MPa]", shrink=0.82)
    _save(pdf, fig)


def _error_map_page(
    pdf: PdfPages, cases: list[dict[str, object]], truth: np.ndarray,
    mask: np.ndarray, extent: list[float]
) -> None:
    errors = [100.0 * (np.asarray(case["map"]) - truth) / truth for case in cases]
    limit = max(15.0, max(float(np.nanpercentile(np.abs(error[mask]), 99.0)) for error in errors))
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    fig.suptitle("Yield-strength percentage-error maps", fontsize=20, weight="bold", color="#17324d")
    image = None
    for ax, case, error in zip(axes.flat, cases, errors, strict=True):
        image = ax.imshow(np.where(mask, error, np.nan), origin="lower", extent=extent,
                          aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        row = case["row"]
        ax.set_title(
            f"{case['title']}\nYielded RMSE {row['yielded_rmse_mpa']:.2f} MPa; "
            f"yielded MAPE {row['yielded_mape_percent']:.2f}%",
            fontsize=10.5,
        )
        ax.set(xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(image, ax=axes, label="100 × (identified − known) / known [%]", shrink=0.82)
    _save(pdf, fig)


def _snapshot_basis_count(snapshot) -> int:
    for item in snapshot.spatial_parameterisations.get("yield_strength", []):
        summary = item.summary
        if summary.get("kind") == "basis_functions":
            return int(summary.get("num_kernels", len(summary.get("kernels", []))))
    return 0


def _solve_evolution_page(
    pdf: PdfPages,
    case: dict[str, object],
    experiment: ExperimentData,
    truth: np.ndarray,
    mask: np.ndarray,
    extent: list[float],
) -> None:
    result = load_identification_result(Path(case["result_path"]))
    solves = [
        solve for solve in result.history.phases[-1].solve_results
        if solve.final_snapshot is not None
    ]
    fig, axes = plt.subplots(2, 4, figsize=(11.69, 8.27), constrained_layout=True)
    fig.suptitle(
        f"Yield-strength error through basis growth: {case['title']}",
        fontsize=19, weight="bold", color="#17324d",
    )
    image = None
    for ax, solve in zip(axes.flat, solves, strict=False):
        partial = evaluate_snapshot_parameter_maps(solve.final_snapshot, experiment)
        identified = np.asarray(partial.get("yield_strength", truth), dtype=float)
        error = 100.0 * (identified - truth) / truth
        basis_count = _snapshot_basis_count(solve.final_snapshot)
        cost = float(solve.final_objective.get("cost", np.nan))
        image = ax.imshow(
            np.where(mask, error, np.nan), origin="lower", extent=extent,
            aspect="auto", cmap="RdBu_r", vmin=-20.0, vmax=20.0,
        )
        status = "accepted" if solve.accepted else "rejected"
        ax.set_title(
            f"Solve {int(solve.solve_iteration) + 1} · {basis_count} BF\n"
            f"J={cost:.4f} · {status}", fontsize=9.5,
        )
        ax.set(xlabel="x [mm]", ylabel="y [mm]")
    for ax in axes.flat[len(solves):]:
        ax.axis("off")
    fig.colorbar(
        image, ax=axes, label="100 × (identified − known) / known [%]",
        ticks=[-20, -10, 0, 10, 20], shrink=0.82, extend="both",
    )
    _save(pdf, fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analysis = args.analysis.resolve()
    output = args.output or analysis / "NOTCHED_EBW_OBJECTIVE_NOISE_CAMPAIGN.pdf"
    rows = _load(analysis / "objective_noise_summary.csv")
    dataset = args.dataset.resolve() if args.dataset else analysis.parents[3]
    cases, truth, mask, extent = _map_data(analysis, rows, dataset)
    experiment = ExperimentData.load_from_file(dataset / "prepared/experiment_data.yaml")
    with PdfPages(output) as pdf:
        _title_page(pdf, rows)
        _comparison_page(pdf, rows)
        _table_page(pdf, rows)
        _yield_map_page(pdf, cases, truth, mask, extent)
        _error_map_page(pdf, cases, truth, mask, extent)
        for case in cases:
            _solve_evolution_page(pdf, case, experiment, truth, mask, extent)
        _interpretation_page(pdf)
        metadata = pdf.infodict()
        metadata["Title"] = "Notched-EBW objective and noise screen"
        metadata["Author"] = "PyVale VFM investigation"
        metadata["Subject"] = "Clean/noisy synthetic identification objective comparison"
    print(output)


if __name__ == "__main__":
    main()

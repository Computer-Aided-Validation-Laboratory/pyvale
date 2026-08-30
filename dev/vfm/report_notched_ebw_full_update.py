"""Create the chronological notched-EBW synthetic-identification overview."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
import numpy as np

from pyvale.vfm import ExperimentData, load_identification_result
from pyvale.vfm.postprocessing import (
    compute_plasticity_diagnostics,
    evaluate_snapshot_parameter_maps,
    load_constitutive_law_from_result,
    load_known_parameter_maps,
)


DATASET = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/"
    "synthetic-fe/wdbn1-idealised-yield/pyvale-vfm"
)
CAMPAIGN = DATASET / "identification/prepared/gate_objective_campaign_20260828"
REPORTS = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/investigation-reports"
)
BEST_CASE = "spd_sensitivity_gate0p0pct_seed00"
BEST_SOLVE = 6

INK = "#18212b"
MUTED = "#596773"
BLUE = "#2369a1"
ORANGE = "#d87520"
GREEN = "#3b7d44"
RED = "#b23a3a"
LIGHT = "#eaf0f5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--reports", type=Path, default=REPORTS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Matplotlib's PDF date serialiser treats aware datetimes as local twice on
    # this host.  A naive local value keeps both the printed and PDF metadata
    # timestamps aligned with the investigation workstation clock.
    generated = datetime.now()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bundle = _load_bundle(args.dataset, args.campaign)
    sources = _report_register(args.reports)

    with PdfPages(
        args.output,
        metadata={
            "Title": "Notched-EBW synthetic VFM identification: consolidated update",
            "Author": "PyVale investigation",
            "CreationDate": generated,
            "Subject": "Chronological synthesis of synthetic inverse-identification investigations",
        },
    ) as pdf:
        _cover(pdf, generated)
        _current_position(pdf)
        _method_page(pdf)
        _chronology_page(pdf)
        _capacity_page(pdf)
        _watering_page(pdf, args.dataset)
        _factorial_page(pdf)
        _campaign_page(pdf, bundle["rows"])
        _best_map_page(pdf, bundle)
        _selector_page(pdf, args.campaign)
        _component_page(pdf, args.campaign)
        _sensitivity_page(pdf, args.campaign)
        _experimental_page(pdf)
        _overnight_page(pdf)
        _next_steps_page(pdf)
        _source_page(pdf, sources)

    print(args.output)


def _load_bundle(dataset: Path, campaign: Path) -> dict[str, object]:
    prepared = dataset / "prepared"
    experiment = ExperimentData.load_from_file(prepared / "experiment_data.yaml")
    known = load_known_parameter_maps(prepared / "known_parameter_maps.npz")
    if known is None:
        raise RuntimeError("Known synthetic maps are required.")
    result = load_identification_result(
        campaign / BEST_CASE / "identification_result.yaml"
    )
    solve = next(
        item
        for item in result.history.phases[-1].solve_results
        if int(item.solve_iteration) == BEST_SOLVE
    )
    if solve.final_snapshot is None:
        raise RuntimeError("Selected solve has no final snapshot.")
    partial = evaluate_snapshot_parameter_maps(solve.final_snapshot, experiment)
    maps = {name: np.asarray(value, dtype=float) for name, value in known.items()}
    maps.update({name: np.asarray(value, dtype=float) for name, value in partial.items()})
    geometry = experiment.specimen_geometry
    mask = geometry.region_of_interest.sample_specimen_mask(geometry.x, geometry.y)
    plasticity = compute_plasticity_diagnostics(
        experiment, load_constitutive_law_from_result(result), known
    )
    if plasticity is None:
        raise RuntimeError("Plasticity diagnostics are unavailable.")
    yielded = np.asarray(plasticity.yielded_datapoints, dtype=bool) & mask
    peak = np.nanmax(np.asarray(plasticity.equivalent_plastic_strain), axis=0)
    high = yielded & (peak >= np.nanpercentile(peak[yielded], 75.0))
    kernels = []
    for item in solve.final_snapshot.spatial_parameterisations["yield_strength"]:
        kernels.extend(item.summary.get("kernels", []))
    with (campaign / "analysis/state_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    selected = next(
        row
        for row in rows
        if row["state_id"] == f"{BEST_CASE}/solve_{BEST_SOLVE}"
    )
    return {
        "experiment": experiment,
        "known": known,
        "maps": maps,
        "mask": mask,
        "yielded": yielded,
        "high": high,
        "kernels": kernels,
        "selected": selected,
        "rows": rows,
    }


def _report_register(path: Path) -> list[tuple[str, str]]:
    return [
        (datetime.fromtimestamp(item.stat().st_mtime).astimezone().strftime("%H:%M"), item.name)
        for item in sorted(path.glob("*.pdf"), key=lambda item: item.stat().st_mtime)
    ]


def _page(title: str, subtitle: str | None = None):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    axis = fig.add_axes((0.09, 0.07, 0.82, 0.86))
    axis.axis("off")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.text(0, 1.0, title, fontsize=20, weight="bold", color=INK, va="top")
    if subtitle:
        axis.text(0, 0.958, subtitle, fontsize=9.5, color=MUTED, va="top")
    return fig, axis


def _save(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig)
    plt.close(fig)


def _heading(axis, y: float, text: str, color: str = INK) -> float:
    axis.text(0, y, text, fontsize=13, weight="bold", color=color, va="top")
    return y - 0.038


def _paragraph(axis, y: float, text: str, *, width: int = 102, size: float = 9.5,
               color: str = INK, bold: bool = False, gap: float = 0.018) -> float:
    wrapped = textwrap.wrap(text, width=width, break_long_words=False)
    axis.text(
        0, y, "\n".join(wrapped), fontsize=size, color=color, va="top",
        linespacing=1.32, weight="bold" if bold else "normal",
    )
    return y - 0.0235 * len(wrapped) - gap


def _bullets(axis, y: float, items: list[str], *, width: int = 96,
             size: float = 9.3, color: str = INK) -> float:
    for item in items:
        wrapped = textwrap.wrap(item, width=width, break_long_words=False)
        axis.text(0.008, y, "•", fontsize=size + 2, color=BLUE, va="top")
        axis.text(
            0.035, y, "\n".join(wrapped), fontsize=size, color=color,
            va="top", linespacing=1.3,
        )
        y -= 0.023 * len(wrapped) + 0.016
    return y


def _callout(axis, y: float, text: str, *, color: str = BLUE) -> float:
    wrapped = textwrap.fill(text, width=96)
    axis.text(
        0.02, y, wrapped, fontsize=11.2, weight="bold", color=INK, va="top",
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.65", "facecolor": LIGHT,
              "edgecolor": color, "linewidth": 1.5},
    )
    return y - 0.12


def _cover(pdf, generated) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.11, 0.86, "Notched-EBW synthetic\nVFM identification", fontsize=27,
             weight="bold", color=INK, va="top", linespacing=1.08)
    fig.text(0.11, 0.755, "Consolidated investigation update", fontsize=17,
             color=BLUE, va="top")
    fig.add_artist(plt.Line2D([0.11, 0.89], [0.72, 0.72], color=BLUE, lw=2.2))
    fig.text(0.11, 0.665, f"{generated:%d %B %Y, %H:%M %Z}", fontsize=12, color=MUTED)
    fig.text(
        0.11, 0.56,
        "Purpose\n\n"
        "Bring the full sequence of synthetic-identification investigations into one current, "
        "chronologically reconciled account before experimental processing. Later analyses are "
        "treated as superseding earlier provisional interpretations.",
        fontsize=12, color=INK, va="top", linespacing=1.45, wrap=True,
    )
    fig.text(
        0.11, 0.34,
        "Current status\n\n"
        "The production EGI-29/57 + FRE objective remains unchanged. The leading route is now a "
        "guarded selector that retains metric, spatial scale and load regime, and uses projected "
        "native-DOF sensitivity to separate yield information from hardening and noise.",
        fontsize=12, color=INK, va="top", linespacing=1.45, wrap=True,
    )
    fig.text(0.11, 0.08, "Synthetic WDBN1 idealised-yield case | PyVale VFM", fontsize=9, color=MUTED)
    _save(pdf, fig)


def _current_position(pdf) -> None:
    fig, axis = _page("Executive summary", "Current conclusions after all 13 reports")
    y = 0.89
    y = _callout(
        axis, y,
        "The algorithm has not yet reached experimental-data readiness. It can fit the truth with its "
        "chosen basis family, but the present scalar objective cannot reliably distinguish the best "
        "late-stage material maps; the remaining limitation is information selection, not raw closure.",
    )
    y = _heading(axis, y, "What is established")
    y = _bullets(axis, y, [
        "Representability is adequate: five directly fitted Gaussians reach 6.83 MPa ROI RMSE; fixed oracle-derived geometry reaches about 13 MPa yielded RMSE and recovers hardening near 3979 MPa.",
        "The objective is watered down: 16.06% of valid residual observations are yielded but contain 74.45% of the positive identified-versus-truth gap; 62.74% of the truth objective comes from unyielded observations.",
        "Exact optimiser-coordinate conditioning is about 3.78×10³ with 11/20 directions above 1%—materially better than the superseded 4.36×10⁵ estimate, but still correlated and path dependent.",
        "SPD log-covariance is the preferred Gaussian geometry. Sensitivity-correction growth is physically interpretable, but lower J does not guarantee a lower property-map error.",
        "Across 15 mature runs, median yielded error remains about 39 MPa and high-plastic error 56–57 MPa. BF7 usually helps; BF8 is the first repeatable sign of objective-driven overfit.",
    ])
    y = _heading(axis, y, "Current direction")
    _bullets(axis, y, [
        "Keep EGI/FRE mechanical closure for now, but delay scalar collapse and carry separate evidence by metric, spatial scale and load regime.",
        "Prioritise EGI-57 onset upper-tail evidence, developed-plasticity coherent FRE, and an EGI-29 onset local sentinel; add projected native-DOF sensitivity to distinguish yield from hardening directions.",
        "Use the ongoing realistic-noise/native-projection campaign to decide which components survive experimental noise before modifying the production objective or growth acceptance rule.",
    ])
    _save(pdf, fig)


def _method_page(pdf) -> None:
    fig, axis = _page("Problem and retained production formulation")
    y = 0.90
    y = _paragraph(axis, y, "Goal: recover a spatial yield-strength field and homogeneous hardening from full-field strains and force, using a compact sequence of Gaussian basis functions.")
    columns = ["Layer", "Retained formulation", "Current interpretation"]
    rows = [
        ["Constitutive", "Plane-stress elastoplastic reconstruction", "Native FE/truth closure is internally consistent"],
        ["Local equilibrium", "EGI-29 and EGI-57", "Complementary local/broad spatial scales"],
        ["Global equilibrium", "63-slice FRE", "Cross-sectional and high-plastic information"],
        ["Time weighting", "Force²", "Good SNR bias; can obscure yield onset"],
        ["Geometry", "Bivariate SPD Gaussians", "Removes angle/axis/positivity degeneracies"],
        ["Optimisation", "Pattern search + global refit", "Path/seed dependence remains"],
        ["Scalar", "Weighted RMS EGI + 0.1 FRE", "Valid closure measure, weak late-map selector"],
    ]
    table = axis.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left",
                       bbox=(0, 0.50, 1, 0.30), colWidths=(0.17, 0.36, 0.47))
    _style_table(table, 8.2)
    y = 0.45
    y = _heading(axis, y, "Success criteria")
    y = _bullets(axis, y, [
        "Known synthetic maps are used only for offline discrimination and release decisions—not inside the production objective.",
        "Judge map quality in yielded and upper-quartile plastic regions, with spatial error maps, seed spread and model-order behaviour; do not select by J alone.",
        "Before experiment, require realistic strain/force noise, load-window omission and model-mismatch checks with frozen thresholds.",
    ])
    _callout(axis, y, "Central design question: which scalar or guarded decision rule preserves the material-identification information present in the full EGI/FRE residual fields?")
    _save(pdf, fig)


def _chronology_page(pdf) -> None:
    fig, axis = _page("Chronology and corrected interpretations", "Later evidence supersedes earlier provisional conclusions")
    entries = [
        ("07:35", "Initial consolidation", "Capacity is adequate; free geometry/seed path is the dominant practical problem. Retain EGI-29/57 after inverse tests supersede the metric-only EGI-17 preference."),
        ("08:28–10:34", "Physics and coordinate audit", "Watering-down is directly quantified. The free-geometry condition number is corrected from 4.36×10⁵ to 3.78×10³ in exact log-normalised optimiser coordinates."),
        ("10:53–14:33", "Growth strategy and factorial", "Move from EGI peaks to signed residual sensitivity; prefer SPD covariance. A 5% gate rejects a useful BF5 trial, motivating gate isolation."),
        ("17:32", "15-run gate campaign", "0.5% is effectively inactive; BF7 helps, BF8 often reduces J while worsening the map. A lower percentage gate alone does not solve selection."),
        ("18:10", "Offline scalar replay", "No tested scalar materially improves selected maps. Preserve metric × load-regime evidence instead of forcing an early weighted sum."),
        ("19:58", "Independent component library", "EGI-57 onset P95 and coherent developed FRE provide complementary late-map discrimination; EGI-29 onset P90 is a specialist local sentinel."),
        ("20:54 onward", "Sensitivity-information pilot", "Projection/orthogonalisation is more promising than unsigned magnitude weighting, especially EGI-15 developed-plasticity evidence for BF5–8 high-plastic differences."),
    ]
    y = 0.91
    for time, label, body in entries:
        axis.text(0.0, y, time, fontsize=9.3, color=BLUE, weight="bold", va="top")
        axis.text(0.11, y, label, fontsize=10.2, color=INK, weight="bold", va="top")
        wrapped = textwrap.wrap(body, width=82)
        axis.text(0.11, y - 0.027, "\n".join(wrapped), fontsize=8.8, color=INK,
                  va="top", linespacing=1.25)
        axis.plot([0.045, 0.045], [y - 0.105, y + 0.005], color="#b8c8d8", lw=1.5)
        axis.scatter([0.045], [y], s=22, color=BLUE, zorder=3)
        y -= 0.122
    _save(pdf, fig)


def _capacity_page(pdf) -> None:
    fig, axis = _page("Capacity and identifiability are different problems")
    chart = fig.add_axes((0.14, 0.52, 0.72, 0.27))
    bases = np.arange(0, 11)
    rmse = [114.84, 39.33, 25.83, 21.39, 21.75, 17.78, 12.54, 12.52, 11.30, 7.12, 6.83]
    chart.plot(bases, rmse, "o-", lw=2, color=BLUE)
    chart.axhline(13, color=GREEN, ls="--", lw=1.5, label="fixed-geometry inverse ≈13 MPa")
    chart.set(xlabel="Direct-fit Gaussian count", ylabel="ROI RMSE [MPa]", xticks=bases,
              title="Direct property-map representability")
    chart.grid(alpha=0.2)
    chart.legend(frameon=False, fontsize=8)
    axis.text(0, 0.40, "Evidence", fontsize=13, weight="bold", color=INK)
    _bullets(axis, 0.355, [
        "Five Gaussians already fit the known map to 17.78 MPa ROI RMSE in the original sequential study; ten reach 6.83 MPa. More capacity can continue improving a direct fit.",
        "With five oracle-derived geometries fixed, inverse optimisation reaches J≈0.0745–0.0749, ≈13 MPa yielded RMSE and H≈3979 MPa. The stress reconstruction and residual pipeline can therefore support much better recovery.",
        "The current free-geometry campaign instead plateaus near 36–39 MPa yielded error. This is an observability, geometry-selection and stopping problem—not proof that the Gaussian family is incapable.",
    ])
    _save(pdf, fig)


def _watering_page(pdf, dataset: Path) -> None:
    fig, axis = _page("Why a good cost can coexist with a poor map", "Exact location-, time- and metric-resolved objective attribution")
    bars = fig.add_axes((0.12, 0.54, 0.36, 0.27))
    bars.bar([0, 1, 2], [0.02170, 0.03802, 0.01982], color="#9ebbd3", label="truth")
    bars.bar([0, 1, 2], [0.00744, 0.00221, 0.00290], bottom=[0.02170, 0.03802, 0.01982], color=ORANGE, label="identified − truth")
    bars.set(xticks=[0, 1, 2], xticklabels=["EGI-29", "EGI-57", "FRE"], ylabel="Objective contribution")
    bars.grid(axis="y", alpha=0.2)
    bars.legend(frameon=False, fontsize=8)
    shares = fig.add_axes((0.57, 0.54, 0.30, 0.27))
    shares.bar([0, 1, 2], [16.06, 74.45, 62.74], color=[BLUE, ORANGE, "#8293a4"])
    shares.set(xticks=[0, 1, 2], xticklabels=["yielded\nobservations", "positive gap\nin yielded rows", "truth J from\nunyielded rows"], ylabel="Share [%]", ylim=(0, 100))
    shares.grid(axis="y", alpha=0.2)
    y = 0.43
    y = _paragraph(axis, y, "The scalar is dominated by a background closure floor, while the difference between correct and incorrect property maps is concentrated in a much smaller active subset. EGI/FRE therefore contain useful information, but the current RMS reduction dilutes it.")
    y = _bullets(axis, y, [
        "EGI-29 contributes 59% of the truth-to-identified gap, EGI-57 18%, and FRE 23%; their spatial roles are not interchangeable.",
        "The known map has J=0.07953, yet mature identified maps can reach J≈0.067–0.069 while retaining ≈36–39 MPa yielded RMSE. Numerical/model discrepancy can be absorbed into the inferred material field.",
        "Sensitivity should distinguish mechanically correctable residual from background closure, but an unsigned weight cannot distinguish yield from hardening or redundant parameter directions.",
    ])
    _save(pdf, fig)


def _factorial_page(pdf) -> None:
    fig, axis = _page("Geometry and growth-policy factorial", "Four controlled cases; production objective otherwise unchanged")
    columns = ["Case", "Geometry", "Growth", "BF", "J", "ROI RMSE", "Yielded RMSE", "H error"]
    rows = [
        ["A", "conventional", "EGI peak", "3", "0.09209", "40.69", "36.97", "−4.26%"],
        ["B", "SPD", "EGI peak", "4", "0.08916", "27.64", "39.95", "−14.49%"],
        ["C", "conventional", "sensitivity", "2", "0.10708", "28.56", "47.20", "−16.39%"],
        ["D", "SPD", "sensitivity", "4", "0.08551", "31.69", "45.31", "−5.04%"],
    ]
    table = axis.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center",
                       bbox=(0, 0.58, 1, 0.24), colWidths=(0.07, 0.17, 0.17, 0.07, 0.13, 0.13, 0.14, 0.12))
    _style_table(table, 7.7)
    y = 0.51
    y = _bullets(axis, y, [
        "SPD covariance is retained because it removes avoidable angle periodicity, axis swapping, positivity constraints and circular-orientation ambiguity—not because it automatically gives the best map.",
        "Case D gives the lowest accepted J but not the lowest map error. Case A gives the lowest yielded RMSE; case B gives the lowest ROI RMSE. Objective and property-error rankings disagree.",
        "Run-D history showed a rejected BF5 trial at J=0.081454 and global/yielded RMSE 30.39/44.60 MPa. Its 4.75% gain narrowly missed the 5% gate, directly confirming that the original gate was too restrictive.",
        "Later gate replication supersedes a simple 'lower the gate' conclusion: 0.5% is effectively inactive and still does not prevent late-stage overfit. Proposal quality and acceptance evidence must be separated.",
    ])
    _callout(axis, y, "Retain SPD + sensitivity-correction as the leading experimental growth mechanism, but do not use training-J percentage improvement as its only acceptance test.")
    _save(pdf, fig)


def _campaign_page(pdf, raw_rows) -> None:
    rows = [row for row in raw_rows if row["source"] == "campaign" and row["accepted"] == "True"]
    fig, axis = _page("Gate campaign and model order", "15 completed runs; 119 solve states; 0% and 0.5% gates")
    chart1 = fig.add_axes((0.12, 0.58, 0.35, 0.24))
    chart2 = fig.add_axes((0.57, 0.58, 0.35, 0.24))
    for gate, color in [("0.0", BLUE), ("0.005", ORANGE)]:
        group = [row for row in rows if row["gate"] == gate]
        bases = sorted({int(row["basis_count"]) for row in group})
        med_j = [np.median([float(row["objective"]) for row in group if int(row["basis_count"]) == bf]) for bf in bases]
        med_e = [np.median([float(row["yielded_rmse_mpa"]) for row in group if int(row["basis_count"]) == bf]) for bf in bases]
        label = "0%" if gate == "0.0" else "0.5%"
        chart1.plot(bases, med_j, "o-", color=color, label=label)
        chart2.plot(bases, med_e, "o-", color=color, label=label)
    chart1.set(xlabel="BF count", ylabel="Median J", title="Mechanical objective")
    chart2.set(xlabel="BF count", ylabel="Median yielded RMSE [MPa]", title="Property-map recovery")
    for chart in (chart1, chart2):
        chart.grid(alpha=0.2)
        chart.legend(frameon=False, fontsize=8)
    y = 0.48
    y = _bullets(axis, y, [
        "Final median yielded RMSE is 39.0 MPa (0%) and 38.9 MPa (0.5%); median high-plastic RMSE is 57.3 and 56.2 MPa. Six of seven paired final maps are identical.",
        "BF6 improves yielded RMSE in 15/15 paired transitions; BF7 in 13/14. BF8 improves only 4/14, with median yielded RMSE worsening 0.30 MPa although J falls 1.50%.",
        "Across all campaign states, J ranks gross progress (ρ=0.845 in the campaign-only report). Among 15 final endpoints, J versus yielded RMSE falls to ρ=0.505 and versus high-plastic RMSE becomes −0.563.",
        "Temporary investigation cap: seven BFs. The cap is a safety decision based on current evidence, not a claim that more than seven basis functions cannot represent the truth.",
    ])
    _save(pdf, fig)


def _best_map_page(pdf, bundle) -> None:
    experiment = bundle["experiment"]
    truth = np.asarray(bundle["known"]["yield_strength"])
    identified = np.asarray(bundle["maps"]["yield_strength"])
    mask = np.asarray(bundle["mask"])
    selected = bundle["selected"]
    x = np.asarray(experiment.specimen_geometry.x)
    y = np.asarray(experiment.specimen_geometry.y)
    extent = [np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)]
    vmin = float(min(np.nanmin(truth[mask]), np.nanmin(identified[mask])))
    vmax = float(max(np.nanmax(truth[mask]), np.nanmax(identified[mask])))
    error = 100 * (identified - truth) / truth
    limit = max(15.0, float(np.nanpercentile(np.abs(error[mask]), 99.0)))
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.suptitle("Best current mature identification", x=0.09, y=0.96, ha="left", fontsize=20, weight="bold", color=INK)
    fig.text(0.09, 0.925, "Lowest yielded-region RMSE among the stored BF5–8 campaign states", fontsize=9.5, color=MUTED)
    axes = [fig.add_axes((0.11, 0.66, 0.78, 0.20)), fig.add_axes((0.11, 0.39, 0.78, 0.20)), fig.add_axes((0.11, 0.12, 0.78, 0.20))]
    im0 = axes[0].imshow(np.where(mask, truth, np.nan), origin="lower", extent=extent, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Known synthetic yield-strength map")
    im1 = axes[1].imshow(np.where(mask, identified, np.nan), origin="lower", extent=extent, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Identified yield-strength map with seven SPD basis functions")
    _draw_kernels(axes[1], bundle["kernels"])
    im2 = axes[2].imshow(np.where(mask, error, np.nan), origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[2].set_title("Percentage error: 100 × (identified − truth) / truth")
    for item in axes:
        item.set(xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im0, ax=axes[:2], label="Yield strength [MPa]", fraction=0.025, pad=0.02)
    fig.colorbar(im2, ax=axes[2], label="Error [%]", fraction=0.025, pad=0.02)
    text = (
        f"Seed 0, BF7 | J={float(selected['objective']):.5f} | ROI RMSE={float(selected['roi_rmse_mpa']):.2f} MPa | "
        f"yielded RMSE={float(selected['yielded_rmse_mpa']):.2f} MPa | high-plastic RMSE={float(selected['high_plastic_rmse_mpa']):.2f} MPa | "
        f"yielded MAPE={float(selected['yielded_mape_percent']):.2f}% | >10% error at {100*float(selected['yielded_above_10pct']):.1f}% of yielded points."
    )
    fig.text(0.09, 0.055, textwrap.fill(text, 125), fontsize=8.5, color=INK, va="top")
    _save(pdf, fig)


def _selector_page(pdf, campaign: Path) -> None:
    fig, axis = _page("Why no replacement scalar has been adopted")
    columns = ["Candidate", "All-state ρ", "BF5–8 ρ", "Final ρ", "BF5–8 high-plastic ρ", "Selected RMSE"]
    rows = [
        ["Current J", "0.845", "0.408", "0.505", "−0.274", "39.0 MPa"],
        ["Time-P90 / max metric", "0.881", "0.293", "−0.025", "−0.221", "39.0 MPa"],
        ["Yield-onset step 3", "0.165", "0.695", "0.798", "0.208", "52.4 MPa"],
        ["Best min-score selector", "—", "—", "—", "—", "38.9 MPa"],
    ]
    table = axis.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center",
                       bbox=(0, 0.63, 1, 0.23), colWidths=(0.27, 0.13, 0.13, 0.12, 0.20, 0.15))
    _style_table(table, 7.6)
    y = 0.56
    y = _bullets(axis, y, [
        "Time-P90 improves gross ranking but not endpoint/model-order selection. Absolute minimisation remains unsafe; the best replay changes one of eight trajectories and gains only 0.10 MPa median yielded RMSE.",
        "Yield onset contains useful yield-strength discrimination that force² temporal weighting can underemphasise, but the onset scalar performs poorly as an absolute global objective and on independent controlled states.",
        "No single ordering is robust across gross convergence, plausible BF5–8 maps, highly plastic material and controlled perturbations. This is the expected failure mode of early scalar collapse.",
    ])
    y = _heading(axis, y, "Design consequence")
    _callout(axis, y, "Preserve a small vector of metric × load-regime evidence through optimisation. Collapse only at a guarded acceptance/decision layer, after thresholds are frozen on development states and validated independently.")
    _save(pdf, fig)


def _component_page(pdf, campaign: Path) -> None:
    fig, axis = _page("Complementary EGI/FRE components", "60 independent perturbations + 32 unique BF5–8 states")
    components = ["EGI-57\nonset P95", "FRE developed\ncoherent RMS", "EGI-29\nonset P90"]
    yielded = [0.596, 0.493, 0.255]
    high = [0.500, 0.776, -0.168]
    chart = fig.add_axes((0.15, 0.56, 0.70, 0.27))
    xx = np.arange(3)
    chart.bar(xx - 0.18, yielded, width=0.36, label="yielded RMSE", color=BLUE)
    chart.bar(xx + 0.18, high, width=0.36, label="high-plastic RMSE", color=ORANGE)
    chart.axhline(0, color="black", lw=0.8)
    chart.set(xticks=xx, xticklabels=components, ylabel="BF5–8 Spearman ρ", ylim=(-0.35, 1.0))
    chart.grid(axis="y", alpha=0.2)
    chart.legend(frameon=False, fontsize=8)
    y = 0.48
    y = _bullets(axis, y, [
        "EGI-57 onset P95 is the primary broad yield-map discriminator: BF5–8 yielded ρ=0.596 and 72.6% pairwise accuracy.",
        "Developed-plasticity coherent FRE is the primary high-plastic guard: BF5–8 high-plastic ρ=0.776 and 80.8% pairwise accuracy. Coherence changes the corresponding raw-RMS association from −0.227 to +0.776.",
        "EGI-29 onset P90 is weak globally but detects compact weld/notch-root errors in controlled development and validation cases. Retain it as a veto/sentinel, not a dominant weighted term.",
        "EGI-57 and coherent FRE are only moderately correlated on BF5–8 states (ρ=0.559); FRE correctly ranks 64.3% of high-plastic pairs missed by EGI-57.",
    ])
    _callout(axis, y, "Smallest current evidence vector: EGI-57 onset P95 + developed coherent FRE + EGI-29 onset P90. No weights or production thresholds are yet justified.")
    _save(pdf, fig)


def _sensitivity_page(pdf, campaign: Path) -> None:
    fig, axis = _page("Sensitivity update: project information,\ndo not merely weight it")
    labels = ["EGI-15 dev.\nyield-unique", "EGI-15 dev.\nfull projection", "EGI-15 late\nyield-unique", "EGI-15 late\nfull projection"]
    within = [0.940, 0.929, 0.887, 0.863]
    adjacent = [0.792, 0.875, 0.750, 0.542]
    bf78 = [0.750, 0.750, 0.625, 0.500]
    chart = fig.add_axes((0.12, 0.57, 0.76, 0.26))
    xx = np.arange(len(labels))
    chart.bar(xx - 0.24, within, 0.24, color=BLUE, label="within-BF ρ")
    chart.bar(xx, adjacent, 0.24, color=ORANGE, label="adjacent accuracy")
    chart.bar(xx + 0.24, bf78, 0.24, color=GREEN, label="BF7→8 accuracy")
    chart.set(xticks=xx, xticklabels=labels, ylim=(0, 1.05), ylabel="High-plastic discrimination")
    chart.grid(axis="y", alpha=0.2)
    chart.legend(frameon=False, fontsize=8, ncol=3)
    y = 0.49
    y = _bullets(axis, y, [
        "Unsigned sensitivity magnitude answers where a parameter matters, but cannot tell whether residual change belongs uniquely to yield strength, is shared with hardening, or is redundant with existing BF directions.",
        "Projecting residuals onto the native parameter-response span—and separating yield-unique from hardening-unique components—directly addresses the identifiability question the optimiser needs.",
        "The generic-direction pilot gives particularly strong BF5–8 high-plastic evidence for EGI-15 during developed plasticity: yield-unique within-BF ρ=0.940, adjacent accuracy 0.792, BF7→8 accuracy 0.750; full projection gives 0.929/0.875/0.750.",
        "These are promising offline diagnostics, not production evidence. The pilot uses generic directions and noise-free synthetic residuals; native BF DOFs, realistic WDBN1 noise and scale stability are the remaining tests.",
    ])
    _save(pdf, fig)


def _experimental_page(pdf) -> None:
    fig, axis = _page("Experimental context and noise implications", "WDBN1 unloaded frames and prepared strain fields")
    columns = ["Quantity", "Compact WDBN1 estimate", "Implication"]
    rows = [
        ["εxx noise", "169.5 µε", "anisotropic spatial correlation"],
        ["εyy noise", "158.1 µε", "anisotropic spatial correlation"],
        ["εxy noise", "114.4 µε", "≈0.106 mm correlation length"],
        ["Force noise", "0.322 N", "small in the unloaded sample"],
        ["DIC spacing", "≈0.0212 mm", "much finer than synthetic 0.2 mm"],
        ["Synthetic EGI supports", "15/29/57 pt = 3.0/5.8/11.4 mm", "map by physical size, not point count"],
    ]
    table = axis.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left",
                       bbox=(0, 0.58, 1, 0.28), colWidths=(0.22, 0.31, 0.47))
    _style_table(table, 8.2)
    y = 0.51
    y = _bullets(axis, y, [
        "A literal 15-point experimental window is only about 0.32 mm across and sits close to the correlated-noise scale. Window definitions must be transferred in millimetres or after controlled resampling.",
        "Force² temporal weighting remains physically reasonable for SNR, but yield-onset and developed-plasticity evidence should stay separate so late high-load states cannot dominate through hardening entanglement.",
        "The previous experimental WDBN1 trials showed widespread yielding and hardening/yield bounds being used to absorb mismatch. That is a warning against deploying an unvalidated flexible scalar objective directly on experiment.",
        "Noise covariance, spatial correlation, masking/edge handling, temporal alignment and metric cross-covariance should be propagated into offline selector validation before production tuning.",
    ])
    _save(pdf, fig)


def _overnight_page(pdf) -> None:
    fig, axis = _page("Ongoing workstation campaign", "Status at report generation: launched successfully; no scientific result yet")
    y = 0.89
    y = _callout(axis, y, "The native-projection/noise campaign is running detached on R0379. All 32 state logs were created, 64 matching launcher/worker processes were visible, and no Traceback/Error/Killed messages were present at the last check.", color=GREEN)
    columns = ["Design item", "Overnight setting"]
    rows = [
        ["State shards", "32 concurrent mature campaign states"],
        ["Native directions", "actual homogeneous, BF height/centre/SPD-shape, hardening DOFs"],
        ["EGI windows", "7, 15, 29, 57 synthetic points"],
        ["Noise scales", "0, 0.5, 1.0, 1.5 × compact WDBN1 model"],
        ["Replicates", "128 correlated strain/force-noise realisations"],
        ["Output", "per-state checkpoints, merged tables, PDF and transfer archive"],
    ]
    table = axis.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left",
                       bbox=(0, 0.47, 1, 0.28), colWidths=(0.28, 0.72))
    _style_table(table, 8.5)
    y = 0.40
    y = _heading(axis, y, "Question it is intended to answer")
    y = _paragraph(axis, y, "Do projected/yield-unique EGI and FRE components retain BF5–8 ranking power when sensitivities use the algorithm’s actual parameter directions and residuals are perturbed with spatially correlated experimental-scale noise?")
    y = _heading(axis, y, "Interpretation rule")
    _bullets(axis, y, [
        "Do not treat 0/32 early checkpoints as failure: a shard writes only after completing its full calculation. Judge health by worker activity, logs and absence of errors.",
        "Do not update the production objective from this report. First transfer the archive, verify its checksum, inspect noise-scale degradation and compare physical window sizes.",
        "The eventual result should either justify a small projected evidence vector for a guarded selector or reject it cleanly before expensive new identifications.",
    ])
    _save(pdf, fig)


def _next_steps_page(pdf) -> None:
    fig, axis = _page("Recommended route to a robust\nidentification algorithm")
    steps = [
        ("1", "Complete native/noise validation", "Analyse rank correlation, pairwise accuracy, BF7→8 decisions and scale stability across noise levels. Select physical—not point-count—EGI supports."),
        ("2", "Freeze a minimal guarded selector", "Start with EGI-57 onset P95, coherent developed FRE and EGI-29 onset P90; add only projected/yield-unique components that improve held-out BF5–8 decisions under noise."),
        ("3", "Implement offline acceptance replay", "Require training J improvement, held-out evidence improvement/non-regression, sufficient projected BF response norm and acceptable conditioning. Tune on development states, then freeze."),
        ("4", "Run a focused confirmation campaign", "SPD + sensitivity-correction only, eight seeds, temporary seven-BF cap, concise progress. Compare against stored 0% trajectories using predeclared map and robustness metrics."),
        ("5", "Stress-test before experiment", "Add strain/force bias, temporal shifts, masking/edge changes, load omissions, thickness error and modest constitutive mismatch. Report seed/noise spread and spatial plastic-zone error."),
        ("6", "Freeze and process experiment", "Only after synthetic release: lock weights/thresholds/window sizes; use experimental data without truth-derived retuning and retain mechanical closure as a non-negotiable guard."),
    ]
    y = 0.91
    for number, title, body in steps:
        axis.text(0.0, y, number, fontsize=15, weight="bold", color="white", ha="center", va="center",
                  bbox={"boxstyle": "circle,pad=0.35", "facecolor": BLUE, "edgecolor": "none"})
        axis.text(0.075, y + 0.012, title, fontsize=11.2, weight="bold", color=INK, va="top")
        wrapped = textwrap.wrap(body, width=86)
        axis.text(0.075, y - 0.020, "\n".join(wrapped), fontsize=8.9, color=INK, va="top", linespacing=1.28)
        y -= 0.135
    axis.text(0, 0.08, "Provisional synthetic release targets", fontsize=12, weight="bold", color=INK)
    axis.text(0, 0.045, "Median yielded RMSE ≤20 MPa | high-plastic RMSE ≤30 MPa | no late-BF tail degradation | stable under realistic noise",
              fontsize=8.7, color=RED, weight="bold")
    _save(pdf, fig)


def _source_page(pdf, sources) -> None:
    fig, axis = _page("Report register and precedence", "Every PDF in investigation-reports was reviewed")
    y = 0.91
    y = _paragraph(axis, y, "The register below is ordered by filesystem modification time. These times are used only to reconstruct the investigation sequence; embedded creation times and the substance of later analyses determine precedence. Duplicate 14:26/14:32 consolidated reports are retained as historical artefacts.", size=8.8)
    for time, name in sources:
        wrapped = textwrap.wrap(name, width=86)
        axis.text(0, y, time, fontsize=7.8, color=BLUE, weight="bold", va="top")
        axis.text(0.09, y, "\n".join(wrapped), fontsize=7.8, color=INK, va="top", linespacing=1.15)
        y -= 0.023 * len(wrapped) + 0.010
    y -= 0.012
    y = _heading(axis, y, "Explicitly superseded observations", RED)
    _bullets(axis, y, [
        "The 4.36×10⁵ free-geometry condition number and 5/20 effective directions are superseded by the exact log-coordinate result: 3.78×10³ and 11/20.",
        "Metric-only preference for EGI-17 is superseded by matched inverse tests; 29/57 remain the production scales, while smaller scales are reconsidered only as separate sensitivity/projection evidence.",
        "The early recommendation to fix Gaussian geometry entirely is softened: use SPD geometry with mechanically informed initialisation and controlled release.",
        "The suggestion simply to lower the 5% gate is superseded by the 15-run campaign: percentage-J gates alone cannot select reliable late-stage maps.",
        "Unsigned sensitivity weighting is not a production default; projected/yield-unique information is the current research direction.",
    ], size=8.4)
    _save(pdf, fig)


def _draw_kernels(axis, kernels) -> None:
    for index, kernel in enumerate(kernels, start=1):
        covariance = np.asarray(kernel["covariance"], dtype=float)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        major = vectors[:, order[0]]
        angle = np.degrees(np.arctan2(major[1], major[0]))
        color = "#ffde59" if float(kernel["height"]) >= 0 else "#5ec8ff"
        ellipse = Ellipse(
            kernel["centre"], 2 * np.sqrt(values[order[0]]), 2 * np.sqrt(values[order[1]]),
            angle=angle, fill=False, color=color, lw=1.4,
        )
        axis.add_patch(ellipse)
        axis.scatter(*kernel["centre"], marker="x", color=color, s=30, linewidth=1.4)
        axis.text(kernel["centre"][0] + 0.2, kernel["centre"][1] + 0.15, f"B{index}", color=color,
                  fontsize=6.5, weight="bold")


def _style_table(table, fontsize: float) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#c8d2db")
        if row == 0:
            cell.set_facecolor(LIGHT)
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_text_props(color=INK)


if __name__ == "__main__":
    main()

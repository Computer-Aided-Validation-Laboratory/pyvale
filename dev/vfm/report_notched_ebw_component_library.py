"""Create the concise PDF report for the residual-component library study."""

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
import numpy as np


INK = "#17202a"
MUTED = "#52606d"
BLUE = "#1565c0"
GREEN = "#2e7d32"
ORANGE = "#ef6c00"
RED = "#c62828"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = _read_rows(args.results / "state_manifest.csv")
    discrimination = _read_rows(
        args.results / "component_discrimination.csv"
    )
    shortlist = json.loads(
        (args.results / "component_shortlist.json").read_text(encoding="utf-8")
    )
    maps = np.load(args.results / "independent_state_maps.npz")
    known = np.load(args.dataset / "prepared/known_parameter_maps.npz")
    x = np.load(args.dataset / "prepared/x.npy")
    y = np.load(args.dataset / "prepared/y.npy")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().astimezone()
    with PdfPages(
        args.output,
        metadata={
            "Title": "Notched-EBW residual-component library study",
            "Author": "PyVale investigation",
            "CreationDate": generated,
        },
    ) as pdf:
        _executive_page(pdf, shortlist, generated)
        _library_page(pdf, manifest, maps, known, x, y)
        _evidence_page(pdf, discrimination, shortlist)
        _recommendation_page(pdf, shortlist)
    print(args.output)


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _page():
    figure = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    axis = figure.add_axes((0.06, 0.06, 0.88, 0.88))
    axis.axis("off")
    return figure, axis


def _title(axis, title, subtitle=None):
    axis.text(0.0, 1.0, title, fontsize=22, fontweight="bold", color=INK, va="top")
    if subtitle:
        axis.text(0.0, 0.945, subtitle, fontsize=10.5, color=MUTED, va="top")


def _wrapped(axis, x, y, text, width=108, **kwargs):
    axis.text(x, y, textwrap.fill(text, width=width), **kwargs)


def _short_label(component):
    metric, regime, summary = component.split("__")
    return f"{metric.upper()} | {regime.replace('_', ' ')} | {summary.replace('_', ' ')}"


def _executive_page(pdf, shortlist, generated):
    figure, axis = _page()
    _title(
        axis,
        "Residual-component library: decision report",
        f"60 independent maps (30 development / 30 validation) + 32 unique BF5–8 states | {generated:%d %B %Y, %H:%M %Z}",
    )
    axis.text(
        0.0, 0.855, "Outcome", fontsize=12, fontweight="bold", color="white",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": BLUE, "edgecolor": "none"},
    )
    _wrapped(
        axis, 0.0, 0.79,
        "Carry three residual components forward as separate evidence. Do not combine or install them as a production objective yet. They discriminate complementary error modes that no single aggregate captures robustly.",
        fontsize=14, fontweight="bold", color=INK, va="top",
    )

    rows = []
    roles = (
        "Late yield-map quality",
        "High-plastic-tail quality",
        "Local weld/notch-root sentinel",
    )
    concise_labels = (
        "EGI-57 onset P95",
        "FRE developed coherent",
        "EGI-29 onset P90",
    )
    for component, concise, role in zip(
        shortlist["selected_components"], concise_labels, roles, strict=True
    ):
        evidence = shortlist["held_out_evidence"][component]
        rows.append([
            concise,
            role,
            f"{evidence['validation']['yielded_rmse_mpa']['spearman_r']:.3f}",
            f"{evidence['validation']['high_plastic_rmse_mpa']['spearman_r']:.3f}",
            f"{evidence['optimiser_bf5_8']['yielded_rmse_mpa']['spearman_r']:.3f}",
            f"{evidence['optimiser_bf5_8']['high_plastic_rmse_mpa']['spearman_r']:.3f}",
        ])
    table = axis.table(
        cellText=rows,
        colLabels=["Component", "Role", "Validation\nyielded ρ", "Validation\nhigh-plastic ρ", "BF5–8\nyielded ρ", "BF5–8\nhigh-plastic ρ"],
        cellLoc="center", colLoc="center", bbox=(0.0, 0.43, 1.0, 0.25),
        colWidths=[0.18, 0.20, 0.15, 0.17, 0.14, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2ec")
        if row == 0:
            cell.set_facecolor("#eaf2f8")
            cell.set_text_props(weight="bold", color=INK)
        elif column in (0, 1):
            cell.set_text_props(ha="left")

    findings = [
        "Development/validation correlations are high because the independent library spans known physical error magnitudes; BF5–8 evidence is the harder and more relevant discriminator.",
        "EGI-57 onset P95 ranks BF5–8 yielded error at ρ=0.596 with 72.6% pairwise accuracy.",
        "Developed-plasticity coherent FRE ranks BF5–8 high-plastic error at ρ=0.776 with 80.8% pairwise accuracy.",
        "EGI-29 onset P90 is weak on the broad BF5–8 population but perfectly orders the two local-error magnitudes in both predeclared splits; retain it only as a specialist sentinel.",
    ]
    axis.text(0.0, 0.365, "What is established", fontsize=13, fontweight="bold", color=INK)
    y = 0.31
    for finding in findings:
        axis.text(0.008, y, "•", fontsize=15, color=BLUE, va="top")
        _wrapped(axis, 0.035, y, finding, fontsize=10.7, color=INK, va="top")
        y -= 0.073
    pdf.savefig(figure)
    plt.close(figure)


def _library_page(pdf, manifest, maps, known, x, y):
    figure = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    figure.suptitle("Independent state library", x=0.06, y=0.97, ha="left", fontsize=22, fontweight="bold", color=INK)
    families = (
        "location_shift", "width_change", "amplitude_error",
        "haz_band_error", "high_plastic_local_error",
        "hardening_compensation", "combined_error",
    )
    descriptions = (
        "±0.4–2.0 mm", "0.50–1.50×", "0.50–1.50× contrast",
        "±25–90 MPa flank bands", "±25–90 MPa notch-root spots",
        "yield ±25–90 MPa; H 0.70–1.30×", "shift + width + amplitude + local + H",
    )
    counts = []
    for family, description in zip(families, descriptions, strict=True):
        counts.append([
            family.replace("_", " "), description,
            sum(row["family"] == family and row["split"] == "development" for row in manifest),
            sum(row["family"] == family and row["split"] == "validation" for row in manifest),
        ])
    table_axis = figure.add_axes((0.06, 0.54, 0.88, 0.32))
    table_axis.axis("off")
    table = table_axis.table(
        cellText=counts,
        colLabels=["Family", "Physical span", "Development", "Validation"],
        cellLoc="center", colLoc="center", bbox=(0.0, 0.0, 1.0, 1.0),
        colWidths=[0.28, 0.42, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.3)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2ec")
        if row == 0:
            cell.set_facecolor("#eaf2f8")
            cell.set_text_props(weight="bold")
        elif column in (0, 1):
            cell.set_text_props(ha="left")

    names = [str(value) for value in maps["names"]]
    examples = [
        ("Truth", np.asarray(known["yield_strength"])),
        ("Location shift", maps["yield_strength"][names.index("validation_location_shift_03")]),
        ("Narrow weld/HAZ", maps["yield_strength"][names.index("validation_width_change_00")]),
        ("Low amplitude", maps["yield_strength"][names.index("validation_amplitude_error_00")]),
        ("Local plastic-zone error", maps["yield_strength"][names.index("validation_high_plastic_local_error_03")]),
        ("Substantial combination", maps["yield_strength"][names.index("validation_combined_error_02")]),
    ]
    minimum = min(float(np.nanmin(values)) for _, values in examples)
    maximum = max(float(np.nanmax(values)) for _, values in examples)
    for index, (label, values) in enumerate(examples):
        axis = figure.add_axes((0.06 + (index % 3) * 0.30, 0.08 + (1 - index // 3) * 0.20, 0.27, 0.16))
        image = axis.imshow(
            values, origin="lower",
            extent=[float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))],
            aspect="auto", cmap="viridis", vmin=minimum, vmax=maximum,
        )
        axis.set_title(label, fontsize=9.5)
        axis.set_xticks([])
        axis.set_yticks([])
    color_axis = figure.add_axes((0.94, 0.08, 0.012, 0.36))
    figure.colorbar(image, cax=color_axis, label="Yield strength [MPa]")
    pdf.savefig(figure)
    plt.close(figure)


def _evidence_page(pdf, discrimination, shortlist):
    figure = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    figure.suptitle("Complementary discrimination", x=0.06, y=0.97, ha="left", fontsize=22, fontweight="bold", color=INK)
    components = [
        "egi57__yield_onset__rms",
        "egi57__yield_onset__p95",
        "fre__developed_plasticity__rms",
        "fre__developed_plasticity__coherent_rms",
        "egi29__yield_onset__p90",
        "egi29__developed_plasticity__p95",
    ]
    labels = [
        "EGI-57 onset RMS", "EGI-57 onset P95",
        "FRE developed RMS", "FRE developed coherent RMS",
        "EGI-29 onset P90", "EGI-29 developed P95",
    ]
    values_yield = []
    values_high = []
    for component in components:
        values_yield.append(_disc(
            discrimination, "optimiser_bf5_8", "yielded_rmse_mpa", component
        )[0])
        values_high.append(_disc(
            discrimination, "optimiser_bf5_8", "high_plastic_rmse_mpa", component
        )[0])
    axis = figure.add_axes((0.08, 0.46, 0.84, 0.38))
    positions = np.arange(len(components))
    width = 0.36
    axis.bar(positions - width / 2, values_yield, width, label="Yielded RMSE", color=BLUE)
    axis.bar(positions + width / 2, values_high, width, label="High-plastic RMSE", color=ORANGE)
    axis.set_xticks(positions, labels, rotation=22, ha="right")
    axis.set_ylim(-0.5, 1.0)
    axis.set_ylabel("BF5–8 Spearman ρ")
    axis.axhline(0.0, color="black", lw=0.8)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)

    comp = shortlist["complementarity"]["optimiser_bf5_8"]
    points = [
        "Upper-tail EGI-57 improves on onset RMS: P95 raises BF5–8 yielded ρ from 0.494 to 0.596 and high-plastic ρ from 0.132 to 0.500.",
        "Spatial coherence is essential for FRE: developed-plasticity RMS has high-plastic ρ=−0.227; coherent RMS changes this to +0.776.",
        "Fine-scale EGI-29 excels on independent local perturbations but developed/late EGI-29 becomes negatively associated with BF5–8 high-plastic quality; it must remain a sentinel, not dominate the score.",
        f"The EGI-57 and coherent-FRE scores are only moderately correlated on BF5–8 states (ρ={comp['score_spearman']:.3f}). Coherent FRE correctly ranks {100*comp['high_plastic_rmse_mpa']['second_rescues_first_misses']:.1f}% of high-plastic pairs missed by EGI-57.",
    ]
    text_axis = figure.add_axes((0.06, 0.06, 0.88, 0.30))
    text_axis.axis("off")
    y = 0.95
    for point in points:
        text_axis.text(0.008, y, "•", fontsize=15, color=BLUE, va="top")
        _wrapped(text_axis, 0.035, y, point, fontsize=10.8, color=INK, va="top")
        y -= 0.23
    pdf.savefig(figure)
    plt.close(figure)


def _disc(rows, subset, target, component):
    row = next(
        row for row in rows
        if row["subset"] == subset
        and row["target"] == target
        and row["component"] == component
    )
    return float(row["spearman_r"]), float(row["pairwise_accuracy"])


def _recommendation_page(pdf, shortlist):
    figure, axis = _page()
    _title(axis, "Smallest evidence vector to carry forward")
    components = shortlist["selected_components"]
    cards = [
        (components[0], "Primary yield-map discriminator", "P95 preserves concentrated onset residuals at the broader EGI scale. Use as held-out evidence for whether a candidate improves plausible late yield maps.", BLUE),
        (components[1], "Primary high-plastic discriminator", "Spatial smoothing rewards coherent cross-sectional mismatch and rejects diffuse/random FRE. Use as a non-regression guard for the plastic tail.", GREEN),
        (components[2], "Local-error sentinel", "Fine-scale onset P90 detects compact notch-root/weld errors that broad EGI and coherent FRE can dilute. Keep separate and give it veto/flag semantics initially.", ORANGE),
    ]
    y = 0.86
    for component, role, explanation, colour in cards:
        axis.text(0.0, y, _short_label(component), fontsize=13, fontweight="bold", color=colour, va="top")
        axis.text(0.43, y, role, fontsize=11.5, fontweight="bold", color=INK, va="top")
        _wrapped(axis, 0.025, y - 0.05, explanation, fontsize=10.7, color=INK, va="top")
        y -= 0.18

    axis.text(0.0, 0.31, "Do not carry forward", fontsize=13, fontweight="bold", color=RED)
    exclusions = [
        "Pre-yield property scoring: mostly tied/constant and therefore misleading; retain pre-yield only for elastic/force calibration.",
        "Coherence fraction alone: unstable and weaker than absolute coherent RMS.",
        "Standalone developed/late EGI-29 as a global selector: strong on controlled maps but anticorrelated with BF5–8 high-plastic quality.",
        "A weighted sum of the three shortlisted components: no weights or acceptance thresholds have yet been validated.",
    ]
    y = 0.255
    for item in exclusions:
        axis.text(0.008, y, "•", fontsize=14, color=RED, va="top")
        _wrapped(axis, 0.035, y, item, fontsize=10.3, color=INK, va="top")
        y -= 0.052

    pdf.savefig(figure)
    plt.close(figure)


if __name__ == "__main__":
    main()

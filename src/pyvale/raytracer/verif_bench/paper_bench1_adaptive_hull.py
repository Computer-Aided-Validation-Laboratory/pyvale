#!/usr/bin/env python3
"""Paper figure/table generator for benchmark case 1 (adaptive hulls).

Consumes the outputs written by ``bench1_adaptive_hull.cpp``. 
It reads the per-frame statistics and
the conservativeness check, aggregates over the deformation frames for which the
projected hull is valid, and writes the LaTeX figure and table that are
\\input by the manuscript.

The study quantifies two benefits of the projected adaptive hull:

  * fewer Newton solver runs - the triangle-fan test rejects rays that do not
    pass through the element before the solver is entered; and
  * fewer Newton iterations per solve - the barycentric seed is closer to the
    solution than the element centre.

Three initial-guess strategies are compared over the same rays:

  * ``adaptive_hull``          - fan test plus barycentric seed (production);
  * ``fan_reject_center_seed`` - fan test with the element-centre seed, which
    isolates the seeding benefit; and
  * ``center_seed``            - no hull at all (the no-hull baseline).
"""

from __future__ import annotations

import csv
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path as _Path
from matplotlib.ticker import ScalarFormatter

from paper_const import repo_root

PAPER_DIR = _Path(
    "/home/wiera/Documents/paper-feebee-raytracer/figures/bench1",
).expanduser()

BENCH_DIR = pathlib.Path("verif_bench/bench/bench1")
STATS_PATH = BENCH_DIR / "adaptive_hull_stats.csv"
CONSERV_PATH = BENCH_DIR / "adaptive_hull_conservativeness.csv"
OUT_FIGS_TEX_PATH = BENCH_DIR / "figs_bench1.tex"
OUT_TABS_TEX_PATH = BENCH_DIR / "tabs_bench1.tex"

ELEMENTS = ("tri6", "quad8", "quad9")
MODE_HULL = "adaptive_hull"
MODE_FAN_CENTER = "fan_reject_center_seed"
MODE_CENTER = "center_seed"
MODES = (MODE_HULL, MODE_FAN_CENTER, MODE_CENTER)

# A frame is treated as well posed for the adaptive hull when the number of
# coverage rays that the hull rejects but the no-hull solver resolves inside the
# reference domain is at most this fraction of the coverage bundle.  Frames that
# fail this check correspond to strongly folded elements.
# \frac{N_{\text{rejected by hull, resolved by no-hull}}} {N_{\text{coverage rays}}} \leq 0.005
WELL_POSED_FRACTION = 0.005

ELEMENT_LABEL = {
    "tri6": "TRI6",
    "quad8": "QUAD8",
    "quad9": "QUAD9",
}

PLOT_DPI = 300.0
BAR_WIDTH = 0.26

FIG_CAPTION = (
    "Adaptive-hull benefit in benchmark case 1. (a) Newton solver runs over "
    "the coverage bundle: the no-hull baseline (centre seed) enters the solver "
    "for every ray, whereas the triangle-fan test rejects rays that miss the "
    "element before any iteration is performed. (b) Mean Newton iterations per "
    "solve over the hit bundle for the three initial-guess strategies, which "
    "isolates the contribution of the barycentric seed. Statistics are "
    "aggregated over the distortion cases and deformation frames for which the "
    "projected hull is conservative."
)

TABLE_CAPTION = "Adaptive-hull benefit and cost in benchmark case 1."


def bench_root() -> pathlib.Path:
    return repo_root() / BENCH_DIR


def run_binary_if_needed() -> None:
    if (repo_root() / STATS_PATH).exists():
        return
    binary = repo_root() / "verif_bench/bench1_adaptive_hull"
    if not binary.exists():
        print(
            "Missing bench1 outputs. Build the case first (from verif_bench/):\n"
            "  g++ -O2 -std=c++20 -fopenmp -o bench1_adaptive_hull \\\n"
            "      bench1_adaptive_hull.cpp cpp/rtrayintersection_modes.cpp\n"
            "  ./bench1_adaptive_hull",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("Missing bench1 outputs, running bench1_adaptive_hull...")
    subprocess.run([str(binary)], cwd=repo_root() / "verif_bench", check=True)


def load_stats() -> list[dict]:
    rows = []
    with open(repo_root() / STATS_PATH, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "case": row["case"],
                    "element": row["element"],
                    "frame": int(row["frame"]),
                    "bundle": row["bundle"],
                    "mode": row["mode"],
                    "rays": int(row["rays"]),
                    "runs": int(row["runs"]),
                    "rejected": int(row["rejected"]),
                    "converged": int(row["converged"]),
                    "in_domain": int(row["in_domain"]),
                    "iters": int(row["total_iters"]),
                    "hull_points": int(row["hull_points"]),
                    "edge_tests": int(row["edge_tests"]),
                    "projected_nodes": int(row["projected_nodes"]),
                    "wall_s": float(row["wall_time_s"]),
                    "wrong_root": int(row["wrong_root"]),
                }
            )
    return rows


def load_conservatism() -> dict:
    table = {}
    with open(repo_root() / CONSERV_PATH, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["case"], row["element"], int(row["frame"]))
            table[key] = {
                "coverage_rays": int(row["coverage_rays"]),
                "hull_hits": int(row["hull_hits"]),
                "center_hits": int(row["center_hits"]),
                "hull_miss_center_hit": int(row["hull_miss_center_hit"]),
            }
    return table


def well_posed_frames(conservatism, fraction: float) -> set:
    good = set()
    for key, info in conservatism.items():
        if info["hull_miss_center_hit"] <= fraction * max(info["coverage_rays"], 1):
            good.add(key)
    return good


def empty_mode_totals() -> dict:
    return {m: {"runs": 0, "iters": 0} for m in MODES}


def aggregate(stats, good_frames) -> dict:
    totals = {}
    for element in ELEMENTS:
        totals[element] = {
            "coverage_rays": 0,
            "hit_rays": 0,
            "hull_points": 0,
            "edge_tests": 0,
            "wall": {m: 0.0 for m in MODES},
            "coverage": empty_mode_totals(),
            "hit": empty_mode_totals(),
        }

    for row in stats:
        key = (row["case"], row["element"], row["frame"])
        if key not in good_frames:
            continue
        agg = totals[row["element"]]
        agg["wall"][row["mode"]] += row["wall_s"]

        if row["bundle"] == "coverage":
            agg["coverage"][row["mode"]]["runs"] += row["runs"]
            agg["coverage"][row["mode"]]["iters"] += row["iters"]
        else:
            agg["hit"][row["mode"]]["runs"] += row["runs"]
            agg["hit"][row["mode"]]["iters"] += row["iters"]

        if row["mode"] == MODE_HULL:
            agg["hull_points"] += row["hull_points"]
            agg["edge_tests"] += row["edge_tests"]
            if row["bundle"] == "coverage":
                agg["coverage_rays"] += row["rays"]
            else:
                agg["hit_rays"] += row["rays"]

    for agg in totals.values():
        for bundle in ("coverage", "hit"):
            for data in agg[bundle].values():
                data["per_solve"] = (
                    data["iters"] / data["runs"] if data["runs"] else 0.0
                )
        rays = agg["hit_rays"] + agg["coverage_rays"]
        agg["hull_points_per_ray"] = agg["hull_points"] / rays if rays else 0.0
        agg["edge_tests_per_ray"] = agg["edge_tests"] / rays if rays else 0.0
    return totals


def pct_reduction(baseline: float, improved: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return 100.0 * (baseline - improved) / baseline


def build_figure(totals) -> str:
    x = np.arange(len(ELEMENTS), dtype=float)
    labels = [ELEMENT_LABEL[e] for e in ELEMENTS]

    cov_hull = [totals[e]["coverage"][MODE_HULL]["runs"] for e in ELEMENTS]
    cov_center = [totals[e]["coverage"][MODE_CENTER]["runs"] for e in ELEMENTS]

    hit_hull = [totals[e]["hit"][MODE_HULL]["per_solve"] for e in ELEMENTS]
    hit_fan = [totals[e]["hit"][MODE_FAN_CENTER]["per_solve"] for e in ELEMENTS]
    hit_center = [totals[e]["hit"][MODE_CENTER]["per_solve"] for e in ELEMENTS]

    fig, (ax_runs, ax_iters) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    ax_runs.bar(x - BAR_WIDTH / 2.0, cov_center, BAR_WIDTH,
                label="no hull (centre seed)", color="#b0b0b0",
                edgecolor="black", linewidth=0.6)
    ax_runs.bar(x + BAR_WIDTH / 2.0, cov_hull, BAR_WIDTH,
                label="adaptive hull", color="#3b6ea5",
                edgecolor="black", linewidth=0.6)
    ax_runs.set_xticks(x)
    ax_runs.set_xticklabels(labels)
    ax_runs.set_ylabel("Newton solver runs")
    ax_runs.set_title("(a) Solver runs (coverage bundle)")
    ax_runs.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_runs.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_runs.legend(fontsize=8.0)

    width = 0.26
    ax_iters.bar(x - width, hit_center, width, label="centre seed",
                 color="#b0b0b0", edgecolor="black", linewidth=0.6)
    ax_iters.bar(x, hit_fan, width, label="hull reject + centre seed",
                 color="#e0a04a", edgecolor="black", linewidth=0.6)
    ax_iters.bar(x + width, hit_hull, width, label="adaptive hull",
                 color="#3b6ea5", edgecolor="black", linewidth=0.6)
    ax_iters.set_xticks(x)
    ax_iters.set_xticklabels(labels)
    ax_iters.set_ylabel("Newton iterations per solve")
    ax_iters.set_title("(b) Iterations per solve (hit bundle)")
    ax_iters.legend(fontsize=8.0)

    for ax in (ax_runs, ax_iters):
        ax.tick_params(axis="both", labelsize=8.0)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PAPER_DIR / "fig_bench1_adaptive_hull.png"
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        "\\includegraphics[width=0.92\\textwidth]{fig_bench1_adaptive_hull.png}\n"
        f"\\caption{{{FIG_CAPTION}}}\n"
        "\\label{fig:bench1_adaptive_hull}\n"
        "\\end{figure}\n"
    )


def build_table(totals) -> str:
    lines = []
    for element in ELEMENTS:
        agg = totals[element]
        runs_hull = agg["coverage"][MODE_HULL]["runs"]
        runs_center = agg["coverage"][MODE_CENTER]["runs"]
        runs_avoided = pct_reduction(runs_center, runs_hull)

        it_hull = agg["hit"][MODE_HULL]["per_solve"]
        it_center = agg["hit"][MODE_CENTER]["per_solve"]
        it_reduced = pct_reduction(it_center, it_hull)

        hull_cost = agg["hull_points_per_ray"]
        time_ratio = (
            agg["wall"][MODE_HULL] / agg["wall"][MODE_CENTER]
            if agg["wall"][MODE_CENTER]
            else 0.0
        )

        lines.append(
            f"\\texttt{{{element.upper()}}} & {runs_hull:.0f} & {runs_center:.0f} & "
            f"{runs_avoided:.1f} & {it_hull:.2f} & {it_center:.2f} & "
            f"{it_reduced:.1f} & {hull_cost:.1f} & {time_ratio:.2f} \\\\"
        )

    body = "\n".join(lines)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{TABLE_CAPTION}}}\n"
        "\\label{tab:bench1_adaptive_hull}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrrrrr}\n"
        "\\hline\n"
        "Element & \\makecell{Runs\\\\hull} & \\makecell{Runs\\\\centre} & "
        "\\makecell{Runs avoided\\\\\\%} & \\makecell{Iters/solve\\\\hull} & "
        "\\makecell{Iters/solve\\\\centre} & "
        "\\makecell{Iters reduced\\\\\\%} & "
        "\\makecell{Hull cost\\\\pts/ray} & \\makecell{Time\\\\ratio} \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )


def write_tex_outputs(figs_tex: str, tabs_tex: str) -> None:
    out_figs_path = repo_root() / OUT_FIGS_TEX_PATH
    out_tabs_path = repo_root() / OUT_TABS_TEX_PATH
    out_figs_path.parent.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    out_figs_path.write_text(figs_tex)
    out_tabs_path.write_text(tabs_tex)
    (PAPER_DIR / OUT_FIGS_TEX_PATH.name).write_text(figs_tex)
    (PAPER_DIR / OUT_TABS_TEX_PATH.name).write_text(tabs_tex)

    print(f"Wrote {out_figs_path}")
    print(f"Wrote {out_tabs_path}")
    print(f"Wrote {PAPER_DIR / OUT_FIGS_TEX_PATH.name}")
    print(f"Wrote {PAPER_DIR / OUT_TABS_TEX_PATH.name}")


def main() -> int:
    run_binary_if_needed()

    stats = load_stats()
    conservatism = load_conservatism()
    good = well_posed_frames(conservatism, WELL_POSED_FRACTION)

    all_frames = set(conservatism.keys())
    excluded = sorted(all_frames - good)
    print(f"Well-posed frames: {len(good)}/{len(all_frames)}")
    if excluded:
        print("Excluded (folded) frames where the hull is not conservative:")
        for case, element, frame in excluded:
            info = conservatism[(case, element, frame)]
            print(
                f"  {case:8s} {element:6s} f{frame:2d}: "
                f"hull missed {info['hull_miss_center_hit']} of "
                f"{info['center_hits']} centre-seed hits"
            )

    totals = aggregate(stats, good)

    print("\nAggregated benefit per element:")
    for element in ELEMENTS:
        agg = totals[element]
        runs_hull = agg["coverage"][MODE_HULL]["runs"]
        runs_center = agg["coverage"][MODE_CENTER]["runs"]
        it_hull = agg["hit"][MODE_HULL]["per_solve"]
        it_center = agg["hit"][MODE_CENTER]["per_solve"]
        time_ratio = agg["wall"][MODE_HULL] / agg["wall"][MODE_CENTER]
        print(
            f"  {element:5s} runs {runs_hull:>8.0f}/{runs_center:<8.0f} "
            f"({pct_reduction(runs_center, runs_hull):5.1f}% avoided)  "
            f"iters/solve {it_hull:.2f}/{it_center:.2f} "
            f"({pct_reduction(it_center, it_hull):5.1f}% reduced)  "
            f"hull {totals[element]['hull_points_per_ray']:.1f} pts/ray  "
            f"time x{time_ratio:.2f}"
        )

    write_tex_outputs(build_figure(totals), build_table(totals))
    print(f"Saved figure assets to {PAPER_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

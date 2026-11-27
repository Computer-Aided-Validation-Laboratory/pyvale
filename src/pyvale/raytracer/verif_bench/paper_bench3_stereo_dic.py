#!/usr/bin/env python3
"""Paper figure/table generator for benchmark case 3 (stereo DIC).

Consumes the CSV and PNG outputs written by
``tests/raytracer/thesis-tests/application_4_fea_compare.py`` and writes the
LaTeX snippets used by the manuscript:

* ``figs_bench3.tex`` -- displacement-field figure, residual maps and a
  quantitative error summary;
* ``tabs_bench3.tex`` -- calibration/shape and pointwise-error tables.

The comparison data are not re-computed here.  Run the comparison first::

    /home/wiera/anaconda3/envs/pyvale/bin/python \
        tests/raytracer/thesis-tests/application_4_fea_compare.py

Then run this script with the same Python environment.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_const import repo_root


PROJECT_ROOT = repo_root().parents[2]
DEFAULT_CASE_BASE = "app4_fea/ssaa_2_8"
DEFAULT_COMPARE_DIR = (
    PROJECT_ROOT
    / "tests/raytracer/thesis-output"
    / DEFAULT_CASE_BASE
    / "compare"
)

PAPER_DIR = Path(
    "/home/wiera/Documents/paper-feebee-raytracer/figures/bench3",
).expanduser()
BENCH_DIR = Path("verif_bench/bench/bench3")

REQUIRED_CSVS = (
    "calibration_and_shape_comparison.csv",
    "dic_vs_fea_errors.csv",
    "dic_intercase_errors.csv",
)

SOURCE_FIGURES = {
    "compare_displacement_fields.png": "fig_bench3_stereo_dic_fields.png",
    "compare_dic_vs_fea_errors.png": "fig_bench3_stereo_dic_residuals.png",
}

CASE_ORDER = ("air_diffuse", "pipe", "water")
CASE_LABEL = {
    "air_diffuse": "Air (diffuse)",
    "pipe": "Pipe (acrylic)",
    "water": "Water + pipe",
}
CASE_SHORT = {
    "air_diffuse": "air",
    "pipe": "pipe",
    "water": "water",
}
CASE_COLOR = {
    "air_diffuse": "#1f77b4",
    "pipe": "#ff7f0e",
    "water": "#2ca02c",
}
COMPONENTS = ("u", "v", "w", "magnitude")
COMPONENT_LABEL = {
    "u": r"$u$",
    "v": r"$v$",
    "w": r"$w$",
    "magnitude": r"$|\mathbf{d}|$",
}
MAPPING_ORDER = ("refracted", "nominal")
MAPPING_LABEL = {
    "refracted": "refracted",
    "nominal": "nominal",
}
PAIR_ORDER = (
    "pipe - air_diffuse",
    "water - air_diffuse",
    "water - pipe",
)
PAIR_LABEL = {
    "pipe - air_diffuse": "pipe $-$ air",
    "water - air_diffuse": "water $-$ air",
    "water - pipe": "water $-$ pipe",
}

PLOT_DPI = 300.0

FIELD_FIG_CAPTION = (
    "Displacement fields in benchmark case 3. Rows are $u$, $v$, $w$ and "
    "$|\\mathbf{d}|$ in the stereo Camera 0 frame; the first column is the "
    "\\gls{fe} ground truth (scaled frame 63), and the remaining columns are stereo 3D "
    "\\gls{dic} for the three optical cases. Each \\gls{dic} subset is mapped to its "
    "true material point through the air, acrylic, or water optical path."
)

RESIDUAL_FIG_CAPTION = (
    "Spatial displacement residuals in benchmark case 3. Rows are $u$, $v$, "
    "$w$ and $|\\mathbf{d}|$; columns are the air, acrylic-pipe and "
    "water-filled-pipe cases. The quantity plotted is "
    "$\\mathrm{DIC}-\\mathrm{FEA}$ at the undeformed material point, using the "
    "refracted back-projection. Positive bias values mean that DIC exceeds FEA."
)

SUMMARY_FIG_CAPTION = (
    "Stereo-DIC errors in benchmark case 3. "
    "(a) Pointwise \\gls{rms} difference between stereo DIC and FEA for each "
    "displacement component, using the refracted true-point mapping. "
    "(b) Pairwise DIC differences in the displacement magnitude on the shared "
    "plate grid."
)

CALIBRATION_TABLE_CAPTION = (
    "Calibration for benchmark case 3. The stereo "
    "calibration uses thirteen rendered target poses through the same optical "
    "path as the measurement. ``Reproj.'' is the mean Camera 0 reprojection "
    "residual. ``$z$ ptp'' is the peak-to-peak variation of the "
    "stereo-triangulated plate depth. "
    "``Lat.\\ refr.'' is the maximum horizontal difference between the "
    "refracted and nominal straight-ray back-projections over the plate width. "
    "$N$ is the number of DIC subsets."
)

ERROR_TABLE_CAPTION = (
    "Pointwise stereo-DIC displacement errors in benchmark case 3. Errors are "
    "the \\gls{rms} of $\\mathrm{DIC}-\\mathrm{FEA}$ at the same undeformed "
    "material point. ``Refracted'' maps "
    "each reference subset through the analytical air/acrylic/water optical "
    "path; ``nominal'' uses the uncorrected straight-ray pinhole "
    "back-projection. The two mappings coincide in air. The final column is "
    "the magnitude error with "
    "$1$\\,px $=39.06$\\,$\\mu$m at the plate. $N$ is the number of matched "
    "subsets."
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def _case_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["case"]: row for row in rows if row.get("case") in CASE_ORDER}


def _error_lookup(rows: list[dict[str, str]]) -> dict:
    lookup = {}
    for row in rows:
        lookup[
            (row["case"], row["mapping"], row["component"])
        ] = row
    return lookup


def load_inputs(compare_dir: Path) -> tuple:
    for name in REQUIRED_CSVS:
        path = compare_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing comparison output: {path}\n"
                "Run application_4_fea_compare.py first, for example:\n"
                "  /home/wiera/anaconda3/envs/pyvale/bin/python "
                "tests/raytracer/thesis-tests/application_4_fea_compare.py"
            )

    calib = _case_rows(
        _read_csv(compare_dir / "calibration_and_shape_comparison.csv")
    )
    errors = _read_csv(compare_dir / "dic_vs_fea_errors.csv")
    intercase = _read_csv(compare_dir / "dic_intercase_errors.csv")
    return calib, errors, intercase


def _copy_source_figures(compare_dir: Path, destinations: tuple[Path, ...]) -> None:
    for source_name, target_name in SOURCE_FIGURES.items():
        source = compare_dir / source_name
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing comparison figure: {source}\n"
                "Run application_4_fea_compare.py without --no-plots."
            )
        for destination in destinations:
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / target_name)


def build_summary_figure(
    errors: list[dict[str, str]],
    intercase: list[dict[str, str]],
    destinations: tuple[Path, ...],
) -> str:
    lookup = _error_lookup(errors)
    case_names = [c for c in CASE_ORDER if any(k[0] == c for k in lookup)]
    if not case_names:
        raise RuntimeError("No DIC-vs-FEA rows found in dic_vs_fea_errors.csv")

    fig, (ax_rmse, ax_inter) = plt.subplots(
        1, 2, figsize=(7.4, 3.05), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    # (a) Pointwise RMSE by component and optical case.
    x = np.arange(len(COMPONENTS), dtype=float)
    group_width = 0.78
    bar_width = group_width / max(len(case_names), 1)
    for index, case in enumerate(case_names):
        values = [
            _float(lookup.get((case, "refracted", comp), {}), "rmse_um")
            for comp in COMPONENTS
        ]
        offset = (index - 0.5 * (len(case_names) - 1)) * bar_width
        ax_rmse.bar(
            x + offset,
            values,
            width=bar_width * 0.92,
            color=CASE_COLOR.get(case, f"C{index}"),
            edgecolor="black",
            linewidth=0.5,
            label=CASE_LABEL.get(case, case),
        )
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels([COMPONENT_LABEL[c] for c in COMPONENTS])
    ax_rmse.set_yscale("log")
    ax_rmse.set_ylabel(r"RMSE$(\mathrm{DIC}-\mathrm{FEA})$ [$\mu$m]")
    ax_rmse.set_title("(a) Pointwise RMSE (refracted mapping)")
    ax_rmse.grid(True, which="both", axis="y", alpha=0.30)
    ax_rmse.legend(fontsize=7.2, loc="upper left", framealpha=0.9)
    ax_rmse.tick_params(axis="both", labelsize=7.5)

    finite = [
        _float(lookup.get((case, "refracted", comp), {}), "rmse_um")
        for case in case_names
        for comp in COMPONENTS
    ]
    finite = [value for value in finite if np.isfinite(value) and value > 0.0]
    if finite:
        ax_rmse.set_ylim(min(finite) * 0.55, max(finite) * 2.4)

    # (b) Pairwise DIC differences in |d| on the shared plate grid.
    pair_rows = {row["comparison"]: row for row in intercase}
    pair_names = [pair for pair in PAIR_ORDER if pair in pair_rows]
    if pair_names:
        x_pair = np.arange(len(pair_names), dtype=float)
        width = 0.34
        rmse = [
            _float(pair_rows[pair], "rmse_um")
            for pair in pair_names
        ]
        mae = [
            _float(pair_rows[pair], "mae_um")
            for pair in pair_names
        ]
        ax_inter.bar(
            x_pair - width / 2.0,
            rmse,
            width,
            color="#3b6ea5",
            edgecolor="black",
            linewidth=0.5,
            label="RMSE",
        )
        ax_inter.bar(
            x_pair + width / 2.0,
            mae,
            width,
            color="#b0b0b0",
            edgecolor="black",
            linewidth=0.5,
            label="MAE",
        )
        ax_inter.set_xticks(x_pair)
        ax_inter.set_xticklabels([PAIR_LABEL[pair] for pair in pair_names])
        ax_inter.set_yscale("log")
        ax_inter.set_ylabel(r"$|\mathbf{d}|$ difference [$\mu$m]")
        ax_inter.set_title("(b) DIC--DIC consistency")
        ax_inter.grid(True, which="both", axis="y", alpha=0.30)
        ax_inter.legend(fontsize=7.2, loc="upper left", framealpha=0.9)
        ax_inter.tick_params(axis="both", labelsize=7.5)
        finite_pair = [value for value in rmse + mae if np.isfinite(value) and value > 0.0]
        if finite_pair:
            ax_inter.set_ylim(min(finite_pair) * 0.55, max(finite_pair) * 2.5)
    else:
        ax_inter.axis("off")

    fig.tight_layout()
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        out_png = destination / "fig_bench3_stereo_dic_summary.png"
        fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        "\\includegraphics[width=0.96\\textwidth]"
        "{fig_bench3_stereo_dic_summary.png}\n"
        f"\\caption{{{SUMMARY_FIG_CAPTION}}}\n"
        "\\label{fig:bench3_stereo_dic_summary}\n"
        "\\end{figure}\n"
    )


def build_figures(summary_tex: str) -> str:
    fields = (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        "\\includegraphics[width=0.96\\textwidth]"
        "{fig_bench3_stereo_dic_fields.png}\n"
        f"\\caption{{{FIELD_FIG_CAPTION}}}\n"
        "\\label{fig:bench3_stereo_dic_fields}\n"
        "\\end{figure}\n"
    )
    residuals = (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        "\\includegraphics[width=0.86\\textwidth]"
        "{fig_bench3_stereo_dic_residuals.png}\n"
        f"\\caption{{{RESIDUAL_FIG_CAPTION}}}\n"
        "\\label{fig:bench3_stereo_dic_residuals}\n"
        "\\end{figure}\n"
    )
    return fields + residuals + summary_tex


def _pass_fail(value: float) -> str:
    return "pass" if value >= 0.5 else "fail"


def build_tables(calib: dict, errors: list[dict[str, str]]) -> str:
    lookup = _error_lookup(errors)

    calib_lines = []
    for case in CASE_ORDER:
        row = calib.get(case)
        if row is None:
            continue
        calib_lines.append(
            "{case} & {status} & {fx:.1f} & {cx:.1f} & {base:.3f} & "
            "{reproj:.3f} & {ptp:.3f} & {lat:.3f} & {n:d} \\\\".format(
                case=CASE_LABEL[case],
                status=_pass_fail(_float(row, "quality_passed")),
                fx=_float(row, "fx0_px"),
                cx=_float(row, "cx0_px"),
                base=_float(row, "baseline_mm"),
                reproj=_float(row, "reproj_mean0_px"),
                ptp=_float(row, "z_world_ptp_mm"),
                lat=_float(row, "lateral_refraction_max_mm"),
                n=int(_float(row, "subsets")),
            )
        )

    calibration_table = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{CALIBRATION_TABLE_CAPTION}}}\n"
        "\\label{tab:bench3_calibration}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrrrrr}\n"
        "\\hline\n"
        "Case & Cal. & \\makecell{$f_{x,0}$\\\\ {[}px{]}} & "
        "\\makecell{$c_{x,0}$\\\\ {[}px{]}} & "
        "\\makecell{Baseline\\\\ {[}mm{]}} & "
        "\\makecell{Reproj.\\\\ {[}px{]}} & "
        "\\makecell{$z$ ptp\\\\ {[}mm{]}} & "
        "\\makecell{Lat.\\ refr.\\\\ {[}mm{]}} & $N$ \\\\\n"
        "\\hline\n"
        f"{chr(10).join(calib_lines)}\n"
        "\\hline\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )

    error_lines = []
    for case in CASE_ORDER:
        mappings = [
            mapping
            for mapping in MAPPING_ORDER
            if any(key[0] == case and key[1] == mapping for key in lookup)
        ]
        # Air has no refraction, so the refracted and nominal rows are
        # identical; report the primary (refracted) row only.
        if case == "air_diffuse" and "refracted" in mappings:
            mappings = ["refracted"]
        for mapping in mappings:
            magnitude_row = lookup.get((case, mapping, "magnitude"))
            if magnitude_row is None:
                continue
            values = [
                _float(lookup.get((case, mapping, comp), {}), "rmse_um")
                for comp in ("u", "v", "w")
            ]
            error_lines.append(
                "{case} & {mapping} & {n:d} & {u:.4f} & {v:.4f} & {w:.4f} & "
                "{mag:.4f} & {magpx:.2f} \\\\".format(
                    case=CASE_LABEL[case],
                    mapping=MAPPING_LABEL[mapping],
                    n=int(_float(magnitude_row, "n_matched")),
                    u=values[0],
                    v=values[1],
                    w=values[2],
                    mag=_float(magnitude_row, "rmse_um"),
                    magpx=_float(magnitude_row, "rmse_px") * 1.0e3,
                )
            )

    error_table = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{ERROR_TABLE_CAPTION}}}\n"
        "\\label{tab:bench3_stereo_dic}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{llrrrrrr}\n"
        "\\hline\n"
        "Case & Mapping & $N$ & \\makecell{RMSE\\\\$u$ [$\\mu$m]} & "
        "\\makecell{RMSE\\\\$v$ [$\\mu$m]} & "
        "\\makecell{RMSE\\\\$w$ [$\\mu$m]} & "
        "\\makecell{RMSE\\\\$|\\mathbf{d}|$ [$\\mu$m]} & "
        "\\makecell{RMSE\\\\$|\\mathbf{d}|$ [mpx]} \\\\\n"
        "\\hline\n"
        f"{chr(10).join(error_lines)}\n"
        "\\hline\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )

    return calibration_table + error_table


def write_outputs(
    figs_tex: str,
    tabs_tex: str,
    paper_dir: Path,
    bench_dir: Path,
) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)
    for directory in (paper_dir, bench_dir):
        (directory / "figs_bench3.tex").write_text(figs_tex)
        (directory / "tabs_bench3.tex").write_text(tabs_tex)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the LaTeX figures and tables for benchmark case 3 "
            "from application_4_fea_compare.py outputs."
        )
    )
    parser.add_argument(
        "--case-base-dir",
        type=str,
        default=DEFAULT_CASE_BASE,
        help=(
            "Case directory under thesis-output/ containing the compare/ "
            f"outputs (default: {DEFAULT_CASE_BASE})"
        ),
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=None,
        help="Override the directory containing the comparison CSVs and PNGs.",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=PAPER_DIR,
        help=f"Paper figure directory (default: {PAPER_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Local snippet output directory (default: "
            f"{repo_root() / BENCH_DIR})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    compare_dir = (
        args.compare_dir.expanduser().resolve()
        if args.compare_dir is not None
        else (
            PROJECT_ROOT
            / "tests/raytracer/thesis-output"
            / args.case_base_dir
            / "compare"
        ).resolve()
    )
    paper_dir = args.paper_dir.expanduser().resolve()
    bench_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (repo_root() / BENCH_DIR).resolve()
    )

    calib, errors, intercase = load_inputs(compare_dir)
    _copy_source_figures(compare_dir, (paper_dir, bench_dir))
    summary_tex = build_summary_figure(errors, intercase, (paper_dir, bench_dir))

    figs_tex = build_figures(summary_tex)
    tabs_tex = build_tables(calib, errors)
    write_outputs(figs_tex, tabs_tex, paper_dir, bench_dir)

    print(f"Read comparison outputs from {compare_dir}")
    print(f"Wrote {paper_dir / 'figs_bench3.tex'}")
    print(f"Wrote {paper_dir / 'tabs_bench3.tex'}")
    print(f"Wrote {bench_dir / 'figs_bench3.tex'}")
    print(f"Wrote {bench_dir / 'tabs_bench3.tex'}")
    print("Figures:")
    for name in (
        "fig_bench3_stereo_dic_fields.png",
        "fig_bench3_stereo_dic_residuals.png",
        "fig_bench3_stereo_dic_summary.png",
    ):
        print(f"  {paper_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

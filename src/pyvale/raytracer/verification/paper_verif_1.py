#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import math
import pathlib
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from paper_const import (
    PAPER_DIR,
    PLOT_AXIS_FONT_SIZE,
    PLOT_LEGEND_FONT_SIZE,
    PLOT_RESOLUTION_DPI,
    PLOT_SQUARE_FIG_SIZE_IN,
    PLOT_TICK_FONT_SIZE,
    PLOT_TITLE_FONT_SIZE,
    SHEAR_REGULAR,
    SHEAR_SHEAR,
    bulge_in_frame,
    bulge_in_limit_frame,
    bulge_out_frame,
    bulge_out_limit_frame,
    bulge_regular_frame,
    repo_root,
)


VERIF_DIR = pathlib.Path("verification/verif/verif_1")
OUT_TABS_TEX_PATH = VERIF_DIR / "verif_1_tabs.tex"
OUT_FIGS_TEX_PATH = VERIF_DIR / "verif_1_figs.tex"
SCI_THRESHOLD = 1.0e-12
VERIF_A_MAP_WIDTH = "0.19\\textwidth"

TABLE_CAPTION = (
    "Newton inverse solver reprojection accuracy for stable "
    "element geometries in verification case 1."
)

BULGE_FIG_CAPTION = (
    "Newton inverse solver reprojection error maps for the bulge tests in "
    "verification case 1."
)

SHEAR_FIG_CAPTION = (
    "Newton inverse solver reprojection error maps for the shear test in "
    "verification case 1."
)


@dataclass(slots=True)
class PlotStyle:
    cmap_seq: str
    resolution: float
    single_fig_size_square: tuple[float, float]
    font_ax_size: float
    font_tick_size: float
    font_head_size: float
    font_leg_size: float


def load_plot_style() -> PlotStyle:
    visualopts_path = pathlib.Path(
        "/home/lloydf/pyvale/src/pyvale/sensorsim/visualopts.py",
    )
    if visualopts_path.exists():
        spec = importlib.util.spec_from_file_location(
            "pyvale_visualopts_local",
            visualopts_path,
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            plot_opts = module.PlotOptsGeneral()
            return PlotStyle(
                cmap_seq=plot_opts.cmap_seq,
                resolution=plot_opts.resolution,
                single_fig_size_square=plot_opts.single_fig_size_square,
                font_ax_size=plot_opts.font_ax_size,
                font_tick_size=plot_opts.font_tick_size,
                font_head_size=plot_opts.font_head_size,
                font_leg_size=plot_opts.font_leg_size,
            )

    return PlotStyle(
        cmap_seq="viridis",
        resolution=PLOT_RESOLUTION_DPI,
        single_fig_size_square=PLOT_SQUARE_FIG_SIZE_IN,
        font_ax_size=PLOT_AXIS_FONT_SIZE,
        font_tick_size=PLOT_TICK_FONT_SIZE,
        font_head_size=PLOT_TITLE_FONT_SIZE,
        font_leg_size=PLOT_LEGEND_FONT_SIZE,
    )


def load_solver_rows(stats_path: pathlib.Path) -> list[dict[str, str]]:
    with stats_path.open(newline="") as stats_file:
        return list(csv.DictReader(stats_file))


def mesh_label(mesh_name: str) -> str:
    if mesh_name == "quad4ibi" or mesh_name == "quad4newton":
        return "quad4"
    return mesh_name


def verif_1_dir(mesh_name: str, distort_name: str) -> pathlib.Path:
    return repo_root() / VERIF_DIR / f"a_distort_{distort_name}_{mesh_name}"


def solver_stats_path(mesh_name: str, distort_name: str, frame_idx: int) -> pathlib.Path:
    return (
        verif_1_dir(mesh_name, distort_name) /
        f"solver_stats_frame{frame_idx}.csv"
    )


def ensure_solver_stats_exists(mesh_name: str, distort_name: str, frame_idx: int) -> pathlib.Path:
    stats_path = solver_stats_path(mesh_name, distort_name, frame_idx)
    if not stats_path.exists():
        raise FileNotFoundError(f"missing {stats_path}")
    return stats_path


def accepted_mask(rows: list[dict[str, str]]) -> np.ndarray:
    mask_vals = []
    for row in rows:
        converged = row["converged"] == "1"
        in_domain = row["in_domain"] == "1"
        reproj_err = float(row["reproj_err"])
        mask_vals.append(converged and in_domain and math.isfinite(reproj_err))
    return np.asarray(mask_vals, dtype=bool)


def get_reproj_err_array(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([float(row["reproj_err"]) for row in rows], dtype=float)
    # return np.asarray([float(row["err_dir"]) for row in rows], dtype=float)

def get_iters_array(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([float(row["iters"]) for row in rows], dtype=float)


def calc_case_stats(rows: list[dict[str, str]]) -> tuple[float, float, float]:
    accepted = accepted_mask(rows)
    reproj_err = get_reproj_err_array(rows)
    iters = get_iters_array(rows)

    reproj_acc = reproj_err[accepted]
    iters_acc = iters[accepted]
    if reproj_acc.size == 0:
        return math.nan, math.nan, math.nan

    rms = float(np.sqrt(np.mean(reproj_acc * reproj_acc)))
    max_err = float(np.max(reproj_acc))
    mean_iters = float(np.mean(iters_acc))
    return rms, max_err, mean_iters


def build_reproj_map(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    row_idx = np.asarray([int(row["row_idx"]) for row in rows], dtype=int)
    col_idx = np.asarray([int(row["col_idx"]) for row in rows], dtype=int)
    xi_vals = np.asarray([float(row["ideal_xi"]) for row in rows], dtype=float)
    eta_vals = np.asarray([float(row["ideal_eta"]) for row in rows], dtype=float)
    reproj_err = get_reproj_err_array(rows)
    accepted = accepted_mask(rows)

    rows_num = int(np.max(row_idx)) + 1
    cols_num = int(np.max(col_idx)) + 1
    err_map = np.full((rows_num, cols_num), np.nan, dtype=float)
    xi_map = np.full((rows_num, cols_num), np.nan, dtype=float)
    eta_map = np.full((rows_num, cols_num), np.nan, dtype=float)

    err_map[row_idx[accepted], col_idx[accepted]] = reproj_err[accepted]
    xi_map[row_idx, col_idx] = xi_vals
    eta_map[row_idx, col_idx] = eta_vals

    extent = (
        float(np.nanmin(xi_map)),
        float(np.nanmax(xi_map)),
        float(np.nanmin(eta_map)),
        float(np.nanmax(eta_map)),
    )
    return err_map, extent


def fmt_sci(value: float) -> str:
    if value == 0.0:
        return "$0$"
    if abs(value) < SCI_THRESHOLD:
        return "$< 1.00 \\times 10^{-12}$"
    exponent = int(math.floor(math.log10(abs(value))))
    scale = 10.0 ** exponent
    mantissa = value / scale
    return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"


def fmt_iters(value: float) -> str:
    if math.isnan(value):
        return "--"
    return f"{value:.3f}"


def make_table_row(
    element_name: str,
    case_name: str,
    rows: list[dict[str, str]],
) -> str:
    rms, max_err, mean_iters = calc_case_stats(rows)
    return (
        f"\\texttt{{{element_name}}} & {case_name} & "
        f"{fmt_sci(rms)} & {fmt_sci(max_err)} & "
        f"{fmt_iters(mean_iters)} \\\\"
    )


def build_case_rows_for_table() -> list[tuple[str, str, str, int, str]]:
    return [
        ("quad4", "quad4newton", "shear", SHEAR_REGULAR, "regular"),
        ("quad4", "quad4newton", "shear", SHEAR_SHEAR, "shear"),
        ("tri6", "tri6", "shear", SHEAR_REGULAR, "regular"),
        ("tri6", "tri6", "shear", SHEAR_SHEAR, "shear"),
        ("tri6", "tri6", "bulge", bulge_in_frame("tri6"), "bulge inward"),
        ("tri6", "tri6", "bulge", bulge_out_frame("tri6"), "bulge outward"),
        ("quad8", "quad8", "shear", SHEAR_REGULAR, "regular"),
        ("quad8", "quad8", "shear", SHEAR_SHEAR, "shear"),
        ("quad8", "quad8", "bulge", bulge_in_frame("quad8"), "bulge inward"),
        (
            "quad8",
            "quad8",
            "bulge",
            bulge_out_frame("quad8"),
            "bulge outward",
        ),
        ("quad9", "quad9", "shear", SHEAR_REGULAR, "regular"),
        ("quad9", "quad9", "shear", SHEAR_SHEAR, "shear"),
        ("quad9", "quad9", "bulge", bulge_in_frame("quad9"), "bulge inward"),
        (
            "quad9",
            "quad9",
            "bulge",
            bulge_out_frame("quad9"),
            "bulge outward",
        ),
    ]


def build_table_tex() -> str:
    body_rows: list[str] = []
    for elem_label, mesh_name, distort_name, frame_idx, case_name in build_case_rows_for_table():
        stats_path = ensure_solver_stats_exists(mesh_name, distort_name, frame_idx)
        rows = load_solver_rows(stats_path)
        body_rows.append(make_table_row(elem_label, case_name, rows))

    body = "\n".join(body_rows)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{TABLE_CAPTION}}}\n"
        "\\label{tab:verification_case_a_reprojection}\n"
        "\\begin{tabular}{llrrr}\n"
        "\\hline\n"
        "Element & Case & RMS $\\epsilon_{rp}$ [px] & "
        "Max $\\epsilon_{rp}$ [px] & Mean iters. \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def save_png_figure(
    mesh_name: str,
    err_map: np.ndarray,
    extent: tuple[float, float, float, float],
    out_name: str,
    style: PlotStyle,
) -> pathlib.Path:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PAPER_DIR / out_name
    svg_path = out_path.with_suffix(".svg")
    finite_vals = err_map[np.isfinite(err_map)]
    scale_exp = 0
    if finite_vals.size > 0:
        max_abs = float(np.max(np.abs(finite_vals)))
        if max_abs > 0.0:
            scale_exp = int(math.floor(math.log10(max_abs)))
    scale = 10.0 ** scale_exp

    fig, ax = plt.subplots(
        figsize=style.single_fig_size_square,
        constrained_layout=False,
    )
    plot_map = np.ma.masked_invalid(err_map / scale)
    im = ax.imshow(
        plot_map,
        origin="lower",
        extent=extent,
        cmap=style.cmap_seq,
        interpolation="none",
        aspect="equal",
    )
    ax.set_box_aspect(1.0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.04)
    cbar = fig.colorbar(im, cax=cax)
    ax.set_xlabel(r"$\xi$", fontsize=PLOT_AXIS_FONT_SIZE)
    ax.set_ylabel(r"$\eta$", fontsize=PLOT_AXIS_FONT_SIZE)
    ax.tick_params(labelsize=PLOT_TICK_FONT_SIZE)
    if mesh_name in {"quad4newton", "quad8", "quad9"}:
        tick_vals = [-1.0, 0.0, 1.0]
        tick_labels = ["-1.0", "0.0", "1.0"]
    else:
        tick_vals = [0.0, 1.0]
        tick_labels = ["0.0", "1.0"]
    ax.set_xticks(tick_vals, tick_labels)
    ax.set_yticks(tick_vals, tick_labels)
    if scale_exp == 0:
        ax.set_title(
            # r"$\epsilon_{rp}$ [px]",
            r"$\epsilon_{rp}$ [\%]",
            fontsize=PLOT_TITLE_FONT_SIZE,
            pad=16.0,
        )
    else:
        ax.set_title(
            # rf"$\epsilon_{{rp}}$ [px] $(\times 10^{{{scale_exp}}})$",
            rf"$\epsilon_{{rp}}$ [%] $(\times 10^{{{scale_exp}}})$",
            fontsize=PLOT_TITLE_FONT_SIZE,
            pad=16.0,
        )
    cbar.ax.tick_params(labelsize=PLOT_TICK_FONT_SIZE)
    fig.subplots_adjust(
        left=0.16,
        right=0.86,
        bottom=0.15,
        top=0.90,
    )
    fig.savefig(
        out_path,
        dpi=style.resolution,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    return out_path


def subfigure_block(
    file_name: str,
    width_str: str,
    label: str,
) -> str:
    return (
        f"\\begin{{subfigure}}[c]{{{width_str}}}\n"
        "\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{{file_name}}}\n"
        "\\caption{}\n"
        f"\\label{{{label}}}\n"
        "\\end{subfigure}"
    )


def build_bulge_fig_tex(style: PlotStyle) -> str:
    del style
    mesh_names = ["tri6", "quad8", "quad9"]
    col_specs = [
        ("Outward", "bulge_out"),
        ("Regular", "bulge_regular"),
        ("Inward", "bulge_in"),
    ]
    row_blocks: list[str] = []
    for mesh_name in mesh_names:
        subfig_blocks: list[str] = []
        for _, suffix in col_specs:
            file_name = f"fig_verifa_{mesh_name}_{suffix}.png"
            subfig_blocks.append(
                subfigure_block(
                    file_name,
                    VERIF_A_MAP_WIDTH,
                    f"fig:verifa_{mesh_name}_{suffix}",
                )
            )
        row_blocks.append(
            "\\makecell[c]{\\texttt{" + mesh_name + "}} & " +
            " & ".join(subfig_blocks) + " \\\\"
        )

    rows_tex = "\n".join(row_blocks)
    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        "\\begin{tabular}{cccc}\n"
        "\\makecell[c]{Element} & \\makecell[c]{Outward} & "
        "\\makecell[c]{Regular} & "
        "\\makecell[c]{Inward}\\\\\n"
        f"{rows_tex}\n"
        "\\end{tabular}\n"
        f"\\caption{{{BULGE_FIG_CAPTION}}}\n"
        "\\label{fig:verification_case_a_bulge}\n"
        "\\end{figure}\n"
    )


def build_shear_fig_tex(style: PlotStyle) -> str:
    del style
    mesh_specs = ["quad4", "tri6", "quad8", "quad9"]
    col_specs = [("Regular", "regular"), ("Shear", "shear")]
    row_blocks: list[str] = []
    for mesh_name in mesh_specs:
        subfig_blocks: list[str] = []
        for _, suffix in col_specs:
            subfig_blocks.append(
                subfigure_block(
                    f"fig_verifa_{mesh_name}_{suffix}.png",
                    VERIF_A_MAP_WIDTH,
                    f"fig:verifa_{mesh_name}_{suffix}",
                )
            )
        row_blocks.append(
            "\\makecell[c]{\\texttt{" + mesh_name + "}} & " +
            " & ".join(subfig_blocks) + " \\\\"
        )

    rows_tex = "\n".join(row_blocks)
    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        "\\begin{tabular}{ccc}\n"
        "\\makecell[c]{Element} & \\makecell[c]{Regular} & "
        "\\makecell[c]{Shear}\\\\\n"
        f"{rows_tex}\n"
        "\\end{tabular}\n"
        f"\\caption{{{SHEAR_FIG_CAPTION}}}\n"
        "\\label{fig:verification_case_a_shear}\n"
        "\\end{figure}\n"
    )


def generate_case_figure(
    mesh_name: str,
    distort_name: str,
    frame_idx: int,
    out_name: str,
    style: PlotStyle,
) -> None:
    stats_path = ensure_solver_stats_exists(mesh_name, distort_name, frame_idx)
    rows = load_solver_rows(stats_path)
    err_map, extent = build_reproj_map(rows)
    save_png_figure(mesh_name, err_map, extent, out_name, style)


def generate_all_figures(style: PlotStyle) -> None:
#     print("Generating bulge RMSE figures...")
#     for mesh_name in ["tri6", "quad8", "quad9"]:
#         frame_cases = [
#             ("bulge_out_limit", bulge_out_limit_frame(mesh_name)),
#             ("bulge_out", bulge_out_frame(mesh_name)),
#             ("bulge_regular", bulge_regular_frame(mesh_name)),
#             ("bulge_in", bulge_in_frame(mesh_name)),
#             ("bulge_in_limit", bulge_in_limit_frame(mesh_name)),
#         ]
#         for suffix, frame_idx in frame_cases:
#             out_name = f"fig_verifa_{mesh_name}_{suffix}.png"
#             print(
#                 f"  {mesh_name} {suffix}: frame {frame_idx} -> {out_name}",
#             )
#             generate_case_figure(
#                 mesh_name,
#                 "bulge",
#                 frame_idx,
#                 out_name,
#                 style,
#             )

    print("Generating shear RMSE figures...")
    # for elem_label, mesh_name in [
    #     ("quad4", "quad4newton"),
    #     ("tri6", "tri6"),
    #     ("quad8", "quad8"),
    #     ("quad9", "quad9"),

    for elem_label, mesh_name in [
        ("tri6", "tri6"),
        ("quad8", "quad8"),
        ("quad9", "quad9"),
    ]:
        for suffix, frame_idx in [("regular", SHEAR_REGULAR), ("shear", SHEAR_SHEAR)]:
            out_name = f"fig_verifa_{elem_label}_{suffix}.png"
            print(
                f"  {elem_label} {suffix}: frame {frame_idx} -> {out_name}",
            )
            generate_case_figure(
                mesh_name,
                "shear",
                frame_idx,
                out_name,
                style,
            )


def write_tex_outputs(
    tabs_tex: str,
    figs_tex: str,
) -> None:
    out_tabs_path = repo_root() / OUT_TABS_TEX_PATH
    out_figs_path = repo_root() / OUT_FIGS_TEX_PATH
    paper_tabs_path = PAPER_DIR / OUT_TABS_TEX_PATH.name
    paper_figs_path = PAPER_DIR / OUT_FIGS_TEX_PATH.name

    out_tabs_path.parent.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    out_tabs_path.write_text(tabs_tex)
    out_figs_path.write_text(figs_tex)
    paper_tabs_path.write_text(tabs_tex)
    paper_figs_path.write_text(figs_tex)

    print(f"Wrote {out_tabs_path}")
    print(f"Wrote {out_figs_path}")
    print(f"Wrote {paper_tabs_path}")
    print(f"Wrote {paper_figs_path}")


def main() -> int:
    print("Loading pyvale-style plot options...")
    style = load_plot_style()
    # print("Building verif_1 LaTeX table...")
    # tables_tex = build_table_tex()
    print("Generating verif_1 figures...")
    generate_all_figures(style)
    print("Building verif_1 figure TeX...")
    figs_tex = build_shear_fig_tex(style) + "\n" + build_bulge_fig_tex(style)
    # write_tex_outputs(tables_tex, figs_tex)
    print(f"Saved PNG figures to {PAPER_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

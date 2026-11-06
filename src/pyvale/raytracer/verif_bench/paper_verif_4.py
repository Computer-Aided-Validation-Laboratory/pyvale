#!/usr/bin/env python3
"""Paper figure/table generator for verification case 4 (depth/occlusion).

Consumes the outputs written by ``verif_4_depth.py`` (the ``both.bmp``,
``frontonly.bmp`` and ``diff.bmp`` images for each ``d_<mesh>_rabbit`` case).
It writes the occlusion-count statistics CSV, exports the tri6 rabbit images
to PNG and produces the LaTeX figure and table.
"""

from __future__ import annotations

import csv
import pathlib
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from paper_const import repo_root
from pathlib import Path as _Path

PAPER_DIR = _Path(
    "/home/wiera/Documents/paper-feebee-raytracer/figures/verif_4",
).expanduser()


VERIF_DIR = pathlib.Path("verif_bench/verif/verif_4")
OUT_STATS_PATH = VERIF_DIR / "verif_4_stats.csv"
OUT_FIGS_TEX_PATH = VERIF_DIR / "figs_verif_4.tex"
OUT_TABS_TEX_PATH = VERIF_DIR / "tabs_verif_4.tex"

RABBIT_MESH_NAMES = ["tri3", "quad4", "tri6", "quad8", "quad9"]
FIGURE_MESH = "tri6"
FRONT_GREY = 255

PANEL_WIDTH = "0.30\\textwidth"
DIFF_FIG_SIZE = (3.0, 2.0)
PLOT_DPI = 300.0

FIG_CAPTION = (
    "Depth (occlusion) verification images for the \\texttt{tri6} rabbit meshes "
    "in verification case 4: (a) both meshes in the scene, (b) the front mesh "
    "only, and (c) the signed difference map between the two renders. The front "
    "mesh is drawn at full scale (255 GL) and the back mesh at half scale "
    "(128 GL); the difference is zero wherever the front mesh occludes the back "
    "mesh, confirming correct depth ordering. The difference is expressed in "
    "8-bit grey levels denoted by GL."
)

TABLE_CAPTION = (
    "Occlusion verification for the rabbit meshes in verification case 4. The "
    "counts are the number of pixels rendered at the front-mesh grey level "
    "(255 GL) with the front mesh alone and with both meshes in the scene. The "
    "difference is zero when the occluded back mesh does not alter the front "
    "mesh's visible footprint."
)


def verif_root() -> pathlib.Path:
    return repo_root() / VERIF_DIR


def run_verif_if_needed() -> None:
    if list(verif_root().glob("d_*_rabbit/both.bmp")):
        return
    print("Missing verif_4 outputs, running verif_4_depth.py...")
    subprocess.run(
        ["python3", "verif_bench/verif_4_depth.py"],
        cwd=repo_root(),
        check=True,
    )


def verif_case_dir(mesh_name: str) -> pathlib.Path:
    return verif_root() / f"d_{mesh_name}_rabbit"


def load_grey(path: pathlib.Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def count_front_pixels(mesh_name: str, image_name: str) -> int:
    image = load_grey(verif_case_dir(mesh_name) / image_name)
    return int(np.count_nonzero(image == FRONT_GREY))


def collect_stats() -> list[dict]:
    stats: list[dict] = []
    for mesh_name in RABBIT_MESH_NAMES:
        count_front = count_front_pixels(mesh_name, "frontonly.bmp")
        count_both = count_front_pixels(mesh_name, "both.bmp")
        stats.append(
            {
                "element": mesh_name,
                "count_front": count_front,
                "count_both": count_both,
                "diff": count_both - count_front,
            }
        )
    return stats


def write_stats_csv(stats: list[dict]) -> None:
    out_path = verif_root() / "verif_4_stats.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["element", "count front", "count both", "diff"])
        for row in stats:
            writer.writerow(
                [
                    row["element"],
                    row["count_front"],
                    row["count_both"],
                    row["diff"],
                ]
            )
    print(f"Wrote {out_path}")


def save_png_from_bmp(src_path: pathlib.Path, dst_name: str) -> pathlib.Path:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    dst_path = PAPER_DIR / dst_name
    with Image.open(src_path) as image:
        image.save(dst_path, format="PNG", optimize=True, dpi=(300, 300))
        fig, ax = plt.subplots(figsize=DIFF_FIG_SIZE, constrained_layout=False)
        ax.imshow(image, cmap="gray")
        ax.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig.savefig(dst_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
    return dst_path


def save_diff_png(diff_mat: np.ndarray, dst_name: str) -> pathlib.Path:
    vmax = float(np.max(np.abs(diff_mat)))
    if vmax == 0.0:
        vmax = 1.0

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    dst_path = PAPER_DIR / dst_name

    fig, ax = plt.subplots(figsize=DIFF_FIG_SIZE, constrained_layout=False)
    im = ax.imshow(
        diff_mat,
        origin="upper",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="none",
        aspect="equal",
    )
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Difference [GL]", fontsize=9.0)
    cbar.ax.tick_params(labelsize=8.0)
    fig.subplots_adjust(left=0.01, right=0.90, bottom=0.01, top=0.99)
    fig.savefig(dst_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(dst_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return dst_path


def export_figures() -> None:
    print(f"Exporting {FIGURE_MESH} rabbit figures...")
    case_dir = verif_case_dir(FIGURE_MESH)
    save_png_from_bmp(case_dir / "both.bmp", "fig_verifd_both.png")
    save_png_from_bmp(case_dir / "frontonly.bmp", "fig_verifd_frontonly.png")

    diff = (
        load_grey(case_dir / "both.bmp").astype(np.int32)
        - load_grey(case_dir / "frontonly.bmp").astype(np.int32)
    )
    save_diff_png(diff, "fig_verifd_diff.png")


def subfigure_block(file_name: str, width_str: str, label: str) -> str:
    return (
        f"\\begin{{subfigure}}[c]{{{width_str}}}\n"
        "\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{{file_name}}}\n"
        "\\caption{}\n"
        f"\\label{{{label}}}\n"
        "\\end{subfigure}"
    )


def build_figs_tex() -> str:
    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        + subfigure_block("fig_verifd_both.png", PANEL_WIDTH, "fig:verifd_both")
        + "\n\\hfill\n"
        + subfigure_block(
            "fig_verifd_frontonly.png", PANEL_WIDTH, "fig:verifd_frontonly"
        )
        + "\n\\hfill\n"
        + subfigure_block("fig_verifd_diff.png", PANEL_WIDTH, "fig:verifd_diff")
        + "\n"
        + f"\\caption{{{FIG_CAPTION}}}\n"
        + "\\label{fig:verification_case_d_tri6}\n"
        + "\\end{figure}\n"
    )


def build_table(stats: list[dict]) -> str:
    body_rows = [
        f"\\texttt{{{row['element']}}} & {row['count_front']} & "
        f"{row['count_both']} & {row['diff']} \\\\"
        for row in stats
    ]
    body = "\n".join(body_rows)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{TABLE_CAPTION}}}\n"
        "\\label{tab:verification_depth}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\hline\n"
        "Element & \\makecell{Count\\\\front} & \\makecell{Count\\\\both} & "
        "Diff.\\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def write_tex_outputs(figs_tex: str, tabs_tex: str) -> None:
    out_figs_path = repo_root() / OUT_FIGS_TEX_PATH
    out_tabs_path = repo_root() / OUT_TABS_TEX_PATH
    paper_figs_path = PAPER_DIR / OUT_FIGS_TEX_PATH.name
    paper_tabs_path = PAPER_DIR / OUT_TABS_TEX_PATH.name

    out_figs_path.parent.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    out_figs_path.write_text(figs_tex)
    out_tabs_path.write_text(tabs_tex)
    paper_figs_path.write_text(figs_tex)
    paper_tabs_path.write_text(tabs_tex)

    print(f"Wrote {out_figs_path}")
    print(f"Wrote {out_tabs_path}")
    print(f"Wrote {paper_figs_path}")
    print(f"Wrote {paper_tabs_path}")


def main() -> int:
    print("Checking verif_4 outputs...")
    run_verif_if_needed()

    print("Collecting occlusion statistics...")
    stats = collect_stats()
    for row in stats:
        print(
            f"  {row['element']}: front={row['count_front']} "
            f"both={row['count_both']} diff={row['diff']}"
        )
    write_stats_csv(stats)

    export_figures()
    write_tex_outputs(build_figs_tex(), build_table(stats))
    print(f"Saved figure assets to {PAPER_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

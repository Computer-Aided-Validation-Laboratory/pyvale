#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import pathlib
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from paper_const import (
    PAPER_DIR,
    SHEAR_REGULAR,
    SHEAR_SHEAR,
    bulge_in_frame,
    bulge_in_limit_frame,
    bulge_out_frame,
    bulge_out_limit_frame,
    bulge_regular_frame,
    repo_root,
)
from pathlib import Path as _Path
PAPER_DIR = _Path("/home/wiera/Documents/paper-feebee-raytracer/figures/verif_2").expanduser()


VERIF_DIR = pathlib.Path("verification/verif/verif_2")
SUMMARY_PATH = VERIF_DIR / "verif_2_summary.csv"
OUT_TABS_TEX_PATH = VERIF_DIR / "tabs_verif_2.tex"
OUT_FIGS_TEX_PATH = VERIF_DIR / "figs_verif_2.tex"
SCI_THRESHOLD = 1.0e-12

SHEAR_TABLE_CAPTION = (
    "Silhouette test result for affine deformation using "
    "pixel sub-sampling $SS=1$ for verification case 2."
)

BULGE_TABLE_CAPTION = (
    "Silhouette verification for the midside node bulge deformation using "
    "pixel sub-sampling $SS=1$ for verification case 2."
)

SHEAR_FIG_CAPTION = (
    "Silhouette verification images for the affine shear cases in "
    "verification case 2."
)

BULGE_FIG_CAPTION = (
    "Silhouette verification images for the nonlinear bulge cases in "
    "verification case 2."
)


def load_summary(summary_path: pathlib.Path) -> list[dict[str, str]]:
    with summary_path.open(newline="") as summary_file:
        reader = csv.DictReader(summary_file)
        return list(reader)


def find_bmp_path(row: dict[str, str]) -> pathlib.Path:
    image_path = repo_root() / row["image_path"]
    return image_path.with_suffix(".bmp")


def ensure_summary_rows() -> list[dict[str, str]]:
    summary_path = repo_root() / SUMMARY_PATH
    if not summary_path.exists():
        print(
            "Missing verif_2 summary, generating it with "
            "scripts/paper_verif_2_helper.py..."
        )
        subprocess.run(
            ["python3", "verification/paper_verif_2_helper.py"],
            cwd=repo_root(),
            check=True,
        )

    rows = load_summary(summary_path)
    needed_pairs = {
        ("tri6", "shear"),
        ("quad8", "shear"),
        ("quad9", "shear"),
        ("tri6", "bulge"),
        ("quad8", "bulge"),
        ("quad9", "bulge"),
    }
    present_pairs = {
        (row["mesh_name"], row["distort_name"])
        for row in rows
    }
    missing_pairs = sorted(needed_pairs - present_pairs)
    if missing_pairs:
        print(
            "verif_2 summary is missing required rows; regenerating with "
            "scripts/paper_verif_2_helper.py...",
        )
        subprocess.run(
            [
                "python3",
                "verification/paper_verif_2_helper.py",
                "--subset",
                "all",
            ],
            cwd=repo_root(),
            check=True,
        )
        rows = load_summary(summary_path)
        present_pairs = {
            (row["mesh_name"], row["distort_name"])
            for row in rows
        }
        missing_pairs = sorted(needed_pairs - present_pairs)
        if missing_pairs:
            raise RuntimeError(
                "verif_2 summary is still missing required rows after "
                "regeneration",
            )
    return rows


def find_row(
    rows: list[dict[str, str]],
    mesh_name: str,
    distort_name: str,
    frame_idx: int,
) -> dict[str, str]:
    for row in rows:
        if (
            row["mesh_name"] == mesh_name and
            row["distort_name"] == distort_name and
            int(row["frame_idx"]) == frame_idx
        ):
            return row
    raise KeyError(
        f"missing row for mesh={mesh_name} distort={distort_name} "
        f"frame={frame_idx}",
    )


def get_row_metric_vals(row: dict[str, str]) -> tuple[float, float, float, float, float]:
    centroid_x = float(row["centroid_diff_x_px"])
    centroid_y = float(row["centroid_diff_y_px"])
    centroid_r = float(row["centroid_dist_px"])
    area_err_pct = abs(float(row["area_err_pct"]))
    mask_diff_pct = float(row["mask_diff_pct"])
    return centroid_x, centroid_y, centroid_r, area_err_pct, mask_diff_pct


def fmt_sci(value: float) -> str:
    if abs(value) < SCI_THRESHOLD:
        return "$< 1.00 \\times 10^{-12}$"
    exponent = int(math.floor(math.log10(abs(value))))
    scale = 10.0 ** exponent
    mantissa = value / scale
    return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"


def fmt_percent(value: float) -> str:
    if value == 0.0:
        return "0.000"
    return f"{value:.3f}"


def make_table_row(
    element_name: str,
    case_name: str,
    row: dict[str, str],
) -> str:
    raw_vals = get_row_metric_vals(row)
    metric_vals = (
        fmt_sci(raw_vals[0]),
        fmt_sci(raw_vals[1]),
        fmt_sci(raw_vals[2]),
        fmt_percent(raw_vals[3]),
        fmt_percent(raw_vals[4]),
    )
    return (
        f"\\texttt{{{element_name}}} & {case_name} & "
        f"{metric_vals[0]} & {metric_vals[1]} & {metric_vals[2]} & "
        f"{metric_vals[3]} & {metric_vals[4]} \\\\"
    )


def build_shear_table(rows: list[dict[str, str]]) -> str:
    mesh_names = ["tri3", "quad4", "tri6", "quad8", "quad9"]
    body_rows: list[str] = []
    for mesh_name in mesh_names:
        row_regular = find_row(rows, mesh_name, "shear", SHEAR_REGULAR)
        row_shear = find_row(rows, mesh_name, "shear", SHEAR_SHEAR)
        body_rows.append(make_table_row(mesh_name, "regular", row_regular))
        body_rows.append(make_table_row(mesh_name, "shear", row_shear))

    body = "\n".join(body_rows)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{SHEAR_TABLE_CAPTION}}}\n"
        "\\label{tab:verification_silhouette_affine}\n"
        "\\begin{tabular}{llrrrrrr}\n"
        "\\hline\n"
        "Element & Case & \\makecell{Centroid\\\\Err. $x$ [px]} & "
        "\\makecell{Centroid\\\\Err. $y$ [px]} & "
        "\\makecell{Centroid\\\\Err. $r$ [px]} & "
        "\\makecell{Area\\\\Err. [\\%]} & "
        "\\makecell{Ref. Mask\\\\Err. [\\%]}\\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def build_bulge_table(rows: list[dict[str, str]]) -> str:
    mesh_names = ["tri6", "quad8", "quad9"]
    body_rows: list[str] = []
    for mesh_name in mesh_names:
        row_inward = find_row(
            rows,
            mesh_name,
            "bulge",
            bulge_in_frame(mesh_name),
        )
        row_regular = find_row(
            rows,
            mesh_name,
            "bulge",
            bulge_regular_frame(mesh_name),
        )
        row_outward = find_row(
            rows,
            mesh_name,
            "bulge",
            bulge_out_frame(mesh_name),
        )
        body_rows.append(make_table_row(mesh_name, "inward bulge", row_inward))
        body_rows.append(make_table_row(mesh_name, "regular", row_regular))
        body_rows.append(make_table_row(mesh_name, "outward bulge", row_outward))

    body = "\n".join(body_rows)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{BULGE_TABLE_CAPTION}}}\n"
        "\\label{tab:verification_silhouette_bulge}\n"
        "\\begin{tabular}{llrrrrrr}\n"
        "\\hline\n"
        "Element & Case & \\makecell{Centroid\\\\Err. $x$ [px]} & "
        "\\makecell{Centroid\\\\Err. $y$ [px]} & "
        "\\makecell{Centroid\\\\Err. $r$ [px]} & "
        "\\makecell{Area\\\\Err. [\\%]} & "
        "\\makecell{Ref. Mask\\\\Err. [\\%]}\\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def save_frame_png(
    row: dict[str, str],
    dst_name: str,
) -> pathlib.Path:
    src_path = find_bmp_path(row)
    if not src_path.exists():
        raise FileNotFoundError(f"missing source bmp {src_path}")

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    dst_path = PAPER_DIR / dst_name
    with Image.open(src_path) as img:
        img.save(
            dst_path,
            format="PNG",
            optimize=True,
            dpi=(300, 300),
        )
        fig, ax = plt.subplots(figsize=(4.0, 4.0), constrained_layout=False)
        ax.imshow(img)
        ax.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig.savefig(
            dst_path.with_suffix(".svg"),
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close(fig)
    return dst_path


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


def build_shear_fig_tex(rows: list[dict[str, str]]) -> str:
    mesh_names = ["tri3", "quad4", "tri6", "quad8", "quad9"]
    row_blocks: list[str] = []
    label_idx = 0
    for mesh_name in mesh_names:
        row_regular = find_row(rows, mesh_name, "shear", SHEAR_REGULAR)
        row_shear = find_row(rows, mesh_name, "shear", SHEAR_SHEAR)
        regular_name = f"fig_verifb_{mesh_name}_shear_regular.png"
        shear_name = f"fig_verifb_{mesh_name}_shear_shear.png"
        save_frame_png(row_regular, regular_name)
        save_frame_png(row_shear, shear_name)

        regular_label = chr(ord("a") + label_idx)
        shear_label = chr(ord("a") + label_idx + 1)
        label_idx += 2
        regular_block = subfigure_block(
            regular_name,
            "0.18\\textwidth",
            f"fig:verifb_{mesh_name}_shear_regular",
        )
        shear_block = subfigure_block(
            shear_name,
            "0.18\\textwidth",
            f"fig:verifb_{mesh_name}_shear_shear",
        )
        row_blocks.append(
            "\\makecell[c]{\\texttt{" + mesh_name + "}} & " +
            regular_block + " & " + shear_block + " \\\\"
        )

    rows_tex = "\n".join(row_blocks)
    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        "\\resizebox{0.48\\textwidth}{!}{%\n"
        "\\begin{tabular}{ccc}\n"
        "\\makecell[c]{Element} & \\makecell[c]{Regular} & "
        "\\makecell[c]{Shear}\\\\\n"
        f"{rows_tex}\n"
        "\\end{tabular}\n"
        "}\n"
        f"\\caption{{{SHEAR_FIG_CAPTION}}}\n"
        "\\label{fig:verification_silhouette_shear}\n"
        "\\end{figure}\n"
    )


def build_bulge_fig_tex(rows: list[dict[str, str]]) -> str:
    mesh_names = ["tri6", "quad8", "quad9"]
    row_blocks: list[str] = []
    label_idx = 0
    for mesh_name in mesh_names:
        frame_cases = [
            ("bulge_out", bulge_out_frame(mesh_name), "outward bulge"),
            ("bulge_regular", bulge_regular_frame(mesh_name), "regular"),
            ("bulge_in", bulge_in_frame(mesh_name), "inward bulge"),
        ]
        subfig_blocks: list[str] = []
        for suffix, frame_idx, _caption_name in frame_cases:
            row = find_row(rows, mesh_name, "bulge", frame_idx)
            file_name = f"fig_verifb_{mesh_name}_{suffix}.png"
            save_frame_png(row, file_name)
            label_idx += 1
            subfig_blocks.append(
                subfigure_block(
                    file_name,
                    "0.18\\textwidth",
                    f"fig:verifb_{mesh_name}_{suffix}",
                )
            )
        row_blocks.append(
            "\\makecell[c]{\\texttt{" + mesh_name + "}} & " +
            " & ".join(subfig_blocks) +
            " \\\\"
        )

    rows_tex = "\n".join(row_blocks)
    return (
        "\\begin{figure*}[htbp]\n"
        "\\centering\n"
        "\\resizebox{0.72\\textwidth}{!}{%\n"
        "\\begin{tabular}{cccc}\n"
        "\\makecell[c]{Element} & \\makecell[c]{Outward} & "
        "\\makecell[c]{Regular} & \\makecell[c]{Inward}\\\\\n"
        f"{rows_tex}\n"
        "\\end{tabular}\n"
        "}\n"
        f"\\caption{{{BULGE_FIG_CAPTION}}}\n"
        "\\label{fig:verification_silhouette_bulge}\n"
        "\\end{figure*}\n"
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
    print("Loading verif_2 summary...")
    rows = ensure_summary_rows()
    print(f"Loaded {len(rows)} summary rows.")

    print("Building LaTeX tables...")
    tables_tex = (
        build_shear_table(rows) +
        "\n" +
        build_bulge_table(rows)
    )
    print("Building figure TeX and exporting PNG figures...")
    figs_tex = (
        build_shear_fig_tex(rows) +
        "\n" +
        build_bulge_fig_tex(rows)
    )
    write_tex_outputs(tables_tex, figs_tex)
    print(f"Saved png figures to {PAPER_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

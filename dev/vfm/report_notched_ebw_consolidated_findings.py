"""Create consolidated notched-EBW findings and run-D basis diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from pyvale.vfm import ExperimentData, load_identification_result
from pyvale.vfm.postprocessing import (
    compute_plasticity_diagnostics,
    load_constitutive_law_from_result,
    load_known_parameter_maps,
)


DATASET = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/"
    "synthetic-fe/wdbn1-idealised-yield/pyvale-vfm"
)
RUN_D = (
    DATASET
    / "identification/prepared/"
    "factorial_D_spd_sensitivity_20260828_113526/identification_result.yaml"
)
LATEST_REPORT = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/investigation-reports/"
    "NOTCHED_EBW_BASIS_GROWTH_FACTORIAL_20260828_1230_BST.pdf"
)


@dataclass(slots=True)
class BasisState:
    centre: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]
    height: float


@dataclass(slots=True)
class SolveState:
    solve_index: int
    accepted: bool
    objective: float
    initial_map: npt.NDArray[np.float64]
    final_map: npt.NDArray[np.float64]
    initial_bases: list[BasisState]
    final_bases: list[BasisState]
    initial_rmse: float
    final_rmse: float
    initial_yielded_rmse: float
    final_yielded_rmse: float


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().astimezone()
    stamp = generated.strftime("%Y%m%d_%H%M_%Z")

    experiment = ExperimentData.load_from_file(args.input / "experiment_data.yaml")
    known = load_known_parameter_maps(args.input / "known_parameter_maps.npz")
    if known is None:
        raise RuntimeError("Known parameter maps are required.")
    result = load_identification_result(args.run_d)
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x,
        experiment.specimen_geometry.y,
    )
    plasticity = compute_plasticity_diagnostics(
        experiment,
        load_constitutive_law_from_result(result),
        known,
    )
    if plasticity is None:
        raise RuntimeError("Plasticity diagnostics are unavailable.")
    yielded = np.asarray(plasticity.yielded_datapoints, dtype=bool) & mask
    truth = np.asarray(known["yield_strength"], dtype=np.float64)
    solves = _reconstruct_solves(result, experiment, truth, mask, yielded)

    diagnostic = args.output / f"NOTCHED_EBW_RUN_D_BASIS_EVOLUTION_{stamp}.pdf"
    cover = args.output / f"_cover_{stamp}.pdf"
    workstation = args.output / f"_workstation_{stamp}.pdf"
    consolidated = args.output / f"NOTCHED_EBW_CONSOLIDATED_FINDINGS_{stamp}.pdf"

    _write_basis_diagnostic(
        diagnostic,
        solves,
        experiment,
        truth,
        mask,
        yielded,
        generated,
    )
    _write_cover(cover, solves, generated)
    _write_workstation_page(workstation, generated)
    subprocess.run(
        [
            "pdfunite",
            str(cover),
            str(args.latest_report),
            str(diagnostic),
            str(workstation),
            str(consolidated),
        ],
        check=True,
    )
    print(f"Basis diagnostic: {diagnostic}")
    print(f"Consolidated report: {consolidated}")


def _reconstruct_solves(result, experiment, truth, mask, yielded) -> list[SolveState]:
    x = np.asarray(experiment.specimen_geometry.x, dtype=np.float64)
    y = np.asarray(experiment.specimen_geometry.y, dtype=np.float64)
    states: list[SolveState] = []
    for solve in result.history.phases[-1].solve_results:
        if solve.final_snapshot is None:
            raise ValueError(f"Solve {solve.solve_iteration} has no final snapshot.")
        basis_snapshot = solve.final_snapshot.spatial_parameterisations[
            "yield_strength"
        ][1]
        references = [
            float(kernel["reference_variance"])
            for kernel in basis_snapshot.summary["kernels"]
        ]
        initial_map, initial_bases = _map_from_dofs(
            solve.initial_dofs, references, x, y
        )
        final_map, final_bases = _map_from_dofs(
            solve.final_dofs, references, x, y
        )
        states.append(
            SolveState(
                solve_index=int(solve.solve_iteration),
                accepted=bool(solve.accepted),
                objective=float(solve.final_objective["cost"]),
                initial_map=initial_map,
                final_map=final_map,
                initial_bases=initial_bases,
                final_bases=final_bases,
                initial_rmse=_rmse(initial_map - truth, mask),
                final_rmse=_rmse(final_map - truth, mask),
                initial_yielded_rmse=_rmse(initial_map - truth, yielded),
                final_yielded_rmse=_rmse(final_map - truth, yielded),
            )
        )
    return states


def _map_from_dofs(dofs, references, x, y):
    values = np.asarray(dofs, dtype=np.float64)
    num_bases = (values.size - 2) // 6
    if values.size != 6 * num_bases + 2 or len(references) != num_bases:
        raise ValueError("Unexpected SPD basis DOF layout.")
    homogeneous = float(values[5 * num_bases])
    heights = values[5 * num_bases + 1 : 6 * num_bases + 1]
    parameter_map = np.full(x.shape, homogeneous, dtype=np.float64)
    bases: list[BasisState] = []
    for index, (height, reference) in enumerate(
        zip(heights, references, strict=True)
    ):
        offset = 5 * index
        centre = values[offset : offset + 2].copy()
        symmetric_log = np.asarray(
            [
                [values[offset + 2], values[offset + 3]],
                [values[offset + 3], values[offset + 4]],
            ],
            dtype=np.float64,
        )
        covariance = float(reference) * expm(symmetric_log)
        inverse = np.linalg.inv(covariance)
        dx = x - centre[0]
        dy = y - centre[1]
        exponent = -0.5 * (
            inverse[0, 0] * dx**2
            + 2.0 * inverse[0, 1] * dx * dy
            + inverse[1, 1] * dy**2
        )
        parameter_map += float(height) * np.exp(exponent)
        bases.append(BasisState(centre, covariance, float(height)))
    return np.clip(parameter_map, 200.0, 2000.0), bases


def _rmse(error, selection) -> float:
    return float(np.sqrt(np.mean(np.asarray(error)[selection] ** 2)))


def _write_basis_diagnostic(
    path,
    solves,
    experiment,
    truth,
    mask,
    yielded,
    generated,
):
    x = np.asarray(experiment.specimen_geometry.x)
    y = np.asarray(experiment.specimen_geometry.y)
    extent = [np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)]
    values = np.concatenate(
        [state.initial_map[mask] for state in solves]
        + [state.final_map[mask] for state in solves]
        + [truth[mask]]
    )
    limits = (float(np.nanpercentile(values, 0.5)), float(np.nanpercentile(values, 99.5)))
    with PdfPages(
        path,
        metadata={
            "Title": "Notched EBW run D basis-function evolution",
            "Author": "PyVale diagnostic report",
            "CreationDate": generated,
        },
    ) as pdf:
        _basis_map_page(pdf, solves[:3], mask, extent, limits, generated)
        _basis_map_page(pdf, solves[3:], mask, extent, limits, generated)
        _accepted_rejected_page(pdf, solves, truth, mask, yielded, extent)
        _basis_interpretation_page(pdf, solves, generated)


def _basis_map_page(pdf, states, mask, extent, limits, generated):
    rows = len(states)
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(11.69, 8.27),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row, state in enumerate(states):
        for column, (label, values, bases, rmse, yielded_rmse) in enumerate(
            (
                (
                    "start",
                    state.initial_map,
                    state.initial_bases,
                    state.initial_rmse,
                    state.initial_yielded_rmse,
                ),
                (
                    "end",
                    state.final_map,
                    state.final_bases,
                    state.final_rmse,
                    state.final_yielded_rmse,
                ),
            )
        ):
            axis = axes[row, column]
            image = axis.imshow(
                np.where(mask, values, np.nan),
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis",
                vmin=limits[0],
                vmax=limits[1],
            )
            _draw_bases(axis, bases)
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
            status = "accepted" if state.accepted else "rejected"
            axis.set_title(
                f"Solve {state.solve_index}: {label} ({len(bases)} BF)\n"
                f"RMSE {rmse:.1f} MPa; yielded {yielded_rmse:.1f} MPa"
                + (f"; {status}" if label == "end" else ""),
                fontsize=10,
            )
            axis.set_xlabel("x [mm]")
            axis.set_ylabel("y [mm]")
    first = states[0].solve_index
    last = states[-1].solve_index
    fig.suptitle(
        f"Run D yield map and SPD Gaussian geometry: solves {first}–{last}\n"
        "ellipses are 1σ covariance contours; × marks centre; red/blue = positive/negative height",
        fontsize=15,
        y=0.985,
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.91))
    if image is not None:
        fig.colorbar(image, ax=axes, label="Yield strength [MPa]", shrink=0.82)
    fig.text(0.01, 0.01, f"Generated {generated:%Y-%m-%d %H:%M %Z}", fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def _draw_bases(axis, bases):
    for index, basis in enumerate(bases, start=1):
        eigenvalues, eigenvectors = np.linalg.eigh(basis.covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        major = eigenvectors[:, order[0]]
        angle = np.degrees(np.arctan2(major[1], major[0]))
        color = "#d62728" if basis.height >= 0.0 else "#1f77b4"
        ellipse = Ellipse(
            xy=basis.centre,
            width=2.0 * np.sqrt(eigenvalues[0]),
            height=2.0 * np.sqrt(eigenvalues[1]),
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=0.22,
            linewidth=1.5,
        )
        axis.add_patch(ellipse)
        is_newest = index == len(bases)
        marker_color = "#ffd92f" if is_newest else "white"
        axis.scatter(
            basis.centre[0],
            basis.centre[1],
            marker="x",
            s=68 if is_newest else 48,
            linewidths=2.2 if is_newest else 1.6,
            color=marker_color,
            zorder=5,
        )
        axis.text(
            basis.centre[0] + 0.18,
            basis.centre[1] + 0.18,
            f"{'NEW ' if is_newest else ''}B{index}",
            color=marker_color,
            fontsize=7.2,
            weight="bold",
            zorder=6,
        )


def _accepted_rejected_page(pdf, solves, truth, mask, yielded, extent):
    accepted = solves[-2]
    rejected = solves[-1]
    accepted_error = 100.0 * (accepted.final_map - truth) / truth
    rejected_error = 100.0 * (rejected.final_map - truth) / truth
    limit = float(
        max(
            np.nanpercentile(np.abs(accepted_error[mask]), 99.5),
            np.nanpercentile(np.abs(rejected_error[mask]), 99.5),
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    for axis, state, label in (
        (axes[0, 0], accepted, "Accepted four-basis model"),
        (axes[0, 1], rejected, "Rejected five-basis trial"),
    ):
        image_map = axis.imshow(
            np.where(mask, state.final_map, np.nan),
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
        )
        _draw_bases(axis, state.final_bases)
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_title(
            f"{label}\nJ={state.objective:.6f}; RMSE={state.final_rmse:.2f} MPa"
        )
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
    for axis, error, state in (
        (axes[1, 0], accepted_error, accepted),
        (axes[1, 1], rejected_error, rejected),
    ):
        image_error = axis.imshow(
            np.where(mask, error, np.nan),
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(
            f"Percentage error; yielded RMSE {state.final_yielded_rmse:.2f} MPa"
        )
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
    fig.suptitle(
        "Run D acceptance gate discarded a lower-cost, slightly more accurate trial",
        fontsize=16,
    )
    fig.colorbar(image_map, ax=axes[0, :], label="Yield strength [MPa]", shrink=0.8)
    fig.colorbar(image_error, ax=axes[1, :], label="Error [%]", shrink=0.8)
    pdf.savefig(fig)
    plt.close(fig)


def _basis_interpretation_page(pdf, solves, generated):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.055, 0.92, "What the run-D basis-growth sequence shows", fontsize=20, weight="bold")
    columns = ["Solve", "BFs", "J(end)", "RMSE start→end", "Yielded start→end", "Decision"]
    rows = [
        [
            state.solve_index,
            len(state.final_bases),
            f"{state.objective:.6f}",
            f"{state.initial_rmse:.1f}→{state.final_rmse:.1f}",
            f"{state.initial_yielded_rmse:.1f}→{state.final_yielded_rmse:.1f}",
            "accept" if state.accepted else "reject",
        ]
        for state in solves
    ]
    axis = fig.add_axes((0.055, 0.58, 0.89, 0.24))
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.65)
    for column in range(len(columns)):
        table[(0, column)].set_facecolor("#dbe9f4")
        table[(0, column)].set_text_props(weight="bold")
    previous = solves[-2]
    trial = solves[-1]
    improvement = 100.0 * (previous.objective - trial.objective) / previous.objective
    lines = [
        "Observed mechanics",
        "• The first broad positive SPD Gaussian builds the high-strength weld band; later sensitivity proposals add narrow negative corrections at the right HAZ, left HAZ, and weld centre.",
        "• Existing Gaussians continue to move and change covariance after every addition. The growth rule proposes an initial missing feature; it does not freeze earlier geometry.",
        "• Every solved endpoint reduces both objective and global yield-map RMSE. This is stronger evidence for sensitivity-informed placement than the final accepted summary alone showed.",
        "",
        "Acceptance-gate finding",
        f"• The fifth-basis solve reduced J from {previous.objective:.6f} to {trial.objective:.6f} ({improvement:.2f}%) and global/yielded RMSE from {previous.final_rmse:.2f}/{previous.final_yielded_rmse:.2f} to {trial.final_rmse:.2f}/{trial.final_yielded_rmse:.2f} MPa.",
        "• It was rejected only because the configured relative-improvement threshold was 5%. The gate therefore hid a mechanically useful refinement that missed the threshold by about 0.25 percentage points.",
        "• This synthetic retrospective does not prove that every smaller objective improvement is physically useful. It does justify testing a lower/no gate and separating proposal quality from acceptance policy.",
    ]
    fig.text(0.07, 0.53, "\n".join(lines), fontsize=10.5, va="top", linespacing=1.42)
    fig.text(0.055, 0.04, f"Generated {generated:%Y-%m-%d %H:%M %Z}", fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def _write_cover(path, solves, generated):
    with PdfPages(
        path,
        metadata={
            "Title": "Notched EBW consolidated VFM identification findings",
            "Author": "PyVale investigation report",
            "CreationDate": generated,
        },
    ) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.055, 0.91, "Notched-EBW synthetic VFM identification", fontsize=23, weight="bold")
        fig.text(
            0.055,
            0.855,
            f"Consolidated findings — {generated:%Y-%m-%d %H:%M %Z}",
            fontsize=15,
        )
        lines = [
            "Current position",
            "• Representability is not the limiting issue: fixed oracle-derived Gaussian geometry previously reached J≈0.0745–0.0749, ≈13 MPa yielded-region RMSE, and H≈3979 MPa.",
            "• The objective is materially watered down. Only 16.06% of valid metric observations are yielded, yet they contain 74.45% of the positive identified-versus-truth gap; 62.74% of the truth objective comes from unyielded observations.",
            "• The corrected free-geometry Jacobian in exact log-normalised optimiser coordinates has condition number 3.78×10³ with 11/20 singular directions above 1%—better than the earlier 4.36×10⁵ audit, but still correlated and path dependent.",
            "",
            "Latest factorial",
            "• SPD covariance coordinates are the cleanest structural improvement: they remove angle periodicity, axis swapping, positivity constraints, and circular-orientation ambiguity while retaining anisotropy.",
            "• Case B (SPD + EGI peak) had the lowest accepted global map RMSE: 27.64 MPa. Case A retained the lowest yielded-region RMSE: 36.97 MPa.",
            "• Case D (SPD + sensitivity) had the lowest accepted scalar objective: 0.085514, but accepted global/yielded RMSE remained 31.69/45.31 MPa. Objective proximity is therefore not sufficient evidence of map accuracy.",
            "",
            "New run-D history finding",
            f"• The rejected five-basis trial reached J={solves[-1].objective:.6f}, global RMSE={solves[-1].final_rmse:.2f} MPa and yielded RMSE={solves[-1].final_yielded_rmse:.2f} MPa—better than the accepted four-basis state on all three measures.",
            "• It was rejected because its 4.75% objective reduction narrowly missed the fixed 5% acceptance gate. Sensitivity placement and refinement acceptance must be evaluated separately.",
        ]
        fig.text(
            0.07,
            0.79,
            "\n".join(_wrap_report_lines(lines, width=118)),
            fontsize=10.7,
            va="top",
            linespacing=1.43,
        )
        fig.text(0.055, 0.04, "The following pages retain the complete 2026-08-28 factorial report, then add solve-by-solve BF diagnostics and the workstation handoff.", fontsize=8.5)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.055, 0.91, "Implications and next investigation", fontsize=21, weight="bold")
        lines = [
            "What is now supported",
            "• Keep the SPD log-covariance kernel as the leading geometry representation.",
            "• Keep sensitivity-correction growth as an experimental policy: its proposal sequence is mechanically interpretable and its solved endpoints improve monotonically in run D.",
            "• Continue reporting objective, global/yielded/high-plastic map error, hardening error, and spatial error maps separately. The current scalar is a mechanical closure measure, not a reliable standalone material-map score.",
            "",
            "Immediate controlled experiment",
            "1. Repeat SPD + sensitivity with the basis-acceptance threshold at 0%, 1%, and the current 5%, holding seed and all other settings fixed.",
            "2. Retain every trial snapshot and compare accepted/best-visited objective and active-region errors. This isolates placement quality from the acceptance gate.",
            "3. Replicate the leading gate setting across optimiser seeds; compare against SPD + EGI peak using identical budgets.",
            "4. Inspect whether sensitivity-derived objective weighting improves active-region recovery. Keep the unweighted mechanical objective as a reported reference until weighting is validated.",
            "",
            "Longer-term algorithm direction",
            "• Preserve residual vectors and their spatial/time/metric identity internally even though the optimiser ultimately consumes a scalar.",
            "• Consider an active-information objective or dual reporting that reduces insensitive background dilution without requiring knowledge of the truth yield mask.",
            "• Profile the threaded candidate path separately: the workstation benchmark peaks near eight workers per run, so large investigation throughput should use concurrent independent runs.",
            "",
            "Decision boundary",
            "Do not select a production default from this single deterministic factorial. The next decision should follow gate isolation plus seed replication, with active-region recovery weighted more heavily than small scalar-objective differences.",
        ]
        fig.text(
            0.07,
            0.84,
            "\n".join(_wrap_report_lines(lines, width=118)),
            fontsize=11.0,
            va="top",
            linespacing=1.5,
        )
        fig.text(0.055, 0.04, f"Generated {generated:%Y-%m-%d %H:%M %Z}", fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)


def _write_workstation_page(path, generated):
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.055, 0.91, "Workstation capability for future investigations", fontsize=21, weight="bold")
        lines = [
            "Available environment",
            "• Company workstation R0379: Ubuntu 24.04 under WSL2, 192 logical CPUs and 251 GiB RAM.",
            "• Native WSL paths: ~/projects/pyvale and ~/projects/pyvale-vfm-test-data; avoid /mnt/c for computation.",
            "• PyVale branch vfm-bf-verif is synchronised from the public GitHub repository over HTTPS.",
            "• cython-stress-recon is installed editable in the PyVale environment; focused PyVale and compiled-adapter tests pass.",
            "",
            "Access and persistence",
            "• Connect to the approved Windows SSH endpoint, then enter WSL with: wsl -d Ubuntu.",
            "• Run investigations inside tmux so they survive SSH disconnection. Do not configure a second SSH service inside WSL.",
            "",
            "Execution defaults",
            "• Use uv run --no-sync until h5py is tracked in pyproject.toml/uv.lock.",
            "• Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1.",
            "• Use --parallel-workers 8 for one identification. A 10-iteration compiled benchmark took 48.22 s and 1.10 GiB; 12/16/32 workers were progressively slower and used more memory.",
            "• The eight-worker run averaged only 274% CPU, so the workstation's main advantage is process-level throughput: run independent configurations concurrently rather than assigning all cores to one pattern search.",
            "",
            "Practical campaign pattern",
            "Start with a modest number of concurrent eight-worker runs, measure aggregate CPU/RAM, and increase cautiously. Give every run a unique output directory and fixed seed/configuration record. Pull committed branch changes before launching a campaign.",
        ]
        fig.text(
            0.07,
            0.84,
            "\n".join(_wrap_report_lines(lines, width=118)),
            fontsize=11.1,
            va="top",
            linespacing=1.52,
        )
        fig.text(0.055, 0.04, f"Setup status recorded {generated:%Y-%m-%d %H:%M %Z}", fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASET / "prepared")
    parser.add_argument("--run-d", type=Path, default=RUN_D)
    parser.add_argument("--latest-report", type=Path, default=LATEST_REPORT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev/vfm/output/notched_ebw_consolidated_findings_20260828"),
    )
    return parser.parse_args()


def _wrap_report_lines(lines: list[str], *, width: int) -> list[str]:
    """Wrap report bullets without losing their visual hierarchy."""
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        if line.startswith("• "):
            wrapped.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    subsequent_indent="  ",
                    break_long_words=False,
                )
            )
            continue
        if line[:2].isdigit() and line[1:3] in {". ", ") "}:
            wrapped.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    subsequent_indent="   ",
                    break_long_words=False,
                )
            )
            continue
        wrapped.extend(textwrap.wrap(line, width=width, break_long_words=False))
    return wrapped


if __name__ == "__main__":
    main()

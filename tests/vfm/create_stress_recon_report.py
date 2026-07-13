from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _masked_extent(x_grid: np.ndarray, y_grid: np.ndarray) -> list[float]:
    return [
        float(np.nanmin(x_grid)),
        float(np.nanmax(x_grid)),
        float(np.nanmin(y_grid)),
        float(np.nanmax(y_grid)),
    ]


def _safe_abs_percent_diff(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    specimen_mask: np.ndarray | None,
) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    percent_diff = np.full_like(reference, np.nan, dtype=np.float64)
    valid = np.isfinite(candidate) & np.isfinite(reference)

    if specimen_mask is not None:
        valid &= specimen_mask

    valid &= np.abs(reference) > 1e-12
    percent_diff[valid] = (
        np.abs(candidate[valid] - reference[valid]) / np.abs(reference[valid])
    ) * 100.0
    return percent_diff


def _field_max_abs(field: np.ndarray, specimen_mask: np.ndarray | None) -> float:
    valid = np.isfinite(field)
    if specimen_mask is not None:
        valid &= specimen_mask

    if not np.any(valid):
        return float("nan")

    return float(np.nanmax(np.abs(field[valid])))


def _field_percentile_limits(
    field: np.ndarray,
    specimen_mask: np.ndarray | None,
    *,
    lower: float = 5.0,
    upper: float = 95.0,
) -> tuple[float, float] | None:
    values = np.asarray(field, dtype=np.float64)
    valid = np.isfinite(values)
    if specimen_mask is not None:
        valid &= specimen_mask

    if not np.any(valid):
        return None

    finite_values = values[valid]
    return (
        float(np.nanpercentile(finite_values, lower)),
        float(np.nanpercentile(finite_values, upper)),
    )


def _add_field_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    clim: tuple[float, float] | None = None,
) -> None:
    image = ax.imshow(
        field,
        extent=_masked_extent(x_grid, y_grid),
        origin="lower",
        aspect="equal",
        cmap=cmap,
    )
    if clim is not None:
        image.set_clim(*clim)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _create_cover_page(
    pdf: PdfPages,
    *,
    report_title: str,
    exodus_file_name: str,
    fe_element_count: int,
    report_step: int,
    grid_divs: int,
    component_metrics: dict[str, dict[str, float]],
    equivalent_metrics: dict[str, float],
    yielded_point_count: int,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), constrained_layout=True)
    fig.patch.set_facecolor("white")

    lines = [
        report_title,
        "",
        f"source Exodus file: {exodus_file_name}",
        f"FE mesh element count: {fe_element_count}",
        f"report step: {report_step}",
        f"ndivs = {grid_divs}",
        f"grid of {grid_divs} x {grid_divs} points",
        "FE element data are interpolated from Exodus element centres to this grid.",
        "",
        "Stress component summaries:",
    ]

    for component_label in ("xx", "yy", "xy"):
        metrics = component_metrics[component_label]
        lines.extend(
            [
                f"{component_label}",
                f"  max abs diff [MPa] = {metrics['max_abs_diff']:.6f}",
                f"  max abs perc diff [%] = {metrics['max_abs_percent_diff']:.6f}",
                "",
            ]
        )

    lines.extend(
        [
            "von Mises",
            f"  max abs diff [MPa] = {equivalent_metrics['max_abs_diff']:.6f}",
            f"  max abs perc diff [%] = {equivalent_metrics['max_abs_percent_diff']:.6f}",
            "",
            f"yielded point count = {yielded_point_count}",
        ]
    )

    fig.text(
        0.08,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _summarise_component_fields(
    *,
    stress_rr: np.ndarray,
    stress_fe: np.ndarray,
    specimen_mask: np.ndarray | None,
) -> dict[str, float]:
    diff_field = np.asarray(stress_rr, dtype=np.float64) - np.asarray(
        stress_fe,
        dtype=np.float64,
    )
    percent_diff_field = _safe_abs_percent_diff(
        stress_rr,
        stress_fe,
        specimen_mask=specimen_mask,
    )

    return {
        "max_abs_diff": _field_max_abs(diff_field, specimen_mask),
        "max_abs_percent_diff": _field_max_abs(percent_diff_field, specimen_mask),
    }


def _create_component_page(
    *,
    component_label: str,
    report_step: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    stress_rr: np.ndarray,
    stress_fe: np.ndarray,
    specimen_mask: np.ndarray | None,
    percentile_clim: bool,
) -> plt.Figure:
    rr_field = np.asarray(stress_rr, dtype=np.float64)
    fe_field = np.asarray(stress_fe, dtype=np.float64)
    diff_field = rr_field - fe_field
    percent_diff_field = _safe_abs_percent_diff(rr_field, fe_field, specimen_mask=specimen_mask)
    metrics = _summarise_component_fields(
        stress_rr=rr_field,
        stress_fe=fe_field,
        specimen_mask=specimen_mask,
    )
    title_suffix = ""
    if percentile_clim:
        title_suffix = " (Colour limits set to 5th and 95th percentiles)"

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    fig.suptitle(
        f"Stress Reconstruction: {component_label} at step {report_step}{title_suffix}"
    )

    _add_field_plot(
        fig,
        axes[0, 0],
        rr_field,
        x_grid,
        y_grid,
        title=f"Radial return {component_label}",
        colorbar_label="stress [MPa]",
        clim=_field_percentile_limits(rr_field, specimen_mask) if percentile_clim else None,
    )
    _add_field_plot(
        fig,
        axes[0, 1],
        fe_field,
        x_grid,
        y_grid,
        title=f"Interpolated FE {component_label}",
        colorbar_label="stress [MPa]",
        clim=_field_percentile_limits(fe_field, specimen_mask) if percentile_clim else None,
    )
    _add_field_plot(
        fig,
        axes[1, 0],
        diff_field,
        x_grid,
        y_grid,
        title=(
            f"Difference {component_label}\n"
            f"max abs diff = {metrics['max_abs_diff']:.6f} MPa"
        ),
        colorbar_label="stress [MPa]",
        cmap="coolwarm",
        clim=_field_percentile_limits(diff_field, specimen_mask) if percentile_clim else None,
    )
    _add_field_plot(
        fig,
        axes[1, 1],
        percent_diff_field,
        x_grid,
        y_grid,
        title=(
            f"Absolute percent difference {component_label}\n"
            f"max abs perc diff = {metrics['max_abs_percent_diff']:.6f} %"
        ),
        colorbar_label="difference [%]",
        cmap="magma",
        clim=(
            _field_percentile_limits(percent_diff_field, specimen_mask)
            if percentile_clim
            else None
        ),
    )

    return fig


def _create_equivalent_page(
    *,
    report_step: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    equivalent_stress_rr: np.ndarray,
    vonmises_stress_fe: np.ndarray,
    specimen_mask: np.ndarray | None,
    percentile_clim: bool,
) -> plt.Figure:
    rr_field = np.asarray(equivalent_stress_rr, dtype=np.float64)
    fe_field = np.asarray(vonmises_stress_fe, dtype=np.float64)
    diff_field = rr_field - fe_field
    percent_diff_field = _safe_abs_percent_diff(rr_field, fe_field, specimen_mask=specimen_mask)
    metrics = _summarise_component_fields(
        stress_rr=rr_field,
        stress_fe=fe_field,
        specimen_mask=specimen_mask,
    )
    title_suffix = ""
    if percentile_clim:
        title_suffix = " (Colour limits set to 5th and 95th percentiles)"

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    fig.suptitle(f"Equivalent Stress Reconstruction at step {report_step}{title_suffix}")

    _add_field_plot(
        fig,
        axes[0, 0],
        rr_field,
        x_grid,
        y_grid,
        title="Radial return von Mises",
        colorbar_label="stress [MPa]",
        clim=_field_percentile_limits(rr_field, specimen_mask) if percentile_clim else None,
    )
    _add_field_plot(
        fig,
        axes[0, 1],
        fe_field,
        x_grid,
        y_grid,
        title="Interpolated FE von Mises",
        colorbar_label="stress [MPa]",
        clim=_field_percentile_limits(fe_field, specimen_mask) if percentile_clim else None,
    )
    _add_field_plot(
        fig,
        axes[1, 0],
        diff_field,
        x_grid,
        y_grid,
        title=f"Difference\nmax abs diff = {metrics['max_abs_diff']:.6f} MPa",
        colorbar_label="stress [MPa]",
        cmap="coolwarm",
        clim=_field_percentile_limits(diff_field, specimen_mask) if percentile_clim else None,
    )
    _add_field_plot(
        fig,
        axes[1, 1],
        percent_diff_field,
        x_grid,
        y_grid,
        title=(
            "Absolute percent difference\n"
            f"max abs perc diff = {metrics['max_abs_percent_diff']:.6f} %"
        ),
        colorbar_label="difference [%]",
        cmap="magma",
        clim=(
            _field_percentile_limits(percent_diff_field, specimen_mask)
            if percentile_clim
            else None
        ),
    )

    return fig


def _create_yield_page(
    *,
    report_step: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    yield_map: np.ndarray,
) -> tuple[plt.Figure, int]:
    yield_field = np.asarray(yield_map, dtype=np.float64)
    yielded_point_count = int(np.nansum(yield_field == 1))

    fig, ax = plt.subplots(figsize=(11.69, 8.27), constrained_layout=True)
    _add_field_plot(
        fig,
        ax,
        yield_field,
        x_grid,
        y_grid,
        title=f"Yield Map at step {report_step}\nyielded point count = {yielded_point_count}",
        colorbar_label="yield state",
        cmap="viridis",
    )

    return fig, yielded_point_count


def create_stress_recon_report(
    report_path: Path,
    *,
    exodus_file_name: str,
    fe_element_count: int,
    report_step: int,
    grid_divs: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    stress_rr: np.ndarray,
    stress_fe: np.ndarray,
    equivalent_stress_rr: np.ndarray,
    vonmises_stress_fe: np.ndarray,
    yield_map: np.ndarray,
    specimen_mask: np.ndarray | None = None,
    report_title: str = "verifying radial return using FE data",
) -> dict[str, object]:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    component_metrics: dict[str, dict[str, float]] = {}
    for component_index, component_label in enumerate(("xx", "yy", "xy")):
        component_metrics[component_label] = _summarise_component_fields(
            stress_rr=stress_rr[report_step, component_index, :, :],
            stress_fe=stress_fe[report_step, component_index, :, :],
            specimen_mask=specimen_mask,
        )

    equivalent_metrics = _summarise_component_fields(
        stress_rr=equivalent_stress_rr[report_step, :, :],
        stress_fe=vonmises_stress_fe[report_step, :, :],
        specimen_mask=specimen_mask,
    )
    yielded_point_count = int(np.nansum(yield_map[report_step, :, :] == 1))

    with PdfPages(report_path) as pdf:
        _create_cover_page(
            pdf,
            report_title=report_title,
            exodus_file_name=exodus_file_name,
            fe_element_count=fe_element_count,
            report_step=report_step,
            grid_divs=grid_divs,
            component_metrics=component_metrics,
            equivalent_metrics=equivalent_metrics,
            yielded_point_count=yielded_point_count,
        )

        for component_index, component_label in enumerate(("xx", "yy", "xy")):
            fig = _create_component_page(
                component_label=component_label,
                report_step=report_step,
                x_grid=x_grid,
                y_grid=y_grid,
                stress_rr=stress_rr[report_step, component_index, :, :],
                stress_fe=stress_fe[report_step, component_index, :, :],
                specimen_mask=specimen_mask,
                percentile_clim=False,
            )
            pdf.savefig(fig)
            plt.close(fig)

            fig = _create_component_page(
                component_label=component_label,
                report_step=report_step,
                x_grid=x_grid,
                y_grid=y_grid,
                stress_rr=stress_rr[report_step, component_index, :, :],
                stress_fe=stress_fe[report_step, component_index, :, :],
                specimen_mask=specimen_mask,
                percentile_clim=True,
            )
            pdf.savefig(fig)
            plt.close(fig)

        fig = _create_equivalent_page(
            report_step=report_step,
            x_grid=x_grid,
            y_grid=y_grid,
            equivalent_stress_rr=equivalent_stress_rr[report_step, :, :],
            vonmises_stress_fe=vonmises_stress_fe[report_step, :, :],
            specimen_mask=specimen_mask,
            percentile_clim=False,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = _create_equivalent_page(
            report_step=report_step,
            x_grid=x_grid,
            y_grid=y_grid,
            equivalent_stress_rr=equivalent_stress_rr[report_step, :, :],
            vonmises_stress_fe=vonmises_stress_fe[report_step, :, :],
            specimen_mask=specimen_mask,
            percentile_clim=True,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, _ = _create_yield_page(
            report_step=report_step,
            x_grid=x_grid,
            y_grid=y_grid,
            yield_map=yield_map[report_step, :, :],
        )
        pdf.savefig(fig)
        plt.close(fig)

    return {
        "report_path": report_path,
        "component_metrics": component_metrics,
        "equivalent_metrics": equivalent_metrics,
        "yielded_point_count": yielded_point_count,
    }


def create_stress_recon_plots(
    *,
    report_step: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    stress_rr: np.ndarray,
    stress_fe: np.ndarray,
    equivalent_stress_rr: np.ndarray,
    vonmises_stress_fe: np.ndarray,
    yield_map: np.ndarray,
    specimen_mask: np.ndarray | None = None,
    plot_stress_rr: bool = True,
    plot_stress_fe: bool = True,
    plot_stress_rr_fe_diff: bool = True,
    plot_stress_rr_fe_perc_diff: bool = True,
    plot_percentile_scaled_diff: bool = True,
    show_equivalent: bool = True,
    show_yield_map: bool = True,
) -> list[plt.Figure]:
    figs: list[plt.Figure] = []

    def build_component_figure(
        component_label: str,
        rr_field: np.ndarray,
        fe_field: np.ndarray,
        *,
        percentile_clim: bool,
    ) -> plt.Figure:
        diff_field = rr_field - fe_field
        perc_diff_field = _safe_abs_percent_diff(
            rr_field,
            fe_field,
            specimen_mask=specimen_mask,
        )

        fields: list[tuple[np.ndarray, str, str, str]] = []
        if plot_stress_rr:
            fields.append((rr_field, f"Radial return {component_label}", "stress [MPa]", "viridis"))
        if plot_stress_fe:
            fields.append((fe_field, f"Interpolated FE {component_label}", "stress [MPa]", "viridis"))
        if plot_stress_rr_fe_diff:
            fields.append((diff_field, f"Difference {component_label}", "stress [MPa]", "coolwarm"))
        if plot_stress_rr_fe_perc_diff:
            fields.append((perc_diff_field, f"Absolute percent difference {component_label}", "difference [%]", "magma"))

        n_fields = len(fields)
        ncols = 2 if n_fields > 1 else 1
        nrows = int(np.ceil(n_fields / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(11.69, 8.27),
            constrained_layout=True,
            squeeze=False,
        )
        suffix = ""
        if percentile_clim:
            suffix = " (Colour limits set to 5th and 95th percentiles)"
        fig.suptitle(f"Stress Reconstruction: {component_label} at step {report_step}{suffix}")

        axes_flat = list(axes.flat)
        for ax, (field, title, colorbar_label, cmap) in zip(axes_flat, fields, strict=False):
            _add_field_plot(
                fig,
                ax,
                field,
                x_grid,
                y_grid,
                title=title,
                colorbar_label=colorbar_label,
                cmap=cmap,
                clim=_field_percentile_limits(field, specimen_mask) if percentile_clim else None,
            )

        for ax in axes_flat[n_fields:]:
            ax.set_visible(False)

        return fig

    for component_index, component_label in enumerate(("xx", "yy", "xy")):
        rr_field = np.asarray(stress_rr[report_step, component_index, :, :], dtype=np.float64)
        fe_field = np.asarray(stress_fe[report_step, component_index, :, :], dtype=np.float64)
        figs.append(
            build_component_figure(component_label, rr_field, fe_field, percentile_clim=False)
        )
        if plot_percentile_scaled_diff:
            figs.append(
                build_component_figure(component_label, rr_field, fe_field, percentile_clim=True)
            )

    if show_equivalent:
        equivalent_fig = _create_equivalent_page(
            report_step=report_step,
            x_grid=x_grid,
            y_grid=y_grid,
            equivalent_stress_rr=equivalent_stress_rr[report_step, :, :],
            vonmises_stress_fe=vonmises_stress_fe[report_step, :, :],
            specimen_mask=specimen_mask,
            percentile_clim=False,
        )
        figs.append(equivalent_fig)
        if plot_percentile_scaled_diff:
            figs.append(
                _create_equivalent_page(
                    report_step=report_step,
                    x_grid=x_grid,
                    y_grid=y_grid,
                    equivalent_stress_rr=equivalent_stress_rr[report_step, :, :],
                    vonmises_stress_fe=vonmises_stress_fe[report_step, :, :],
                    specimen_mask=specimen_mask,
                    percentile_clim=True,
                )
            )

    if show_yield_map:
        yield_fig, _ = _create_yield_page(
            report_step=report_step,
            x_grid=x_grid,
            y_grid=y_grid,
            yield_map=yield_map[report_step, :, :],
        )
        figs.append(yield_fig)

    return figs

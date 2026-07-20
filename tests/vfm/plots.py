import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

STRESS_COMPONENT_LABELS = ("xx", "yy", "xy")


def _plot_stress_abs_diff(
    x_grid: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    abs_diff: npt.NDArray[np.float64],
) -> None:
    """Plot the abs difference of each stress component at a single timestep."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, label, component in zip(
        axes, STRESS_COMPONENT_LABELS, range(3), strict=True
    ):
        field = abs_diff[component, :, :]
        image = ax.pcolormesh(x_grid, y_grid, field)
        fig.colorbar(image, ax=ax, label="|calc - FE| [MPa]")
        ax.set_title(f"stress_{label} abs diff")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.invert_yaxis()
    plt.show()


def _plot_stress_abs_perc_diff(
    x_grid: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    abs_perc_diff: npt.NDArray[np.float64],
) -> None:
    """Plot the absolute percentage difference of each stress component."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, label, component in zip(
        axes, STRESS_COMPONENT_LABELS, range(3), strict=True
    ):
        field = abs_perc_diff[component, :, :]
        image = ax.pcolormesh(x_grid, y_grid, field)
        fig.colorbar(image, ax=ax, label="|calc - FE| / |FE| [%]")
        image.set_clim(np.nanpercentile(field, 5), np.nanpercentile(field, 95))
        ax.set_title(f"stress_{label} abs % diff")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.invert_yaxis()
    plt.show()


def _plot_metric_virtual_work(
    internal_virtual_work_a: npt.NDArray[np.float64],
    external_virtual_work_a: npt.NDArray[np.float64],
    internal_virtual_work_b: npt.NDArray[np.float64],
    external_virtual_work_b: npt.NDArray[np.float64],
    label_a: str,
    label_b: str,
    sbvf_labels: tuple[str, ...],
) -> None:
    """Compare the internal/external virtual work of two metric evaluations.

    Each virtual work array has shape (num_virtual_fields, timesteps). One row
    of plots is drawn per SBVF, showing the IVW, EVW, abs difference and
    percentage difference between the two evaluations. Each SBVF corresponds to
    the single degree of freedom of one homogeneous constitutive parameter, so
    ``sbvf_labels`` names the parameter driving each row.
    """
    num_virtual_fields = internal_virtual_work_a.shape[0]

    # Figure 1: per-SBVF comparison of IVW and EVW between the two evaluations.
    fig_work, axes = plt.subplots(
        num_virtual_fields,
        4,
        figsize=(18, 3.5 * num_virtual_fields),
        constrained_layout=True,
        squeeze=False,
    )

    # Figure 2: per-SBVF comparison of the PVW residual |IVW - EVW| between the
    # two evaluations. Shown at the same time as figure 1.
    fig_residual, residual_axes = plt.subplots(
        num_virtual_fields,
        3,
        figsize=(13.5, 3.5 * num_virtual_fields),
        constrained_layout=True,
        squeeze=False,
    )

    for vf in range(num_virtual_fields):
        ivw_a = internal_virtual_work_a[vf]
        ivw_b = internal_virtual_work_b[vf]
        evw_a = external_virtual_work_a[vf]
        evw_b = external_virtual_work_b[vf]

        ivw_abs_diff = np.abs(ivw_b - ivw_a)
        evw_abs_diff = np.abs(evw_b - evw_a)
        # Guard against division by zero (e.g. zero virtual work at the first
        # timestep), leaving those points as NaN so they are skipped in the plot
        ivw_percentage_diff = np.divide(
            ivw_abs_diff * 100.0,
            np.abs(ivw_a),
            out=np.full_like(ivw_abs_diff, np.nan),
            where=ivw_a != 0.0,
        )
        evw_percentage_diff = np.divide(
            evw_abs_diff * 100.0,
            np.abs(evw_a),
            out=np.full_like(evw_abs_diff, np.nan),
            where=evw_a != 0.0,
        )

        # PVW residual magnitude |IVW - EVW| for each evaluation, and the
        # residual as a percentage of EVW, separately for each evaluation.
        ivw_evw_diff_a = np.abs(ivw_a - evw_a)
        ivw_evw_diff_b = np.abs(ivw_b - evw_b)
        ivw_evw_percentage_diff_a = np.divide(
            ivw_evw_diff_a * 100.0,
            np.abs(evw_a),
            out=np.full_like(evw_a, np.nan),
            where=evw_a != 0.0,
        )
        ivw_evw_percentage_diff_b = np.divide(
            ivw_evw_diff_b * 100.0,
            np.abs(evw_b),
            out=np.full_like(evw_b, np.nan),
            where=evw_b != 0.0,
        )

        sbvf_label = sbvf_labels[vf]

        axes[vf, 0].plot(ivw_a, marker=".", label=label_a)
        axes[vf, 0].plot(ivw_b, marker=".", label=label_b)
        axes[vf, 0].set_title(f"{sbvf_label} IVW")
        axes[vf, 0].set_ylabel("internal virtual work")
        axes[vf, 0].legend()

        axes[vf, 1].plot(evw_a, marker=".", label=label_a)
        axes[vf, 1].plot(evw_b, marker=".", label=label_b)
        axes[vf, 1].set_title(f"{sbvf_label} EVW")
        axes[vf, 1].set_ylabel("external virtual work")
        axes[vf, 1].legend()

        axes[vf, 2].plot(ivw_abs_diff, marker=".", label="IVW")
        axes[vf, 2].plot(evw_abs_diff, marker=".", label="EVW")
        axes[vf, 2].set_title(f"{sbvf_label} abs diff")
        axes[vf, 2].set_ylabel(f"|{label_b} - {label_a}|")
        axes[vf, 2].legend()

        axes[vf, 3].plot(ivw_percentage_diff, marker=".", label="IVW")
        axes[vf, 3].plot(evw_percentage_diff, marker=".", label="EVW")
        axes[vf, 3].set_title(f"{sbvf_label} percentage diff")
        axes[vf, 3].set_ylabel("% diff")
        axes[vf, 3].legend()

        # Evaluation a (e.g. known) in blue, b (e.g. calc) in orange; IVW solid,
        # EVW dashed.
        residual_axes[vf, 0].plot(
            ivw_a, marker=".", color="blue", linestyle="-", label=f"IVW {label_a}"
        )
        residual_axes[vf, 0].plot(
            evw_a, marker=".", color="blue", linestyle="--", label=f"EVW {label_a}"
        )
        residual_axes[vf, 0].plot(
            ivw_b, marker=".", color="orange", linestyle="-", label=f"IVW {label_b}"
        )
        residual_axes[vf, 0].plot(
            evw_b, marker=".", color="orange", linestyle="--", label=f"EVW {label_b}"
        )
        residual_axes[vf, 0].set_title(f"{sbvf_label} IVW & EVW")
        residual_axes[vf, 0].set_ylabel("virtual work")
        residual_axes[vf, 0].legend()

        residual_axes[vf, 1].plot(ivw_evw_diff_a, marker=".", label=label_a)
        residual_axes[vf, 1].plot(ivw_evw_diff_b, marker=".", label=label_b)
        residual_axes[vf, 1].set_title(f"{sbvf_label} |IVW - EVW|")
        residual_axes[vf, 1].set_ylabel("|IVW - EVW|")
        residual_axes[vf, 1].legend()

        residual_axes[vf, 2].plot(ivw_evw_percentage_diff_a, marker=".", label=label_a)
        residual_axes[vf, 2].plot(ivw_evw_percentage_diff_b, marker=".", label=label_b)
        residual_axes[vf, 2].set_title(f"{sbvf_label} |IVW - EVW| percentage diff")
        residual_axes[vf, 2].set_ylabel("% diff (IVW vs EVW)")
        residual_axes[vf, 2].legend()

        for column in range(4):
            axes[vf, column].set_xlabel("timestep")
        for column in range(3):
            residual_axes[vf, column].set_xlabel("timestep")

    plt.show()


def _plot_map_comparison(
    x_grid: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    reference_map: npt.NDArray[np.float64],
    comparison_map: npt.NDArray[np.float64],
    reference_label: str,
    comparison_label: str,
) -> None:
    """Compare two parameter maps defined on the same grid.

    Draws four panels in a single row: the reference map, the comparison map,
    the absolute difference ``|comparison - reference|`` and the absolute
    percentage difference ``|comparison - reference| / |reference| * 100``.
    The percentage difference is left as NaN wherever the reference is zero so
    those points are skipped in the plot.
    """
    abs_diff = np.abs(comparison_map - reference_map)
    abs_perc_diff = np.divide(
        abs_diff * 100.0,
        np.abs(reference_map),
        out=np.full_like(abs_diff, np.nan),
        where=reference_map != 0.0,
    )

    panels = (
        (reference_map, reference_label, reference_label),
        (comparison_map, comparison_label, comparison_label),
        (abs_diff, "abs diff", f"|{comparison_label} - {reference_label}|"),
        (abs_perc_diff, "abs % diff", "|diff| / |reference| [%]"),
    )

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    for ax, (field, title, colorbar_label) in zip(axes, panels, strict=True):
        image = ax.pcolormesh(x_grid, y_grid, field)
        fig.colorbar(image, ax=ax, label=colorbar_label)
        # Clip the colour scale for the percentage panel so a few large values
        # near zero-reference regions don't wash out the rest.
        if title == "abs % diff":
            image.set_clim(
                np.nanpercentile(field, 5), np.nanpercentile(field, 95)
            )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.invert_yaxis()
    plt.show()


def _plot_identification_diff(
    x_grid: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    identified_maps: dict[str, npt.NDArray[np.float64]],
    known_maps: dict[str, npt.NDArray[np.float64]],
) -> None:
    """Plot the difference between the identified and known parameter maps."""
    fig, axes = plt.subplots(
        1, len(known_maps), figsize=(16, 4), constrained_layout=True
    )
    for ax, param_name in zip(axes, known_maps, strict=True):
        field = identified_maps[param_name] - known_maps[param_name]
        image = ax.pcolormesh(x_grid, y_grid, field)
        fig.colorbar(image, ax=ax, label="identified - known")
        ax.set_title(param_name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.invert_yaxis()
    plt.show()

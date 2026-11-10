# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""PyVista views of validation points over optional simulation geometry/fields."""

from dataclasses import dataclass

import numpy as np
import pyvista as pv

from pyvale.dataio.simdata import SimData
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_vis
from pyvale.sensorsim.sensordescriptor import SensorDescriptor
from pyvale.sensorsim.visualopts import VisOptsSimSensors
from pyvale.sensorsim.visualtools import create_pv_plotter
from pyvale.valid.metricstrategy import EMetricType
from pyvale.valid.valengine import EValResAxis, ValResult
from pyvale.valid.valtools import create_metric_label, extract_metric_values
from pyvale.valid.valtypes import EValMetricPart


@dataclass(slots=True)
class ValSimField:
    """One background nodal scalar field, shaped (n_nodes,), at an explicit step.

    Values can be a stored component or a separately calculated invariant. The
    descriptor specifies the background's physical units, independently of the
    validation metric. Coordinate units/frame are supplied to the plot function.
    """

    values: np.ndarray
    time_index: int
    descriptor: SensorDescriptor
    cmap: str = "viridis"
    colour_lims: tuple[float, float] | None = None


@dataclass(slots=True)
class ValVisOpts:
    """Metric/mesh appearance; camera, fonts and marker sizes use SensorSim options.

    point_labels covers all original validation points, even when selecting a
    subset. None disables labels. Generate sensor tags with SensorDescriptor.
    """

    metric_colour_lims: tuple[float, float] | None = None
    metric_cmap: str | None = None
    mesh_opacity: float = 0.6
    point_labels: tuple[str, ...] | None = None
    labels_always_visible: bool = False
    title: str | None = None
    metric_label: str | None = None


def _calculate_colour_limits(
    values: np.ndarray,
    limits: tuple[float, float] | None,
    *,
    signed: bool,
    magnitude: bool,
) -> tuple[float, float]:
    if limits is not None:
        lower, upper = limits
        if not np.all(np.isfinite(limits)) or lower >= upper:
            raise ValueError("Colour limits must be finite and strictly increasing.")
        if signed and lower != -upper:
            raise ValueError(
                "Signed metric colour limits must be symmetric about zero."
            )
        if magnitude and lower < 0.0:
            raise ValueError("Magnitude metric colour limits cannot be negative.")
        return lower, upper
    finite = values[np.isfinite(values)]
    if signed:
        limit = float(np.max(np.abs(finite))) if len(finite) else 0.0
        limit = limit if limit > 0.0 else 1.0
        return -limit, limit
    lower = 0.0 if magnitude or not len(finite) else float(np.min(finite))
    upper = float(np.max(finite)) if len(finite) else lower
    if upper == lower:
        padding = abs(lower) * 0.01 if lower != 0.0 else 1.0
        upper += padding
        if not magnitude:
            lower -= padding
    return lower, upper


def plot_val_metric_on_sim(
    result: ValResult,
    observable: str,
    metric: EMetricType,
    *,
    selectors: dict[EValResAxis, int] | None = None,
    part: EValMetricPart = EValMetricPart.TOTAL,
    points: tuple[int, ...] | None = None,
    sim_data: SimData | None = None,
    sim_field: ValSimField | None = None,
    sim_coordinate_units: str = "",
    sim_frame: str = "",
    vis_opts: VisOptsSimSensors | None = None,
    metric_opts: ValVisOpts | None = None,
    plotter: pv.Plotter | None = None,
) -> pv.Plotter:
    """Colour xyz validation points, optionally over an independently coloured mesh.

    The caller supplies the background run and declares its coordinate units and
    frame; both must match the observable geometry. A time-resolved result must
    match the supplied background step exactly. No nearest-time selection,
    coordinate transformation, scenario reduction or mesh interpolation occurs.
    NaN metrics remain grey points. Signed balance can cancel despite mismatch.
    A newly constructed plotter is off-screen by default; showing/saving is explicit.
    """
    options = VisOptsSimSensors(interactive=False) if vis_opts is None else vis_opts
    metric_options = ValVisOpts() if metric_opts is None else metric_opts
    field = result.get(observable, metric)
    values = extract_metric_values(field, selectors=selectors, part=part)
    observable_data = result.domain.observables[observable]
    geometry = observable_data.geometry
    coords = geometry.coords
    if coords is None or not np.all(np.isfinite(coords)):
        raise ValueError("Metric visualisation needs finite xyz point coordinates.")
    if np.any(np.isinf(values)):
        raise ValueError("Metric values cannot contain infinity.")
    if (
        not np.isfinite(metric_options.mesh_opacity)
        or not 0 <= metric_options.mesh_opacity <= 1
    ):
        raise ValueError("Mesh opacity must be between zero and one.")
    if points is None:
        point_indices = np.arange(len(coords))
    else:
        if not points or any(
            isinstance(point, (bool, np.bool_))
            or not isinstance(point, (int, np.integer))
            or not 0 <= point < len(coords)
            for point in points
        ):
            raise ValueError("Point subset must contain valid original point indices.")
        point_indices = np.asarray(points)
    if metric_options.point_labels is not None and len(
        metric_options.point_labels
    ) != len(coords):
        raise ValueError("Point labels must cover all original validation points.")
    if sim_field is not None and sim_data is None:
        raise ValueError("A background field requires simulation data.")
    selected_values = values[point_indices]
    signed = part is EValMetricPart.BALANCE
    metric_limits = _calculate_colour_limits(
        selected_values,
        metric_options.metric_colour_lims,
        signed=signed,
        magnitude=not signed,
    )
    mesh = None
    background_limits = None
    if sim_data is not None:
        if (
            sim_coordinate_units != geometry.coordinate_units
            or sim_frame != geometry.frame
        ):
            raise ValueError(
                "Simulation and validation coordinate units/frame must match."
            )
        if sim_data.coords is None or not np.all(np.isfinite(sim_data.coords)):
            raise ValueError("Simulation needs finite xyz coordinates.")
        if sim_field is not None:
            if sim_field.values.shape != (len(sim_data.coords),):
                raise ValueError("Background scalar values must have shape (n_nodes,).")
            if np.any(np.isinf(sim_field.values)):
                raise ValueError("Background scalar values cannot contain infinity.")
            index = sim_field.time_index
            if (
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or sim_data.time is None
                or not 0 <= index < len(sim_data.time)
            ):
                raise ValueError("Select an explicit valid background time index.")
            if observable_data.times is not None:
                time_index = (selectors or {}).get(EValResAxis.TIME, 0)
                if sim_data.time[index] != observable_data.times[time_index]:
                    raise ValueError(
                        "Background and validation physical times must match."
                    )
            background_limits = _calculate_colour_limits(
                sim_field.values,
                sim_field.colour_lims,
                signed=False,
                magnitude=False,
            )
        mesh = simdata_to_pyvista_vis(sim_data, sim_data.num_spat_dims)
    pv_plot = create_pv_plotter(options) if plotter is None else plotter
    if mesh is not None:
        if sim_field is None:
            pv_plot.add_mesh(
                mesh,
                name="validation-background",
                reset_camera=False,
                color="lightgrey",
                opacity=metric_options.mesh_opacity,
                show_edges=options.show_edges,
                show_scalar_bar=False,
            )
        else:
            mesh.point_data["background_scalar"] = sim_field.values.copy()
            pv_plot.add_mesh(
                mesh,
                name="validation-background",
                reset_camera=False,
                scalars="background_scalar",
                opacity=metric_options.mesh_opacity,
                lighting=False,
                show_edges=options.show_edges,
                cmap=sim_field.cmap,
                clim=background_limits,
                show_scalar_bar=options.colour_bar_show,
                scalar_bar_args={
                    "title": f"Simulation: {sim_field.descriptor.create_label_flat()}",
                    "vertical": options.colour_bar_vertical,
                    "position_x": 0.02,
                    "position_y": 0.12,
                    "width": 0.10 if options.colour_bar_vertical else 0.40,
                    "height": 0.70 if options.colour_bar_vertical else 0.08,
                    "color": options.font_colour,
                    "title_font_size": options.colour_bar_font_size,
                    "label_font_size": options.colour_bar_font_size,
                },
            )
    point_mesh = pv.PolyData(coords[point_indices].copy())
    point_mesh.point_data["validation_metric"] = selected_values.copy()
    point_mesh.point_data["validation_point_id"] = point_indices.copy()
    units = (
        "1"
        if metric in (EMetricType.KS, EMetricType.CVM, EMetricType.RELATIVE_ERROR)
        else observable_data.units
    )
    metric_label = metric_options.metric_label or create_metric_label(metric, part)
    bar_label = (
        "MAVM d+ − d−"
        if signed and metric_options.metric_label is None
        else metric_label
    )
    component = (selectors or {}).get(EValResAxis.COMPONENT, 0)
    quantity_label = observable_data.components[component]
    pv_plot.add_mesh(
        point_mesh,
        name="validation-points",
        reset_camera=False,
        scalars="validation_metric",
        style="points",
        render_points_as_spheres=True,
        point_size=options.sens_point_size,
        lighting=False,
        cmap=metric_options.metric_cmap or ("RdBu_r" if signed else "magma"),
        clim=metric_limits,
        nan_color="grey",
        show_scalar_bar=options.colour_bar_show,
        scalar_bar_args={
            "title": f"Validation: {quantity_label}\n{bar_label} [{units}]",
            "vertical": options.colour_bar_vertical,
            "position_x": 0.85 if options.colour_bar_vertical else 0.55,
            "position_y": 0.12,
            "width": 0.10 if options.colour_bar_vertical else 0.40,
            "height": 0.70 if options.colour_bar_vertical else 0.08,
            "color": options.font_colour,
            "title_font_size": options.colour_bar_font_size,
            "label_font_size": options.colour_bar_font_size,
            "nan_annotation": bool(np.any(np.isnan(selected_values))),
            "n_labels": 3 if signed else 5,
        },
    )
    if metric_options.point_labels is not None:
        labels = [metric_options.point_labels[index] for index in point_indices]
        pv_plot.add_point_labels(
            point_mesh,
            labels,
            name="validation-labels",
            show_points=False,
            font_size=options.sens_label_font_size,
            text_color=options.font_colour,
            shape_color=options.sens_label_colour,
            always_visible=metric_options.labels_always_visible,
        )
    title = metric_options.title or f"{quantity_label}: {metric_label} [{units}]"
    pv_plot.add_text(
        title,
        position="upper_edge",
        font_size=options.time_label_font_size,
        color=options.font_colour,
        name="validation-title",
    )
    if options.time_label_pos is not None and observable_data.times is not None:
        index = (selectors or {}).get(EValResAxis.TIME, 0)
        pv_plot.add_text(
            f"Time: {observable_data.times[index]:g} {observable_data.time_units}",
            position=options.time_label_pos,
            font_size=options.time_label_font_size,
            color=options.font_colour,
            name="validation-time",
        )
    if plotter is None:
        pv_plot.camera_position = options.camera_position
        pv_plot.reset_camera()
        if sim_data is not None and sim_data.num_spat_dims == 3:
            pv_plot.camera.zoom(0.8)
    return pv_plot

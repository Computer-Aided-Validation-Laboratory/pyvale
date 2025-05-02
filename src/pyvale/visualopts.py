# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from pathlib import Path
import enum
from dataclasses import dataclass
import numpy as np
import matplotlib as plt


@dataclass(slots=True)
class PlotOptsGeneral:
    """ Dataclass for controlling the properties of figures and graphs such as
    figure size, resolution, font sizes, marker sizes, line widths and
    colormaps.
    """
    aspect_ratio: float = 1.62
    # These are in inches because of matplotlib
    a4_width: float = 8.25
    a4_height: float = 11.75
    a4_margin_width: float = 0.5
    a4_margin_height: float = 0.5
    a4_print_width: float = a4_width-2*a4_margin_width
    a4_print_height: float = a4_height-2*a4_margin_height

    single_fig_scale: float = 0.5

    single_fig_size_square: tuple[float,float] = (
        a4_print_width*single_fig_scale,
        a4_print_width*single_fig_scale
     )
    single_fig_size_portrait: tuple[float,float] = (
        a4_print_width*single_fig_scale/aspect_ratio,
        a4_print_width*single_fig_scale
     )
    single_fig_size_landscape: tuple[float,float] = (
        a4_print_width*single_fig_scale,
        a4_print_width*single_fig_scale/aspect_ratio
     )

    resolution: float = 300.0

    font_name: str = 'Liberation Sans'
    font_def_weight: str = 'normal'
    font_def_size: float = 8.0
    font_tick_size: float = 8.0
    font_head_size: float = 9.0
    font_ax_size: float = 8.0
    font_leg_size: float = 8.0

    ms: float = 3.2
    lw: float = 0.8

    cmap_seq: str = "cividis"
    cmap_div: str = "RdBu"

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_colors = len(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    def __post_init__(self) -> None:
        plt.rc('font', size=self.font_def_size)
        plt.rc('axes', titlesize=self.font_def_size)


@dataclass(slots=True)
class TraceOptsSensor:
    """Dataclass for controlling the properties of sensor trace plots.
    """
    legend: bool = True

    x_label: str = r"x [$m$]"
    y_label: str = r"y [$m$]"
    z_label: str = r"z [$m$]"
    time_label: str = r"Time, $t$ [$s$]"

    truth_line: str | None = "-"
    sim_line: str | None = "-"
    meas_line: str = "--o"

    sensors_to_plot: np.ndarray | None = None
    time_min_max: tuple[float,float] | None = None


@dataclass(slots=True)
class TraceOptsExperiment:
    """Dataclass for contorlling the properties of sensor trace plots from
    batches of simulated experiments.
    """
    legend: bool = True

    x_label: str = r"x [$m$]"
    y_label: str = r"y [$m$]"
    z_label: str = r"z [$m$]"
    time_label: str = r"Time, $t$ [$s$]"

    truth_line: str | None = None
    sim_line: str | None = None
    exp_mean_line: str = "-"
    exp_marker_line: str = "+"

    sensors_to_plot: np.ndarray | None = None
    time_min_max: tuple[float,float] | None = None

    centre: str = "mean"
    plot_all_exp_points: bool = False
    fill_between: str | None = "3std"


@dataclass(slots=True)
class VisOptsSimSensors:
    """Dataclass for controlling displays of the simulation mesh and sensor
    locations using pyvista.
    """
    # pyvista ops
    window_size_px: tuple[int,int] = (1280,800)
    camera_position: np.ndarray | str = "xy"
    show_edges: bool = True
    interactive: bool = True

    font_colour: str = "black"
    background_colour: str = "white" # "white"

    time_label_font_size: float = 12
    time_label_position: str = "upper_left"
    time_label_show: bool = True

    colour_bar_font_size: float = 18
    colour_bar_show: bool = True
    colour_bar_lims: tuple[float,float] | None = None
    colour_bar_vertical: bool = True

    # pyvale ops
    show_perturbed_pos: bool = True
    sens_colour_nom: str = "red"
    sens_colour_pert: str = "blue"
    sens_point_size: float = 20
    sens_label_font_size: float = 30
    sens_label_colour: str = "grey"


class EImageType(enum.Enum):
    PNG = enum.auto()
    SVG = enum.auto()

@dataclass(slots=True)
class VisOptsImageSave:
    """Dataclass for image save options.
    """
    path: Path | None = None
    image_type: EImageType = EImageType.PNG
    transparent_background: bool = False


class EAnimationType(enum.Enum):
    MP4 = enum.auto()
    GIF = enum.auto()

@dataclass(slots=True)
class VisOptsAnimation:
    """Dataclass for animation save options.
    """
    frames_per_second: float = 10.0
    off_screen: bool = False

    # save options
    save_animation: EAnimationType | None = None
    save_path: Path | None = None










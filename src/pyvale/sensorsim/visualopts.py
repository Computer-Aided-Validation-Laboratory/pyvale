# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module contains options dataclasses for controlling the appearance of
visualisations in pyvale.
"""
import platform
from pathlib import Path
import enum
from dataclasses import dataclass, field
import numpy as np
import matplotlib as plt


@dataclass(slots=True)
class PlotOptsGeneral:
    """Dataclass for controlling the properties of figures and graphs such as
    figure size, resolution, font sizes, marker sizes, line widths and
    colormaps. This dataclass is used to interact with matplotlib and pyvista
    so units conform to these packages. The defaults set in this dataclass are
    selected based on producing print quality figures for journal articles.
    """

    aspect_ratio: float = 1.62
    """Aspect ratio of the figure canvas.
    """

    single_fig_scale: float = 0.5
    """Scaling for a single column figure, defaults to a half (0.5) page width.
    """

    resolution: float = 300.0
    """Figure resolution in dpi, defaults to 300dpi for print quality.
    """

    font_def_weight: str = "normal"
    """Default weight for fonts on plots.
    """

    font_def_size: float = 8.0
    """Default font size for plots.
    """

    font_tick_size: float = 8.0
    """Default font tick label size
    """

    font_head_size: float = 9.0
    """Default font size for headings/titles on plots.
    """

    font_ax_size: float = 8.0
    """Default axis label font size.
    """

    font_leg_size: float = 8.0
    """Default font size for legends.
    """

    ms: float = 3.2
    """Marker size for points on plots
    """

    lw: float = 0.8
    """Line width for traces on plots.
    """

    cmap_seq: str = "cividis"
    """The colormap to use for monotonic fields
    """

    cmap_div: str = "RdBu"
    """The colormap to use for diverging fields, defaults to Red-Blue.
    """

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    """Color cycle for lines on plots.
    """

    # These are in inches because of matplotlib
    a4_width: float = 8.25
    """Width of an A4 page in inches.
    """

    a4_height: float = 11.75
    """Height of an A4 page in inches.
    """

    a4_margin_width: float = 0.5
    """Margin width on an A4 page in inches.
    """

    a4_margin_height: float = 0.5
    """Margin heigh on an A4 page in inches.
    """

    font_name: str = field(init=False)
    """Does not need to be initialised, calculated from other inputs. Name of
    the font to use. Defaults to Arial on Windows/Mac and Liberation Sans on
    Linux.
    """

    a4_print_width: float = field(init=False)
    """Does not need to be initialised, calculated from other inputs. Printable
    width of an A4 page in inches based on subtracting twice the margin width.
    """

    a4_print_height: float = field(init=False)
    """Does not need to be initialised, calculated from other inputs. Printable
    height of an A4 page in inches based on subtracting twice the margin width.
    """

    single_fig_size_square: tuple[float,float] = field(init=False)
    """Does not need to be initialised, calculated from other inputs. Uses the
    printable A4 width and the single figure scaling to create a square canvas
    that fits in a single column of a two column journal article.
    """

    single_fig_size_portrait: tuple[float,float] = field(init=False)
    """Does not need to be initialised, calculated from other inputs. Uses the
    printable A4 width and the single figure scaling to create a potrait canvas
    that fits in a single column of a two column journal article.
    """

    single_fig_size_landscape: tuple[float,float] = field(init=False)
    """Does not need to be initialised, calculated from other inputs. Uses the
    printable A4 width and the single figure scaling to create a landscape
    canvas that fits in a single column of a two column journal article.
    """

    colors_num: int = field(init=False)
    """Does not need to be initialised, calculated from other inputs. The number
    of colors in the line color cycle.
    """


    def __post_init__(self) -> None:

        self.font_name = "Arial"
        if platform.system() == "Linux":
            self.font_name = "Liberation Sans"

        self.a4_print_width = self.a4_width-2*self.a4_margin_width
        self.a4_print_height = self.a4_height-2*self.a4_margin_height

        self.single_fig_size_square = (
            self.a4_print_width*self.single_fig_scale,
            self.a4_print_width*self.single_fig_scale
        )

        self.single_fig_size_portrait = (
            self.a4_print_width*self.single_fig_scale/self.aspect_ratio,
            self.a4_print_width*self.single_fig_scale
        )
        self.single_fig_size_landscape = (
            self.a4_print_width*self.single_fig_scale,
            self.a4_print_width*self.single_fig_scale/self.aspect_ratio
        )

        self.colors_num = len(self.colors)

        plt.rc("font", size=self.font_def_size)
        plt.rc("axes", titlesize=self.font_def_size)


@dataclass(slots=True)
class TraceOptsSensor:
    """Dataclass for controlling the appearance of sensor trace plots including
    axis labels, line styles and time over which to plot the sensor traces. Note
    that latex symbols can be used in label strings by using a python raw string
    . For example: r"strain, $\epsilon$ [-]".
    """

    legend_loc: str | None = "best"
    """Set the legend location based on matplotlib legend location string. If
    None then no legend is added. The legend lists the sensors by tag
    """

    x_label: str = r"x [$mm$]"
    """Label for the x axis defaults to: r"x [$mm$]".
    """

    y_label: str = r"y [$mm$]"
    """Label for the y axis defaults to: r"y [$mm$]".
    """

    z_label: str = r"z [$mm$]"
    """Label for the z axis defaults to: r"z [$mm$]".
    """

    time_label: str = r"Time, $t$ [$s$]"
    """Label for the time axis for traces pots which is assumed to be the
    horizontal axis.
    """

    truth_line: str | None = "-"
    """Matplotlib line style string for the ground truth virtual sensor values.
    If None then the truth line is not plotted for all virtual sensors.
    """

    sim_line: str | None = None
    """Matplotlib line style for the simulation output at the virtual sensor
    locations. If None then the line is not plotted for all virtual sensors.
    """

    meas_line: str = "--+"
    """Matplotlib line style for the virtual sensor measurement traces.
    """

    sensors_to_plot: np.ndarray | None = None
    """Array (1D) of indices for the sensors to plot. If None then all sensors
    are plotted. Defaults to None.
    """

    time_min_max: tuple[float,float] | None = None
    """Time range over which to plot the sensor traces. If None then the full
    time range is plotted. Defaults to None.
    """


class EExpVisCentre(enum.Enum):
    """Enumeration for plotting the center of the distribution of a series of
    virtual sensor experiment traces.
    """

    MEAN = enum.auto()
    """Mean over all virtual experiments for plotting the center of the virtual
    experiment traces.
    """

    MEDIAN = enum.auto()
    """Median over all virtual experiments for plotting the center of the
    virtual experiment traces.
    """


class EExpVisBounds(enum.Enum):
    """Enumeration for plotting the uncertainty bounds of a series of virtual
    sensor experiment traces. The uncertainty bounds are shown by filling
    between the given upper and lower bounds. See the experient trace opts
    dataclass which also allows for a scaling factor to be set to allow for
    plotting a given multiple of the standard deviation.
    """

    MINMAX = enum.auto()
    """Minimum and maximum over all virtual experiments for each sampling point.
    """

    QUARTILE = enum.auto()
    """Lower 25% and upper 75% quartiles over all virtual experiments for each
    sampling point.
    """

    MAD = enum.auto()
    """Median absolute deviation over all virtual experiments for each sampling
    point.
    """

    STD = enum.auto()
    """Standard deviation over all virtual sensor experiments for each sampling
    point.
    """



@dataclass(slots=True)
class TraceOptsExperiment:
    """Dataclass for controlling the properties of sensor trace plots from
    batches of simulated experiments.
    """

    legend_loc: str | None = "best"
    """Set the legend location based on matplotlib legend location string. If
    None then no legend is added. The legend lists the sensors by tag
    """

    x_label: str = r"x [$mm$]"
    """Label for the x axis defaults to: r"x [$mm$]".
    """

    y_label: str = r"y [$mm$]"
    """Label for the y axis defaults to: r"y [$mm$]".
    """

    z_label: str = r"z [$mm$]"
    """Label for the z axis defaults to: r"z [$mm$]".
    """

    time_label: str = r"Time, $t$ [$s$]"
    """Label for the time axis for traces pots which is assumed to be the
    horizontal axis.
    """

    truth_line: str | None = "-"
    """Matplotlib line style string for the ground truth virtual sensor values.
    If None then the truth line is not plotted for all virtual sensors.
    """

    sim_line: str | None = None
    """Matplotlib line style for the simulation output at the virtual sensor
    locations. If None then the line is not plotted for all virtual sensors.
    """

    exp_centre_line: str = "-"
    """Matplotlib line style string for the experiment centre line.
    """

    exp_marker_line: str = "+"
    """Maplotlib line style string use for plotting all experiments.
    """

    sensors_to_plot: np.ndarray | None = None
    """Array (1D) of indices for the sensors to plot. If None then all sensors
    are plotted. Defaults to None.
    """

    time_min_max: tuple[float,float] | None = None
    """Time range over which to plot the sensor traces. If None then the full
    time range is plotted. Defaults to None.
    """

    centre: EExpVisCentre = EExpVisCentre.MEAN
    """Specifies the summary statistic to use for the center line of the sensor
    trace distribution. Defaults to EExpVisCentre.MEAN.
    """

    fill_between: EExpVisBounds | None = EExpVisBounds.MINMAX
    """Specifies the summary statistic to use for plotting the uncertainty
    bounds for the virtual sensor traces. Defaults to EExpVisBounds.MINMAX.
    Note that this statistic will be multipled by the fill_scale parameter.
    """

    fill_scale: float = 1.0
    """Scaling factor multiplied by the uncertainty bound summary statistic for
    showing filled uncertainty bounds on sensor traces plots. Defaults to 1.0.
    A common setting would be 2.0 or 3.0 while setting fill_between =
    EExpVisBounds.STD (standard deviation).
    """

    plot_all_exp_points: bool = False
    """Allows all experiment points to be plotted. Note that for more than 100
    experiments for a given sensor array this will be slow. Defaults to False.
    """


@dataclass(slots=True)
class VisOptsSimSensors:
    """Dataclass for controlling displays of the simulation mesh and sensor
    locations using pyvista.
    """
    # pyvista ops
    window_size_px: tuple[int,int] = (1280,800)
    """Window size for pyvista canvas in pixels: (horizontal_px,vertical_px).
    """

    camera_position: np.ndarray | str = "xy"
    """Camera position for the pyvista view either as a string of axis labels
    or as a 3x3 rotation matrix. Defaults to viewing the x-y plane with "xy".
    """

    show_edges: bool = True
    """Flag to show the element edges in visualisations. Defaults to True.
    """

    interactive: bool = True
    """Flag to allow interactive viewing of the plot. Defaults to True.
    """

    font_colour: str = "black"
    """Font colour string. Useful for creating "dark mode" style plots with
    "white" font and a "black background". Defaults to "light mode" with "black"
    font.
    """

    background_colour: str = "white"
    """Background colour string. Useful for creating "dark mode" style plots
    with "white" font and a "black background". Defaults to "light mode" with
    "black" font.
    """

    time_label_pos: str | None = "upper_left"
    """Position of the simulation time step label. If None then the simulation
    time step label is not shown. Defaults to "upper_left".
    """

    time_label_font_size: int = 12
    """Font size for the simulation time step label on the canvas. Defaults to
    12.
    """

    colour_bar_show: bool = True
    """Flag to show the colourbar for the simulation field. Defaults to True.
    """

    colour_bar_font_size: int = 18
    """Font size for the colourbar. Defaults to 18.
    """

    colour_bar_lims: tuple[float,float] | None = None
    """Max and min limits for the colour bar. If None the default limits are
    used.
    """

    colour_bar_vertical: bool = True
    """Flag to set the colourbar to vertical instead of horizontal. Defaults to
    True.
    """

    # pyvale ops
    show_perturbed_pos: bool = True
    """Flag to show the perturbed sensor positions if field errors are used.
    Defaults to True.
    """

    sens_colour_nom: str = "red"
    """Colour for the markers showing the nominal sensor locations.
    """

    sens_colour_pert: str = "blue"
    """Colour for the markers showing the perturbed sensor locations.
    """

    sens_point_size: float = 20.0
    """Size for the markers used to show the sensor locations on the mesh.
    """

    sens_label_font_size: int = 30
    """Font size for the sensor marker labels.
    """

    sens_label_colour: str = "grey"
    """Colour for the sensor labels. Note that this needs to provide reasonable
    contrast with the selected font colour so "grey" is the default.
    """


class EImageType(enum.Enum):
    """NOTE: This is a feature under developement.

    Enumeration for specifying the format for saving images.
    """
    PNG = enum.auto()
    SVG = enum.auto()


@dataclass(slots=True)
class VisOptsImageSave:
    """NOTE: This is a feature under developement.

    Dataclass for image saving options.
    """

    path: Path | None = None
    image_type: EImageType = EImageType.PNG
    transparent_background: bool = False



class EAnimationType(enum.Enum):
    """NOTE: This is a feature under developement.

    Enumeration for specifying the save file type for animations.
    """
    MP4 = enum.auto()
    GIF = enum.auto()


@dataclass(slots=True)
class VisOptsAnimation:
    """NOTE: This is a feature under developement.

    Dataclass for animation save options.
    """

    frames_per_second: float = 10.0
    off_screen: bool = False
    save_animation: EAnimationType | None = None
    save_path: Path | None = None












# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module contains utility functions used for creating pyvale visualisations.
"""

from pathlib import Path
import numpy as np
import vtk #NOTE: has to be here to fix latex bug in pyvista/vtk
# See: https://github.com/pyvista/pyvista/discussions/2928
# NOTE: causes output to console to be suppressed unfortunately
# NOTE: May2025 still needs include but does not suppress console output
import pyvista as pv
from pyvale.visualopts import (VisOptsSimSensors,
                                    VisOptsImageSave,
                                    EImageType,
                                    VisOptsAnimation,
                                    EAnimationType)

def create_pv_plotter(vis_opts: VisOptsSimSensors) -> pv.Plotter:
    """Creates a pyvista plotter based on the input options.

    Parameters
    ----------
    vis_opts : VisOptsSimSensors
        Dataclass containing the visualisation options for creating the plotter.

    Returns
    -------
    pv.Plotter
        Blank pyvista plotter object with the given settings.
    """
    pv_plot = pv.Plotter(window_size=vis_opts.window_size_px)
    pv_plot.set_background(vis_opts.background_colour)
    pv.global_theme.font.color = vis_opts.font_colour
    pv_plot.add_axes_at_origin(labels_off=True)
    return pv_plot


def get_colour_lims(component_data: np.ndarray,
                    colour_bar_lims: tuple[float,float] | None
                    ) -> tuple[float,float]:
    """Gets the colourbar limits based on the input component data array.

    Parameters
    ----------
    component_data : np.ndarray
        Array of data for the field component of interest. Can be any shape as
        the array is flattened for the limit calculations
    colour_bar_lims : tuple[float,float] | None
        Forces the colourbar limits to be the values give in the tuple. If None
        then the colorbar limits are calculated based on the input data array.

    Returns
    -------
    tuple[float,float]
        Colourbar limits in the form: (min,max).
    """
    if colour_bar_lims is None:
        min_comp = np.min(component_data.flatten())
        max_comp = np.max(component_data.flatten())
        colour_bar_lims = (min_comp,max_comp)

    assert colour_bar_lims[1] > colour_bar_lims[0], ("Colourbar minimum must be"
    + " smaller than the colourbar maximum.")

    return colour_bar_lims


def save_pv_image(pv_plot: pv.Plotter,
               image_save_opts: VisOptsImageSave) -> None:
    """Saves an image of a pyvista visualisation to disk based on the input
    options.

    Parameters
    ----------
    pv_plot : pv.Plotter
        Pyvista plotter object to save the image from.
    image_save_opts : VisOptsImageSave
        Dataclass containing the options to save the image.
    """

    if image_save_opts.path is None:
        image_save_opts.path = Path.cwd() / "pyvale-image"

    if image_save_opts.image_type == EImageType.PNG:
        image_save_opts.path = image_save_opts.path.with_suffix(".png")
        pv_plot.screenshot(image_save_opts.path,
                           image_save_opts.transparent_background)

    elif image_save_opts.image_type == EImageType.SVG:
        image_save_opts.path = image_save_opts.path.with_suffix(".svg")
        pv_plot.save_graphic(image_save_opts.path)


def set_animation_writer(pv_plot: pv.Plotter,
                         anim_opts: VisOptsAnimation) -> pv.Plotter:
    """Sets the animation writer and output path for a virtual sensor simulation
    visualisation.

    Parameters
    ----------
    pv_plot : pv.Plotter
        Pyvistas plot object which will be used to create the animation.
    anim_opts : VisOptsAnimation
        Dataclass containing the options for creating the animation.

    Returns
    -------
    pv.Plotter
        Pyvista plotter with the given animation writer opened.
    """
    if anim_opts.save_animation is None:
          return pv_plot

    if anim_opts.save_path is None:
        anim_opts.save_path = Path.cwd() / "pyvale-animation"

    if anim_opts.save_animation == EAnimationType.GIF:
        anim_opts.save_path = anim_opts.save_path.with_suffix(".gif")
        pv_plot.open_gif(anim_opts.save_path,
                         loop=0,
                         fps=anim_opts.frames_per_second)

    elif anim_opts.save_animation == EAnimationType.MP4:
        anim_opts.save_path = anim_opts.save_path.with_suffix(".mp4")
        pv_plot.open_movie(anim_opts.save_path,
                           anim_opts.frames_per_second)

    return pv_plot



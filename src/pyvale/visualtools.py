# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from pathlib import Path
import numpy as np
#import vtk #NOTE: has to be here to fix latex bug in pyvista/vtk
# See: https://github.com/pyvista/pyvista/discussions/2928
#NOTE: causes output to console to be suppressed unfortunately
import pyvista as pv
from pyvale.visualopts import (VisOptsSimSensors,
                                    VisOptsImageSave,
                                    EImageType,
                                    VisOptsAnimation,
                                    EAnimationType)

#TODO: Docstrings

def create_pv_plotter(vis_opts: VisOptsSimSensors) -> pv.Plotter:
    pv_plot = pv.Plotter(window_size=vis_opts.window_size_px)
    pv_plot.set_background(vis_opts.background_colour)
    pv.global_theme.font.color = vis_opts.font_colour
    pv_plot.add_axes_at_origin(labels_off=True)
    return pv_plot


def get_colour_lims(component_data: np.ndarray,
                     colour_bar_lims: tuple[float,float] | None
                     ) -> tuple[float,float]:

    if colour_bar_lims is None:
        min_comp = np.min(component_data.flatten())
        max_comp = np.max(component_data.flatten())
        colour_bar_lims = (min_comp,max_comp)

    return colour_bar_lims


def save_image(pv_plot: pv.Plotter,
               image_save_opts: VisOptsImageSave) -> None:
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
        print(80*"=")
        print(f"{anim_opts.save_path=}")
        print(80*"=")
        pv_plot.open_movie(anim_opts.save_path,
                           anim_opts.frames_per_second)

    return pv_plot



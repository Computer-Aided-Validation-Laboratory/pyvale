# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: this module is a feature under developement.
"""

from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from pyvale.camera import CameraBasic2D
from pyvale.visualopts import PlotOptsGeneral

# TODO: this only works for a 2D camera, maybe this should be deprecated
def plot_measurement_image(camera: CameraBasic2D,
                           component: str,
                           time_step: int = -1,
                           plot_opts: PlotOptsGeneral | None = None
                           ) -> tuple[Any,Any]:

    if plot_opts is None:
        plot_opts = PlotOptsGeneral()

    comp_ind = camera.get_field().get_component_index(component)
    meas_image = camera.get_measurement_images()[:,:,comp_ind,time_step]
    descriptor = camera.get_descriptor()

    (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_square,
                             layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    cset = plt.imshow(meas_image,
                      cmap=plt.get_cmap(plot_opts.cmap_seq),
                      origin='lower')
    ax.set_aspect('equal','box')

    fig.colorbar(cset,
                 label=descriptor.create_label_flat(comp_ind))

    title = f"Time: {camera.get_sample_times()[time_step]}s"
    ax.set_title(title,fontsize=plot_opts.font_head_size)
    ax.set_xlabel(r"x ($px$)",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax.set_ylabel(r"y ($px$)",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

    return (fig,ax)


def plot_field_image(image: np.ndarray,
                     title_str: str | None = None,
                     plot_opts: PlotOptsGeneral | None = None
                     ) -> tuple[Any,Any]:

    if plot_opts is None:
        plot_opts = PlotOptsGeneral()

        (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_square,
                                layout='constrained')
        fig.set_dpi(plot_opts.resolution)
        cset = plt.imshow(image,
                          cmap=plt.get_cmap(plot_opts.cmap_seq))
                          #origin='lower')
        ax.set_aspect('equal','box')
        fig.colorbar(cset)
        if title_str is not None:
            ax.set_title(title_str,fontsize=plot_opts.font_head_size)
        ax.set_xlabel(r"x ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
        ax.set_ylabel(r"y ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

        return (fig,ax)

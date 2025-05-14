# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: this module is a feature under developement.
"""

import numpy as np
import matplotlib.pyplot as plt

class ImageDefDiags:
    @staticmethod
    def plot_speckle_image(image: np.ndarray,
                   title: str = "",
                   cmap: str = "gray") -> None:
        fig, ax = plt.subplots()
        cset = plt.imshow(image,cmap=plt.get_cmap(cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title(title,fontsize=12)
        fig.colorbar(cset)

    @staticmethod
    def plot_vector_image(image: np.ndarray,
                   title: str = "",
                   cmap: str = "plasma") -> None:
        fig, ax = plt.subplots()
        cset = plt.imshow(image,cmap=plt.get_cmap(cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title(title,fontsize=12)
        fig.colorbar(cset)


    @staticmethod
    def plot_image_xy(image: np.ndarray,
                      extent: tuple[float,float,float,float],
                      title: str = "",
                      cmap: str = "plasma",
                      ax_units: str = "mm") -> None:

        fig, ax = plt.subplots()
        cset = plt.imshow(image,
                        aspect='auto',interpolation='none',
                        origin='lower',cmap=plt.get_cmap(cmap),
                        extent=extent)
        ax.set_aspect('equal','box')
        ax.set_title(title,fontsize=12)
        ax.set_xlabel(f'x [{ax_units}]',fontsize=12)
        ax.set_ylabel(f'y [{ax_units}]',fontsize=12)
        fig.colorbar(cset)

    @staticmethod
    def plot_all_diags(def_image: np.ndarray,
                    def_mask: np.ndarray | None,
                    def_image_subpx: np.ndarray,
                    subpx_disp_x: np.ndarray,
                    subpx_disp_y: np.ndarray,
                    subpx_grid_xm: np.ndarray,
                    subpx_grid_ym: np.ndarray) -> None:
        image_map = "gray"
        vector_map = "plasma"

        if def_mask is not None:
            ImageDefDiags.plot_speckle_image('Def. Mask',def_mask,image_map)

        ImageDefDiags.plot_speckle_image('Subpx Def. Image',def_image_subpx,image_map)
        ImageDefDiags.plot_speckle_image('Def. Image',def_image,image_map)

        ext = tuple(np.array([subpx_grid_xm.min(),subpx_grid_xm.max(),
                        subpx_grid_ym.min(),subpx_grid_ym.max()])*10**3)
        ImageDefDiags.plot_image_xy('Sub Pixel Disp X',subpx_disp_x,ext,vector_map)
        ImageDefDiags.plot_image_xy('Sub Pixel Disp Y',subpx_disp_y,ext,vector_map)

# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class StrainResults:
    """
    Data container for Strain analysis results. This dataclass stores the strain 
    window coordinates, 2D deformation gradient and strain values.

    Attributes
    ----------

    window_x : np.ndarray
        The x-coordinates of the strain window centre. shape=(img_num,y,x)
    window_y : np.ndarray
        The y-coordinates of the strain window centre. shape=(img_num,y,x)
    x_mm : np.ndarray | None
        X-coordinate of the strain window centre in physical units relative to cam0.
    y_mm : np.ndarray | None
        Y-coordinate of the strain window centre in physical units relative to cam0.
    z_mm : np.ndarray | None
        Z-coordinate of the strain window centre in physical units relative to cam0.
    def_00 : np.ndarray
        The xx component (1 + ∂u/∂x) of the deformation gradient matrix. shape=(img_num, y, x)
    def_01 : np.ndarray
        The xy component (∂u/∂y) of the deformation gradient matrix. shape=(img_num, y, x)
    def_10 : np.ndarray
        The yx component (∂v/∂x) of the deformation gradient matrix. shape=(img_num, y, x)
    def_11 : np.ndarray
        The yy component (1 + ∂v/∂y) of the deformation gradient matrix. shape=(img_num, y, x)
    def_20 : np.ndarray
        The zx component (∂w/∂x) of the deformation gradient matrix. shape=(img_num, y, x)
    def_21 : np.ndarray
        The zy component (∂w/∂y) of the deformation gradient matrix. shape=(img_num, y, x)
    eps_xx : np.ndarray
        The xx component of the surface strain tensor. shape=(img_num, y, x)
    eps_xy : np.ndarray
        The xy component of the surface strain tensor. shape=(img_num, y, x)
    eps_yx : np.ndarray
        The yx component of the surface strain tensor. shape=(img_num, y, x)
    eps_yy : np.ndarray
        The yy component of the surface strain tensor. shape=(img_num, y, x)
    filenames : list[str]
        name of Strain result files that have been found
    """

    window_x: np.ndarray
    window_y: np.ndarray
    def_00: np.ndarray
    def_01: np.ndarray
    def_10: np.ndarray
    def_11: np.ndarray
    def_20: np.ndarray
    def_21: np.ndarray
    eps_xx: np.ndarray
    eps_xy: np.ndarray
    eps_yx: np.ndarray
    eps_yy: np.ndarray
    filenames: list[str]
    x_mm: np.ndarray | None = None
    y_mm: np.ndarray | None = None
    z_mm: np.ndarray | None = None

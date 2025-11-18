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
    Data container for Strain analysis results.

    This dataclass stores the strain window coordinates, 2D deformation gradient
    and strain values.

    Attributes
    ----------
    window_x : np.ndarray
        The x-coordinates of the strain window centre. shape=(img_num,y,x)
    window_y : np.ndarray
        The y-coordinates of the strain window centre. shape=(img_num,y,x)
    def_xx : np.ndarray
        The xx component of the 2D deformation gradient. shape=(img_num,y,x)
    def_xy : np.ndarray
        The xy component of the 2D deformation gradient. shape=(img_num,y,x)
    def_yx : np.ndarray
        The yx component of the 2D deformation gradient. shape=(img_num,y,x)
    def_yy : np.ndarray
        The yy component of the 2D deformation gradient. shape=(img_num,y,x)
    eps_xx : np.ndarray
        The xx component of the 2D strain tensor. shape=(img_num,y,x)
     eps_xy : np.ndarray
        The xy component of the 2D strain tensor. shape=(img_num,y,x)
     eps_yx : np.ndarray
        The yx component of the 2D strain tensor. shape=(img_num,y,x)
     eps_yy : np.ndarray
        The yy component of the 2D strain tensor. shape=(img_num,y,x)
    filenames : list[str]
        name of Strain result files that have been found
    """

    window_x: np.ndarray
    window_y: np.ndarray
    def_xx: np.ndarray
    def_xy: np.ndarray
    def_yx: np.ndarray
    def_yy: np.ndarray
    eps_xx: np.ndarray
    eps_xy: np.ndarray
    eps_yx: np.ndarray
    eps_yy: np.ndarray
    filenames: list[str]

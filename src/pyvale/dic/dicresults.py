# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class StereoResults:
    """
    Data container for stereo DIC results
    """
    
    u_px: np.ndarray
    """Horizontal displacement in pixels to the right image. shape=(img_num,y,x)"""

    v_px: np.ndarray
    """Vertical displacement in pixels to the right image. shape=(img_num,y,x)"""

    u_mm: np.ndarray
    """Horizontal displacement in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    v_mm: np.ndarray
    """Vertical displacement in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    w_mm: np.ndarray
    """Axial displacement in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    x_mm: np.ndarray
    """X-coordinate in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    y_mm: np.ndarray
    """Y-coordinate in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""
    
    z_mm: np.ndarray
    """Z-coordinate in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""
    
    mag_px: np.ndarray | None = None
    """Displacement magnitude to the right image, typically computed as sqrt(disp_u_px^2 + disp_v_px^2). shape=(img_num,y,x)"""

    converged: np.ndarray | None = None
    """boolean value for whether the subset has converged or not. shape=(img_num,y,x)"""

    cost: np.ndarray | None = None
    """Final cost or residual value from the correlation between subset in left and right image as calculated using ZNCC. shape=(img_num,y,x)"""

    ftol: np.ndarray | None = None
    """Final `ftol` value from the optimization routine, indicating function tolerance. shape=(img_num,y,x)"""
    
    xtol: np.ndarray | None = None
    """Final `xtol` value from the optimization routine, indicating solution tolerance. shape=(img_num,y,x)"""
    
    niter: np.ndarray | None = None
    """Number of iterations taken to converge for each subset point. shape=(img_num,y,x)"""

@dataclass(slots=True)
class Results:
    """
    Data container for DIC analysis results.

    This dataclass stores the displacements, convergence info, and correlation data
    associated with a DIC computation.
    """

    ss_x: np.ndarray
    """The x-coordinates of the subset centers (in pixels). shape=(img_num,y,x)"""
    ss_y: np.ndarray
    """The y-coordinates of the subset centers (in pixels). shape=(img_num,y,x)"""

    u_px: np.ndarray
    """Horizontal displacements in pixels at each subset location. shape=(img_num,y,x)"""

    v_px: np.ndarray
    """Vertical displacements in pixels at each subset location. shape=(img_num,y,x)"""

    u_mm: np.ndarray | None = None
    """Horizontal displacement in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    v_mm: np.ndarray | None = None
    """Vertical displacement in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""
    
    x_mm: np.ndarray | None = None
    """X-coordinate in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    y_mm: np.ndarray | None = None
    """Y-coordinate in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    mag_px: np.ndarray | None = None
    """Displacement magnitude in mm at each subset location, typically computed as sqrt(u^2 + v^2). shape=(img_num,y,x)"""

    converged: np.ndarray | None = None
    """boolean value for whether the subset has converged or not. shape=(img_num,y,x)"""

    cost: np.ndarray | None = None
    """Final cost or residual value from the correlation optimization as calculated using ZNCC. shape=(img_num,y,x)"""

    ftol: np.ndarray | None = None
    """Final `ftol` value from the optimization routine, indicating function tolerance. shape=(img_num,y,x)"""

    xtol: np.ndarray | None = None
    """Final `xtol` value from the optimization routine, indicating solution tolerance. shape=(img_num,y,x)"""

    niter: np.ndarray | None = None
    """Number of iterations taken to converge for each subset point. shape=(img_num,y,x)"""

    filenames: list[str] | None = None
    """name of DIC result files that have been found"""

    stereo: StereoResults | None = None
    """Optional field to store stereo DIC results if available."""


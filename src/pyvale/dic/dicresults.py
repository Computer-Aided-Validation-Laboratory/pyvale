# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class Results:
    """
    Data container for Digital Image Correlation (DIC) analysis results.

    This dataclass stores the displacements, convergence info, and correlation data
    associated with a DIC computation.

    Attributes
    ----------
    ss_x : np.ndarray
        The x-coordinates of the subset centers (in pixels). shape=(img_num,y,x)
    ss_y : np.ndarray
        The y-coordinates of the subset centers (in pixels). shape=(img_num,y,x)
    u : np.ndarray
        Horizontal displacements at each subset location. shape=(img_num,y,x)
    v : np.ndarray
        Vertical displacements at each subset location. shape=(img_num,y,x)
    mag : np.ndarray
        Displacement magnitude at each subset location, typically computed as sqrt(u^2 + v^2). shape=(img_num,y,x)
    converged : np.ndarray
        boolean value for whether the subset has converged or not. shape=(img_num,y,x)
    cost : np.ndarray
        Final cost or residual value from the correlation optimization (e.g., ZNSSD). shape=(img_num,y,x)
    ftol : np.ndarray
        Final `ftol` value from the optimization routine, indicating function tolerance. shape=(img_num,y,x)
    xtol : np.ndarray
        Final `xtol` value from the optimization routine, indicating solution tolerance. shape=(img_num,y,x)
    niter : np.ndarray
        Number of iterations taken to converge for each subset point. shape=(img_num,y,x)
    shape_params : np.ndarray | None
        Optional shape parameters if output during DIC calculation (e.g., affine, rigid). shape=(img_num,y,x)
    filenames : list[str]
        name of DIC result files that have been found
    """

    ss_x: np.ndarray
    ss_y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    mag: np.ndarray
    converged: np.ndarray
    cost: np.ndarray
    ftol: np.ndarray
    xtol: np.ndarray
    niter: np.ndarray
    shape_params: np.ndarray
    filenames: list[str]

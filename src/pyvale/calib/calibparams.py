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
        The x-coordinates of the subset centers (in pixels).
    ss_y : np.ndarray
        The y-coordinates of the subset centers (in pixels).
    u : np.ndarray
        Horizontal displacements at each subset location.
    v : np.ndarray
        Vertical displacements at each subset location.
    mag : np.ndarray
        Displacement magnitude at each subset location, typically computed as sqrt(u^2 + v^2).
    converged : np.ndarray
        boolean value for whether the subset has converged or not.
    cost : np.ndarray
        Final cost or residual value from the correlation optimization (e.g., ZNSSD).
    ftol : np.ndarray
        Final `ftol` value from the optimization routine, indicating function tolerance.
    xtol : np.ndarray
        Final `xtol` value from the optimization routine, indicating solution tolerance.
    niter : np.ndarray
        Number of iterations taken to converge for each subset point.
    shape_params : np.ndarray | None
        Optional shape parameters if output during DIC calculation (e.g., affine, rigid).
    filenames : list[str]
        name of DIC result files that have been found
    """

    

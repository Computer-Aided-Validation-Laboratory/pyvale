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

    This dataclass stores the strain window coordinates, deformation gradient
    and strain values.

    Attributes
    ----------
    window_x : np.ndarray
        The x-coordinates of the strain window centre.
    window_y : np.ndarray
        The y-coordinates of the strain window centre.
    def_grad : np.ndarray
        The 2D deformation gradient.
    eps : np.ndarray
        The 2D strain tensor.
    filenames : list[str]
        name of Strain result files that have been found
    """

    window_x: np.ndarray
    window_y: np.ndarray
    def_grad: np.ndarray
    eps: np.ndarray
    filenames: list[str]

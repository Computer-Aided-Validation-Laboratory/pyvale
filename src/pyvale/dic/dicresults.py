# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class DICResults:
    ss_x: np.ndarray
    ss_y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    mag: np.ndarray
    cost: np.ndarray
    ftol: np.ndarray
    xtol: np.ndarray
    niter: np.ndarray

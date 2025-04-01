"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""


from dataclasses import dataclass
import numpy as np

@dataclass
class DICResults:
    niter: np.ndarray
    subsets: np.ndarray
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    ftol: np.ndarray
    xtol: np.ndarray

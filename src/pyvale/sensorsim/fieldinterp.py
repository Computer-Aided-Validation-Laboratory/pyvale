# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation

class FieldInterp(ABC):
    @abstractmethod
    def interp_field(self,
                    points: np.ndarray,
                    times: np.ndarray | None = None,
                    angles: tuple[Rotation,...] | None = None,
                    ) -> np.ndarray:
        pass

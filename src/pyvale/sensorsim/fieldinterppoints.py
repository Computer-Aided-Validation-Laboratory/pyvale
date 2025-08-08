# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from pyvale.sensorsim.fieldinterp import FieldInterp
import numpy as np


class FieldInterpPoints(FieldInterp):

    def __init__(self) -> None:
        pass

    def interp_field(self,
                    points: np.ndarray,
                    times: np.ndarray | None = None,
                    ) -> np.ndarray:
        pass
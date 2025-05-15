# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: this module is a feature under developement.
"""

from abc import ABC, abstractmethod
import numpy as np

# NOTE:
# - Need to render a single frame static/deformed
# - Need to render all frames static/deformed
class IRaster(ABC):
    @abstractmethod
    def render_static_frame() -> None:
        pass

    @abstractmethod
    def render_deformed_frame() -> None:
        pass


# NOTE:
# - Manages parallelisation/saving with different rendering backends
# - Uses multi-processing
class Raster:
    pass

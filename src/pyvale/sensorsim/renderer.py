#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from pyvale.sensorsim.renderscene import RenderScene

# NOTE: This module is a feature under developement.

class IRenderer(ABC):
    @abstractmethod
    def render(self,
               scene: RenderScene,
               cam_ind: int = 0,
               frame_ind: int = 0,
               field_ind: int = 0) -> np.ndarray:
        pass

    @abstractmethod
    def render_to_disk(self,
                       scene: RenderScene,
                       cam_ind: int = 0,
                       frame_ind: int = 0,
                       field_ind: int = 0,
                       save_path: Path | None = None,
                       ) -> None:
        pass

    @abstractmethod
    def render_all(self, scene: RenderScene) -> list[np.ndarray]:
        pass

    @abstractmethod
    def render_all_to_disk(self,
                          scene: RenderScene,
                          save_path: Path | None = None) -> None:
        pass








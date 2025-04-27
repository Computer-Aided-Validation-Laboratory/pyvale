#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from pyvale.renderscene import RenderScene

# NOTE: This module is a feature under developement.

#===============================================================================
class IRenderEngine(ABC):
    @abstractmethod
    def render(self,
               scene: RenderScene,
               frame_ind: int = 0) -> np.ndarray:
        pass

    @abstractmethod
    def render_to_disk(self,
                       scene: RenderScene,
                       save_path: Path | None = None,
                       frame_ind: int = 0) -> None:
        pass

    @abstractmethod
    def render_all(self, scene: RenderScene) -> list[np.ndarray]:
        pass

    @abstractmethod
    def render_all_to_disk(self,
                          scene: RenderScene,
                          save_path: Path | None = None) -> None:
        pass






#===============================================================================
# TODO
#-------------------------------------------------------------------------------
# TO DISK / TO RAM
# ONE FRAME / ALL FRAMES

# Highest level abstraction - need to have python and non-python parallel by
# frame and by camera
class Renderer:
    __slots__ = ("scene","engine")

    def __init__(self,
                 scene: RenderScene,
                 engine: IRenderEngine) -> None:
        self.scene = scene
        self.engine = engine

    def render_one(self, frame_ind: int = 0) -> None:
        pass

    def render_all(self, parallel: int | None = None) -> None:
        pass






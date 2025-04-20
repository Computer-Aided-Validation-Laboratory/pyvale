"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from pyvale.core.renderscene import RenderScene

# NOTE: This module is a feature under developement.

#===============================================================================
class IRenderEngine(ABC):
    @abstractmethod
    def render_frame(self, frame_ind: int = 0) -> list[np.ndarray]:
        pass

    @abstractmethod
    def render_frame_to_disk(self,
                          save_path: Path | None = None,
                          frame_ind: int = 0) -> None:
        pass

    @abstractmethod
    def render_all_frames(self) -> list[np.ndarray]:
        pass

    @abstractmethod
    def render_all_frames_to_disk(self, save_path: Path | None = None) -> None:
        pass



#===============================================================================
@dataclass(slots=True)
class RenderOpts:
    save_path: Path | None = None
    image_tag: str = "image"
    #image_formats: tuple[ImageFormat,...]
    bits_per_unit: int = 1
    parallel: int | None = None

    # def __post_init__(self) -> None:
    #     if save_path is None:


#===============================================================================
# TODO
#-------------------------------------------------------------------------------
# TO DISK / TO RAM
# ONE FRAME / ALL FRAMES
#
# Parallelisation:
#   - By frame
#   - By camera
#

# Highest level abstraction - need to have python and non-python parallel by
# frame and by camera
class Renderer:
    __slots__ = ("scene","engine","opts")

    def __init__(self,
                 scene: RenderScene,
                 engine: IRenderEngine,
                 opts: RenderOpts) -> None:
        self.scene = scene
        self.engine = engine
        self.opts = opts

    def render_one_frame(self, frame_ind: int = 0) -> None:
        pass

    def render_all_frames(self, parallel: int | None = None) -> None:
        pass






# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from pyvale.cameradata import CameraData
from pyvale.output import Outputs

#TODO: docstrings

class RenderEngine(Enum):
    """Different render engines on Blender
    """
    CYCLES = "CYCLES"
    EEVEE = "BLENDER_EEVEE_NEXT"
    WORKBENCH = "BLENDER_WORKBENCH"

@dataclass(slots=True)
class RenderData:
    cam_data: CameraData | tuple[CameraData, CameraData]
    base_dir: Path = Outputs.base_dir
    samples: int = 2
    engine: RenderEngine = RenderEngine.CYCLES
    max_bounces: int = 12
    bit_size: int = 8
    threads:int = 4
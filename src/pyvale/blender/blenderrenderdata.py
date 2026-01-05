# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Pyvale
from pyvale.sensorsim.cameradata import CameraData


class RenderEngine(Enum):
    """Different render engines on Blender
    """
    CYCLES = "CYCLES"
    EEVEE = "BLENDER_EEVEE_NEXT"
    WORKBENCH = "BLENDER_WORKBENCH"

@dataclass(slots=True)
class RenderData:
    cam_data: CameraData | tuple[CameraData, CameraData]
    base_dir: Path | None = None
    dir_name: str = "images"
    samples: int = 2
    engine: RenderEngine = RenderEngine.CYCLES
    max_bounces: int = 12
    bit_size: int = 8
    threads:int = 4

    def __post_init__(self) -> None:
        if self.base_dir is None:
            self.base_dir = Path.cwd()

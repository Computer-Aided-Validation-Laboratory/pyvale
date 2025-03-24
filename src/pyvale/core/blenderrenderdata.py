from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from pyvale.core.cameradata import CameraData

class RenderEngine(Enum):
    """Different render engines on Blender
    """
    CYCLES = "CYCLES"
    EEVEE = "BLENDER_EEVEE_NEXT"
    WORKBENCH = "BLENDER_WORKBENCH"

@dataclass
class RenderData:
    cam_data: CameraData | tuple[CameraData, CameraData]
    save_dir: Path | None
    save_name : str | None
    samples: int | None = None
    engine: RenderEngine | None = None
    max_bounces: int | None = None

    def __post_init__(self) -> None:
        if self.max_bounces is None:
            self.max_bounces = 12
        if self.engine is None:
            self.engine = RenderEngine.CYCLES
        if self.samples is None:
            self.samples = 2
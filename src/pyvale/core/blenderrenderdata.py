from enum import Enum
from dataclasses import dataclass
from pyvale.core.cameradata import CameraData

class RenderEngine(Enum):
    """Different render engines on Blender
    """
    CYCLES = "CYCLES"
    EEVEE = "BLENDER_EEVEE_NEXT"
    WORKBENCH = "BLENDER_WORKBENCH"

@dataclass
class RenderData:
    samples: int | None = None
    engine: RenderEngine | None = None
    max_bounces: int | None = None
    cam_data: CameraData | tuple[CameraData, CameraData]

    def __post_init__(self) -> None:
        self.max_bounces = 12
        self.engine = RenderEngine.CYCLES
        self.samples = 2
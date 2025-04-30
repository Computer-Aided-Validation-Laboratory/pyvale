from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from pyvale.cameradata import CameraData

class RenderEngine(Enum):
    """Different render engines on Blender
    """
    CYCLES = "CYCLES"
    EEVEE = "BLENDER_EEVEE_NEXT"
    WORKBENCH = "BLENDER_WORKBENCH"

@dataclass
class RenderData:
    cam_data: CameraData | tuple[CameraData, CameraData]
    save_dir: Path | None = None
    save_name: str| None = None
    samples: int = 2
    engine: RenderEngine = RenderEngine.CYCLES
    max_bounces: int = 12
    bit_size: int = 8
    threads:int = 4

    def __post_init__(self) -> None:
        if self.save_dir is None:
            self.save_dir = Path.cwd() / "blenderimages"
            if not self.save_dir.is_dir():
                self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.save_name is None:
            self.save_name = "blenderimage"
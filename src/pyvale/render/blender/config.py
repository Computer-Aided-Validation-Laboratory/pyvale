"""Blender-specific unified renderer configuration."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class EBlenderEngine(Enum):
    """Blender render engines supported by the unified adapter."""

    CYCLES = "CYCLES"
    EEVEE = "BLENDER_EEVEE"
    WORKBENCH = "BLENDER_WORKBENCH"


@dataclass(frozen=True, slots=True)
class BlenderConfig:
    """Stable Blender controls accepted by the unified adapter.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory used for optional TIFF and Blender-project outputs.
    engine : EBlenderEngine, optional
        Blender render engine.
    samples : int, optional
        Per-pixel render samples.
    max_bounces : int, optional
        Cycles maximum light-bounce count.
    threads : int, optional
        Blender render worker count.
    render_deformed : bool, optional
        Render each nodal-displacement frame instead of a static scene.
    save_images : bool, optional
        Persist TIFFs and return their paths rather than retaining image arrays.
    save_scene : bool, optional
        Persist the constructed Blender scene as a ``.blend`` project file.
    """

    output_dir: Path
    engine: EBlenderEngine = EBlenderEngine.CYCLES
    samples: int = 2
    max_bounces: int = 12
    threads: int = 1
    render_deformed: bool = False
    save_images: bool = False
    save_scene: bool = False

    def __post_init__(self) -> None:
        """Normalise the output directory to a path object."""
        object.__setattr__(self, "output_dir", Path(self.output_dir))

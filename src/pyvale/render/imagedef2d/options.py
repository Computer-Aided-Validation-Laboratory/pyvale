"""Configuration for finite-element-driven image deformation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, kw_only=True)
class ImageDefOpts:
    """Options controlling finite-element image deformation."""

    save_path: Path | None = None
    save_tag: str = "defimage"
    mask_input_image: bool = True
    add_static_ref: bool = False
    fe_interp: str = "linear"
    fe_rescale: bool = True
    fe_extrap_outside_fov: bool = True
    image_def_order: int = 3
    image_def_extrap: str = "nearest"
    image_def_extval: float = 0.0
    def_complex_geom: bool = True


__all__ = ["ImageDefOpts"]

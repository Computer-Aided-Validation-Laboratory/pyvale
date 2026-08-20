"""Blender-owned mesh shader definitions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class BlenderMaterial:
    """Principled-material controls owned by the Blender backend.

    Parameters
    ----------
    roughness : float, optional
        Blender principled-BSDF roughness.
    metallic : float, optional
        Blender principled-BSDF metallic weight.
    interpolant : str, optional
        Blender image-texture interpolation mode.
    """

    roughness: float = 1.0
    metallic: float = 0.0
    interpolant: str = "Cubic"


@dataclass(frozen=True, slots=True)
class BlenderTextureShader:
    """Image texture controls for a Blender mesh material.

    Parameters
    ----------
    image_path : pathlib.Path
        Speckle or surface texture image supplied to Blender.
    millimetres_per_pixel : float
        Texture scale used by Blender UV unwrapping.
    material : BlenderMaterial, optional
        Backend-owned principled material controls.
    """

    image_path: Path
    millimetres_per_pixel: float
    material: BlenderMaterial = BlenderMaterial()

    def __post_init__(self) -> None:
        """Normalise the texture path."""
        object.__setattr__(self, "image_path", Path(self.image_path))


@dataclass(frozen=True, slots=True)
class BlenderImageShader:
    """In-memory texture-image controls for a Blender mesh material.

    Parameters
    ----------
    image : numpy.ndarray
        Greyscale or RGBA image texture supplied directly to Blender.
    millimetres_per_pixel : float
        Texture scale used by Blender UV unwrapping.
    material : BlenderMaterial, optional
        Backend-owned principled material controls.
    """

    image: np.ndarray
    millimetres_per_pixel: float
    material: BlenderMaterial = BlenderMaterial()

    def __post_init__(self) -> None:
        """Store an immutable contiguous image representation."""
        image = np.asarray(self.image)
        if np.issubdtype(image.dtype, np.floating):
            upper = 1.0 if image.size and np.nanmax(image) <= 1.0 else 255.0
            image = np.clip(image * (255.0 / upper), 0.0, 255.0)
        object.__setattr__(
            self, "image", np.ascontiguousarray(image, dtype=np.uint8),
        )

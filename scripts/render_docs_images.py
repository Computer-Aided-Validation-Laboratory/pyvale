#!/usr/bin/env python3
# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Build documentation artifacts from Render2D and Render3D examples.

Run the render examples first so their standard directories exist below
``pyvale-output``. This script selects the documented first-frame outputs,
converts them to PNG, combines stereo pairs side by side, and links the
artifacts into the Sphinx ``_static`` directory.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "pyvale-output"
IMAGES_ROOT = REPO_ROOT / "images"
STATIC_ROOT = REPO_ROOT / "docs" / "source" / "_static"
STEREO_GAP_PX = 16


@dataclass(frozen=True, slots=True)
class ImageArtifact:
    """One documentation image selected from an example output directory."""

    example: str
    sources: tuple[str, ...]

    @property
    def filename(self) -> str:
        """Return the canonical documentation image filename."""
        return f"{self.example}.png"


# Sources are explicit by design: this records the representative variant,
# first frame, and stereo camera ordering used in the documentation.
IMAGE_ARTIFACTS = (
    ImageArtifact(
        "render2d_ex1a_pixint_grid_one_elem_types",
        ("quad9.npy",),
    ),
    ImageArtifact(
        "render2d_ex1b_pixint_grid_mesh",
        ("warped_images.npy",),
    ),
    ImageArtifact(
        "render2d_ex1c_pixint_speck_newton",
        ("warped_images.npy",),
    ),
    ImageArtifact(
        "render2d_ex2a_imagedef2d_planar_deformation",
        ("warped_images.npy",),
    ),
    ImageArtifact(
        "render3d_ex1a_riley_quickstart",
        ("cam0_frame0_field0.bmp",),
    ),
    ImageArtifact(
        "render3d_ex1b_riley_sphere200",
        ("cam0_frame0_field0.bmp",),
    ),
    ImageArtifact(
        "render3d_ex1c_riley_psf",
        ("global_subpx_full/cam0_frame0_field0.bmp",),
    ),
    ImageArtifact(
        "render3d_ex1d_riley_rabbits",
        ("cam0_frame0_field0.bmp",),
    ),
    ImageArtifact(
        "render3d_ex1e_riley_dicuq",
        ("cam0_frame0_field0.bmp", "cam1_frame0_field0.bmp"),
    ),
    ImageArtifact(
        "render3d_ex1f_riley_dic_from_exodus",
        ("cam0_frame0_field0.bmp", "cam1_frame0_field0.bmp"),
    ),
    ImageArtifact(
        "render3d_ex1g_riley_stereocal",
        ("cam0_frame0_field0.bmp", "cam1_frame0_field0.bmp"),
    ),
    ImageArtifact(
        "render3d_ex2a_blender_scene",
        ("images/blenderimage_0.tiff",),
    ),
    ImageArtifact(
        "render3d_ex2b_blender_deformation",
        ("images/blenderimage_0.tiff",),
    ),
    ImageArtifact(
        "render3d_ex2c_blender_stereo",
        (
            "images/blenderimage_0_0.tiff",
            "images/blenderimage_0_1.tiff",
        ),
    ),
    ImageArtifact(
        "render3d_ex2e_blender_stereo_deformation",
        (
            "images/blenderimage_0_0.tiff",
            "images/blenderimage_0_1.tiff",
        ),
    ),
    ImageArtifact(
        "render3d_ex2f_blender_calibration_target",
        (
            "calimages/blendercal_0_0.tiff",
            "calimages/blendercal_1_1.tiff",
        ),
    ),
)

CALIBRATION_EXAMPLE = "render3d_ex2d_blender_calibration"
CALIBRATION_SOURCE = Path("calibration") / "calibration.yaml"
CALIBRATION_FILENAME = f"{CALIBRATION_EXAMPLE}.yaml"


def load_first_frame(path: Path) -> np.ndarray:
    """Load one persisted image or the first canonical frame from NumPy."""
    if path.suffix.lower() == ".npy":
        image = np.load(path)
        if image.ndim == 5:
            image = image[0, 0, :, :, 0]
        elif image.ndim != 2:
            raise ValueError(
                f"Unsupported NumPy image shape {image.shape} in {path}."
            )
        return image

    image = np.asarray(iio.imread(path))
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape {image.shape} in {path}.")
    return image


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert one finite greyscale or colour image to documentation PNG data."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255

    finite = np.isfinite(image)
    if not np.any(finite):
        raise ValueError("Cannot scale an image containing no finite values.")

    values = image.astype(np.float64)
    value_min = float(np.min(values[finite]))
    value_max = float(np.max(values[finite]))
    values[~finite] = value_min

    if value_min >= 0.0 and value_max <= 1.0:
        scaled = values * 255.0
    elif value_max > value_min:
        scaled = 255.0 * (values - value_min) / (value_max - value_min)
    else:
        scaled = np.zeros_like(values)
    return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)


def combine_stereo(images: tuple[np.ndarray, ...]) -> np.ndarray:
    """Return one image, or combine a camera pair with a white separator."""
    if len(images) == 1:
        return images[0]
    if len(images) != 2:
        raise ValueError("Documentation artifacts support one or two cameras.")

    camera_0, camera_1 = images
    if camera_0.ndim != camera_1.ndim:
        raise ValueError("Stereo images must have matching channel layouts.")
    if camera_0.shape[0] != camera_1.shape[0]:
        raise ValueError("Stereo images must have matching heights.")

    separator_shape = list(camera_0.shape)
    separator_shape[1] = STEREO_GAP_PX
    separator = np.full(separator_shape, 255, dtype=np.uint8)
    return np.concatenate((camera_0, separator, camera_1), axis=1)


def replace_static_link(artifact: Path) -> Path:
    """Create the Sphinx static symlink for one generated artifact."""
    link = STATIC_ROOT / artifact.name
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink: {link}")

    target = Path(os.path.relpath(artifact, start=STATIC_ROOT))
    link.symlink_to(target)
    return link


def validate_sources() -> None:
    """Fail before writing anything when an expected example output is absent."""
    missing: list[Path] = []
    for artifact in IMAGE_ARTIFACTS:
        example_dir = OUTPUT_ROOT / artifact.example
        for source in artifact.sources:
            path = example_dir / source
            if not path.is_file():
                missing.append(path)

    calibration = OUTPUT_ROOT / CALIBRATION_EXAMPLE / CALIBRATION_SOURCE
    if not calibration.is_file():
        missing.append(calibration)

    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Run all Render2D and Render3D examples before generating "
            f"documentation artifacts. Missing:\n{paths}"
        )


def main() -> None:
    """Generate documentation images, calibration text, and static links."""
    validate_sources()
    IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
    STATIC_ROOT.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for artifact in IMAGE_ARTIFACTS:
        example_dir = OUTPUT_ROOT / artifact.example
        images = tuple(
            image_to_uint8(load_first_frame(example_dir / source))
            for source in artifact.sources
        )
        output_path = IMAGES_ROOT / artifact.filename
        iio.imwrite(output_path, combine_stereo(images))
        replace_static_link(output_path)
        generated.append(output_path)

    calibration_source = (
        OUTPUT_ROOT / CALIBRATION_EXAMPLE / CALIBRATION_SOURCE
    )
    calibration_output = IMAGES_ROOT / CALIBRATION_FILENAME
    shutil.copy2(calibration_source, calibration_output)
    replace_static_link(calibration_output)
    generated.append(calibration_output)

    print(f"Generated {len(generated)} documentation artifacts:")
    for path in generated:
        print(f"  {path.relative_to(REPO_ROOT)}")
    print(f"Static links: {STATIC_ROOT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

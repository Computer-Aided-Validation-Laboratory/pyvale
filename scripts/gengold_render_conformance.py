#!/usr/bin/env python3
"""Generate the committed 32-pixel render conformance gold data."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

import pyvale.render as render
from pyvale.verif.renderconformance import (
    conformance_cases,
    preview_range,
    render_backend_case,
)


BACKENDS = (
    "blender",
    "imagedef2d",
    "pixint_grid",
    "pixint_speck",
    "riley",
)
GOLD_DIRECTORIES = {
    backend: f"gold_{backend}"
    for backend in BACKENDS
}


def main() -> None:
    """Render and deliberately write selected conformance baselines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write committed NumPy gold arrays",
    )
    parser.add_argument(
        "--tiff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write scaled 8-bit TIFF previews (default: enabled)",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=BACKENDS,
        help="generate only this backend; may be supplied more than once",
    )
    arguments = parser.parse_args()
    if not arguments.write:
        parser.error("choose --write to refresh committed gold data")

    repository_root = Path(__file__).resolve().parents[1]
    render_root = repository_root / "tests" / "render"
    selected = tuple(arguments.backend or BACKENDS)

    for backend in selected:
        if backend == "blender" and not render.blender_available():
            print(
                "Skipping Blender because its optional backend is unavailable.",
            )
            continue

        gold_dir = render_root / GOLD_DIRECTORIES[backend]
        gold_dir.mkdir(parents=True, exist_ok=True)

        for case in conformance_cases():
            with TemporaryDirectory() as temporary_dir:
                images = render_backend_case(
                    backend,
                    case,
                    Path(temporary_dir),
                )
            array_path = gold_dir / f"{case.name}.npy"
            np.save(array_path, images)
            print(array_path)

            if arguments.tiff:
                _write_tiffs(
                    gold_dir,
                    case.name,
                    images,
                    preview_range(backend),
                )


def _write_tiffs(
    output_dir: Path,
    case_name: str,
    images: np.ndarray,
    intensity_range: tuple[float, float],
) -> None:
    """Write every frame, camera, and channel as a scaled 8-bit TIFF."""
    lower, upper = intensity_range
    scaled = np.clip((images - lower) / (upper - lower), 0.0, 1.0)
    scaled = np.rint(255.0 * scaled).astype(np.uint8)

    for frame in range(scaled.shape[0]):
        for camera in range(scaled.shape[1]):
            for channel in range(scaled.shape[4]):
                name = (
                    f"{case_name}_f{frame:03d}_c{camera:02d}"
                    f"_ch{channel:02d}.tiff"
                )
                Image.fromarray(
                    scaled[frame, camera, :, :, channel],
                ).save(output_dir / name)


if __name__ == "__main__":
    main()

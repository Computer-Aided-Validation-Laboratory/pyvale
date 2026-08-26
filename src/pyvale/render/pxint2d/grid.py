# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Analytic eggbox rendering through validated PixInt2D maps."""

import numpy as np
from scipy.ndimage import gaussian_filter

from ..camera import Camera2D
from ..capabilities import RenderCapabilities
from ..imagewarp2d import IImageWarp2D
from ..mesh import EElementType, Mesh2D
from ..result import ImageWarpResult
from ..scene import Scene2D
from ..verifyinput import mesh_convention_issues
from .mapping import map_points
from .model import (
    AnalyticRule,
    Eggbox,
    EPxIntMapping,
    PxInt2DOpts,
    quadrature_points,
)


class PixIntGrid2D(IImageWarp2D):
    """Render an analytic eggbox texture over a deforming 2D mesh.

    Parameters
    ----------
    texture : Eggbox or None, optional
        Analytic texture. ``None`` selects the default eggbox.
    options : PxInt2DOpts or None, optional
        Mapping, integration, and execution controls.
    """

    capabilities = RenderCapabilities(
        element_types=frozenset(EElementType),
        supports_lights=False,
        supports_camera_distortion=False,
        supports_psf=True,
    )

    def __init__(
        self,
        texture: Eggbox | None = None,
        options: PxInt2DOpts | None = None,
    ) -> None:
        """Create a Grid2D renderer."""
        self.texture = Eggbox() if texture is None else texture
        self.options = PxInt2DOpts() if options is None else options

    def verify_input(self, scene: Scene2D) -> None:
        """Validate a Grid2D request before quadrature allocation.

        Parameters
        ----------
        scene : Scene2D
            Complete planar rendering request.

        Raises
        ------
        ValueError
            If the scene is invalid or unsupported.
        """
        mesh = scene.mesh
        camera = scene.camera

        convention_issues = mesh_convention_issues(
            mesh.coords,
            mesh.connectivity,
            "mesh",
        )
        if convention_issues:
            raise ValueError(convention_issues[0].message)

        if (
            not np.isfinite(mesh.coords).all()
            or not np.isfinite(
                mesh.displacement,
            ).all()
        ):
            raise ValueError(
                "mesh coordinates and displacements must be finite.",
            )

        if np.any(camera.pixels_count <= 0) or camera.pixel_size <= 0.0:
            raise ValueError("camera geometry must be positive.")

        if (
            self.options.mapping is EPxIntMapping.NEWTON_ONE_ELEM
            and mesh.connectivity.shape[0] != 1
        ):
            raise ValueError("NEWTON_ONE_ELEM requires exactly one element.")

        if (
            isinstance(self.options.integration, AnalyticRule)
            and self.options.mapping is not EPxIntMapping.AFFINE
        ):
            raise ValueError(
                "analytic Grid2D integration requires AFFINE mapping.",
            )

    def _render(self, scene: Scene2D) -> ImageWarpResult:
        """Render every displacement frame in a validated Grid2D request.

        Parameters
        ----------
        scene : Scene2D
            Previously validated planar rendering request.

        Returns
        -------
        ImageWarpResult
            Rendered images and masks.
        """
        mesh = scene.mesh
        camera = scene.camera
        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []

        for frame in range(mesh.displacement.shape[0]):
            image, mask = self._render_frame(mesh, camera, frame)
            images.append(image)
            masks.append(mask)

        return ImageWarpResult(
            images=np.asarray(images)[:, None, :, :, None],
            masks=np.asarray(masks)[:, None, :, :, None],
            output_paths=(),
        )

    def _render_frame(
        self,
        mesh: Mesh2D,
        camera: Camera2D,
        frame: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render a numerical image and validity mask for one frame."""
        if isinstance(self.options.integration, AnalyticRule):
            return self._analytic_frame(mesh, camera, frame)

        x_origin, y_origin, pixel_x, pixel_y = _pixel_geometry(camera)
        quad_x, quad_y, weights = quadrature_points(self.options.integration)
        points_per_pixel = len(weights)

        query_x = (x_origin[:, None] + pixel_x * quad_x).ravel()
        query_y = (y_origin[:, None] + pixel_y * quad_y).ravel()
        reference_x, reference_y, valid = map_points(
            mesh,
            frame,
            query_x,
            query_y,
            self.options.mapping,
        )
        values = self.texture.evaluate(reference_x, reference_y)
        values[~valid] = camera.background

        pixels = values.reshape(-1, points_per_pixel) @ weights
        mask = valid.reshape(-1, points_per_pixel).all(axis=1)
        image = pixels.reshape(camera.pixels_count[1], camera.pixels_count[0])
        mask = mask.reshape(image.shape)
        if self.options.psf is not None:
            psf = self.options.psf
            image = gaussian_filter(
                image,
                psf.sigma_pixels,
                mode="constant",
                cval=camera.background,
                radius=round(psf.sigma_pixels * psf.support_sigmas),
            )

        return np.flipud(image), np.flipud(mask)

    def _analytic_frame(
        self,
        mesh: Mesh2D,
        camera: Camera2D,
        frame: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the exact affine eggbox pixel integral."""
        x_origin, y_origin, pixel_x, pixel_y = _pixel_geometry(camera)
        deformed = mesh.coords + mesh.displacement[frame]
        design = np.column_stack((deformed, np.ones(len(deformed))))
        coeff, _, _, _ = np.linalg.lstsq(design, mesh.coords, rcond=None)
        if np.max(np.abs(design @ coeff - mesh.coords)) > 1.0e-8:
            raise ValueError(
                "analytic Grid2D integration requires affine motion."
            )
        x_centre = x_origin + 0.5 * pixel_x
        y_centre = y_origin + 0.5 * pixel_y

        ax, ay = coeff[0, 0], coeff[0, 1]
        bx, by = coeff[1, 0], coeff[1, 1]
        off_x, off_y = coeff[2, 0], coeff[2, 1]
        wave_x = 2.0 * np.pi / self.texture.period[0]
        wave_y = 2.0 * np.pi / self.texture.period[1]

        def average(freq_x: float, freq_y: float, phase: float) -> np.ndarray:
            factor = np.sinc(freq_x * pixel_x / (2.0 * np.pi)) * np.sinc(
                freq_y * pixel_y / (2.0 * np.pi)
            )
            return factor * np.cos(
                freq_x * x_centre + freq_y * y_centre + phase
            )

        cos_x = average(
            wave_x * ax, wave_x * bx, wave_x * off_x + self.texture.phase[0]
        )
        cos_y = average(
            wave_y * ay, wave_y * by, wave_y * off_y + self.texture.phase[1]
        )
        plus = average(
            wave_x * ax + wave_y * ay,
            wave_x * bx + wave_y * by,
            wave_x * off_x
            + wave_y * off_y
            + self.texture.phase[0]
            + self.texture.phase[1],
        )
        minus = average(
            wave_x * ax - wave_y * ay,
            wave_x * bx - wave_y * by,
            wave_x * off_x
            - wave_y * off_y
            + self.texture.phase[0]
            - self.texture.phase[1],
        )
        image = (
            self.texture.mean
            - 0.5 * self.texture.contrast
            + 0.5 * self.texture.contrast * (cos_x + cos_y)
            + 0.25 * self.texture.contrast * (plus + minus)
        )
        image = image.reshape(camera.pixels_count[1], camera.pixels_count[0])
        return np.flipud(image), np.flipud(np.ones_like(image, dtype=bool))


def _pixel_geometry(
    camera: Camera2D,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return bottom-left pixel origins and square physical pixel size."""
    width, height = (int(value) for value in camera.pixels_count)
    pixel_x = camera.pixel_size
    pixel_y = camera.pixel_size
    x_coords = camera.roi_cent_world[0] - 0.5 * width * pixel_x
    y_coords = camera.roi_cent_world[1] - 0.5 * height * pixel_y
    identifiers = np.arange(width * height)
    return (
        x_coords + (identifiers % width) * pixel_x,
        y_coords + (identifiers // width) * pixel_y,
        pixel_x,
        pixel_y,
    )


__all__ = ["PixIntGrid2D"]

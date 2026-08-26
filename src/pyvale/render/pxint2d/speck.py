# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Deterministic analytic speckle rendering through PixInt2D maps."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from ..camera import Camera2D
from ..capabilities import RenderCapabilities
from ..imagewarp2d import IImageWarp2D
from ..mesh import EElementType, Mesh2D
from ..result import ImageWarpResult
from ..scene import Scene2D
from ..verifyinput import mesh_convention_issues
from .grid import _pixel_geometry
from .mapping import map_points
from .model import (
    AnalyticRule,
    EPxIntMapping,
    PxInt2DOpts,
    quadrature_points,
)


@dataclass(slots=True)
class AdditiveSpeckles:
    """Finite deterministic lattice of additive disk or Gaussian speckles.

    Parameters
    ----------
    kind : str
        Speckle shape: ``"disk"`` or ``"gaussian"``.
    pitch : float
        Lattice spacing in world units.
    diameter : float
        Speckle diameter in world units.
    centres : numpy.ndarray
        Speckle centres with shape ``(count, 2)``.
    intensity_mean : float, optional
        Mean normalised intensity.
    intensity_contrast : float, optional
        Peak-to-trough texture contrast.
    gaussian_edge_fraction : float, optional
        Relative intensity at the Gaussian edge (used to compute sigma).
    tail_sigmas : float, optional
        Finite support radius for Gaussian speckles in standard deviations.
    """

    kind: str
    pitch: float
    diameter: float
    centres: np.ndarray
    intensity_mean: float = 0.5
    intensity_contrast: float = 0.4
    gaussian_edge_fraction: float = 0.1
    tail_sigmas: float = 6.0

    def calculate_radius(self) -> float:
        """Return disk radius in world units."""
        return 0.5 * self.diameter

    def calculate_sigma(self) -> float:
        """Return Gaussian standard deviation in world units."""
        radius = self.calculate_radius()
        return radius / np.sqrt(-2.0 * np.log(self.gaussian_edge_fraction))

    @classmethod
    def jittered_lattice(
        cls,
        *,
        kind: str,
        speckle_diameter: float,
        black_area_fraction: float,
        jitter_pdf: str,
        jitter: float,
        seed: int,
        bounds: tuple[float, float, float, float],
        intensity_mean: float = 0.5,
        intensity_contrast: float = 0.4,
        gaussian_edge_fraction: float = 0.1,
        tail_sigmas: float = 6.0,
    ) -> "AdditiveSpeckles":
        """Create a reproducible jittered square lattice of speckles."""
        if kind not in {"disk", "gaussian"}:
            raise ValueError("kind must be 'disk' or 'gaussian'.")

        if jitter_pdf not in {"uniform", "gaussian"}:
            raise ValueError("jitter_pdf must be 'uniform' or 'gaussian'.")

        pitch = speckle_diameter * np.sqrt(np.pi / (4.0 * black_area_fraction))
        radius = 0.5 * speckle_diameter
        sigma = radius / np.sqrt(-2.0 * np.log(gaussian_edge_fraction))
        support = radius if kind == "disk" else tail_sigmas * sigma
        xmin, xmax, ymin, ymax = bounds
        margin = support + 4.0 * jitter * pitch
        ids_x = np.arange(
            np.floor((xmin - margin) / pitch),
            np.ceil((xmax + margin) / pitch) + 1,
        )
        ids_y = np.arange(
            np.floor((ymin - margin) / pitch),
            np.ceil((ymax + margin) / pitch) + 1,
        )
        grid_x, grid_y = np.meshgrid(ids_x * pitch, ids_y * pitch)
        centres = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        generator = np.random.default_rng(seed)

        if jitter_pdf == "uniform":
            offsets = generator.uniform(-jitter, jitter, centres.shape)
        else:
            offsets = generator.normal(0.0, jitter, centres.shape)

        centres += offsets * pitch
        return cls(
            kind,
            pitch,
            speckle_diameter,
            np.ascontiguousarray(centres),
            intensity_mean,
            intensity_contrast,
            gaussian_edge_fraction,
            tail_sigmas,
        )

    def coverage(self, x_coord: np.ndarray, y_coord: np.ndarray) -> np.ndarray:
        """Evaluate unclamped additive speckle coverage at point coordinates."""
        points = np.column_stack((np.ravel(x_coord), np.ravel(y_coord)))
        delta = points[:, None, :] - self.centres[None, :, :]
        radii_squared = np.sum(delta * delta, axis=2)

        if self.kind == "disk":
            radius = self.calculate_radius()
            coverage = np.sum(radii_squared <= radius * radius, axis=1)
        else:
            sigma = self.calculate_sigma()
            support = self.tail_sigmas * sigma
            contribution = np.exp(-0.5 * radii_squared / sigma**2)
            contribution[radii_squared > support * support] = 0.0
            coverage = contribution.sum(axis=1)

        return coverage.reshape(np.shape(x_coord))

    def to_intensity(self, coverage: np.ndarray) -> np.ndarray:
        """Map raw additive coverage to normalised greyscale intensity."""
        return np.clip(
            self.intensity_mean
            + self.intensity_contrast
            * (1.0 - 2.0 * np.clip(coverage, 0.0, 1.0)),
            0.0,
            1.0,
        )


class PixIntSpeck2D(IImageWarp2D):
    """Render an additive speckle pattern over a deforming 2D mesh.

    Parameters
    ----------
    pattern : AdditiveSpeckles
        Deterministic speckle pattern to render.
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
        pattern: AdditiveSpeckles,
        options: PxInt2DOpts | None = None,
    ) -> None:
        """Create a speckle renderer with one deterministic pattern."""
        self.pattern = pattern
        self.options = PxInt2DOpts() if options is None else options

    def verify_input(self, scene: Scene2D) -> None:
        """Validate a Speck2D request before expensive point evaluation.

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

        if self.pattern.centres.ndim != 2 or self.pattern.centres.shape[1] != 2:
            raise ValueError("speckle centres must have shape (count, 2).")

        if self.pattern.pitch <= 0.0 or self.pattern.diameter <= 0.0:
            raise ValueError("speckle pitch and diameter must be positive.")

        if (
            self.options.mapping is EPxIntMapping.NEWTON_ONE_ELEM
            and mesh.connectivity.shape[0] != 1
        ):
            raise ValueError("NEWTON_ONE_ELEM requires exactly one element.")

        if isinstance(self.options.integration, AnalyticRule):
            raise TypeError(
                "analytic Speck2D integration is not yet available."
            )

        if np.any(camera.pixels_num <= 0) or camera.pixels_size <= 0.0:
            raise ValueError("camera geometry must be positive.")

    def _render(self, scene: Scene2D) -> ImageWarpResult:
        """Render every displacement frame in a validated Speck2D request.

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
        """Render one numerical speckle image and its validity mask."""
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
        coverage = self.pattern.coverage(reference_x, reference_y)
        coverage[~valid] = 0.0

        raw = coverage.reshape(-1, points_per_pixel) @ weights
        mask = valid.reshape(-1, points_per_pixel).all(axis=1)
        raw = raw.reshape(camera.pixels_num[1], camera.pixels_num[0])
        mask = mask.reshape(raw.shape)
        if self.options.psf is not None:
            psf = self.options.psf
            raw = gaussian_filter(
                raw,
                psf.sigma_pixels,
                mode="constant",
                cval=0.0,
                radius=round(psf.sigma_pixels * psf.support_sigmas),
            )

        return np.flipud(self.pattern.to_intensity(raw)), np.flipud(mask)


__all__ = ["AdditiveSpeckles", "PixIntSpeck2D"]

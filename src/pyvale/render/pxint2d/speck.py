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
from ..imagewarp2d import IImageWarp2D
from ..mesh2d import DisplacementSeries2D, Mesh2D
from ..result import ImageWarpResult
from .grid import _pixel_geometry
from .mapping import map_points
from .model import AnalyticRule, PxInt2DOpts, quadrature_points


@dataclass(frozen=True, slots=True)
class AdditiveSpeckles:
    """Finite deterministic lattice of additive disk or Gaussian speckles."""

    kind: str
    pitch: float
    diameter: float
    centres: np.ndarray
    intensity_mean: float = 0.5
    intensity_contrast: float = 0.4
    gaussian_edge_fraction: float = 0.1
    tail_sigmas: float = 6.0

    @property
    def radius(self) -> float:
        """Return disk radius in world units."""
        return 0.5 * self.diameter

    @property
    def sigma(self) -> float:
        """Return Gaussian standard deviation in world units."""
        return self.radius / np.sqrt(-2.0 * np.log(self.gaussian_edge_fraction))

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
        ids_x = np.arange(np.floor((xmin-margin)/pitch),
                          np.ceil((xmax+margin)/pitch) + 1)
        ids_y = np.arange(np.floor((ymin-margin)/pitch),
                          np.ceil((ymax+margin)/pitch) + 1)
        grid_x, grid_y = np.meshgrid(ids_x * pitch, ids_y * pitch)
        centres = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        generator = np.random.default_rng(seed)

        if jitter_pdf == "uniform":
            offsets = generator.uniform(-jitter, jitter, centres.shape)
        else:
            offsets = generator.normal(0.0, jitter, centres.shape)

        centres += offsets * pitch
        return cls(kind, pitch, speckle_diameter, np.ascontiguousarray(centres),
                   intensity_mean, intensity_contrast, gaussian_edge_fraction,
                   tail_sigmas)

    def coverage(self, x_coord: np.ndarray, y_coord: np.ndarray) -> np.ndarray:
        """Evaluate unclamped additive speckle coverage at point coordinates."""
        points = np.column_stack((np.ravel(x_coord), np.ravel(y_coord)))
        delta = points[:, None, :] - self.centres[None, :, :]
        radii_squared = np.sum(delta * delta, axis=2)

        if self.kind == "disk":
            coverage = np.sum(radii_squared <= self.radius * self.radius, axis=1)
        else:
            support = self.tail_sigmas * self.sigma
            contribution = np.exp(-0.5 * radii_squared / self.sigma ** 2)
            contribution[radii_squared > support * support] = 0.0
            coverage = contribution.sum(axis=1)

        return coverage.reshape(np.shape(x_coord))

    def to_intensity(self, coverage: np.ndarray) -> np.ndarray:
        """Map raw additive coverage to normalised greyscale intensity."""
        return np.clip(
            self.intensity_mean
            + self.intensity_contrast * (1.0 - 2.0 * np.clip(coverage, 0.0, 1.0)),
            0.0, 1.0,
        )


@dataclass(frozen=True, slots=True)
class _SpeckPlan:
    """Validated Speck2D render request."""

    mesh: Mesh2D
    camera: Camera2D
    displacements: DisplacementSeries2D


class PixIntSpeck2D(IImageWarp2D):
    """Render an additive speckle pattern over a deforming 2D mesh."""

    def __init__(
        self,
        pattern: AdditiveSpeckles,
        options: PxInt2DOpts | None = None,
    ) -> None:
        """Create a speckle renderer with one deterministic pattern."""
        self.pattern = pattern
        self.options = PxInt2DOpts() if options is None else options

    def verify_input(
        self,
        mesh: Mesh2D,
        camera: Camera2D,
        displacements: DisplacementSeries2D,
    ) -> _SpeckPlan:
        """Validate a Speck2D request before expensive point evaluation."""
        if displacements.values.shape[1] != mesh.coords.shape[0]:
            raise ValueError("displacement nodes must match mesh nodes.")

        if not np.isfinite(mesh.coords).all() or not np.isfinite(
                displacements.values).all():
            raise ValueError("mesh coordinates and displacements must be finite.")

        if self.pattern.centres.ndim != 2 or self.pattern.centres.shape[1] != 2:
            raise ValueError("speckle centres must have shape (count, 2).")

        if self.pattern.pitch <= 0.0 or self.pattern.diameter <= 0.0:
            raise ValueError("speckle pitch and diameter must be positive.")

        if (self.options.mapping.value == "newton_one_elem"
                and mesh.connectivity.shape[0] != 1):
            raise ValueError("NEWTON_ONE_ELEM requires exactly one element.")

        if isinstance(self.options.integration, AnalyticRule):
            raise ValueError("analytic Speck2D integration is not yet available.")
        return _SpeckPlan(mesh, camera, displacements)

    def _render(self, render_plan: object) -> ImageWarpResult:
        """Render every displacement frame in a validated Speck2D request."""
        if not isinstance(render_plan, _SpeckPlan):
            raise TypeError("PixIntSpeck2D received an invalid render plan.")
        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []

        for frame in range(render_plan.displacements.values.shape[0]):
            image, mask = self._render_frame(render_plan, frame)
            images.append(image)
            masks.append(mask)
        return ImageWarpResult(
            images=np.asarray(images)[:, None, :, :, None],
            masks=np.asarray(masks)[:, None, :, :, None],
        )

    def _render_frame(
        self,
        plan: _SpeckPlan,
        frame: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render one numerical speckle image and its validity mask."""
        x_origin, y_origin, pixel_x, pixel_y = _pixel_geometry(plan.camera)
        quad_x, quad_y, weights = quadrature_points(self.options.integration)
        points_per_pixel = len(weights)

        query_x = (x_origin[:, None] + pixel_x * quad_x).ravel()
        query_y = (y_origin[:, None] + pixel_y * quad_y).ravel()
        reference_x, reference_y, valid = map_points(
            plan.mesh, plan.displacements, frame, query_x, query_y,
            self.options.mapping,
        )
        coverage = self.pattern.coverage(reference_x, reference_y)
        coverage[~valid] = 0.0

        raw = coverage.reshape(-1, points_per_pixel) @ weights
        mask = valid.reshape(-1, points_per_pixel).all(axis=1)
        raw = raw.reshape(plan.camera.pixels_count[1], plan.camera.pixels_count[0])
        mask = mask.reshape(raw.shape)
        if self.options.psf is not None:
            psf = self.options.psf
            raw = gaussian_filter(
                raw, psf.sigma_pixels, mode="constant", cval=0.0,
                radius=round(psf.sigma_pixels * psf.support_sigmas),
            )

        return np.flipud(self.pattern.to_intensity(raw)), np.flipud(mask)


__all__ = ["AdditiveSpeckles", "PixIntSpeck2D"]

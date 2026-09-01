"""Correlated measurement-noise realisations for VFM guard preparation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import enum

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from pyvale.vfm.experimentdata import ExperimentData


class MeasurementNoiseMode(enum.StrEnum):
    """Supported sources for guard measurement-noise floors."""

    CALIBRATED = "calibrated"
    USER = "user"
    PARENT_ONLY = "parent-only"


@dataclass(slots=True, frozen=True)
class MeasurementNoiseFloorConfig:
    """Frozen measurement-noise ensemble used once per accepted parent.

    Strain standard deviations are stored in microstrain because that is the
    physically meaningful user-facing unit. Spatial filter widths are Gaussian
    sigmas in physical millimetres, ordered ``(y, x)`` for each of ``exx``,
    ``eyy`` and ``exy``. Parent-only mode deliberately requires no noise model.
    """

    mode: MeasurementNoiseMode = MeasurementNoiseMode.PARENT_ONLY
    seeds: tuple[int, ...] = ()
    strain_std_microstrain: tuple[float, float, float] | None = None
    force_std_n: float | None = None
    strain_filter_sigmas_mm_yx: tuple[tuple[float, float], ...] | None = None
    component_correlation: tuple[tuple[float, float, float], ...] | None = None
    quantile: float = 0.95
    model_source: str | None = None

    def __post_init__(self) -> None:
        mode = MeasurementNoiseMode(self.mode)
        object.__setattr__(self, "mode", mode)
        if not np.isclose(self.quantile, 0.95, rtol=0.0, atol=1.0e-15):
            raise ValueError("Guard measurement-noise floor quantile must be 0.95.")
        if mode is MeasurementNoiseMode.PARENT_ONLY:
            if self.seeds:
                raise ValueError("parent-only noise mode must not define seeds.")
            return
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Measurement-noise seeds must be non-empty and unique.")
        if self.strain_std_microstrain is None or len(self.strain_std_microstrain) != 3:
            raise ValueError("Three strain-noise standard deviations are required.")
        if any(not np.isfinite(value) or value < 0.0 for value in self.strain_std_microstrain):
            raise ValueError("Strain-noise standard deviations must be finite and non-negative.")
        if self.force_std_n is None or not np.isfinite(self.force_std_n) or self.force_std_n < 0.0:
            raise ValueError("A finite non-negative force-noise standard deviation is required.")
        filters = self.strain_filter_sigmas_mm_yx
        if filters is None or len(filters) != 3 or any(
            len(pair) != 2 or any(not np.isfinite(value) or value <= 0.0 for value in pair)
            for pair in filters
        ):
            raise ValueError(
                "A positive calibrated y/x spatial-filter sigma is required for every strain component; IID fallback is not allowed."
            )
        correlation = np.asarray(
            np.eye(3) if self.component_correlation is None else self.component_correlation,
            dtype=np.float64,
        )
        if correlation.shape != (3, 3) or np.any(~np.isfinite(correlation)):
            raise ValueError("component_correlation must be a finite 3x3 matrix.")
        if not np.allclose(correlation, correlation.T, rtol=0.0, atol=1.0e-12):
            raise ValueError("component_correlation must be symmetric.")
        if not np.allclose(np.diag(correlation), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("component_correlation must have a unit diagonal.")
        if np.min(np.linalg.eigvalsh(correlation)) < -1.0e-12:
            raise ValueError("component_correlation must be positive semidefinite.")

    def metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "number_of_realisations": len(self.seeds),
            "seeds": list(self.seeds),
            "quantile": self.quantile,
            "strain_std_microstrain": (
                None if self.strain_std_microstrain is None
                else list(self.strain_std_microstrain)
            ),
            "force_std_n": self.force_std_n,
            "strain_filter_sigmas_mm_yx": (
                None if self.strain_filter_sigmas_mm_yx is None
                else [list(pair) for pair in self.strain_filter_sigmas_mm_yx]
            ),
            "component_correlation": (
                None if self.component_correlation is None
                else [list(row) for row in self.component_correlation]
            ),
            "model_source": self.model_source,
            "interpretation": "measurement noise only",
        }


def measurement_noise_realisation(
    experiment: ExperimentData,
    config: MeasurementNoiseFloorConfig,
    seed: int,
    *,
    force_axis: str,
) -> ExperimentData:
    """Return one correlated strain/force measurement-noise realisation."""

    if config.mode is MeasurementNoiseMode.PARENT_ONLY:
        raise ValueError("parent-only mode does not generate noise realisations.")
    if seed not in config.seeds:
        raise ValueError(f"Seed {seed} is not part of the frozen noise ensemble.")
    if force_axis not in {"x", "y"}:
        raise ValueError("force_axis must be 'x' or 'y'.")
    assert config.strain_std_microstrain is not None
    assert config.strain_filter_sigmas_mm_yx is not None
    assert config.force_std_n is not None

    result = copy.deepcopy(experiment)
    x = np.asarray(experiment.specimen_geometry.x, dtype=np.float64)
    y = np.asarray(experiment.specimen_geometry.y, dtype=np.float64)
    dx = float(np.nanmedian(np.abs(np.diff(x, axis=1))))
    dy = float(np.nanmedian(np.abs(np.diff(y, axis=0))))
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        raise ValueError("Experiment grid spacing must be finite and positive.")
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(x, y)
    rng = np.random.default_rng(seed)
    strain = np.asarray(experiment.strain, dtype=np.float64).copy()
    target_sigmas = 1.0e-6 * np.asarray(config.strain_std_microstrain, dtype=np.float64)
    target_correlation = np.asarray(
        np.eye(3) if config.component_correlation is None else config.component_correlation,
        dtype=np.float64,
    )
    target_root = _symmetric_matrix_root(target_correlation)

    for frame in range(strain.shape[0]):
        filtered = np.empty((3, int(np.count_nonzero(mask))), dtype=np.float64)
        for component, sigma_mm in enumerate(config.strain_filter_sigmas_mm_yx):
            sample = gaussian_filter(
                rng.standard_normal(mask.shape),
                sigma=(sigma_mm[0] / dy, sigma_mm[1] / dx),
                mode="reflect",
            )
            values = sample[mask]
            values -= float(np.mean(values))
            scale = float(np.std(values))
            if scale <= np.finfo(np.float64).eps:
                raise RuntimeError("Generated a zero-variance correlated strain-noise field.")
            filtered[component] = values / scale

        # Whiten the finite sampled fields before imposing the calibrated
        # zero-lag component correlation. This retains the calibrated spatial
        # filters while preventing their finite-sample cross-correlation from
        # silently replacing the declared component covariance.
        empirical = np.cov(filtered, bias=True)
        whitened = _symmetric_inverse_root(empirical) @ filtered
        correlated = target_root @ whitened
        correlated -= np.mean(correlated, axis=1, keepdims=True)
        correlated /= np.std(correlated, axis=1, keepdims=True)
        for component in range(3):
            strain[frame, component, mask] += target_sigmas[component] * correlated[component]

    force = np.asarray(experiment.boundary_conditions.force, dtype=np.float64).copy()
    force_component = 0 if force_axis == "x" else 1
    force[:, force_component] += rng.normal(0.0, config.force_std_n, force.shape[0])
    result.strain = strain
    result.boundary_conditions.force = force
    return result


def _symmetric_matrix_root(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    clipped = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(clipped)) @ eigenvectors.T


def _symmetric_inverse_root(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    floor = max(float(np.max(eigenvalues)) * 1.0e-12, np.finfo(np.float64).eps)
    return (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, floor)))) @ eigenvectors.T

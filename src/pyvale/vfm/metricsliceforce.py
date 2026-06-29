from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.metric import IMetric
from pyvale.vfm.spatialparamslicewise import SlicePartition
from pyvale.vfm.spatialparam import ISpatialParameterisation


@dataclass(slots=True)
class SliceForceReconstructionDiagnostics:
    reconstructed_force: npt.NDArray[np.float64]
    force_reconstruction_error: npt.NDArray[np.float64]
    normalised_force_reconstruction_error: npt.NDArray[np.float64]
    temporal_weights: npt.NDArray[np.float64]
    spatial_weights: npt.NDArray[np.float64]
    weighted_temporal_rms: npt.NDArray[np.float64]
    weighted_spatiotemporal_rms: float
    target_force: npt.NDArray[np.float64]


@dataclass(slots=True)
class SliceWiseForceReconstructionMetric(IMetric):
    slice_partition: SlicePartition

    def evaluate(
        self,
        stress: npt.NDArray[np.float64],
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, ISpatialParameterisation],
        experiment_data: ExperimentData,
    ) -> npt.NDArray[np.float64]:
        diagnostics = self.evaluate_diagnostics(stress, experiment_data)
        weighted_residual = (
            diagnostics.normalised_force_reconstruction_error
            * np.sqrt(diagnostics.temporal_weights)[:, np.newaxis]
            * np.sqrt(diagnostics.spatial_weights)[np.newaxis, :]
        )
        return weighted_residual.ravel()

    def evaluate_diagnostics(
        self,
        stress: npt.NDArray[np.float64],
        experiment_data: ExperimentData,
    ) -> SliceForceReconstructionDiagnostics:
        if stress.ndim != 4:
            raise ValueError(f"Expected stress with shape (timesteps, components, y, x), got {stress.shape}.")

        loading_axis = self.slice_partition.axis
        stress_component_index = 0 if loading_axis == "x" else 1
        target_force = _extract_force_component(experiment_data.boundary_conditions.force, loading_axis)
        if target_force.shape[0] != stress.shape[0]:
            raise ValueError(
                f"Force history length {target_force.shape[0]} does not match stress timesteps {stress.shape[0]}."
            )

        reconstructed_force = np.zeros((stress.shape[0], self.slice_partition.num_slices), dtype=np.float64)
        stress_component = stress[:, stress_component_index, :, :].reshape(stress.shape[0], -1)
        pixel_area_flat = experiment_data.specimen_geometry.pixel_area.ravel()

        for slice_index, point_indices in enumerate(self.slice_partition.slice_point_indices):
            if point_indices.size == 0:
                continue
            slice_span = self.slice_partition.spans[slice_index]
            if slice_span <= 0.0:
                raise ValueError(f"Slice {slice_index} has non-positive span {slice_span}.")
            slice_force = np.nan_to_num(
                stress_component[:, point_indices]
                * pixel_area_flat[point_indices][np.newaxis, :],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            reconstructed_force[:, slice_index] = (
                experiment_data.specimen_geometry.thickness
                * np.sum(slice_force, axis=1)
                / slice_span
            )

        force_reconstruction_error = reconstructed_force - target_force[:, np.newaxis]
        force_scale = float(np.max(np.abs(target_force)))
        if force_scale <= 0.0:
            force_scale = 1.0
        normalised_force_reconstruction_error = force_reconstruction_error / force_scale

        temporal_weights = _normalise_weights(np.abs(target_force))
        spatial_weights = _normalise_weights(self.slice_partition.widths)

        weighted_temporal_rms = np.sqrt(
            np.sum(temporal_weights[:, np.newaxis] * normalised_force_reconstruction_error**2, axis=0)
        )
        weighted_spatiotemporal_rms = float(
            np.sqrt(
                np.sum(
                    temporal_weights[:, np.newaxis]
                    * spatial_weights[np.newaxis, :]
                    * normalised_force_reconstruction_error**2
                )
            )
        )

        return SliceForceReconstructionDiagnostics(
            reconstructed_force=reconstructed_force,
            force_reconstruction_error=force_reconstruction_error,
            normalised_force_reconstruction_error=normalised_force_reconstruction_error,
            temporal_weights=temporal_weights,
            spatial_weights=spatial_weights,
            weighted_temporal_rms=weighted_temporal_rms,
            weighted_spatiotemporal_rms=weighted_spatiotemporal_rms,
            target_force=target_force,
        )


def _extract_force_component(
    force: npt.NDArray[np.float64],
    axis: str,
) -> npt.NDArray[np.float64]:
    resolved_force = np.asarray(force, dtype=np.float64)
    if resolved_force.ndim == 1:
        return resolved_force
    if resolved_force.ndim != 2:
        raise ValueError(f"Unsupported force array shape {resolved_force.shape}.")

    component_index = 0 if axis == "x" else 1
    if resolved_force.shape[1] <= component_index:
        raise ValueError(
            f"Force array shape {resolved_force.shape} does not contain the required {axis}-component."
        )
    return resolved_force[:, component_index]


def _normalise_weights(raw_weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    resolved = np.asarray(raw_weights, dtype=np.float64)
    resolved = np.where(np.isfinite(resolved) & (resolved > 0.0), resolved, 0.0)
    total = float(np.sum(resolved))
    if total <= 0.0:
        return np.full(resolved.shape, 1.0 / resolved.size, dtype=np.float64)
    return resolved / total

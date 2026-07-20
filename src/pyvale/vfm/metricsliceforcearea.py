from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.metric import IMetric
from pyvale.vfm.metricsliceforce import (
    SliceForceReconstructionDiagnostics,
    SliceLocalForceReconstructionProblem,
    _extract_force_component,
    _normalise_weights,
)
from pyvale.vfm.spatialparam import ISpatialParameterisation
from pyvale.vfm.spatialparamslicewisearea import SliceAreaPartition


@dataclass(slots=True)
class SliceWiseAreaForceReconstructionMetric(IMetric):
    """Area-weighted slice force reconstruction metric.

    The reconstructed slice force is the discrete analogue of

        F_r = h / L_r * integral_{S_r} sigma dA

    where the area integral is approximated by precomputed overlap areas
    between each slice region and the native DIC support cells.
    """

    slice_partition: SliceAreaPartition

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

    def build_local_problem(
        self,
        experiment_data: ExperimentData,
        slice_index: int,
    ) -> SliceLocalForceReconstructionProblem:
        if slice_index < 0 or slice_index >= self.slice_partition.num_slices:
            raise IndexError(f"Slice index {slice_index} is out of range.")

        # extract global applied force in longitudinal direction
        target_force = _extract_force_component(experiment_data.boundary_conditions.force, self.slice_partition.axis)
        # best to apply weighting in cost function instead??
        temporal_weights = _normalise_weights(np.abs(target_force))
        spatial_weights = _normalise_weights(self.slice_partition.widths)

        point_indices = self.slice_partition.slice_force_point_indices[slice_index]
        point_weights = self.slice_partition.slice_force_point_weights[slice_index]
        finite_strain_points = np.all(np.isfinite(experiment_data.strain), axis=(0, 1)).ravel()
        if point_indices.size > 0:
            valid_operator_points = finite_strain_points[point_indices]
            point_indices = point_indices[valid_operator_points]
            point_weights = point_weights[valid_operator_points]

        if point_indices.size == 0:
            global_point_indices = np.zeros(0, dtype=np.int64)
            local_point_indices = np.zeros(0, dtype=np.int64)
        else:
            global_point_indices, local_point_indices = np.unique(point_indices, return_inverse=True)
            global_point_indices = global_point_indices.astype(np.int64, copy=False)
            local_point_indices = local_point_indices.astype(np.int64, copy=False)

        return SliceLocalForceReconstructionProblem(
            slice_index=slice_index,
            axis=self.slice_partition.axis,
            stress_component_index=0 if self.slice_partition.axis == "x" else 1,
            global_point_indices=global_point_indices,
            local_point_indices=local_point_indices,
            point_weights=point_weights,
            target_force=target_force,
            temporal_weights=temporal_weights,
            spatial_weight=float(spatial_weights[slice_index]),
        )

    def evaluate_local(
        self,
        stress: npt.NDArray[np.float64],
        experiment_data: ExperimentData,
        local_problem: SliceLocalForceReconstructionProblem,
    ) -> npt.NDArray[np.float64]:
        if stress.ndim != 4:
            raise ValueError(f"Expected stress with shape (timesteps, components, y, x), got {stress.shape}.")
        if stress.shape[0] != local_problem.target_force.shape[0]:
            raise ValueError(
                "Local stress history does not match the force history length: "
                f"{stress.shape[0]} vs {local_problem.target_force.shape[0]}."
            )

        stress_component = stress[:, local_problem.stress_component_index, :, :].reshape(stress.shape[0], -1)
        reconstructed_force = np.zeros(local_problem.target_force.shape, dtype=np.float64)
        if local_problem.local_point_indices.size > 0 and local_problem.point_weights.size > 0:
            reconstructed_force = (
                float(experiment_data.specimen_geometry.thickness)
                * np.sum(
                    stress_component[:, local_problem.local_point_indices]
                    * local_problem.point_weights[np.newaxis, :],
                    axis=1,
                )
            )

        force_error = reconstructed_force - local_problem.target_force
        force_scale = float(np.max(np.abs(local_problem.target_force)))
        if force_scale <= 0.0:
            force_scale = 1.0

        weighted_residual = (
            (force_error / force_scale)
            * np.sqrt(local_problem.temporal_weights)
            * np.sqrt(local_problem.spatial_weight)
        )
        return weighted_residual

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
        thickness = float(experiment_data.specimen_geometry.thickness)

        for slice_index, (point_indices, point_weights) in enumerate(
            zip(
                self.slice_partition.slice_force_point_indices,
                self.slice_partition.slice_force_point_weights,
                strict=True,
            )
        ):
            if point_indices.size == 0 or point_weights.size == 0:
                continue
            reconstructed_force[:, slice_index] = (
                thickness * np.sum(stress_component[:, point_indices] * point_weights[np.newaxis, :], axis=1)
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

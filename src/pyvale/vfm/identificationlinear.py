"""Direct homogeneous plane-stress linear-elastic VFM identification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

from pyvale.vfm.experimentdata import EEdgeCondition, ExperimentData
from pyvale.vfm.vfmesh import (
    generate_virtual_fields_from_mesh,
    generate_virtual_fields_mesh,
)


@dataclass(slots=True, frozen=True)
class LinearElasticIdentificationResult:
    """Result and numerical diagnostics from the direct ``q11``/``q12`` solve."""

    q11: float
    q12: float
    elastic_modulus: float
    poissons_ratio: float
    timestep_indices: tuple[int, ...]
    virtual_mesh_size: tuple[int, int]
    spatial_stride: tuple[int, int]
    analysis_grid_shape: tuple[int, int]
    system_matrix: npt.NDArray[np.float64]
    right_hand_side: npt.NDArray[np.float64]
    scaled_system_matrix: npt.NDArray[np.float64]
    scaled_right_hand_side: npt.NDArray[np.float64]
    rank: int
    singular_values: npt.NDArray[np.float64]
    condition_number: float
    residual_norm: float
    scaled_residual_norm: float

    def to_summary(self) -> dict[str, object]:
        return {
            "q11_mpa": self.q11,
            "q12_mpa": self.q12,
            "elastic_modulus_mpa": self.elastic_modulus,
            "poissons_ratio": self.poissons_ratio,
            "timestep_indices": list(self.timestep_indices),
            "virtual_mesh_size": list(self.virtual_mesh_size),
            "spatial_stride": list(self.spatial_stride),
            "analysis_grid_shape": list(self.analysis_grid_shape),
            "system_shape": list(self.system_matrix.shape),
            "rank": self.rank,
            "singular_values": self.singular_values.tolist(),
            "condition_number": self.condition_number,
            "residual_norm": self.residual_norm,
            "scaled_residual_norm": self.scaled_residual_norm,
        }

    def save_summary(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_summary(), indent=2) + "\n", encoding="utf-8")
        return path


def identify_isotropic_linear_elasticity_plane_stress(
    experiment_data: ExperimentData,
    *,
    timestep_indices: Sequence[int] | None = None,
    virtual_mesh_size: tuple[int, int] = (4, 4),
    spatial_stride: tuple[int, int] = (1, 1),
) -> LinearElasticIdentificationResult:
    """Identify homogeneous isotropic plane-stress elasticity by one SVD solve.

    The unknown linear coefficients are ``q11`` and ``q12``::

        sigma_xx = q11 * eps_xx + q12 * eps_yy
        sigma_yy = q12 * eps_xx + q11 * eps_yy
        sigma_xy = (q11 - q12) * eps_xy

    ``eps_xy`` is tensorial shear strain. Engineering constants are recovered
    after the solve as ``nu=q12/q11`` and ``E=q11*(1-nu**2)``.
    """

    strain = np.asarray(experiment_data.strain, dtype=np.float64)
    if strain.ndim != 4 or strain.shape[1] != 3:
        raise ValueError("strain must have shape (timesteps, 3, y, x)")
    indices = _resolve_timestep_indices(strain.shape[0], timestep_indices)
    mesh_size = _resolve_mesh_size(virtual_mesh_size)
    resolved_stride = _resolve_spatial_stride(spatial_stride)
    selected_strain = strain[np.asarray(indices, dtype=np.int64)]

    geometry = experiment_data.specimen_geometry
    specimen_mask = geometry.region_of_interest.sample_specimen_mask(
        geometry.x,
        geometry.y,
    )
    specimen_mask &= np.all(np.isfinite(selected_strain), axis=(0, 1))
    if not np.any(specimen_mask):
        raise ValueError("Selected timesteps have no common finite specimen points")

    analysis_strain, analysis_x, analysis_y, analysis_area, analysis_mask = (
        _spatially_average_for_analysis(
            selected_strain,
            np.asarray(geometry.x, dtype=np.float64),
            np.asarray(geometry.y, dtype=np.float64),
            np.asarray(geometry.pixel_area, dtype=np.float64),
            specimen_mask,
            resolved_stride,
        )
    )

    virtual_mesh = generate_virtual_fields_mesh(
        analysis_x,
        analysis_y,
        analysis_mask,
        experiment_data.boundary_conditions.edge_conditions,
        np.asarray(mesh_size, dtype=np.uint32),
    )
    stress_bases = plane_stress_stiffness_bases(analysis_strain)
    pixel_volume = analysis_area * float(geometry.thickness)
    forces = np.asarray(experiment_data.boundary_conditions.force, dtype=np.float64)[
        np.asarray(indices, dtype=np.int64)
    ]
    traction_edge_index = _traction_edge_index(
        experiment_data.boundary_conditions.edge_conditions
    )

    rows: list[npt.NDArray[np.float64]] = []
    rhs: list[float] = []
    for target_basis in stress_bases:
        virtual_fields = generate_virtual_fields_from_mesh(
            target_basis,
            virtual_mesh,
        )
        for local_timestep in range(len(indices)):
            virtual_strain = virtual_fields.virtual_strain[local_timestep]
            row = np.asarray([
                np.nansum(
                    basis[local_timestep]
                    * virtual_strain
                    * pixel_volume[np.newaxis, :, :]
                )
                for basis in stress_bases
            ], dtype=np.float64)
            edge_displacement = virtual_fields.virtual_displacement_edge[
                local_timestep, :, traction_edge_index
            ]
            external_work = float(forces[local_timestep] @ edge_displacement)
            rows.append(row)
            rhs.append(external_work)

    system_matrix = np.vstack(rows)
    right_hand_side = np.asarray(rhs, dtype=np.float64)
    scales = np.maximum(
        np.linalg.norm(system_matrix, axis=1),
        np.abs(right_hand_side),
    )
    nonzero = scales > np.finfo(np.float64).eps
    if np.count_nonzero(nonzero) < 2:
        raise ValueError("Direct linear system has fewer than two nonzero equations")
    scaled_matrix = system_matrix[nonzero] / scales[nonzero, np.newaxis]
    scaled_rhs = right_hand_side[nonzero] / scales[nonzero]
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        scaled_matrix,
        scaled_rhs,
        rcond=None,
    )
    if rank < 2:
        raise ValueError(
            "Direct linear system is rank deficient: "
            f"rank {rank} for two unknown coefficients"
        )

    q11, q12 = (float(value) for value in coefficients)
    if abs(q11) <= np.finfo(np.float64).eps:
        raise ValueError("Identified q11 is zero; engineering constants are undefined")
    poissons_ratio = q12 / q11
    elastic_modulus = q11 * (1.0 - poissons_ratio**2)
    residual = system_matrix @ coefficients - right_hand_side
    scaled_residual = scaled_matrix @ coefficients - scaled_rhs
    condition_number = float(
        np.inf
        if singular_values[-1] <= np.finfo(np.float64).eps
        else singular_values[0] / singular_values[-1]
    )
    return LinearElasticIdentificationResult(
        q11=q11,
        q12=q12,
        elastic_modulus=float(elastic_modulus),
        poissons_ratio=float(poissons_ratio),
        timestep_indices=indices,
        virtual_mesh_size=mesh_size,
        spatial_stride=resolved_stride,
        analysis_grid_shape=analysis_mask.shape,
        system_matrix=system_matrix,
        right_hand_side=right_hand_side,
        scaled_system_matrix=scaled_matrix,
        scaled_right_hand_side=scaled_rhs,
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=np.float64),
        condition_number=condition_number,
        residual_norm=float(np.linalg.norm(residual)),
        scaled_residual_norm=float(np.linalg.norm(scaled_residual)),
    )


def plane_stress_stiffness_bases(
    strain: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return stress influence maps for homogeneous ``q11`` and ``q12``."""

    values = np.asarray(strain, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("strain must have shape (timesteps, 3, y, x)")
    q11 = np.empty_like(values)
    q12 = np.empty_like(values)
    q11[:, 0] = values[:, 0]
    q11[:, 1] = values[:, 1]
    q11[:, 2] = values[:, 2]
    q12[:, 0] = values[:, 1]
    q12[:, 1] = values[:, 0]
    q12[:, 2] = -values[:, 2]
    return q11, q12


def _resolve_timestep_indices(
    timestep_count: int,
    timestep_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if timestep_indices is None:
        resolved = tuple(range(timestep_count))
    else:
        resolved = tuple(dict.fromkeys(int(index) for index in timestep_indices))
    if not resolved:
        raise ValueError("At least one timestep must be selected")
    if min(resolved) < 0 or max(resolved) >= timestep_count:
        raise IndexError("timestep_indices extend outside the strain history")
    return resolved


def _resolve_mesh_size(mesh_size: tuple[int, int]) -> tuple[int, int]:
    if len(mesh_size) != 2:
        raise ValueError("virtual_mesh_size must contain (rows, columns)")
    resolved = tuple(int(value) for value in mesh_size)
    if min(resolved) < 1:
        raise ValueError("virtual_mesh_size entries must be positive")
    return resolved


def _resolve_spatial_stride(stride: tuple[int, int]) -> tuple[int, int]:
    if len(stride) != 2:
        raise ValueError("spatial_stride must contain (rows, columns)")
    resolved = tuple(int(value) for value in stride)
    if min(resolved) < 1:
        raise ValueError("spatial_stride entries must be positive")
    return resolved


def _spatially_average_for_analysis(
    strain: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    pixel_area: npt.NDArray[np.float64],
    specimen_mask: npt.NDArray[np.bool_],
    stride: tuple[int, int],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]:
    """Area-average blocks while preserving their integration weights.

    The compact grid is used only to construct and evaluate the virtual
    fields.  Each retained point carries the summed area of its source block,
    so spatial reduction does not change the represented specimen volume.
    """

    if stride == (1, 1):
        return strain, x, y, pixel_area, specimen_mask

    valid_rows, valid_columns = np.nonzero(specimen_mask)
    row_min, row_max = int(valid_rows.min()), int(valid_rows.max())
    column_min, column_max = int(valid_columns.min()), int(valid_columns.max())
    source_row_count = row_max - row_min + 1
    source_column_count = column_max - column_min + 1
    block_rows = source_row_count // stride[0]
    block_columns = source_column_count // stride[1]
    if block_rows < 2 or block_columns < 2:
        raise ValueError("spatial_stride leaves fewer than two blocks per axis")
    used_row_count = block_rows * stride[0]
    used_column_count = block_columns * stride[1]
    source_slice = (
        slice(row_min, row_min + used_row_count),
        slice(column_min, column_min + used_column_count),
    )
    source_mask = specimen_mask[source_slice]
    source_area = np.where(source_mask, pixel_area[source_slice], 0.0)
    group_shape = (block_rows, block_columns)
    aggregated_area = source_area.reshape(
        block_rows,
        stride[0],
        block_columns,
        stride[1],
    ).sum(axis=(1, 3))
    aggregated_mask = aggregated_area > np.finfo(np.float64).eps
    aggregated_strain = np.full(
        (strain.shape[0], strain.shape[1], *group_shape),
        np.nan,
        dtype=np.float64,
    )
    for timestep in range(strain.shape[0]):
        for component in range(strain.shape[1]):
            source_values = strain[timestep, component][source_slice]
            weighted_values = np.where(
                source_mask,
                source_values * source_area,
                0.0,
            )
            totals = weighted_values.reshape(
                block_rows,
                stride[0],
                block_columns,
                stride[1],
            ).sum(axis=(1, 3))
            aggregated_strain[timestep, component, aggregated_mask] = (
                totals[aggregated_mask] / aggregated_area[aggregated_mask]
            )

    aggregated_mask &= np.all(
        np.isfinite(aggregated_strain),
        axis=(0, 1),
    )
    aggregated_strain[:, :, ~aggregated_mask] = np.nan
    source_x_axis = np.nanmedian(x[source_slice], axis=0)
    source_y_axis = np.nanmedian(y[source_slice], axis=1)
    sampled_x_axis = source_x_axis.reshape(
        block_columns,
        stride[1],
    ).mean(axis=1)
    sampled_y_axis = source_y_axis.reshape(
        block_rows,
        stride[0],
    ).mean(axis=1)
    sampled_x, sampled_y = np.meshgrid(sampled_x_axis, sampled_y_axis)
    return (
        aggregated_strain,
        sampled_x,
        sampled_y,
        aggregated_area,
        aggregated_mask,
    )


def _traction_edge_index(edge_conditions) -> int:
    edge_names = ("min_y_edge", "min_x_edge", "max_y_edge", "max_x_edge")
    indices: list[int] = []
    for index, name in enumerate(edge_names):
        edge = getattr(edge_conditions, name)
        if edge.x is EEdgeCondition.Traction or edge.y is EEdgeCondition.Traction:
            indices.append(index)
    if len(indices) != 1:
        raise ValueError(
            "Direct linear identification requires exactly one traction edge, "
            f"found {len(indices)}"
        )
    return indices[0]

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import correlate2d

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import EEdgeCondition, ExperimentData
from pyvale.vfm.metric import IMetric, MetricResult
from pyvale.vfm.spatialparam import ISpatialParameterisation


EquilibriumGapVirtualFieldType = Literal[
    "single_pos_pos",
    "single_pos_neg",
    "two_averaged",
]


@dataclass(slots=True, frozen=True)
class EquilibriumGapResult:
    """Equilibrium-gap residuals and commonly used scaled diagnostics."""

    metric_result: MetricResult
    raw_gap: npt.NDArray[np.float64]
    normalised_gap: npt.NDArray[np.float64]
    weighted_temporal_rms: npt.NDArray[np.float64]
    weighted_spatiotemporal_rms: float


@dataclass(slots=True, frozen=True)
class _EquilibriumGapOperator:
    virtual_strain_fields: npt.NDArray[np.float64]
    volume: npt.NDArray[np.float64]
    window_point_counts: npt.NDArray[np.float64]
    valid_centre_mask: npt.NDArray[np.bool_]
    longitudinal_force: npt.NDArray[np.float64]
    force_weights: npt.NDArray[np.float64]


@dataclass(slots=True)
class EquilibriumGapMetric(IMetric):
    """Equilibrium gap indicator (EGI) metric.

    The metric rasterises a 9-node, 4-element virtual window over the stress
    field. Each window returns a local internal virtual-work residual. The
    raw residual is returned as the metric residual; normalised and weighted
    fields are included in ``additional_fields`` for objective scaling or
    plotting.
    """

    window_size: npt.NDArray[np.uint32]
    sliding_pitch: npt.NDArray[np.uint32]
    virtual_field_type: EquilibriumGapVirtualFieldType
    normalise_virtual_strain: bool
    pixel_area_scale: float
    exclude_non_free_edge_margin: bool
    _operator: _EquilibriumGapOperator | None

    def __init__(
        self,
        window_size: npt.NDArray[np.uint32] | tuple[int, int] = (29, 29),
        sliding_pitch: npt.NDArray[np.uint32] | tuple[int, int] = (1, 1),
        *,
        virtual_field_type: EquilibriumGapVirtualFieldType = "single_pos_pos",
        normalise_virtual_strain: bool = True,
        pixel_area_scale: float = 1.0,
        exclude_non_free_edge_margin: bool = True,
    ) -> None:
        self.window_size = np.asarray(window_size, dtype=np.uint32)
        self.sliding_pitch = np.asarray(sliding_pitch, dtype=np.uint32)
        self.virtual_field_type = virtual_field_type
        self.normalise_virtual_strain = normalise_virtual_strain
        self.pixel_area_scale = pixel_area_scale
        self.exclude_non_free_edge_margin = exclude_non_free_edge_margin
        self._operator = None
        _validate_window_definition(self.window_size, self.sliding_pitch)

    def initialise(
        self,
        experiment_data: ExperimentData,
    ) -> None:
        self._operator = _build_equilibrium_gap_operator(
            experiment_data,
            window_size=self.window_size,
            sliding_pitch=self.sliding_pitch,
            virtual_field_type=self.virtual_field_type,
            normalise_virtual_strain=self.normalise_virtual_strain,
            pixel_area_scale=self.pixel_area_scale,
            exclude_non_free_edge_margin=self.exclude_non_free_edge_margin,
        )

    def evaluate(
        self,
        stress: npt.NDArray[np.float64],
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        experiment_data: ExperimentData,
    ) -> MetricResult:
        return self.evaluate_equilibrium_gap(stress).metric_result

    def evaluate_equilibrium_gap(
        self,
        stress: npt.NDArray[np.float64],
    ) -> EquilibriumGapResult:
        """Evaluate raw EGI and derived normalised RMS diagnostics."""

        if self._operator is None:
            raise RuntimeError(
                "Equilibrium gap operator has not been prepared. "
                "Call initialise(...) before evaluate(...)."
            )
        if stress.ndim != 4 or stress.shape[1] != 3:
            raise ValueError(
                "Expected stress with shape (timesteps, 3, y, x), "
                f"got {stress.shape}."
            )
        if stress.shape[0] != self._operator.longitudinal_force.shape[0]:
            raise ValueError(
                "Stress history length does not match force history length: "
                f"{stress.shape[0]} vs {self._operator.longitudinal_force.shape[0]}."
            )
        if stress.shape[2:] != self._operator.valid_centre_mask.shape:
            raise ValueError(
                "Stress spatial shape does not match prepared operator shape: "
                f"{stress.shape[2:]} vs {self._operator.valid_centre_mask.shape}."
            )

        raw_gap = _evaluate_raw_gap(stress, self._operator)
        raw_gap[:, ~self._operator.valid_centre_mask] = np.nan
        normalised_gap = _normalise_raw_gap(raw_gap, self._operator)
        weighted_temporal_rms = _calculate_weighted_temporal_rms(
            normalised_gap,
            self._operator.force_weights,
        )
        weighted_spatiotemporal_rms = _calculate_nan_rms(
            normalised_gap
            * np.sqrt(self._operator.force_weights)[:, np.newaxis, np.newaxis]
        )

        finite_raw_gap = raw_gap[np.isfinite(raw_gap)]
        metric_result = MetricResult(
            residual=finite_raw_gap,
            additional_fields={
                "raw_gap": raw_gap,
                "normalised_gap": normalised_gap,
                "weighted_temporal_rms": weighted_temporal_rms,
                "weighted_spatiotemporal_rms": weighted_spatiotemporal_rms,
                "force_weights": self._operator.force_weights,
                "longitudinal_force": self._operator.longitudinal_force,
                "window_point_counts": self._operator.window_point_counts,
                "valid_centre_mask": self._operator.valid_centre_mask,
                "virtual_strain_fields": self._operator.virtual_strain_fields,
            },
        )
        return EquilibriumGapResult(
            metric_result=metric_result,
            raw_gap=raw_gap,
            normalised_gap=normalised_gap,
            weighted_temporal_rms=weighted_temporal_rms,
            weighted_spatiotemporal_rms=weighted_spatiotemporal_rms,
        )


def _validate_window_definition(
    window_size: npt.NDArray[np.uint32],
    sliding_pitch: npt.NDArray[np.uint32],
) -> None:
    if window_size.shape != (2,):
        raise ValueError("window_size must contain [rows, columns].")
    if sliding_pitch.shape != (2,):
        raise ValueError("sliding_pitch must contain [rows, columns].")
    if np.any(window_size < 3):
        raise ValueError("Equilibrium gap windows need at least 3 rows and columns.")
    if np.any(window_size % 2 == 0):
        raise ValueError("Equilibrium gap window rows and columns must be odd.")
    if np.any(sliding_pitch < 1):
        raise ValueError("sliding_pitch values must be at least 1.")


def _build_equilibrium_gap_operator(
    experiment_data: ExperimentData,
    *,
    window_size: npt.NDArray[np.uint32],
    sliding_pitch: npt.NDArray[np.uint32],
    virtual_field_type: EquilibriumGapVirtualFieldType,
    normalise_virtual_strain: bool,
    pixel_area_scale: float,
    exclude_non_free_edge_margin: bool,
) -> _EquilibriumGapOperator:
    specimen_geometry = experiment_data.specimen_geometry
    valid_point_mask = (
        specimen_geometry.specimen_mask
        & np.isfinite(specimen_geometry.x)
        & np.isfinite(specimen_geometry.y)
        & np.isfinite(specimen_geometry.pixel_area)
    )
    window_point_counts = _correlate_same(
        valid_point_mask.astype(np.float64),
        np.ones(tuple(window_size), dtype=np.float64),
    )
    valid_centre_mask = valid_point_mask & (window_point_counts > 0.0)
    valid_centre_mask &= _build_pitch_mask(valid_centre_mask.shape, sliding_pitch)
    if exclude_non_free_edge_margin:
        valid_centre_mask &= _build_non_free_edge_mask(
            experiment_data,
            window_size,
        )

    volume = (
        np.asarray(specimen_geometry.pixel_area, dtype=np.float64)
        * float(pixel_area_scale)
        * float(specimen_geometry.thickness)
    )
    volume = np.where(valid_point_mask, volume, 0.0)
    longitudinal_force = _extract_longitudinal_force(experiment_data)
    force_weights = _calculate_force_weights(longitudinal_force)
    virtual_strain_fields = _build_virtual_strain_fields(
        specimen_geometry.x,
        specimen_geometry.y,
        window_size,
        virtual_field_type=virtual_field_type,
        normalise_virtual_strain=normalise_virtual_strain,
    )
    return _EquilibriumGapOperator(
        virtual_strain_fields=virtual_strain_fields,
        volume=volume,
        window_point_counts=window_point_counts,
        valid_centre_mask=valid_centre_mask,
        longitudinal_force=longitudinal_force,
        force_weights=force_weights,
    )


def _build_pitch_mask(
    shape: tuple[int, int],
    sliding_pitch: npt.NDArray[np.uint32],
) -> npt.NDArray[np.bool_]:
    pitch_mask = np.zeros(shape, dtype=bool)
    pitch_mask[:: int(sliding_pitch[0]), :: int(sliding_pitch[1])] = True
    return pitch_mask


def _build_non_free_edge_mask(
    experiment_data: ExperimentData,
    window_size: npt.NDArray[np.uint32],
) -> npt.NDArray[np.bool_]:
    edge_conditions = experiment_data.boundary_conditions.edge_conditions
    mask = np.ones(experiment_data.specimen_geometry.x.shape, dtype=bool)
    row_margin = int(window_size[0] // 2)
    col_margin = int(window_size[1] // 2)

    if _edge_can_do_external_work(edge_conditions.min_x_edge):
        mask[:, :col_margin] = False
    if _edge_can_do_external_work(edge_conditions.max_x_edge):
        mask[:, mask.shape[1] - col_margin :] = False
    if _edge_can_do_external_work(edge_conditions.min_y_edge):
        mask[:row_margin, :] = False
    if _edge_can_do_external_work(edge_conditions.max_y_edge):
        mask[mask.shape[0] - row_margin :, :] = False
    return mask


def _edge_can_do_external_work(edge) -> bool:
    return edge.x is not EEdgeCondition.Free or edge.y is not EEdgeCondition.Free


def _edge_has_traction(edge) -> bool:
    return edge.x is EEdgeCondition.Traction or edge.y is EEdgeCondition.Traction


def _extract_longitudinal_force(
    experiment_data: ExperimentData,
) -> npt.NDArray[np.float64]:
    force = np.asarray(experiment_data.boundary_conditions.force, dtype=np.float64)
    if force.ndim != 2 or force.shape[1] < 2:
        raise ValueError(
            "EquilibriumGapMetric expects force with shape (timesteps, 2)."
        )

    edge_conditions = experiment_data.boundary_conditions.edge_conditions
    if (
        _edge_has_traction(edge_conditions.min_x_edge)
        or _edge_has_traction(edge_conditions.max_x_edge)
    ):
        return force[:, 0]
    if (
        _edge_has_traction(edge_conditions.min_y_edge)
        or _edge_has_traction(edge_conditions.max_y_edge)
    ):
        return force[:, 1]
    raise ValueError("No traction edge found for equilibrium gap normalisation.")


def _calculate_force_weights(
    longitudinal_force: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    force_squared = np.asarray(longitudinal_force, dtype=np.float64) ** 2
    valid = np.isfinite(force_squared) & (force_squared > 0.0)
    weights = np.zeros(force_squared.shape, dtype=np.float64)
    if not np.any(valid):
        return np.ones(force_squared.shape, dtype=np.float64)
    weights[valid] = force_squared[valid] / float(np.mean(force_squared[valid]))
    return weights


def _build_virtual_strain_fields(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    window_size: npt.NDArray[np.uint32],
    *,
    virtual_field_type: EquilibriumGapVirtualFieldType,
    normalise_virtual_strain: bool,
) -> npt.NDArray[np.float64]:
    if virtual_field_type == "single_pos_pos":
        fields = [
            _build_virtual_strain_field(
                x,
                y,
                window_size,
                centre_dof_x=1.0,
                centre_dof_y=1.0,
            )
        ]
    elif virtual_field_type == "single_pos_neg":
        fields = [
            _build_virtual_strain_field(
                x,
                y,
                window_size,
                centre_dof_x=1.0,
                centre_dof_y=-1.0,
            )
        ]
    elif virtual_field_type == "two_averaged":
        fields = [
            _build_virtual_strain_field(
                x,
                y,
                window_size,
                centre_dof_x=1.0,
                centre_dof_y=1.0,
            ),
            _build_virtual_strain_field(
                x,
                y,
                window_size,
                centre_dof_x=1.0,
                centre_dof_y=-1.0,
            ),
        ]
    else:
        raise ValueError(f"Unsupported virtual_field_type '{virtual_field_type}'.")

    virtual_strain_fields = np.asarray(fields, dtype=np.float64)
    if normalise_virtual_strain:
        virtual_strain_fields = np.asarray(
            [
                _normalise_virtual_strain_field(field)
                for field in virtual_strain_fields
            ],
            dtype=np.float64,
        )
    return virtual_strain_fields


def _build_virtual_strain_field(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    window_size: npt.NDArray[np.uint32],
    *,
    centre_dof_x: float,
    centre_dof_y: float,
) -> npt.NDArray[np.float64]:
    rows = int(window_size[0])
    cols = int(window_size[1])
    row_mid = rows // 2
    col_mid = cols // 2
    window_x = np.asarray(x[:rows, :cols], dtype=np.float64)
    window_y = np.asarray(y[:rows, :cols], dtype=np.float64)

    node_rows = np.asarray([0, row_mid, rows - 1], dtype=np.int64)
    node_cols = np.asarray([0, col_mid, cols - 1], dtype=np.int64)
    node_coordinates = np.asarray(
        [
            [window_x[row, col], window_y[row, col]]
            for col in node_cols
            for row in node_rows
        ],
        dtype=np.float64,
    )
    element_nodes = np.asarray(
        (
            (2, 5, 4, 1),
            (5, 8, 7, 4),
            (4, 7, 6, 3),
            (1, 4, 3, 0),
        ),
        dtype=np.int64,
    )
    element_dofs = np.asarray(
        (
            (4, 5, 10, 11, 8, 9, 2, 3),
            (10, 11, 16, 17, 14, 15, 8, 9),
            (8, 9, 14, 15, 12, 13, 6, 7),
            (2, 3, 8, 9, 6, 7, 0, 1),
        ),
        dtype=np.int64,
    )

    num_points = rows * cols
    b_xx = np.zeros((num_points, 18), dtype=np.float64)
    b_yy = np.zeros((num_points, 18), dtype=np.float64)
    b_xy = np.zeros((num_points, 18), dtype=np.float64)
    element_count = np.zeros(num_points, dtype=np.float64)
    point_coordinates = np.column_stack((window_x.ravel(), window_y.ravel()))

    for nodes, dofs in zip(element_nodes, element_dofs, strict=True):
        coords = node_coordinates[nodes]
        in_element = _points_in_axis_aligned_element(point_coordinates, coords)
        for point_index in np.flatnonzero(in_element):
            xi, eta = _coordinate_transform(coords, point_coordinates[point_index])
            _, shape_derivative_local = _shape_functions(xi, eta)
            jacobian = shape_derivative_local.T @ coords
            shape_derivative_global = shape_derivative_local @ np.linalg.inv(jacobian)
            b_matrix = np.asarray(
                (
                    (
                        shape_derivative_global[0, 0],
                        0.0,
                        shape_derivative_global[1, 0],
                        0.0,
                        shape_derivative_global[2, 0],
                        0.0,
                        shape_derivative_global[3, 0],
                        0.0,
                    ),
                    (
                        0.0,
                        shape_derivative_global[0, 1],
                        0.0,
                        shape_derivative_global[1, 1],
                        0.0,
                        shape_derivative_global[2, 1],
                        0.0,
                        shape_derivative_global[3, 1],
                    ),
                    (
                        shape_derivative_global[0, 1],
                        shape_derivative_global[0, 0],
                        shape_derivative_global[1, 1],
                        shape_derivative_global[1, 0],
                        shape_derivative_global[2, 1],
                        shape_derivative_global[2, 0],
                        shape_derivative_global[3, 1],
                        shape_derivative_global[3, 0],
                    ),
                ),
                dtype=np.float64,
            )
            b_xx[point_index, dofs] += b_matrix[0, :]
            b_yy[point_index, dofs] += b_matrix[1, :]
            b_xy[point_index, dofs] += b_matrix[2, :]
            element_count[point_index] += 1.0

    if np.any(element_count == 0.0):
        raise ValueError("Some equilibrium-gap window points were not in any element.")
    b_xx /= element_count[:, np.newaxis]
    b_yy /= element_count[:, np.newaxis]
    b_xy /= element_count[:, np.newaxis]

    virtual_displacement = np.zeros(18, dtype=np.float64)
    virtual_displacement[8] = centre_dof_x
    virtual_displacement[9] = centre_dof_y

    virtual_strain = np.empty((3, rows, cols), dtype=np.float64)
    virtual_strain[0, :, :] = (b_xx @ virtual_displacement).reshape(rows, cols)
    virtual_strain[1, :, :] = (b_yy @ virtual_displacement).reshape(rows, cols)
    virtual_strain[2, :, :] = (b_xy @ virtual_displacement).reshape(rows, cols)
    return virtual_strain


def _points_in_axis_aligned_element(
    points: npt.NDArray[np.float64],
    node_coordinates: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    tolerance = 1.0e-10
    x_min = float(np.min(node_coordinates[:, 0])) - tolerance
    x_max = float(np.max(node_coordinates[:, 0])) + tolerance
    y_min = float(np.min(node_coordinates[:, 1])) - tolerance
    y_max = float(np.max(node_coordinates[:, 1])) + tolerance
    return (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )


def _coordinate_transform(
    node_coordinates: npt.NDArray[np.float64],
    point_coordinates: npt.NDArray[np.float64],
) -> tuple[float, float]:
    xi = (
        2.0
        * (point_coordinates[0] - node_coordinates[0, 0])
        / (node_coordinates[1, 0] - node_coordinates[0, 0])
        - 1.0
    )
    eta = (
        2.0
        * (point_coordinates[1] - node_coordinates[0, 1])
        / (node_coordinates[3, 1] - node_coordinates[0, 1])
        - 1.0
    )
    return float(xi), float(eta)


def _shape_functions(
    xi: float,
    eta: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    shape_function = np.asarray(
        (
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ),
        dtype=np.float64,
    )
    shape_derivative_local = np.asarray(
        (
            (-0.25 * (1.0 - eta), -0.25 * (1.0 - xi)),
            (0.25 * (1.0 - eta), -0.25 * (1.0 + xi)),
            (0.25 * (1.0 + eta), 0.25 * (1.0 + xi)),
            (-0.25 * (1.0 + eta), 0.25 * (1.0 - xi)),
        ),
        dtype=np.float64,
    )
    return shape_function, shape_derivative_local


def _normalise_virtual_strain_field(
    virtual_strain: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    min_value = float(np.nanmin(virtual_strain))
    max_value = float(np.nanmax(virtual_strain))
    if np.isclose(max_value, min_value):
        return virtual_strain.copy()
    normalised = 2.0 * (virtual_strain - min_value) / (max_value - min_value) - 1.0
    normalised[np.isclose(virtual_strain, 0.0) & np.isclose(normalised, 0.0)] = 0.0
    return normalised


def _evaluate_raw_gap(
    stress: npt.NDArray[np.float64],
    operator: _EquilibriumGapOperator,
) -> npt.NDArray[np.float64]:
    raw_gap_by_field = []
    for virtual_strain in operator.virtual_strain_fields:
        current_gap = np.zeros(
            (stress.shape[0], stress.shape[2], stress.shape[3]),
            dtype=np.float64,
        )
        for component_index in range(3):
            stress_volume = np.nan_to_num(
                stress[:, component_index, :, :] * operator.volume[np.newaxis, :, :],
                nan=0.0,
            )
            for timestep_index in range(stress.shape[0]):
                current_gap[timestep_index, :, :] += _correlate_same(
                    stress_volume[timestep_index, :, :],
                    virtual_strain[component_index, :, :],
                )
        raw_gap_by_field.append(current_gap)

    if len(raw_gap_by_field) == 1:
        return raw_gap_by_field[0]

    return 0.5 * (
        np.abs(raw_gap_by_field[0])
        + np.abs(raw_gap_by_field[1])
    )


def _normalise_raw_gap(
    raw_gap: npt.NDArray[np.float64],
    operator: _EquilibriumGapOperator,
) -> npt.NDArray[np.float64]:
    force = np.abs(operator.longitudinal_force)
    denominator = (
        force[:, np.newaxis, np.newaxis]
        * operator.window_point_counts[np.newaxis, :, :]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised_gap = raw_gap / denominator
    normalised_gap[:, operator.window_point_counts <= 0.0] = np.nan
    normalised_gap[force <= 0.0, :, :] = np.nan
    normalised_gap[:, ~operator.valid_centre_mask] = np.nan
    return normalised_gap


def _calculate_weighted_temporal_rms(
    normalised_gap: npt.NDArray[np.float64],
    force_weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    weighted_gap_squared = (
        normalised_gap**2
        * force_weights[:, np.newaxis, np.newaxis]
    )
    valid_counts = np.sum(np.isfinite(weighted_gap_squared), axis=0)
    weighted_sum = np.nansum(weighted_gap_squared, axis=0)
    temporal_rms = np.full(valid_counts.shape, np.nan, dtype=np.float64)
    valid = valid_counts > 0
    temporal_rms[valid] = np.sqrt(weighted_sum[valid] / valid_counts[valid])
    return temporal_rms


def _calculate_nan_rms(
    values: npt.NDArray[np.float64],
) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite_values**2)))


def _correlate_same(
    values: npt.NDArray[np.float64],
    kernel: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return correlate2d(
        values,
        kernel,
        mode="same",
        boundary="fill",
        fillvalue=0.0,
    )

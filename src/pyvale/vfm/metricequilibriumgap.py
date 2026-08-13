from __future__ import annotations

from dataclasses import dataclass
import enum
import warnings

import numpy as np
import numpy.typing as npt
from scipy.signal import correlate2d

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import EEdgeCondition, ExperimentData
from pyvale.vfm.metric import IMetric, MetricResult
from pyvale.vfm.spatialparam import ISpatialParameterisation

# TODO
# should all normalisation be done in objective function, not in metric? (e.g. normalised_gap, weighted_temporal_rms, weighted_spatiotemporal_rms)
# suitable default for window_size and sliding_pitch? (e.g. fraction of domain, 1x1)
# remove pixel area scale? when is it used?

class EquilibriumGapVirtualFieldType(enum.StrEnum):
    SINGLE_POS_POS = "single_pos_pos"
    SINGLE_POS_NEG = "single_pos_neg"
    TWO_AVERAGED = "two_averaged"


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

    The metric rasterises a set of virtual strain fields, defined by a 
    9-node, 4-element virtual window over the stress field, computing a
    scalar value of the internal virtual work (equilibrium gap) for each window. 
    The raw residual is returned as the metric residual; normalised and weighted
    fields are included in ``additional_fields`` for objective scaling or
    plotting.   
      
    Parameters
    ----------
    window_size : tuple[int, int] or npt.NDArray[np.uint32], optional
        The number of rows and columns in the virtual window, must be odd and at least 3. Default is (29, 29).
    sliding_pitch : tuple[int, int] or npt.NDArray[np.uint32], optional
        The number of rows and columns to slide the window for each evaluation, must be at least 1. Default is (1, 1).
    virtual_field_type : EquilibriumGapVirtualFieldType, optional
        The type of virtual strain field to use for the equilibrium gap evaluation. 
        single_pos_pos: single virtual field with positive x and y displacements at the centre node.
        single_pos_neg: single virtual field with positive x and negative y displacements at the centre node.
        two_averaged: average of the two virtual fields above.
    normalise_virtual_strain : bool, optional
        Whether to normalise the virtual strain fields. Default is True.
        Normalise virtual strain fields to the range [-1, 1] based on the minimum and maximum values of each field. 
        This ensures that the virtual strain fields have a consistent scale, which can improve the stability and 
        interpretability of the equilibrium gap metric.
    pixel_area_scale : float, optional
        Scale factor for the pixel area when computing the volume. Default is 1.0.
        This is only required if mismatch in units between the pixel area and the stress field, 
        e.g. if the pixel area is in mm^2 and the stress is in Pa, then a scale factor of 1e-6 is 
        required to convert the pixel area to m^2.
    _operator : _EquilibriumGapOperator | None
        Internal operator for evaluating the equilibrium gap, initialised in ``initialise()``.
        This operator contains the virtual strain fields, volume, window point counts, valid centre mask,
          longitudinal force, and force weights. Hence does not need to be recomputed for each evaluation, improving performance.
    """

    # Define attributes of EquilibriumGapMetric class with type annotations
    window_size: npt.NDArray[np.uint32]
    sliding_pitch: npt.NDArray[np.uint32]
    virtual_field_type: EquilibriumGapVirtualFieldType
    normalise_virtual_strain: bool
    pixel_area_scale: float
    _operator: _EquilibriumGapOperator | None
    plot_virtual_field_schematic: bool
    plot_virtual_window_raster: bool

    def __init__(
        self,
        window_size: npt.NDArray[np.uint32] | tuple[int, int] = (29, 29),
        sliding_pitch: npt.NDArray[np.uint32] | tuple[int, int] = (1, 1),
        *,
        virtual_field_type: EquilibriumGapVirtualFieldType = EquilibriumGapVirtualFieldType.TWO_AVERAGED,
        normalise_virtual_strain: bool = True,
        pixel_area_scale: float = 1.0,
        plot_virtual_field_schematic: bool = False,
        plot_virtual_window_raster: bool = False,
    ) -> None:
        self.window_size = np.asarray(window_size, dtype=np.uint32)
        self.sliding_pitch = np.asarray(sliding_pitch, dtype=np.uint32)
        self.virtual_field_type = virtual_field_type
        self.normalise_virtual_strain = normalise_virtual_strain
        self.pixel_area_scale = pixel_area_scale
        self.plot_virtual_field_schematic = plot_virtual_field_schematic
        self.plot_virtual_window_raster = plot_virtual_window_raster
        self._operator = None
        _validate_window_definition(self.window_size, self.sliding_pitch)

    def initialise(
        self,
        experiment_data: ExperimentData,
    ) -> None:
        """Precompute the equilibrium gap operator.
         
        The operator contains the virtual strain fields, integration volumes,
        valid window mask."""
        self._operator = _build_equilibrium_gap_operator(
            experiment_data,
            window_size=self.window_size,
            sliding_pitch=self.sliding_pitch,
            virtual_field_type=self.virtual_field_type,
            normalise_virtual_strain=self.normalise_virtual_strain,
            pixel_area_scale=self.pixel_area_scale,
            plot_virtual_field_schematic=self.plot_virtual_field_schematic,
            plot_virtual_window_raster=self.plot_virtual_window_raster,
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
    plot_virtual_field_schematic: bool,
    plot_virtual_window_raster: bool,
) -> _EquilibriumGapOperator:

    # Unpack specimen geometry for convenience
    specimen_geometry = experiment_data.specimen_geometry

    # Build mask of valid points (points within specimen and with finite x, y, area)
    specimen_mask = specimen_geometry.region_of_interest.sample_specimen_mask(specimen_geometry.x,specimen_geometry.y)
    valid_point_mask = (
        specimen_mask
        & np.isfinite(specimen_geometry.x)
        & np.isfinite(specimen_geometry.y)
        & np.isfinite(specimen_geometry.pixel_area)
    )
    
    # Raster a kernel of ones, with window size, over the valid point mask to count the number of valid points
    # in each window. The resulting 2D array has same shape as the valid point mask, with each element containing
    # the count of valid points in the window centred on that element. So windows in specimen centre will have counts 
    # equal to the window size, windows outside specimen will be zero, while windows at the edge of the specimen will
    # have counts less than the window size but greater than zero.
    window_point_counts = _correlate_same(
        valid_point_mask.astype(np.float64),
        np.ones(tuple(window_size), dtype=np.float64),
    )

    # Debug: plot map of window counts
    # import matplotlib.pyplot as plt
    # plt.imshow(window_point_counts, cmap='viridis')
    # plt.colorbar(); plt.title('Window Point Counts'); plt.show()

    # Combine the valid point mask and the window point counts to create a mask of valid window centres.
    valid_centre_mask = valid_point_mask & (window_point_counts > 0.0)

    # Compute a mask of valid window centres based on the sliding pitch. 
    # This ensures that only windows that are spaced by the sliding pitch are considered valid.
    pitch_mask = np.zeros(valid_centre_mask.shape, dtype=bool)
    pitch_mask[:: int(sliding_pitch[0]), :: int(sliding_pitch[1])] = True

    # Combine the valid centre mask and the pitch mask to create a final mask of valid window centres.
    valid_centre_mask &= pitch_mask
 
    # Exclude border of half the window size around non-free edges, 
    # as these windows are not valid for equilibrium gap evaluation.
    # Note: current implementation is simple and assumes that the 
    # bounding box of the specimen is rectangular and aligned with the x and y axes.
    valid_centre_mask &= _build_non_free_edge_mask(
        experiment_data.specimen_geometry.x.shape,
        experiment_data.boundary_conditions.edge_conditions,
        int(window_size[0] // 2),
        int(window_size[1] // 2)
    )

    # Compute volume of each pixel in specimen (area * thickness)
    # The pixel_area_scale is used to convert the pixel area to the same units as the stress field, if necessary.
    volume = (
        np.asarray(specimen_geometry.pixel_area, dtype=np.float64)
        * float(pixel_area_scale)
        * float(specimen_geometry.thickness)
    )
    # Set the volume of pixels outside the specimen to zero, as these pixels are not valid for equilibrium gap evaluation.
    volume = np.where(valid_point_mask, volume, 0.0)

    # Compute virtual strain fields for the equilibrium gap evaluation. 
    # These are 3D arrays of shape (3, window_rows, window_cols),
    # where the first dimension corresponds to the three strain components (xx, yy, xy).
    virtual_strain_fields = _build_virtual_strain_fields(
        specimen_geometry.x,
        specimen_geometry.y,
        window_size,
        virtual_field_type=virtual_field_type,
        normalise_virtual_strain=normalise_virtual_strain,
        plot_virtual_field_schematic=plot_virtual_field_schematic,
    )

    # debug: plot the virtual window raster
    plot_virtual_window_raster = True
    if plot_virtual_window_raster:
        _plot_virtual_window_raster(
            specimen_geometry.x,
            specimen_geometry.y,
            valid_point_mask,
            valid_centre_mask,
            window_size,
        )

    # Extract the longitudinal force from the experiment data, which is used to normalise the equilibrium gap metric.
    # Unsure if this should be done here or in the objective function, but it is done here for now.
    longitudinal_force = _extract_longitudinal_force(experiment_data)
    force_weights = _calculate_force_weights(longitudinal_force)

    return _EquilibriumGapOperator(
        virtual_strain_fields=virtual_strain_fields,
        volume=volume,
        window_point_counts=window_point_counts,
        valid_centre_mask=valid_centre_mask,
        longitudinal_force=longitudinal_force,
        force_weights=force_weights,
    )


def _build_non_free_edge_mask(
    mask_shape: tuple[int, int],
    edge_conditions,
    row_margin: int,
    col_margin: int,
) -> np.ndarray[np.bool_]:
    """
    Build a mask that excludes the border of half the window size around 
    traction edges, as evaluation of equilibrium gap is not valid across
    traction edges.
    """
    mask = np.ones(mask_shape, dtype=bool)
    if _edge_is_non_free(edge_conditions.min_x_edge):
        mask[:, :col_margin] = False
    if _edge_is_non_free(edge_conditions.max_x_edge):
        mask[:, mask.shape[1] - col_margin :] = False
    if _edge_is_non_free(edge_conditions.min_y_edge):
        mask[:row_margin, :] = False
    if _edge_is_non_free(edge_conditions.max_y_edge):
        mask[mask.shape[0] - row_margin :, :] = False

    return mask


def _edge_is_non_free(edge) -> bool:
    return edge.x is not EEdgeCondition.Free or edge.y is not EEdgeCondition.Free


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
    normalise_virtual_strain: bool = True,
    plot_virtual_field_schematic: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Build the virtual strain fields for the equilibrium gap evaluation.

    Returns an array of shape (num_fields, 3, window_rows, window_cols), where the
    leading dimension indexes the virtual field set and the second dimension indexes
    the strain components (xx, yy, xy). 
    If virtual_field_type is TWO_AVERAGED, then num_fields=2, otherwise num_fields=1.
    
    NOTE: The magnitude of the virtual strain fields scales with the window size, so 
    normalisation, in addition to he normalise_virtual_strain normalisation done here,
    is recommended to ensure that the equilibrium gap values are independent of the window size.
    """

    if not normalise_virtual_strain:
        warnings.warn(
            "EquilibriumGapMetric: normalise_virtual_strain is False. "
            "Normalising the virtual strain is reccommended to ensure " \
            "EG values are independent of coordinates or window size.",
            UserWarning,
        )

    if virtual_field_type == EquilibriumGapVirtualFieldType.SINGLE_POS_POS:
        fields = [
            _build_virtual_strain_field(
                x,
                y,
                window_size,
                centre_dof_x=1.0,
                centre_dof_y=1.0,
            )
        ]
    elif virtual_field_type == EquilibriumGapVirtualFieldType.SINGLE_POS_NEG:
        fields = [
            _build_virtual_strain_field(
                x,
                y,
                window_size,
                centre_dof_x=1.0,
                centre_dof_y=-1.0,
            )
        ]
    elif virtual_field_type == EquilibriumGapVirtualFieldType.TWO_AVERAGED:
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

    # Convert the list of virtual strain fields to a 3D numpy array
    # of shape (num_fields, 3, window_rows, window_cols). 
    # If virtual_field_type is TWO_AVERAGED, then num_fields=2, otherwise num_fields=1.
    virtual_strain_fields = np.asarray(fields, dtype=np.float64)

    # If requested, normalise the virtual strain fields to the range [-1, 1]
    # based on the minimum and maximum values of each field.
    if normalise_virtual_strain:
        normalised_fields = []
        # Loop over each virtual strain field
        for field in virtual_strain_fields:
            min_value = float(np.nanmin(field))
            max_value = float(np.nanmax(field))
            # If the field is constant (min_value == max_value), then return a copy of the field
            if np.isclose(max_value, min_value):
                normalised_fields.append(field.copy())
                continue
            # Normalise the field to the range [-1, 1] based on the minimum and maximum values of the field
            normalised = 2.0 * (field - min_value) / (max_value - min_value) - 1.0
            # If original and normalised value is near zero, set the normalised value to zero. 
            zero_mask = np.isclose(field, 0.0, atol=1e-8) & np.isclose(normalised, 0.0, atol=1e-8)
            normalised[zero_mask] = 0.0
            # Append the normalised field to the list of normalised fields
            normalised_fields.append(normalised)
        # Convert the list of normalised fields to a 3D numpy array of shape (num_fields, 3, window_rows, window_cols)
        virtual_strain_fields = np.asarray(normalised_fields, dtype=np.float64)

    # Debug: plot the window, mesh, virtual displacements and virtual strains 
    plot_virtual_field_schematic = False 
    if plot_virtual_field_schematic:
        _plot_virtual_field_schematic(
            x,
            y,
            window_size,
            virtual_strain_fields[0], # plot the first virtual field
            centre_dof_x=1.0,
            centre_dof_y=1.0,
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
    """
    Build a single virtual strain field for the equilibrium gap evaluation, 
    based on a 9-node, 4-element virtual window.

    EG WINDOW: 4 elements per window
    
    ELEMENT ORDER:
                   N3
       N0 x--------x--------x N6
          |        |        |
          |   E3   |   E2   |
       N1 x--------xN4------x N7
          |        |        |
          |   E0   |   E1   |
       N2 x--------x--------x N8
                   N5
    
    NODE ORDER:
    
      3 x-------x 2
        |       |    
        |       |  
      0 x-------x 1

      
    The virtual strain fields are returned as a 3D array of shape (3, window_rows, window_cols), where the
    first dimension indexes the strain components (xx, yy, xy) and the second and third dimensions index 
    the rows and columns of the window.
    """

    # Define window 
    rows = int(window_size[0])
    cols = int(window_size[1])
    num_points = rows * cols
    row_mid = rows // 2
    col_mid = cols // 2
    window_x = np.asarray(x[:rows, :cols], dtype=np.float64)
    window_y = np.asarray(y[:rows, :cols], dtype=np.float64)

    # Define 9 nodes
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

    # Define nodes associated with each element
    element_nodes = np.asarray(
        (
            (2, 5, 4, 1), #E0
            (5, 8, 7, 4), #E1
            (4, 7, 6, 3), #E2
            (1, 4, 3, 0), #E3
        ),
        dtype=np.int64,
    )

    # Define degrees of freedom associated with each element (2 DOF per node)
    # The DOF ordering is [u1, v1, u2, v2, u3, v3, u4, v4] for each element,
    # x dof (u) index = node index * 2 
    # y dof (v) index = node index * 2 + 1
    element_dofs = np.asarray(
        (
            (4, 5, 10, 11, 8, 9, 2, 3),
            (10, 11, 16, 17, 14, 15, 8, 9),
            (8, 9, 14, 15, 12, 13, 6, 7),
            (2, 3, 8, 9, 6, 7, 0, 1),
        ),
        dtype=np.int64,
    )

    # Define strain-displacement (B) matrices for each point in the window, initialised to zero.
    # Each matrix has shape num_points x 18, where 18 is the number of DOF in the 9-node window (2 DOF per node).
    b_xx = np.zeros((num_points, 18), dtype=np.float64)
    b_yy = np.zeros((num_points, 18), dtype=np.float64)
    b_xy = np.zeros((num_points, 18), dtype=np.float64)

    # Count the number of elements that contribute to each point in the window, initialised to zero.
    element_count = np.zeros(num_points, dtype=np.float64)

    # Create an array of point coordinates for each point in the window, with shape num_points x 2.
    point_coordinates = np.column_stack((window_x.ravel(), window_y.ravel()))

    # Loop over elements and assemble the B matrices for each point in the window. 

    for nodes, dofs in zip(element_nodes, element_dofs, strict=True):
        # Get node coordinates for current element
        coords = node_coordinates[nodes]
        # Check which points are inside current element
        in_element = _points_in_axis_aligned_element(point_coordinates, coords)

        # Loop over points inside current element
        for point_index in np.flatnonzero(in_element):
            # Compute local coordinates (xi, eta) for current point
            xi, eta = _coordinate_transform(coords, point_coordinates[point_index])
            # Compute shape function derivatives in local coordinates
            _, shape_derivative_local = _shape_functions(xi, eta)
            # Compute Jacobian matrix for current element
            jacobian = shape_derivative_local.T @ coords
            # Transform shape function derivatives to global coordinates
            shape_derivative_global = shape_derivative_local @ np.linalg.inv(jacobian)
            # Assemble the B matrix for the current point in the current element, 
            # with shape 3 x 8 (3 strain components, 8 DOF).
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
            # Assign the B matrix values to the corresponding rows in the 
            # global b_xx, b_yy, and b_xy matrices for the current point.
            b_xx[point_index, dofs] += b_matrix[0, :]
            b_yy[point_index, dofs] += b_matrix[1, :]
            b_xy[point_index, dofs] += b_matrix[2, :]

            # Increment the element count for the current point
            element_count[point_index] += 1.0

    # Check if any points in the window were not inside any element. If so, raise an error.
    if np.any(element_count == 0.0):
        raise ValueError("Some equilibrium-gap window points were not in any element.")

    # Normalize the B matrices by the number of elements contributing to each point, 
    # so that the B matrices represent the average contribution of each element to the point.
    # This is equivalent to averaging the data on the element boundaries,
    # and ensures that the virtual strain field is continuous across element boundaries.
    b_xx /= element_count[:, np.newaxis]
    b_yy /= element_count[:, np.newaxis]
    b_xy /= element_count[:, np.newaxis]

    # Define the virtual displacement vector for the 9-node window, with 18 DOF (2 DOF per node).
    # All DOFs are initially set to zero, except for the centre node DOFs which are set to the specified values.
    virtual_displacement = np.zeros(18, dtype=np.float64)
    virtual_displacement[8] = centre_dof_x
    virtual_displacement[9] = centre_dof_y

    # Compute the virtual strain field for the entire window by multiplying
    # the strain-displacementB matrices with the virtual displacement vector.
    virtual_strain = np.empty((3, rows, cols), dtype=np.float64)
    virtual_strain[0, :, :] = (b_xx @ virtual_displacement).reshape(rows, cols)
    virtual_strain[1, :, :] = (b_yy @ virtual_displacement).reshape(rows, cols)
    virtual_strain[2, :, :] = (b_xy @ virtual_displacement).reshape(rows, cols)

    return virtual_strain


def _points_in_axis_aligned_element(
    points: npt.NDArray[np.float64],
    node_coordinates: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """
    Check which points are inside an axis-aligned quadrilateral element defined by its node coordinates.
    The element is defined by 4 nodes in the order: bottom-left, bottom-right, top-right, top-left.
    The function returns a boolean array of the same length as points, where True indicates that the point is inside the element.
    """
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
    """
    Transform global coordinates of a point to local coordinates (xi, eta) in the reference element.
    The reference element is defined in the local coordinate system with xi and eta ranging from -1 to 1.
    The transformation is based on the node coordinates of the element and the point coordinates.
    Node coordinates are expected in the order: bottom-left, bottom-right, top-right, top-left.
    """

    # xi = m-X1 / (X2 - X1) * 2 - 1
    # where:
    # m is the x-coordinate of the point, 
    # X1 is the x-coordinate of the bottom-left node
    # X2 is the x-coordinate of the bottom-right node.
    # 2 is the local element length (-1 to 1)
    # -1 centres the local coordinate system at the element centre.
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
    """
    Compute the shape functions and their derivatives for a 4-node quadrilateral element at local coordinates (xi, eta).
    
    The shape functions are defined in the local coordinate system with xi and eta ranging from -1 to 1.
    Shape functions are returned as a 1D array of shape (4,) corresponding to the 4 nodes of the element. 
    The derivatives are returned as a 2D array of shape (4, 2) where the first column corresponds to 
    the derivative with respect to xi and the second column corresponds to the derivative with respect to eta.    
    """

    # Shape functions for a 4-node quadrilateral element (bilinear shape functions)
    # Shape function has shape (4,) corresponding to the 4 nodes of the element.
    shape_function = np.asarray(
        (
            0.25 * (1.0 - xi) * (1.0 - eta), # if xi = -1, eta = -1, N1 = 1 (lower left)
            0.25 * (1.0 + xi) * (1.0 - eta), # if xi = 1,  eta = -1,  N2 = 1 (lower right)
            0.25 * (1.0 + xi) * (1.0 + eta), # if xi = 1,  eta = 1,   N3 = 1 (upper right)
            0.25 * (1.0 - xi) * (1.0 + eta), # if xi = -1, eta = 1,  N4 = 1 (upper left)
        ),
        dtype=np.float64,
    )

    # Derivatives of shape functions with respect to local coordinates (xi, eta)
    # The derivatives have shape (4, 2) corresponding to the 4 nodes and 2 local coordinates.
    # [
    #   [dN1/dxi,  dN1/deta],
    #   [dN2/dxi,  dN2/deta],
    #   [dN3/dxi,  dN3/deta],
    #   [dN4/dxi,  dN4/deta],
    # ]
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
    """Correlate 2D array with kernel, returning an array of the same shape
    as the input values. 
    
    The correlation is performed with zero-padding at the boundaries.
    In other words, the output at each point is the sum of the element-wise
    product of the kernel and the overlapping values, with missing values treated as zero.
    This is equivalent to a convolution with the kernel flipped in both dimensions.
    This is used to compute the sum of values in a sliding window defined by the kernel,
    which is useful for computing the equilibrium gap metric over a spatial domain.
    """
    return correlate2d(
        values,
        kernel,
        mode="same",
        boundary="fill",
        fillvalue=0.0,
    )


def _plot_virtual_field_schematic(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    window_size: npt.NDArray[np.uint32],
    virtual_strain: npt.NDArray[np.float64],
    *,
    centre_dof_x: float,
    centre_dof_y: float,
) -> None:
    """Plot the virtual mesh, displacement fields, and strain fields."""
    import matplotlib.pyplot as plt

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

    point_coordinates = np.column_stack((window_x.ravel(), window_y.ravel()))
    virtual_displacement_x = np.zeros(rows * cols, dtype=np.float64)
    virtual_displacement_y = np.zeros(rows * cols, dtype=np.float64)
    element_count = np.zeros(rows * cols, dtype=np.float64)

    for nodes in element_nodes:
        coords = node_coordinates[nodes]
        in_element = _points_in_axis_aligned_element(point_coordinates, coords)

        local_displacement_x = np.zeros(4, dtype=np.float64)
        local_displacement_y = np.zeros(4, dtype=np.float64)

        for local_index, global_node_index in enumerate(nodes):
            if global_node_index == 4:
                local_displacement_x[local_index] = centre_dof_x
                local_displacement_y[local_index] = centre_dof_y

        for point_index in np.flatnonzero(in_element):
            xi, eta = _coordinate_transform(coords, point_coordinates[point_index])
            shape_function, _ = _shape_functions(xi, eta)
            virtual_displacement_x[point_index] += shape_function @ local_displacement_x
            virtual_displacement_y[point_index] += shape_function @ local_displacement_y
            element_count[point_index] += 1.0

    if np.any(element_count == 0.0):
        raise ValueError("Some equilibrium-gap window points were not in any element.")

    virtual_displacement_x = (virtual_displacement_x / element_count).reshape(rows, cols)
    virtual_displacement_y = (virtual_displacement_y / element_count).reshape(rows, cols)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), constrained_layout=True)
    axes = np.asarray(axes)

    external_node_mask = np.ones(9, dtype=bool)
    external_node_mask[4] = False

    ax = axes[0, 0]
    ax.scatter(window_x, window_y, marker="x", color="0.25", s=18)

    for nodes in element_nodes:
        polygon = node_coordinates[np.asarray([*nodes, nodes[0]])]
        ax.plot(polygon[:, 0], polygon[:, 1], color="black", linewidth=1.5)

    ax.scatter(
        node_coordinates[external_node_mask, 0],
        node_coordinates[external_node_mask, 1],
        color="tab:red",
        s=45,
        zorder=3,
    )
    ax.scatter(
        node_coordinates[4, 0],
        node_coordinates[4, 1],
        color="tab:green",
        s=55,
        zorder=4,
    )
    x_min = float(np.nanmin(window_x))
    x_max = float(np.nanmax(window_x))
    y_min = float(np.nanmin(window_y))
    y_max = float(np.nanmax(window_y))
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_padding = 0.15 * x_span
    y_padding = 0.15 * y_span
    x_arrow = 0.58 * x_span
    y_arrow = 0.58 * y_span
    x_label_offset = 0.025 * x_span
    y_label_offset = 0.025 * y_span
    centre_x = node_coordinates[4, 0]
    centre_y = node_coordinates[4, 1]
    ax.quiver(
        centre_x,
        centre_y,
        x_arrow,
        0.0,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="tab:blue",
        zorder=5,
    )
    ax.quiver(
        centre_x,
        centre_y,
        0.0,
        y_arrow,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="tab:blue",
        zorder=5,
    )
    ax.text(
        centre_x + x_arrow + x_label_offset,
        centre_y,
        "x",
        color="blue",
        ha="left",
        va="center",
    )
    ax.text(
        centre_x,
        centre_y + y_arrow + y_label_offset,
        "y",
        color="blue",
        ha="center",
        va="top",
    )
    ax.set_title("(a) mesh")
    ax.set_aspect("equal")
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_max + y_padding, y_min - y_padding)
    ax.axis("off")

    image_specs = (
        (axes[0, 1], virtual_displacement_x, r"(b) $u_x^*$"),
        (axes[0, 2], virtual_displacement_y, r"(c) $u_y^*$"),
        (axes[1, 0], virtual_strain[0], r"(d) $\epsilon_{xx}^*$"),
        (axes[1, 1], virtual_strain[1], r"(e) $\epsilon_{yy}^*$"),
        (axes[1, 2], virtual_strain[2], r"(f) $\epsilon_{xy}^*$"),
    )

    def _centres_to_edges(centres: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if centres.size < 2:
            return np.asarray([centres[0] - 0.5, centres[0] + 0.5], dtype=np.float64)

        diffs = np.diff(centres)
        edges = np.empty(centres.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
        edges[0] = centres[0] - 0.5 * diffs[0]
        edges[-1] = centres[-1] + 0.5 * diffs[-1]
        return edges

    x_edges = _centres_to_edges(window_x[0, :])
    y_edges = _centres_to_edges(window_y[:, 0])

    for ax, values, title in image_specs:
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            cmap="viridis",
            shading="flat",
        )
        ax.scatter(window_x, window_y, marker="x", color="magenta", s=8)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[-1], y_edges[0])
        ax.axis("off")
        fig.colorbar(image, ax=ax, shrink=0.85)

    plt.show()

def _plot_virtual_window_raster(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    valid_point_mask: npt.NDArray[np.bool_],
    valid_centre_mask: npt.NDArray[np.bool_],
    window_size: npt.NDArray[np.uint32],
) -> None:
    """Plot the first three valid EGI windows rastered across the specimen."""
    import matplotlib.pyplot as plt

    rows = int(window_size[0])
    cols = int(window_size[1])
    row_half = rows // 2
    col_half = cols // 2

    centre_indices = np.argwhere(valid_centre_mask)
    if centre_indices.shape[0] < 3:
        raise ValueError(
            "At least three valid EGI window centres are required for the raster plot."
        )

    full_window_mask = (
        (centre_indices[:, 0] >= row_half)
        & (centre_indices[:, 0] < x.shape[0] - row_half)
        & (centre_indices[:, 1] >= col_half)
        & (centre_indices[:, 1] < x.shape[1] - col_half)
    )
    display_centre_indices = centre_indices[full_window_mask]
    if display_centre_indices.shape[0] < 3:
        display_centre_indices = centre_indices

    centre_x = x[display_centre_indices[:, 0], display_centre_indices[:, 1]]
    centre_y = y[display_centre_indices[:, 0], display_centre_indices[:, 1]]
    x_targets = np.linspace(float(np.nanmin(centre_x)), float(np.nanmax(centre_x)), 3)
    y_target = 0.5 * (float(np.nanmin(centre_y)) + float(np.nanmax(centre_y)))
    x_scale = max(float(np.nanmax(centre_x) - np.nanmin(centre_x)), 1.0)
    y_scale = max(float(np.nanmax(centre_y) - np.nanmin(centre_y)), 1.0)

    selected_centre_indices = []
    available_mask = np.ones(display_centre_indices.shape[0], dtype=bool)
    for x_target in x_targets:
        scores = (
            ((centre_x - x_target) / x_scale) ** 2
            + ((centre_y - y_target) / y_scale) ** 2
        )
        scores[~available_mask] = np.inf
        selected_index = int(np.argmin(scores))
        selected_centre_indices.append(display_centre_indices[selected_index])
        available_mask[selected_index] = False

    centre_indices = np.asarray(selected_centre_indices, dtype=np.int64)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), constrained_layout=True)
    axes = np.asarray(axes)

    for window_index, ax in enumerate(axes):
        centre_row, centre_col = centre_indices[window_index]

        row_start = centre_row - row_half
        row_stop = centre_row + row_half + 1
        col_start = centre_col - col_half
        col_stop = centre_col + col_half + 1
        if row_start < 0 or row_stop > x.shape[0] or col_start < 0 or col_stop > x.shape[1]:
            row_start = max(row_start, 0)
            row_stop = min(row_stop, x.shape[0])
            col_start = max(col_start, 0)
            col_stop = min(col_stop, x.shape[1])

        window_x = x[row_start:row_stop, col_start:col_stop]
        window_y = y[row_start:row_stop, col_start:col_stop]
        window_valid_point_mask = valid_point_mask[row_start:row_stop, col_start:col_stop]

        node_rows = np.asarray([row_start, centre_row, row_stop - 1], dtype=np.int64)
        node_cols = np.asarray([col_start, centre_col, col_stop - 1], dtype=np.int64)
        node_coordinates = np.asarray(
            [
                [x[row, col], y[row, col]]
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

        ax.scatter(
            x[valid_point_mask],
            y[valid_point_mask],
            marker=".",
            color="0.75",
            s=1,
        )
        ax.scatter(
            window_x[window_valid_point_mask],
            window_y[window_valid_point_mask],
            marker="x",
            color="tab:blue",
            s=10,
        )

        for nodes in element_nodes:
            polygon = node_coordinates[np.asarray([*nodes, nodes[0]])]
            ax.plot(polygon[:, 0], polygon[:, 1], color="black", linewidth=1.5)

        external_node_mask = np.ones(9, dtype=bool)
        external_node_mask[4] = False
        ax.scatter(
            node_coordinates[external_node_mask, 0],
            node_coordinates[external_node_mask, 1],
            color="tab:red",
            s=45,
            zorder=3,
        )
        ax.scatter(
            node_coordinates[4, 0],
            node_coordinates[4, 1],
            color="tab:green",
            s=55,
            zorder=4,
        )

        ax.set_title(f"Window {window_index + 1}")
        ax.set_aspect("equal")
        x_min = float(np.nanmin(window_x))
        x_max = float(np.nanmax(window_x))
        y_min = float(np.nanmin(window_y))
        y_max = float(np.nanmax(window_y))
        x_padding = 0.20 * (x_max - x_min)
        y_padding = 0.20 * (y_max - y_min)
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_max + y_padding, y_min - y_padding)
        ax.axis("off")

    plt.show()

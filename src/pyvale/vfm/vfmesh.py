from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pyvale.vfm.experimentdata import Edge, EdgeConditions, EEdgeCondition


@dataclass(slots=True)
class VirtualFieldsMesh:
    """Virtual-field helper mesh and the matrices derived from it."""
    virtual_node_coordinates_x: npt.NDArray[np.float64]
    virtual_node_coordinates_y: npt.NDArray[np.float64]
    specimen_point_indices: npt.NDArray[np.int64]
    edge_conditions: EdgeConditions
    virtual_node_ids: npt.NDArray[np.int64]
    virtual_element_node_ids: npt.NDArray[np.int64]
    data_point_virtual_element_ids: npt.NDArray[np.int64]
    global_shape_function_matrix: npt.NDArray[np.float64]
    global_strain_displacement_matrix: npt.NDArray[np.float64]
    global_strain_displacement_matrix_pseudoinverse: npt.NDArray[np.float64]
    active_degrees_of_freedom: npt.NDArray[np.int64]


@dataclass(slots=True)
class MeshNodalCoordinates:
    """Nodal coordinates defining a mesh."""
    nodal_coord_x: npt.NDArray[np.float64]   # shape (n_points_y + 1, n_points_x + 1)
    nodal_coord_y: npt.NDArray[np.float64]   # shape (n_points_y + 1, n_points_x + 1)


@dataclass(slots=True)
class GlobalVirtualFields:
    """Virtual strains and displacements generated from reference map using VF mesh."""
    virtual_strain: npt.NDArray[np.float64]
    virtual_displacement_edge: npt.NDArray[np.float64]
    virtual_displacement: npt.NDArray[np.float64]

@dataclass(slots=True)
class _EdgeDofConstraintDefinition:
    """DOF condensation metadata for one named edge."""
    edge_name: str
    edge_condition: Edge
    edge_nodes: npt.NDArray[np.int64]
    master_node: int
    slave_nodes: npt.NDArray[np.int64]


@dataclass(slots=True)
class _BoundaryConstraintInfo:
    """Metadata for plotting node-level DOF constraints."""
    node_dof_conditions: dict[int, tuple[EEdgeCondition, EEdgeCondition]]


def _extend_centroid_grid(
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Pad a centroid grid of coordinates by one layer using linear extrapolation."""

    if values.ndim != 2:
        raise ValueError("Expected a 2D centroid grid.")
    if values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("Need at least a 2x2 centroid grid to build a data mesh.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Centroid grid must contain only finite values.")

    ny, nx = values.shape
    # Initialise extended grid of coordinates
    extended = np.empty((ny + 2, nx + 2), dtype=np.float64)
    # Populate interior with original values
    extended[1:-1, 1:-1] = values
    
    # Populate edges by linear extrapolation from the interior
    # v_edge = v_boundary + (v_boundary - v_adjacent) = 2 * v_boundary - v_adjacent
    extended[0, 1:-1]  = 2.0 * extended[1, 1:-1]  - extended[2, 1:-1]  # top row = 2 * first interior row - second interior row
    extended[-1, 1:-1] = 2.0 * extended[-2, 1:-1] - extended[-3, 1:-1] # bottom row = 2 * last interior row - second to last interior row
    extended[1:-1, 0]  = 2.0 * extended[1:-1, 1]  - extended[1:-1, 2]  # left column = 2 * first interior column - second interior column
    extended[1:-1, -1] = 2.0 * extended[1:-1, -2] - extended[1:-1, -3] # right column = 2 * last interior column - second to last interior column
    
    # Populate corners by linear extrapolation from the edges
    extended[0, 0] = extended[0, 1] + extended[1, 0] - extended[1, 1]           # top-left corner = top edge + left edge - first interior point
    extended[0, -1] = extended[0, -2] + extended[1, -1] - extended[1, -2]       # top-right corner = top edge + right edge - first interior point on the right
    extended[-1, 0] = extended[-2, 0] + extended[-1, 1] - extended[-2, 1]       # bottom-left corner = bottom edge + left edge - last interior point on the left
    extended[-1, -1] = extended[-2, -1] + extended[-1, -2] - extended[-2, -2]   # bottom-right corner = bottom edge + right edge - last interior point on the right

    return extended


def _fill_missing_1d_axis(axis_values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    axis = np.asarray(axis_values, dtype=np.float64).copy()
    finite_mask = np.isfinite(axis)
    if not np.any(finite_mask):
        raise ValueError("Could not infer a structured coordinate axis from all-NaN values.")
    if np.all(finite_mask):
        return axis

    indices = np.arange(axis.size, dtype=np.float64)
    finite_indices = indices[finite_mask]
    finite_values = axis[finite_mask]
    axis[~finite_mask] = np.interp(indices[~finite_mask], finite_indices, finite_values)

    if finite_indices.size >= 2:
        first_finite_index = int(finite_indices[0])
        last_finite_index = int(finite_indices[-1])
        leading_slope = (finite_values[1] - finite_values[0]) / (finite_indices[1] - finite_indices[0])
        trailing_slope = (finite_values[-1] - finite_values[-2]) / (finite_indices[-1] - finite_indices[-2])

        for index in range(first_finite_index - 1, -1, -1):
            axis[index] = axis[index + 1] - leading_slope
        for index in range(last_finite_index + 1, axis.size):
            axis[index] = axis[index - 1] + trailing_slope

    return axis


def _structured_coordinate_grid_from_measurements(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Infer structured coordinate grids from noisy/partially-missing measurements."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.ndim != 2:
        raise ValueError("x and y must be 2D arrays.")

    x_axis = np.empty(x.shape[1], dtype=np.float64)
    for col_index in range(x.shape[1]):
        finite_values = x[:, col_index][np.isfinite(x[:, col_index])]
        x_axis[col_index] = float(np.median(finite_values)) if finite_values.size > 0 else np.nan
    x_axis = _fill_missing_1d_axis(x_axis)

    y_axis = np.empty(y.shape[0], dtype=np.float64)
    for row_index in range(y.shape[0]):
        finite_values = y[row_index, :][np.isfinite(y[row_index, :])]
        y_axis[row_index] = float(np.median(finite_values)) if finite_values.size > 0 else np.nan
    y_axis = _fill_missing_1d_axis(y_axis)

    structured_x = np.broadcast_to(x_axis[None, :], x.shape).copy()
    structured_y = np.broadcast_to(y_axis[:, None], y.shape).copy()
    return structured_x, structured_y


def _generate_data_mesh_nodal_coord(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> MeshNodalCoordinates:
    """Construct the fine mesh of data-point elements from centroid coordinates.

    Assumes:
    - `x` and `y` are 2D centroid grids with shape (n_points_y, n_points_x)
    - row index increases downward
    - the coordinate convention in the test data is already the intended one
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.ndim != 2:
        raise ValueError("x and y must be 2D arrays.")
    if x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("Need at least a 2x2 measurement grid.")

    structured_x, structured_y = _structured_coordinate_grid_from_measurements(x, y)
    x_ext = _extend_centroid_grid(structured_x)
    y_ext = _extend_centroid_grid(structured_y)

    nodal_coord_x = 0.25 * (
        x_ext[:-1, :-1]
        + x_ext[:-1, 1:]
        + x_ext[1:, :-1]
        + x_ext[1:, 1:]
    )
    nodal_coord_y = 0.25 * (
        y_ext[:-1, :-1]
        + y_ext[:-1, 1:]
        + y_ext[1:, :-1]
        + y_ext[1:, 1:]
    )

    return MeshNodalCoordinates(
        nodal_coord_x=nodal_coord_x,
        nodal_coord_y=nodal_coord_y,
    )


def _plot_grid_lines(
    ax,
    grid_x: npt.NDArray[np.float64],
    grid_y: npt.NDArray[np.float64],
    color: str,
    linewidth: float,
) -> None:
    for row in range(grid_x.shape[0]):
        ax.plot(grid_x[row, :], grid_y[row, :], color=color, linewidth=linewidth)
    for col in range(grid_x.shape[1]):
        ax.plot(grid_x[:, col], grid_y[:, col], color=color, linewidth=linewidth)


def _compute_glyph_half_size_from_spacing(
    coordinates: npt.NDArray[np.float64],
    axis: int,
    fraction_of_spacing: float = 0.3,
) -> float:
    """Return a glyph half-size based on the smallest adjacent node spacing."""

    spacing = np.abs(np.diff(coordinates, axis=axis))
    finite_positive_spacing = spacing[np.isfinite(spacing) & (spacing > 0.0)]
    if finite_positive_spacing.size > 0:
        return 0.5 * fraction_of_spacing * float(np.min(finite_positive_spacing))

    span = float(np.max(coordinates) - np.min(coordinates))
    if span > 0.0:
        return 0.5 * fraction_of_spacing * span

    return 0.1


def _estimate_positive_spacing_tolerance(
    coordinates: npt.NDArray[np.float64],
    *,
    fraction_of_spacing: float = 0.05,
) -> float:
    spacing = np.abs(np.diff(np.asarray(coordinates, dtype=np.float64)))
    finite_positive_spacing = spacing[np.isfinite(spacing) & (spacing > 0.0)]
    if finite_positive_spacing.size == 0:
        return 1.0e-12
    return max(1.0e-12, fraction_of_spacing * float(np.min(finite_positive_spacing)))


def _plot_node_constraint_glyphs(
    ax,
    virtual_grid_x: npt.NDArray[np.float64],
    virtual_grid_y: npt.NDArray[np.float64],
    node_dof_conditions: dict[int, tuple[EEdgeCondition, EEdgeCondition]],
) -> None:
    """Overlay per-DOF constraint glyphs at constrained nodes."""

    half_width = _compute_glyph_half_size_from_spacing(virtual_grid_x, axis=1)
    half_height = _compute_glyph_half_size_from_spacing(virtual_grid_y, axis=0)

    x_flat = virtual_grid_x.ravel()
    y_flat = virtual_grid_y.ravel()

    for node_id, (x_condition, y_condition) in node_dof_conditions.items():
        node_x = float(x_flat[node_id])
        node_y = float(y_flat[node_id])

        if x_condition is EEdgeCondition.Fixed:
            ax.plot(
                [node_x - half_width, node_x + half_width],
                [node_y, node_y],
                color="red",
                linewidth=2.0,
                solid_capstyle="round",
                zorder=5,
            )
        elif x_condition is EEdgeCondition.Traction:
            ax.annotate(
                "",
                xy=(node_x + half_width, node_y),
                xytext=(node_x - half_width, node_y),
                arrowprops=dict(
                    arrowstyle="<->",
                    color="green",
                    linewidth=4,
                    shrinkA=0.0,
                    shrinkB=0.0,
                ),
                zorder=5,
            )

        if y_condition is EEdgeCondition.Fixed:
            ax.plot(
                [node_x, node_x],
                [node_y - half_height, node_y + half_height],
                color="red",
                linewidth=2.0,
                solid_capstyle="round",
                zorder=5,
            )
        elif y_condition is EEdgeCondition.Traction:
            ax.annotate(
                "",
                xy=(node_x, node_y + half_height),
                xytext=(node_x, node_y - half_height),
                arrowprops=dict(
                    arrowstyle="<->",
                    color="green",
                    linewidth=1.4,
                    shrinkA=0.0,
                    shrinkB=0.0,
                ),
                zorder=5,
            )


def _generate_vf_mesh_nodal_coord(
    data_mesh:MeshNodalCoordinates,
    mesh_size:npt.NDArray[np.uint32],  # row_count, column_count
) -> MeshNodalCoordinates:
    """Snap user-defined virtual mesh onto the measured x/y grid lines."""
    
    if mesh_size.shape != (2,):
        raise ValueError("mesh_size must contain [row_count, column_count].")
    if mesh_size[0] < 1 or mesh_size[1] < 1:
        raise ValueError("mesh_size must be at least [1, 1].")
    
    # Check data mesh nodal y coordinates are constant along rows (within 0.1% of mean row value)
    if not np.all(np.isclose(data_mesh.nodal_coord_y.mean(axis=1), data_mesh.nodal_coord_y[:, 0], rtol=0.001)):
        raise ValueError("Data mesh nodal y coordinates are not constant along rows. Current implementation assumes they should be.")
    # Check data mesh nodal x coordinates are constant along columns (within 0.1% of mean row value)
    if not np.all(np.isclose(data_mesh.nodal_coord_x.mean(axis=0), data_mesh.nodal_coord_x[0, :], rtol=0.001)):
        raise ValueError("Data mesh nodal x coordinates are not constant along columns. Current implementation assumes they should be.")

    # Extract the structured 1D nodal coordinate lines from the data mesh
    data_mesh_nodal_coord_x_1d = data_mesh.nodal_coord_x[0, :]
    data_mesh_nodal_coord_y_1d = data_mesh.nodal_coord_y[:, 0]

    mean_data_dx = np.nanmean(np.diff(data_mesh_nodal_coord_x_1d,axis=0))
    mean_data_dy = np.nanmean(np.diff(data_mesh_nodal_coord_y_1d,axis=0))

    # Initialise vf mesh nodal coord by linearly spacing along specimen dimensions, ensuring one on each edge
    vf_mesh_nodal_coord_y_1d=np.linspace(data_mesh_nodal_coord_y_1d[0],data_mesh_nodal_coord_y_1d[-1],mesh_size[0]+1)
    vf_mesh_nodal_coord_x_1d=np.linspace(data_mesh_nodal_coord_x_1d[0],data_mesh_nodal_coord_x_1d[-1],mesh_size[1]+1)
    

    # Compute x distance from vf mesh nodes to data mesh nodes

    # Reshape vf_mesh_nodal_coord_x_1d to (n, 1)
    vf_mesh_nodal_coord_x_reshaped = vf_mesh_nodal_coord_x_1d[:, np.newaxis]  # Shape (n, 1)

    # Reshape data_mesh_nodal_coord_x_1d to (1, m)
    data_mesh_nodal_coord_x_reshaped = data_mesh_nodal_coord_x_1d[np.newaxis, :]  # Shape (1, m)

    # Broadcast the reshaped arrays to (n, m)
    vf_mesh_broadcasted = np.broadcast_to(vf_mesh_nodal_coord_x_reshaped, (vf_mesh_nodal_coord_x_reshaped.shape[0], data_mesh_nodal_coord_x_reshaped.shape[1]))
    data_mesh_broadcasted = np.broadcast_to(data_mesh_nodal_coord_x_reshaped, (vf_mesh_nodal_coord_x_reshaped.shape[0], data_mesh_nodal_coord_x_reshaped.shape[1]))

    # Subtract the broadcasted arrays
    x_diff = vf_mesh_broadcasted - data_mesh_broadcasted  # Shape (n, m)

    # Compute the absolute values of the differences in x coordinates
    x_distances = np.abs(x_diff)  # Shape (n, m)

    # Compute absolute minimum distance in x from each vf mesh node to the data mesh nodes
    x_distances_closest = np.min(x_distances, axis=1)  # Shape (n,)

    # Closest data mesh nodal x coordinate idx for each vf mesh node
    closest_x_idx = np.argmin(x_distances, axis=1)

    # Check max x distance is less than half the mean data mesh spacing (just ensures virtual nodes lie within data region)
    if np.any(x_distances_closest > 0.5 * mean_data_dx):
        raise ValueError(
            "mesh_size is too coarse in the x direction: some virtual nodes are more than half a data-mesh spacing away from the closest data-mesh node."
        )


    # Compute y distance from vf mesh nodes to data mesh nodes

    # Reshape vf_mesh_nodal_coord_y_1d to (n, 1)
    vf_mesh_nodal_coord_y_reshaped = vf_mesh_nodal_coord_y_1d[:, np.newaxis]  # Shape (n, 1)

    # Reshape data_mesh_nodal_coord_y_1d to (1, m)
    data_mesh_nodal_coord_y_reshaped = data_mesh_nodal_coord_y_1d[np.newaxis, :]  # Shape (1, m)

    # Broadcast the reshaped arrays to (n, m)
    vf_mesh_broadcasted_y = np.broadcast_to(vf_mesh_nodal_coord_y_reshaped, (vf_mesh_nodal_coord_y_reshaped.shape[0], data_mesh_nodal_coord_y_reshaped.shape[1]))
    data_mesh_broadcasted_y = np.broadcast_to(data_mesh_nodal_coord_y_reshaped, (vf_mesh_nodal_coord_y_reshaped.shape[0], data_mesh_nodal_coord_y_reshaped.shape[1]))

    # Subtract the broadcasted arrays
    y_diff = vf_mesh_broadcasted_y - data_mesh_broadcasted_y  # Shape (n, m)

    # Compute the absolute values of the differences in y coordinates
    y_distances = np.abs(y_diff)  # Shape (n, m)

    # Compute absolute minimum distance in y from each vf mesh node to the data mesh nodes
    y_distances_closest = np.min(y_distances, axis=1)  # Shape (n,)

    # Closest data mesh nodal y coordinate idx for each vf mesh node
    closest_y_idx = np.argmin(y_distances, axis=1)

    # Check max y distance is less than half the mean data mesh spacing (just ensures virtual nodes lie within data region)
    if np.any(y_distances_closest > 0.5 * mean_data_dy):
        raise ValueError(
            "mesh_size is too coarse in the y direction: some virtual nodes are more than half a data-mesh spacing away from the closest data-mesh node."
        )

    # Force the outer virtual mesh nodes to lie exactly on the specimen edges
    closest_x_idx[0] = 0
    closest_x_idx[-1] = data_mesh_nodal_coord_x_1d.size - 1
    closest_y_idx[0] = 0
    closest_y_idx[-1] = data_mesh_nodal_coord_y_1d.size - 1

    # Check snapping has not collapsed neighbouring virtual nodes onto the same data-mesh node
    if np.unique(closest_x_idx).size != closest_x_idx.size:
        raise ValueError(
            "mesh_size is too fine in the x direction: multiple virtual nodes snapped to the same data-mesh x coordinate."
        )
    if np.unique(closest_y_idx).size != closest_y_idx.size:
        raise ValueError(
            "mesh_size is too fine in the y direction: multiple virtual nodes snapped to the same data-mesh y coordinate."
        )

    # Updated vf mesh nodal coordinates after snapping to closest data mesh nodes (so no data element is split)
    vf_mesh_nodal_coord_x_1d = data_mesh_nodal_coord_x_1d[closest_x_idx]
    vf_mesh_nodal_coord_y_1d = data_mesh_nodal_coord_y_1d[closest_y_idx]

    # Expand vf mesh nodal coord to 2D grid
    vf_mesh_nodal_coord_x, vf_mesh_nodal_coord_y = np.meshgrid(
        vf_mesh_nodal_coord_x_1d,
        vf_mesh_nodal_coord_y_1d,
    )
    
    return MeshNodalCoordinates(
        nodal_coord_x=vf_mesh_nodal_coord_x,
        nodal_coord_y=vf_mesh_nodal_coord_y,
    )


def _evaluate_bilinear_shape_functions(
    xi: float,
    eta: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate bilinear quad shape functions at local coordinates.

    Node order is:
    1. lower-left  -> (-1, -1)
    2. lower-right -> (+1, -1)
    3. upper-right -> (+1, +1)
    4. upper-left  -> (-1, +1)
    """

    shape_function_values = np.array(
        [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ],
        dtype=np.float64,
    )

    shape_function_derivatives = np.array(
        [
            [-0.25 * (1.0 - eta), -0.25 * (1.0 - xi)],
            [0.25 * (1.0 - eta), -0.25 * (1.0 + xi)],
            [0.25 * (1.0 + eta), 0.25 * (1.0 + xi)],
            [-0.25 * (1.0 + eta), 0.25 * (1.0 - xi)],
        ],
        dtype=np.float64,
    )

    return shape_function_values, shape_function_derivatives


def _compute_local_element_coordinates(
    point_coordinates: npt.NDArray[np.float64],
    element_node_coordinates: npt.NDArray[np.float64],
    tolerance: float = 1.0e-10,
    max_iterations: int = 25,
) -> tuple[float, float]:
    """Find local element coordinates for a point inside a bilinear quad.

    An iterative Newton-Raphson method is used to solve the nonlinear mapping 
    from local to global coordinates. It is possible to used a closed form solution,
    as per the previous MATLAB implementation of this code (see below). However,
    the iterative method is more general and readible and the performance is unlikely
    to be an issue.

    % Closed form solution based on "Chongyu Hua, An inverse transformation for 
    % quadrilateral isoparametric elements: Analysis and application, 
    % Finite Elements in Analysis and Design, vol 7, 1990"

    % Node ordering is of type (d) in the article
    %%Function
    A = [-1 1 -1 1; -1 -1 1 1; 1 -1 -1 1]*nodeCoords; %eq (12)
    d = [4*ptCoords(1)-(sum(nodeCoords(:,1))); 4*ptCoords(2)-(sum(nodeCoords(:,2)))]; %eq (8)

    d1 = d(1); d2=d(2);
    a1 = A(1,1); a2 = A(1,2); b1 = A(2,1); b2 = A(2,2); c1 = A(3,1); c2 = A(3,2);
    % Parameters from table 1
    ab = a1*b2-a2*b1; ac = a1*c2-a2*c1; ad = a1*d2-a2*d1; cb = c1*b2-c2*b1;
    da = d1*a2-d2*a1; dc = d1*c2-d2*c1; ba = b1*a2-b2*a1; db = d1*b2-d2*b1;
    bd = b1*d2-b2*d1; bc = b1*c2-b2*c1;

    % Algorithms from table 1
    if a1*a2*ab*ac ~= 0 || (a1==0 && a2*c1 ~= 0) || (a2 == 0 && a1*b2 ~= 0)
        xi = roots([ab, cb+da, dc]);
        xi = xi(abs(xi)<1);
        eta = (ad+ba*xi)/ac;
    elseif a1*a2 ~= 0 && ab == 0
        xi = (a1*dc)/(b1*ac+a1*ad);
        eta = ad/ac;
    elseif a1*a2 ~= 0 && ac == 0
        xi = ad/ab;
        eta = (a1*db)/(c1*ab+a1*ad);
    else
        xi = dc/(a1*d2+bc);
        eta = bd/(a2*d1+bc);
    end

    """

    # Initial guess at local coordinates is the element center (0, 0) 
    xi = 0.0
    eta = 0.0

    for _ in range(max_iterations):
        # Evaluate shape functions and their derivatives at current local coordinates
        shape_function_values, shape_function_derivatives = _evaluate_bilinear_shape_functions(xi,eta)
        # Evaluate global coordinates for the current identified shape function matrix
        mapped_coordinates = shape_function_values @ element_node_coordinates
        # compute residual between known global point coordinates and evaluated
        residual = point_coordinates - mapped_coordinates
        # jacobian = dNdxi / dNdx??
        jacobian = shape_function_derivatives.T @ element_node_coordinates
        # compute step (unsure what solve does exatly)
        update = np.linalg.solve(jacobian, residual)
        # update guess of local coordinates
        xi += float(update[0])
        eta += float(update[1])

        if np.linalg.norm(update, ord=np.inf) < tolerance:
            tol = 1e-3
            if not (-1.0 - tol <= xi <= 1.0 + tol and -1.0 - tol <= eta <= 1.0 + tol):
                raise ValueError("Point mapped outside its assigned virtual element.")
            
            return xi, eta

    raise ValueError(
        "Could not determine local element coordinates for a measurement point."
    )


def _assemble_strain_displacement_matrix(
    shape_function_gradients_global: npt.NDArray[np.float64],
    use_nlgeom: bool = False,
) -> npt.NDArray[np.float64]:
    """Assemble the strain-displacement matrix ("B matrix") for a 4-node quad.
    
    shape_function_gradients_global is (4,2) with each row corresponding to a node, and the two columns corresponding to the derivatives with respect to x and y, respectively.
    [
    [dN1/dx, dN1/dy],
    [dN2/dx, dN2/dy],
    [dN3/dx, dN3/dy],
    [dN4/dx, dN4/dy],
    ]
    
    The element displacement vector is assumed to be ordered as
    u_e = [u1, v1, u2, v2, u3, v3, u4, v4]^T.

    If `use_nlgeom` is False:
        return B matrix with shape (3, 8) that maps nodal displacements to the small-strain vector
        [epsilon_xx, epsilon_yy, gamma_xy]^T = [du/dx, dv/dy, du/dy + dv/dx]^T.

        B =
            [
                [dN1/dx, 0,      dN2/dx, 0,      dN3/dx, 0,      dN4/dx, 0     ],
                [0,      dN1/dy, 0,      dN2/dy, 0,      dN3/dy, 0,      dN4/dy],
                [dN1/dy, dN1/dx, dN2/dy, dN2/dx, dN3/dy, dN3/dx, dN4/dy, dN4/dx],
            ]

    If `use_nlgeom` is True:
        return B matrix with shape (4, 8) that maps nodal displacements to the separate displacement-gradient components
        [epsilon_xx, epsilon_yy, du/dy, dv/dx]^T = [du/dx, dv/dy, du/dy, dv/dx]^T.

        B =
            [
                [dN1/dx, 0,      dN2/dx, 0,      dN3/dx, 0,      dN4/dx, 0     ],
                [0,      dN1/dy, 0,      dN2/dy, 0,      dN3/dy, 0,      dN4/dy],
                [dN1/dy, 0,      dN2/dy, 0,      dN3/dy, 0,      dN4/dy, 0     ],
                [0,      dN1/dx, 0,      dN2/dx, 0,      dN3/dx, 0,      dN4/dx],
            ]

    Returns
    -------
    ndarray
        Strain-displacement matrix with columns ordered as
        [u1, v1, u2, v2, u3, v3, u4, v4].

    """

    if use_nlgeom:
        return np.array(
            [
                [
                    shape_function_gradients_global[0, 0], 0.0,
                    shape_function_gradients_global[1, 0], 0.0,
                    shape_function_gradients_global[2, 0], 0.0,
                    shape_function_gradients_global[3, 0], 0.0,
                ],
                [
                    0.0, shape_function_gradients_global[0, 1],
                    0.0, shape_function_gradients_global[1, 1],
                    0.0, shape_function_gradients_global[2, 1],
                    0.0, shape_function_gradients_global[3, 1],
                ],
                [
                    shape_function_gradients_global[0, 1], 0.0,
                    shape_function_gradients_global[1, 1], 0.0,
                    shape_function_gradients_global[2, 1], 0.0,
                    shape_function_gradients_global[3, 1], 0.0,
                ],
                [
                    0.0, shape_function_gradients_global[0, 0],
                    0.0, shape_function_gradients_global[1, 0],
                    0.0, shape_function_gradients_global[2, 0],
                    0.0, shape_function_gradients_global[3, 0],
                ],
            ],
            dtype=np.float64,
        )

    return np.array(
        [
            [
                shape_function_gradients_global[0, 0], 0.0,
                shape_function_gradients_global[1, 0], 0.0,
                shape_function_gradients_global[2, 0], 0.0,
                shape_function_gradients_global[3, 0], 0.0,
            ],
            [
                0.0, shape_function_gradients_global[0, 1],
                0.0, shape_function_gradients_global[1, 1],
                0.0, shape_function_gradients_global[2, 1],
                0.0, shape_function_gradients_global[3, 1],
            ],
            [
                shape_function_gradients_global[0, 1], shape_function_gradients_global[0, 0],
                shape_function_gradients_global[1, 1], shape_function_gradients_global[1, 0],
                shape_function_gradients_global[2, 1], shape_function_gradients_global[2, 0],
                shape_function_gradients_global[3, 1], shape_function_gradients_global[3, 0],
            ],
        ],
        dtype=np.float64,
    )


def _compute_point_shape_function_values_and_strain_displacement_matrices(
    point_coordinates: npt.NDArray[np.float64],
    element_node_coordinates: npt.NDArray[np.float64],
    use_nlgeom: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate shape functions and the strain-displacement matrix at one point."""

    # Compute local element coordinates for datapoint
    xi, eta = _compute_local_element_coordinates(point_coordinates, element_node_coordinates)

    # Compute shape functions and derivatives at this points local coordinates (could this use derivates from above to save recalc?)
    shape_function_values, shape_function_local_derivatives = _evaluate_bilinear_shape_functions(xi,eta)

    # Note: shape_function_local_derivatives (4,2) are with respect to local element coordinates, 
    # so we need to transform to global coordinates before assembling the strain-displacement matrix.
    # Each row of shape_function_local_derivatives corresponds to a node, and the two columns correspond to 
    # the derivatives with respect to xi and eta, respectively.
    # element_node_coordinates is (4,2) with each row corresponding to a node and the two columns 
    # corresponding to x and y coordinates of the node, respectively.
    # Hence: shape_function_local_derivatives.T is (2,4) and element_node_coordinates is (4,2), 
    # so the matrix multiplication gives a (2,2) jacobian matrix.
    # jacobian =[ [dx/dxi,  dy/dxi ],  [dx/deta, dy/deta] ]

    # Compute the Jacobian of the local-to-physical coordinate mapping (matrix multiplication of shape function derivatives with element node coordinates)
    jacobian = shape_function_local_derivatives.T @ element_node_coordinates  
    # Solve linear system to get shape function gradients with respect to global coordinates 
    # J.T @ shape_function_gradients_global.T = shape_function_local_derivatives.T
    # Ax=b where A is jacobian.T, x is shape_function_gradients_global.T and b is shape_function_local_derivatives.T
    shape_function_gradients_global = np.linalg.solve(
        jacobian.T,
        shape_function_local_derivatives.T,
    ).T

    # Assemble strain displacement matrix
    strain_displacement_matrix = _assemble_strain_displacement_matrix(
        shape_function_gradients_global,
        use_nlgeom=use_nlgeom,
    )

    return shape_function_values, strain_displacement_matrix




def _apply_direction_constraint(
    constraint: EEdgeCondition,
    edge_dofs: npt.NDArray[np.int64],
    master_dof: int,
    slave_edge_dofs: npt.NDArray[np.int64],
 ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return fixed, slave, and master DOFs for one constrained direction."""

    empty_dofs = np.empty(0, dtype=np.int64)

    if constraint is EEdgeCondition.Free:
        return empty_dofs, empty_dofs, empty_dofs

    if constraint is EEdgeCondition.Fixed:
        return edge_dofs.astype(np.int64, copy=False), empty_dofs, empty_dofs

    if constraint is EEdgeCondition.Traction:
        return (
            empty_dofs,
            slave_edge_dofs.astype(np.int64, copy=False),
            np.array([master_dof], dtype=np.int64),
        )

    raise ValueError(f"Unsupported edge boundary condition: {constraint!r}")


def _build_boundary_constraint_debug_info(
    fixed_dofs: set[int],
    slave_dofs: set[int],
    master_dofs: set[int],
) -> _BoundaryConstraintInfo:
    """Build per-node x/y DOF conditions from constrained DOF sets."""

    traction_dofs = slave_dofs.union(master_dofs)
    constrained_dofs = fixed_dofs.union(traction_dofs)
    constrained_node_ids = sorted({dof // 2 for dof in constrained_dofs})

    node_dof_conditions: dict[
        int, tuple[EEdgeCondition, EEdgeCondition]
    ] = {}
    for node_id in constrained_node_ids:
        x_dof = 2 * node_id
        y_dof = x_dof + 1

        if x_dof in fixed_dofs:
            x_condition = EEdgeCondition.Fixed
        elif x_dof in traction_dofs:
            x_condition = EEdgeCondition.Traction
        else:
            x_condition = EEdgeCondition.Free

        if y_dof in fixed_dofs:
            y_condition = EEdgeCondition.Fixed
        elif y_dof in traction_dofs:
            y_condition = EEdgeCondition.Traction
        else:
            y_condition = EEdgeCondition.Free

        node_dof_conditions[node_id] = (x_condition, y_condition)

    return _BoundaryConstraintInfo(node_dof_conditions=node_dof_conditions)


def _compute_constrained_strain_displacement_matrix(
    global_strain_displacement_matrix: npt.NDArray[np.float64],
    virtual_node_ids: npt.NDArray[np.int64],
    edge_conditions: EdgeConditions,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    _BoundaryConstraintInfo,
]:
    """Apply virtual boundary constraints to the global strain-displacement matrix.

    - `Free` leaves the edge DOFs untouched.
    - `Fixed` removes the edge DOFs from the active unknowns.
    - `Traction` enforces a constant virtual displacement along the edge by
      condensing slave edge DOFs into one master edge DOF.

    Input global_strain_displacement_matrix has shape (3 * n_specimen_points, 2 * n_nodes)
    
    The rows correspond to virtual strain components at specimen data points:
        rows 0                    : n_specimen_points      -> eps_xx
        rows n_specimen_points    : 2*n_specimen_points    -> eps_yy
        rows 2*n_specimen_points  : 3*n_specimen_points    -> gamma_xy

    The columns correspond to nodal virtual displacement DOFs:
        [ux_0, uy_0, ux_1, uy_1, ..., ux_n, uy_n]

    For a Traction condition we want to impose constant virtual displacement along the edge,
    which means all dofs (e.g. x dofs) on that edge are tied together. To do this we designate
    one master DOF (e.g. x dof of first node on edge) and treat the rest as slave DOFs. The 
    slave DOFs are then condensed into the master DOF by summing the corresponding columns in the constrained matrix.
    
    Before applying the constraint, the virtual strain field contains separate
    contributions from the master and slave DOFs:
        eps = B_m ux_m + B_s1 ux_s1 + B_s2 ux_s2 + ...
    
    A Traction condition on this edge enforces:
        ux_s1 = ux_m
        ux_s2 = ux_m
        ...
    
    Substituting this into the strain expression gives:
        eps = (B_m + B_s1 + B_s2 + ...) ux_m
    
    Therefore, the slave columns are summed and added to the master column.
    The slave columns are then removed later because they are no longer
    independent unknowns.
        
    """

    # Initialise the constrained matrix as a copy of the global matrix, and empty sets to track which DOFs are fixed, slave, or master.
    constrained_matrix = np.array(global_strain_displacement_matrix, copy=True)
    fixed_dofs: set[int] = set()
    slave_dofs: set[int] = set()
    master_dofs: set[int] = set()

    edge_dof_constraints = (
        _EdgeDofConstraintDefinition(
            edge_name="min_x_edge",
            edge_condition=edge_conditions.min_x_edge,
            edge_nodes=virtual_node_ids[:, 0],
            master_node=int(virtual_node_ids[0, 0]), # min-x edge master node is first node in column
            slave_nodes=virtual_node_ids[:, 0][1:],
        ),
        _EdgeDofConstraintDefinition(
            edge_name="min_y_edge",
            edge_condition=edge_conditions.min_y_edge,
            edge_nodes=virtual_node_ids[0, :],
            master_node=int(virtual_node_ids[0, 0]), # min-y edge master node is first node in row
            slave_nodes=virtual_node_ids[0, :][1:],
        ),
        _EdgeDofConstraintDefinition(
            edge_name="max_x_edge",
            edge_condition=edge_conditions.max_x_edge,
            edge_nodes=virtual_node_ids[:, -1],
            master_node=int(virtual_node_ids[-1, -1]), # max-x edge master node is last node in column
            slave_nodes=virtual_node_ids[:, -1][:-1],
        ),
        _EdgeDofConstraintDefinition(
            edge_name="max_y_edge",
            edge_condition=edge_conditions.max_y_edge,
            edge_nodes=virtual_node_ids[-1, :],
            master_node=int(virtual_node_ids[-1, -1]), # max-y edge master node is last node in row
            slave_nodes=virtual_node_ids[-1, :][:-1],
        ),
    )

    for edge_dof_constraint in edge_dof_constraints:
        # Gather the DOF indices for the edge nodes, master node, and slave nodes. Each node has two DOFs (x and y).
        edge_dofs_x = 2 * edge_dof_constraint.edge_nodes
        edge_dofs_y = edge_dofs_x + 1
        master_dof_x = int(2 * edge_dof_constraint.master_node)
        master_dof_y = master_dof_x + 1
        slave_dofs_x = 2 * edge_dof_constraint.slave_nodes
        slave_dofs_y = slave_dofs_x + 1

        # Impose x direction constraint and return the affected DOFs
        fixed_dofs_x, slave_dofs_x, master_dofs_x = _apply_direction_constraint(
            edge_dof_constraint.edge_condition.x,
            edge_dofs_x,
            master_dof_x,
            slave_dofs_x,
        )

        # Impose y direction constraint and return the affected DOFs
        fixed_dofs_y, slave_dofs_y, master_dofs_y = _apply_direction_constraint(
            edge_dof_constraint.edge_condition.y,
            edge_dofs_y,
            master_dof_y,
            slave_dofs_y,
        )

        # Update DOF sets according to imposed constraints
        fixed_dofs.update(fixed_dofs_x.tolist())
        fixed_dofs.update(fixed_dofs_y.tolist())
        slave_dofs.update(slave_dofs_x.tolist())
        slave_dofs.update(slave_dofs_y.tolist())

        # Condense slave DOFs into master DOF by summing corresponding columns in constrained matrix (see docstring)
        if master_dofs_x.size > 0:
            constrained_matrix[:, master_dofs_x[0]] += np.sum(
                constrained_matrix[:, slave_dofs_x],
                axis=1,
            )
            # Keep track of master DOFs
            master_dofs.update(master_dofs_x.tolist())

        # Condense slave DOFs into master DOF by summing corresponding columns in constrained matrix (see docstring)
        if master_dofs_y.size > 0:
            constrained_matrix[:, master_dofs_y[0]] += np.sum(
                constrained_matrix[:, slave_dofs_y],
                axis=1,
            )
            # Keep track of master DOFs
            master_dofs.update(master_dofs_y.tolist())

    # Check for any conflicting constraints.
    conflicting_master_dofs = master_dofs.intersection(fixed_dofs)
    if conflicting_master_dofs:
        raise ValueError(
            "Incompatible boundary conditions: a master DOF cannot also be fixed."
        )

    # Remove fixed and slave DOFs from the active DOF mask
    active_dof_mask = np.ones(constrained_matrix.shape[1], dtype=bool)
    if fixed_dofs:
        active_dof_mask[np.fromiter(fixed_dofs, dtype=np.int64)] = False
    if slave_dofs:
        active_dof_mask[np.fromiter(slave_dofs, dtype=np.int64)] = False
    active_dof_ids = np.flatnonzero(active_dof_mask).astype(np.int64)

    # Retain only the active DOF columns in the constrained matrix (i.e. remove fixed and slave DOFs)
    constrained_matrix = constrained_matrix[:, active_dof_ids]

    # Build debug info for plotting node-level DOF conditions (e.g. to verify correct application of boundary conditions)
    boundary_constraint_debug_info = _build_boundary_constraint_debug_info(
        fixed_dofs,
        slave_dofs,
        master_dofs,
    )

    return constrained_matrix, active_dof_ids, boundary_constraint_debug_info



def _apply_edge_conditions(
    virtual_displacement: npt.NDArray[np.float64],
    edge_conditions: EdgeConditions,
    virtual_node_ids: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    updated = virtual_displacement.copy()

    for edge in range(4):
        if edge == 0: 
            edge_nodes = virtual_node_ids[0, :]
            master_node = virtual_node_ids[0, 0]
            slave_nodes = edge_nodes[1:]
            x_condition = edge_conditions.min_y_edge.x
            y_condition = edge_conditions.min_y_edge.y
        elif edge == 1:
            edge_nodes = virtual_node_ids[:, 0]
            master_node = virtual_node_ids[0, 0]
            slave_nodes = edge_nodes[1:]
            x_condition = edge_conditions.min_x_edge.x
            y_condition = edge_conditions.min_x_edge.y
        elif edge == 2:
            edge_nodes = virtual_node_ids[-1, :]
            master_node = virtual_node_ids[-1, -1]
            slave_nodes = edge_nodes[:-1]
            x_condition = edge_conditions.max_y_edge.x
            y_condition = edge_conditions.max_y_edge.y
        else:
            edge_nodes = virtual_node_ids[:, -1]
            master_node = virtual_node_ids[-1, -1]
            slave_nodes = edge_nodes[:-1]
            x_condition = edge_conditions.max_x_edge.x
            y_condition = edge_conditions.max_x_edge.y

        edge_dofs_x = 2 * edge_nodes
        edge_dofs_y = edge_dofs_x + 1
        master_dof_x = 2 * master_node
        master_dof_y = master_dof_x + 1
        slave_dofs_x = 2 * slave_nodes
        slave_dofs_y = slave_dofs_x + 1

        if x_condition is EEdgeCondition.Fixed:
            updated[edge_dofs_x] = 0.0
        elif x_condition is EEdgeCondition.Traction:
            updated[slave_dofs_x] = updated[master_dof_x]

        if y_condition is EEdgeCondition.Fixed:
            updated[edge_dofs_y] = 0.0
        elif y_condition is EEdgeCondition.Traction:
            updated[slave_dofs_y] = updated[master_dof_y]

    return updated


def _plot_generated_virtual_fields(
    reference_map: npt.NDArray[np.float64],
    virtual_displacement: npt.NDArray[np.float64],
    virtual_strain: npt.NDArray[np.float64],
) -> None:
    """Plot one summary figure per selected timestep."""

    import matplotlib.pyplot as plt

    n_timesteps, n_components, _, _ = reference_map.shape
    # Select up to 5 timesteps, evenly spaced (but always include the first and last timesteps)
    n_timesteps_to_plot = min(5, n_timesteps)
    timestep_indices = np.linspace(
        0,
        n_timesteps - 1,
        n_timesteps_to_plot,
        dtype=int,
    )
    timestep_indices = np.unique(timestep_indices)

    # Create figure for each select timestep
    for timestep in timestep_indices:
        # 3 x 3 grid of subplots. 
        # Row 0: ref map, Row 1: virtual strain, Row 2: virtual displacement. 
        # Cols: components
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))

        component_titles = ["xx", "yy", "xy"]
        n_components_to_plot = min(3, n_components)

        for component in range(n_components_to_plot):
            # Plot ref map for each component
            reference_field = reference_map[timestep, component, :, :]
            reference_field_finite = reference_field[np.isfinite(reference_field)]
            ref_plot_kwargs: dict[str, float] = {}
            # Set color limits to 2nd and 98th percentile to avoid outliers dominating
            if reference_field_finite.size > 0:
                ref_plot_kwargs["vmin"] = float(np.percentile(reference_field_finite, 2.0))
                ref_plot_kwargs["vmax"] = float(np.percentile(reference_field_finite, 98.0))
                if np.isclose(ref_plot_kwargs["vmin"], ref_plot_kwargs["vmax"]):
                    ref_plot_kwargs = {}

            ref_im = axes[0, component].imshow(reference_field, **ref_plot_kwargs)
            axes[0, component].set_title(f"Reference map {component_titles[component]} t{timestep}")
            fig.colorbar(ref_im, ax=axes[0, component])

            # Plot virtual strain for each component
            strain_im = axes[1, component].imshow(virtual_strain[timestep, component, :, :])
            axes[1, component].set_title(f"Virtual strain {component_titles[component]}")
            fig.colorbar(strain_im, ax=axes[1, component])

        # Plot virtual displacement for first component
        disp_x_im = axes[2, 0].imshow(virtual_displacement[timestep, 0, :, :])
        axes[2, 0].set_title("Virtual displacement x")
        fig.colorbar(disp_x_im, ax=axes[2, 0])

        # Plot virtual displacement for second component
        disp_y_im = axes[2, 1].imshow(virtual_displacement[timestep, 1, :, :])
        axes[2, 1].set_title("Virtual displacement y")
        fig.colorbar(disp_y_im, ax=axes[2, 1])

        axes[2, 2].axis("off")

        fig.tight_layout()
        plt.show()


def plot_virtual_fields_mesh(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    virtual_fields_mesh: MeshNodalCoordinates | None = None,
    data_mesh: MeshNodalCoordinates | None = None,
    specimen_mask: npt.NDArray[np.bool_] | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
    plot_data_points: bool = True,
    node_ids: npt.NDArray[np.int32] | None = None,
    element_node_ids: npt.NDArray[np.int32] | None = None,
    node_dof_conditions: dict[int, tuple[EEdgeCondition, EEdgeCondition]] | None = None,
) -> Path | None:
    """Plot data points plus whichever of the data mesh and VF mesh are provided."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if data_mesh is None and virtual_fields_mesh is None:
        raise ValueError("At least one of data_mesh or virtual_fields_mesh must be provided.")

    fig, ax = plt.subplots(figsize=(10, 4))

    legend_handles: list[Line2D] = []

    if data_mesh is not None:
        _plot_grid_lines(
            ax,
            data_mesh.nodal_coord_x,
            data_mesh.nodal_coord_y,
            color="#49dbe6",
            linewidth=0.7,
        )
        legend_handles.append(
            Line2D([0], [0], color="#49dbe6", linewidth=1.0, label="Data elements")
        )

    if virtual_fields_mesh is not None:
        if isinstance(virtual_fields_mesh, VirtualFieldsMesh):
            virtual_grid_x = virtual_fields_mesh.virtual_node_coordinates_x
            virtual_grid_y = virtual_fields_mesh.virtual_node_coordinates_y
        else:
            virtual_grid_x = virtual_fields_mesh.nodal_coord_x
            virtual_grid_y = virtual_fields_mesh.nodal_coord_y
        _plot_grid_lines(ax, virtual_grid_x, virtual_grid_y, color="black", linewidth=0.9)
        legend_handles.append(
            Line2D([0], [0], color="black", linewidth=1.0, label="Virtual elements")
        )

    if plot_data_points:
        if specimen_mask is None:
            point_x = x
            point_y = y
        else:
            point_x = x[specimen_mask]
            point_y = y[specimen_mask]
        ax.scatter(
            point_x,
            point_y,
            s=5,
            marker="x",
            linewidths=0.4,
            color="red",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="None",
                markersize=6,
                markeredgewidth=0.8,
                color="red",
                label="Data points",
            )
        )

    if node_dof_conditions is not None:
        _plot_node_constraint_glyphs(
            ax,
            virtual_grid_x,
            virtual_grid_y,
            node_dof_conditions,
        )
        legend_handles.append(
            Line2D([0], [0], color="red", linewidth=2.0, label="Fixed DOF")
        )
        legend_handles.append(
            Line2D([0], [0], color="green", linewidth=1.4, label="Traction DOF")
        )

    if node_ids is not None:
        for row in range(node_ids.shape[0]):
            for col in range(node_ids.shape[1]):
                node_id = int(node_ids[row, col])
                if node_dof_conditions is not None and node_id not in node_dof_conditions:
                    continue
                ax.text(
                    virtual_grid_x[row, col],
                    virtual_grid_y[row, col],
                    str(node_id),
                    color="blue",
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
                )

    if element_node_ids is not None:
        for element_id in range(element_node_ids.shape[0]):
            element_nodes = element_node_ids[element_id]
            elem_x = virtual_grid_x.ravel()[element_nodes]
            elem_y = virtual_grid_y.ravel()[element_nodes]
            ax.text(
                np.mean(elem_x),
                np.mean(elem_y),
                f"E{element_id}",
                color="black",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0),
            )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="best")
    ax.set_title("Mesh Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    saved_path: Path | None = None
    if output_path is not None:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    # plt.close(fig)
    return saved_path


def generate_virtual_fields_mesh(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    specimen_mask: npt.NDArray[np.bool_],
    edge_conditions: EdgeConditions,
    mesh_size: npt.NDArray[np.uint32],
    use_nlgeom: bool = False,
    generate_plots: bool = False,
):
    """Construct a mesh over the test data to be used for virtual field generation.
    
    Parameters
    ----------
    x : ndarray
        Shape (n_points_y, n_points_x).
        The x coordinates of the measurement points.
    y : ndarray
        Shape (n_points_y, n_points_x).
        The y coordinates of the measurement points.
    specimen_mask : ndarray of bool
        Shape (n_points_y, n_points_x).
        A mask indicating the specimen region (True for points inside the specimen).
    boundary_conditions : BoundaryConditions
        The boundary conditions associated with the test data.
    mesh_size : ndarray
        Shape (2,1) 
        The number of virtual elements in the y (n rows) and x directions (n columns), respectively.   

    Returns
    -------
    VirtualFieldsMesh
        A dataclass containing the virtual fields mesh and related matrices.


    Workflow
    --------
    1. Construct fine mesh consisting of 'data point elements' (associated area of each point)
    2. Construct coarse virtual mesh by snapping a regular grid onto the data point element edges
    3. Assemble connectivity matrix (defining associations between virtual elements and data points)
    4. Assemble global shape function matrix (Nglob) and global strain-displacement matrix (Bglob) for the virtual mesh
    5. Impose virtual boundary conditions to Bglob to get Bbar 
    6. Compute pseudo-inverse of Bbar (Binv)
    7. Return virtual fields mesh object containing required data

    """

    # Check physical y coordinates increase as array row increases
    if y[0,0] > y[-1,0]:
        raise ValueError("Coordinate data is not in the expected format. y-coordinate should increase as array row increases.")
    
    # Construct fine mesh around data points
    data_mesh_nodal_coord =_generate_data_mesh_nodal_coord(x,y)

    # Debug: plot data mesh overlaid on data points 
    if generate_plots:
        plot_virtual_fields_mesh(
            x,
            y,
            data_mesh=data_mesh_nodal_coord,
            specimen_mask=specimen_mask,
            plot_data_points=True,
        )

    # Construct coarse virtual mesh (of user-defined size) by snapping a regular grid onto the data point element edges
    vf_mesh_nodal_coord = _generate_vf_mesh_nodal_coord(data_mesh_nodal_coord,mesh_size)
    assignment_tolerance_x = _estimate_positive_spacing_tolerance(data_mesh_nodal_coord.nodal_coord_x[0, :])
    assignment_tolerance_y = _estimate_positive_spacing_tolerance(data_mesh_nodal_coord.nodal_coord_y[:, 0])

    # Debug: plot virtual fields mesh and data mesh overlaid on data points 
    if generate_plots:
        plot_virtual_fields_mesh(
            x,
            y,
            virtual_fields_mesh=vf_mesh_nodal_coord,
            data_mesh=data_mesh_nodal_coord,
            specimen_mask=specimen_mask,
            plot_data_points=True,
            show=True,
        )

    # Define counts
    n_datapoints = x.size # all datapoints including outside the specimen mask
    n_elem_rows = mesh_size[0]
    n_elem_cols = mesh_size[1]
    n_node_cols = n_elem_cols + 1
    n_node_rows = n_elem_rows + 1
    n_elements = n_elem_cols * n_elem_rows
    n_nodes = n_node_cols * n_node_rows

    # Define node IDs
    vf_mesh_node_ids = np.arange(0,n_nodes,1) # start, end (exclusive), step
    vf_mesh_node_ids = vf_mesh_node_ids.reshape(n_node_rows, n_node_cols)

    # Define nodes associated with each element ("connectivity matrix")
    #
    # For each element, node order is: lower-left, lower-right, upper-right, upper-left
    # vf_element_node_ids has:
    #   a row for each element (row 0 corresponds to element 0)columns
    #   NODES_PER_ELEMENT columns in the node ordered defined above (col 0 is lower-left node etc.)

    NODES_PER_ELEMENT= 4 # current implementation assumes linear quadrilateral elements
    vf_element_node_ids = np.empty((n_elements,NODES_PER_ELEMENT),dtype=np.int32) # element index, node index within element


    # Providing physical y coords increase downwards
    elem_rows_in_numbering_order = range(0, n_elem_rows)


    element_id = 0
    for elem_row in elem_rows_in_numbering_order:
        for elem_col in range(n_elem_cols):
            lower_left = vf_mesh_node_ids[elem_row, elem_col]
            lower_right = vf_mesh_node_ids[elem_row, elem_col + 1]
            upper_right = vf_mesh_node_ids[elem_row + 1, elem_col + 1]
            upper_left = vf_mesh_node_ids[elem_row + 1, elem_col]


            vf_element_node_ids[element_id, :] = [
                lower_left,
                lower_right,
                upper_right,
                upper_left,
            ]
            element_id += 1


    # Debug: plot virtual fields mesh and data mesh overlaid on data points with node and elem ids annotated
    if generate_plots:
        plot_virtual_fields_mesh(
            x,
            y,
            virtual_fields_mesh=vf_mesh_nodal_coord,
            data_mesh=data_mesh_nodal_coord,
            specimen_mask=specimen_mask,
            plot_data_points=True,
            show=True,
            node_ids=vf_mesh_node_ids,
            element_node_ids=vf_element_node_ids,
        )


    # == Define element associated with each datapoint ==

    # Flatten datapoints and specimen mask in row-major order
    x_points = x.ravel()
    y_points = y.ravel()
    specimen_mask_flat = specimen_mask.ravel().astype(bool)

    # Initialise data_point_element_ids defining element associated with each datapoint
    # Use -1 for points outside the specimen or not assigned.
    data_point_element_ids = np.full(n_datapoints, -1, dtype=np.int32)


    # Loop over virtual elements and assign contained data points.
    # Element connectivity local order is [lower_left, lower_right, upper_right, upper_left].
    for element_id in range(n_elements):
        element_node_ids = vf_element_node_ids[element_id]
        element_node_coords_x = vf_mesh_nodal_coord.nodal_coord_x.ravel()[element_node_ids]
        element_node_coords_y = vf_mesh_nodal_coord.nodal_coord_y.ravel()[element_node_ids]

        x_min = np.min(element_node_coords_x)
        x_max = np.max(element_node_coords_x)
        y_min = np.min(element_node_coords_y)
        y_max = np.max(element_node_coords_y)

        points_in_element = (
            specimen_mask_flat
            & (x_points >= x_min - assignment_tolerance_x)
            & (x_points <= x_max + assignment_tolerance_x)
            & (y_points >= y_min - assignment_tolerance_y)
            & (y_points <= y_max + assignment_tolerance_y)
        )

        data_point_element_ids[points_in_element] = element_id



    # One displacement DOF pair per node: [ux, uy]
    DOF_PER_NODE = 2
    node_dof_ids = np.empty((n_nodes, DOF_PER_NODE), dtype=np.int32)
    node_dof_ids[:, 0] = 2 * np.arange(n_nodes, dtype=np.int32)
    node_dof_ids[:, 1] = 2 * np.arange(n_nodes, dtype=np.int32) + 1

    # Element DOFs gathered from the element-node connectivity
    # Local order: [ux1, uy1, ux2, uy2, ux3, uy3, ux4, uy4]
    n_dof_per_element = NODES_PER_ELEMENT * DOF_PER_NODE
    element_dof_ids = np.empty((n_elements, n_dof_per_element), dtype=np.int32)
    for element_id in range(n_elements):
        element_node_ids = vf_element_node_ids[element_id]
        element_dof_ids[element_id, :] = node_dof_ids[element_node_ids, :].reshape(1,n_dof_per_element)


    specimen_point_indices = np.flatnonzero(specimen_mask_flat)
    n_specimen_points = specimen_point_indices.size
    n_total_dofs = DOF_PER_NODE * n_nodes

    unassigned_specimen_points = specimen_point_indices[data_point_element_ids[specimen_point_indices] < 0]
    if unassigned_specimen_points.size > 0:
        vf_node_x_1d = vf_mesh_nodal_coord.nodal_coord_x[0, :]
        vf_node_y_1d = vf_mesh_nodal_coord.nodal_coord_y[:, 0]
        fallback_cols = np.searchsorted(vf_node_x_1d, x_points[unassigned_specimen_points], side="right") - 1
        fallback_rows = np.searchsorted(vf_node_y_1d, y_points[unassigned_specimen_points], side="right") - 1
        fallback_cols = np.clip(fallback_cols, 0, n_elem_cols - 1)
        fallback_rows = np.clip(fallback_rows, 0, n_elem_rows - 1)
        data_point_element_ids[unassigned_specimen_points] = fallback_rows * n_elem_cols + fallback_cols

    # Initialise global shape function matrix (which computes displacements at datapoints from nodal displacements)
    global_shape_function_matrix = np.zeros((n_specimen_points, n_nodes), dtype=np.float64)
    # Initialise global strain-displacement matrix (which computes strains at datapoints from nodal displacements)
    global_strain_displacement_matrix = np.zeros((3 * n_specimen_points, n_total_dofs), dtype=np.float64)

    if np.any(data_point_element_ids[specimen_point_indices] < 0):
        raise ValueError(
            "Some specimen data points were not assigned to a virtual element."
        )

    vf_node_x_flat = vf_mesh_nodal_coord.nodal_coord_x.ravel()
    vf_node_y_flat = vf_mesh_nodal_coord.nodal_coord_y.ravel()

    # Assemble global shape-function and strain-displacement matrices 
    # TODO: vectorise this loop to increase speed (or even loop over elements rather than pts)
    for datapoint_row, datapoint_idx in enumerate(specimen_point_indices):  #does datapoint_row ever differ from datapoint_idx?
        
        # Gather x and y coordinates of datapoint
        point_coordinates = np.array(
            [x_points[datapoint_idx], y_points[datapoint_idx]],
            dtype=np.float64,
        )

        # Gather associated element id, nodal ids and nodal coordinates for datapoint
        element_id = data_point_element_ids[datapoint_idx]
        element_node_ids = vf_element_node_ids[element_id, :]
        # element_node_coordinates has shape n_nodes_per_elem x 2 (e.g. 4 x 2).
        element_node_coordinates = np.column_stack(
            (
                vf_node_x_flat[element_node_ids],
                vf_node_y_flat[element_node_ids],
            )
        )

        # Compute shape function values and strain-displacement matrix for the datapoint
        shape_function_values, strain_displacement_matrix = (
            _compute_point_shape_function_values_and_strain_displacement_matrices(
                point_coordinates,
                element_node_coordinates,
                use_nlgeom=use_nlgeom,
            )
        )

        # Populate global shape function matrix for this datapoint
        global_shape_function_matrix[datapoint_row, element_node_ids] = (
            shape_function_values
        )


        # Populate global strain-displacement matrix for this datapoint shape: float64[(3*n_specimen_points, 2*n_nodes)]
        element_dofs = element_dof_ids[element_id, :]
        global_strain_displacement_matrix[
            datapoint_row,
            element_dofs,
        ] = strain_displacement_matrix[0, :]
        global_strain_displacement_matrix[
            datapoint_row + n_specimen_points,
            element_dofs,
        ] = strain_displacement_matrix[1, :]
        global_strain_displacement_matrix[
            datapoint_row + 2 * n_specimen_points,
            element_dofs,
        ] = strain_displacement_matrix[2, :]

    # Impose virtual boundary conditions to get constrained strain-displacement matrix and active DOF ids
    constrained_strain_displacement_matrix, active_dof_ids, boundary_constraint_debug_info = (
        _compute_constrained_strain_displacement_matrix(
            global_strain_displacement_matrix,
            vf_mesh_node_ids,
            edge_conditions,
        )
    )


    # Debug: plot virtual fields mesh and data mesh overlaid on data points with node and elem ids annotated and B.Cs
    # fixed nodes are red squares, traction master nodes are green stars and slave nodes are green triangles. Shown in legend.
    if generate_plots:
        plot_virtual_fields_mesh(
            x,
            y,
            virtual_fields_mesh=vf_mesh_nodal_coord,
            data_mesh=data_mesh_nodal_coord,
            specimen_mask=specimen_mask,
            plot_data_points=True,
            show=True,
            node_ids=vf_mesh_node_ids,
            element_node_ids=vf_element_node_ids,
            node_dof_conditions=boundary_constraint_debug_info.node_dof_conditions,
        )

    # Compute pseudo-inverse of constrained strain-displacement matrix for use in virtual field generation
    strain_displacement_pseudoinverse = np.linalg.pinv(
        constrained_strain_displacement_matrix
    )

    return VirtualFieldsMesh(
        virtual_node_coordinates_x=vf_mesh_nodal_coord.nodal_coord_x,
        virtual_node_coordinates_y=vf_mesh_nodal_coord.nodal_coord_y,
        global_strain_displacement_matrix=global_strain_displacement_matrix,
        global_strain_displacement_matrix_pseudoinverse=strain_displacement_pseudoinverse,
        active_degrees_of_freedom=active_dof_ids,
        virtual_element_node_ids=vf_element_node_ids.astype(np.int64, copy=False),
        virtual_node_ids=vf_mesh_node_ids.astype(np.int64, copy=False),
        edge_conditions=edge_conditions,
        specimen_point_indices=specimen_point_indices.astype(np.int64, copy=False),
        global_shape_function_matrix=global_shape_function_matrix,
        data_point_virtual_element_ids=data_point_element_ids.astype(np.int64, copy=False),
    )


def generate_virtual_fields_from_mesh(
    reference_map: npt.NDArray[np.float64],
    virtual_fields_mesh: VirtualFieldsMesh,
    plot_fields: bool = False,
) -> GlobalVirtualFields:
    """Generate virtual fields that replicate the reference map as closely as possible, 
    while also enforcing any required virtual boundary conditions.
    
    Parameters
    ----------
    reference_map : ndarray
        Shape (n_timesteps, n_components, n_points_y, n_points_x).
        The reference strain map from which to generate virtual fields. e.g. stress sensitivity map
    virtual_fields_mesh : VirtualFieldsMesh
        The virtual fields mesh data containing global strain-displacement matrix etc. for 
        construction of virtual fields.
        
    Returns
    -------
    GlobalVirtualFields
        The generated virtual fields, including virtual strain fields over the specimen,
        virtual displacements at the specimen edges, and (optionally) full-field virtual displacements over the specimen.

    """

    # Gather dimensions
    n_timesteps, n_components, size_y, size_x = reference_map.shape
    n_measured_points = int(virtual_fields_mesh.specimen_point_indices.size)
    n_dofs = int(virtual_fields_mesh.global_strain_displacement_matrix.shape[1])
    # Initialise output arrays
    virtual_strain = np.full((n_timesteps, n_components, size_y, size_x), np.nan, dtype=np.float64)
    virtual_displacement_edge = np.zeros((n_timesteps, 2, 4), dtype=np.float64)
    virtual_displacement = np.full((n_timesteps, 2, size_y, size_x), np.nan, dtype=np.float64)

    # Loop over timesteps and generate virtual fields for each timestep's reference map
    for timestep in range(n_timesteps):

        # Get strain map for this timestep (shape: n_components, n_y, n_x)
        strain_map = reference_map[timestep]  # (n_components, n_y, n_x)
        # Extract strain values at valid (non-NaN) specimen data points (shape: n_components, n_specimen_points)
        strain_at_points = strain_map.reshape(n_components, -1)[
            :, virtual_fields_mesh.specimen_point_indices]
        # Flatten to 1D array (shape: n_components * n_specimen_points) where 
        # target_strain(0:n_specimen_points) correspond to component 0 
        # target_strain(n_specimen_points:2*n_specimen_points) corresponds to component 1 etc. 
        # This is the format expected by the global strain-displacement matrix
        target_strain = strain_at_points.ravel()

        # Set NaN values to zero. Will mask out later
        # TODO: This is a bit hacky and will bias the generated virtual fields
        # towards replicating zero strain at the NaN points, which may not be desirable.
        # Ideally would modify the least squares solve to ignore NaN values rather than 
        # setting to zero, but the pesudoinverse is built for full system.
        target_strain = np.nan_to_num(target_strain, nan=0.0)

        # Compute virtual displacement vector that replicates the target strain as closely as possible in a least squares sense
        virtual_displacement_vector = np.zeros(n_dofs, dtype=np.float64)
        virtual_displacement_vector[virtual_fields_mesh.active_degrees_of_freedom] = (
            virtual_fields_mesh.global_strain_displacement_matrix_pseudoinverse @ target_strain
        )

        # Impose virtual boundary conditions on the virtual displacement vector
        virtual_displacement_vector = _apply_edge_conditions(
            virtual_displacement_vector,
            virtual_fields_mesh.edge_conditions,
            virtual_fields_mesh.virtual_node_ids,
        )

        # Recompute virtual strain with the constrained virtual displacement vector to ensure BCs are satisifed
        reconstructed_virtual_strain = (
            virtual_fields_mesh.global_strain_displacement_matrix @ virtual_displacement_vector
        )

        # Map reconstructed virtual strains back to specimen grid (excluding NaN datapoints)
        for component in range(n_components):
            component_map = np.full(size_x * size_y, np.nan, dtype=np.float64)
            start = component * n_measured_points
            stop = (component + 1) * n_measured_points
            component_map[virtual_fields_mesh.specimen_point_indices] = (
                reconstructed_virtual_strain[start:stop]
            )
            virtual_strain[timestep, component, :, :] = component_map.reshape(
                (size_y, size_x),
            )

        # Map virtual displacements to specimen grid using global shape function matrix (excluding NaN datapoints)
        x_displacement = (
            virtual_fields_mesh.global_shape_function_matrix @ virtual_displacement_vector[0::2]
        )
        y_displacement = (
            virtual_fields_mesh.global_shape_function_matrix @ virtual_displacement_vector[1::2]
        )
        flat_x = np.full(size_x * size_y, np.nan, dtype=np.float64)
        flat_y = np.full(size_x * size_y, np.nan, dtype=np.float64)
        flat_x[virtual_fields_mesh.specimen_point_indices] = x_displacement
        flat_y[virtual_fields_mesh.specimen_point_indices] = y_displacement
        virtual_displacement[timestep, 0, :, :] = flat_x.reshape((size_y, size_x))
        virtual_displacement[timestep, 1, :, :] = flat_y.reshape((size_y, size_x))

        # Extract edge displacements by averaging virtual displacements at nodes along each edge of the virtual mesh
        virtual_displacement_edge[timestep, 0, 0] = np.mean(
            virtual_displacement_vector[2 * virtual_fields_mesh.virtual_node_ids[0, :]] # x displacement on upper (min y) edge
        )
        virtual_displacement_edge[timestep, 0, 1] = np.mean(
            virtual_displacement_vector[2 * virtual_fields_mesh.virtual_node_ids[:, 0]] # x displacement on left (min x) edge
        )
        virtual_displacement_edge[timestep, 0, 2] = np.mean(
            virtual_displacement_vector[2 * virtual_fields_mesh.virtual_node_ids[-1, :]] # x displacement on lower (max y) edge
        )
        virtual_displacement_edge[timestep, 0, 3] = np.mean(
            virtual_displacement_vector[2 * virtual_fields_mesh.virtual_node_ids[:, -1]] # x displacement on right (max x) edge
        )

        virtual_displacement_edge[timestep, 1, 0] = np.mean(
            virtual_displacement_vector[2 * virtual_fields_mesh.virtual_node_ids[0, :] + 1] # y displacement on upper (min y) edge
        )
        virtual_displacement_edge[timestep, 1, 1] = np.mean(
            virtual_displacement_vector[2 * virtual_fields_mesh.virtual_node_ids[:, 0] + 1] # y displacement on left (min x) edge
        )
        virtual_displacement_edge[timestep, 1, 2] = np.mean(
            virtual_displacement_vector[2 * virtual_fields_mesh.virtual_node_ids[-1, :] + 1] # y displacement on lower (max y) edge
        )
        virtual_displacement_edge[timestep, 1, 3] = np.mean(
            virtual_displacement_vector[2 * virtual_fields_mesh.virtual_node_ids[:, -1] + 1] # y displacement on right (max x) edge
        )

    # Debug: plot reference map, virtual displacement, and virtual strain for selected timesteps
    if plot_fields:
        _plot_generated_virtual_fields(
            reference_map,
            virtual_displacement,
            virtual_strain,
        )

    return GlobalVirtualFields(
        virtual_strain=virtual_strain,
        virtual_displacement_edge=virtual_displacement_edge,
        virtual_displacement=virtual_displacement,
    )

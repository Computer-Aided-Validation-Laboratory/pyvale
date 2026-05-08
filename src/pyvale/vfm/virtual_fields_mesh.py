import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt



@dataclass(slots=True)
class VirtualFieldsMesh:
    """Virtual-field helper mesh and the matrices derived from it."""

    # vXcoord 
    x: npt.NDArray[np.float64]
    # vYcoord 
    y: npt.NDArray[np.float64]
    # Bglob 
    b_glob: npt.NDArray[np.float64]
    # Binv 
    b_inv: npt.NDArray[np.float64]
    # actDofs 
    # TODO: rename to active_degrees_of_freedom
    act_dofs: npt.NDArray[np.int64]
    # virtConnectivity 
    virtual_element_connectivity: npt.NDArray[np.uint32]
    # vcoordGrid 
    virtual_elements: npt.NDArray[np.int64]
    # BC_settings 
    boundary_condition_settings: npt.NDArray[np.uint32]
    # indexlist 
    # TODO: should we use the specimen mask here instead?
    specimen_mask: npt.NDArray[np.uint32]
    # NGlob 
    # TODO: I think this is some kind of global shape function, probably worth a rename
    n_glob: npt.NDArray[np.float64]
    # YDownDecreaseFlag 
    # ptElemAss
    virtual_element_point_mapping: npt.NDArray[np.uint32]
    # freeDof
    # TODO: rename to free_degrees_of_freedom
    free_dof: npt.NDArray[np.int64]



@dataclass(slots=True)
class MeshNodalCoordinates:
    """Nodal coordinates defining a mesh."""
    nodal_coord_x: npt.NDArray[np.float64]   # shape (num_points_y + 1, num_points_x + 1)
    nodal_coord_y: npt.NDArray[np.float64]   # shape (num_points_y + 1, num_points_x + 1)



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



def _generate_data_mesh_nodal_coord(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> MeshNodalCoordinates:
    """Construct the fine mesh of data-point elements from centroid coordinates.

    Assumes:
    - `x` and `y` are 2D centroid grids with shape (num_points_y, num_points_x)
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

    x_ext = _extend_centroid_grid(x)
    y_ext = _extend_centroid_grid(y)

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
            virtual_grid_shape = virtual_fields_mesh.virtual_elements.shape
            virtual_grid_x = virtual_fields_mesh.x.reshape(virtual_grid_shape, order="F")
            virtual_grid_y = virtual_fields_mesh.y.reshape(virtual_grid_shape, order="F")
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

    if node_ids is not None:
        for row in range(node_ids.shape[0]):
            for col in range(node_ids.shape[1]):
                ax.text(
                    virtual_grid_x[row, col],
                    virtual_grid_y[row, col],
                    str(node_ids[row, col]),
                    color="blue",
                    fontsize=8,
                    ha="center",
                    va="center",
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

    # Note: np.newaxis is used to broadcast the 1D arrays into 2D. Python then broadcasts arrays to  for pairwise distance calculation
    # x_distances = np.abs(
    #     vf_mesh_nodal_coord_x_1d[:, np.newaxis]
    #     - data_mesh_nodal_coord_x_1d[np.newaxis, :]
    # )

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
    # y_distances = np.abs(
    #     vf_mesh_nodal_coord_y_1d[:, np.newaxis]
    #     - data_mesh_nodal_coord_y_1d[np.newaxis, :]
    # )

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


def _evaluate_linear_shape_functions(
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

    shape_functions = np.array(
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

    return shape_functions, shape_function_derivatives


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
    the iterative method is more general and readible and the performance is not a 
    concern given the small number of points we need to evaluate this for.

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
        shape_functions, shape_function_derivatives = _evaluate_linear_shape_functions(xi,eta)
        # Evaluate global coordinates for the current identified shape function matrix
        mapped_coordinates = shape_functions @ element_node_coordinates
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
    """Assemble the strain-displacement matrix for a 4-node quad."""

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


def _compute_point_shape_and_strain_matrices(
    point_coordinates: npt.NDArray[np.float64],
    element_node_coordinates: npt.NDArray[np.float64],
    use_nlgeom: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate shape functions and the strain-displacement matrix at one point."""

    # Compute local element coordinates for datapoint
    xi, eta = _compute_local_element_coordinates(point_coordinates, element_node_coordinates)

    # Compute shape functions and derivatives for datapoint (could this use derivates from above to save recalc?)
    shape_functions, shape_function_derivatives = _evaluate_linear_shape_functions(
        xi,
        eta,
    )

    # Compute derivatives of shape functions
    jacobian = shape_function_derivatives.T @ element_node_coordinates
    shape_function_gradients_global = np.linalg.solve(
        jacobian.T,
        shape_function_derivatives.T,
    ).T

    # Assemble strain displacement matrix
    strain_displacement_matrix = _assemble_strain_displacement_matrix(
        shape_function_gradients_global,
        use_nlgeom=use_nlgeom,
    )

    return strain_displacement_matrix, shape_functions

def generate_virtual_fields_mesh(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    specimen_mask: npt.NDArray[np.uint32],
    boundary_conditions: dict[str, str],
    mesh_size: npt.NDArray[np.uint32],
):
    """Construct a mesh over the test data to be used for virtual field generation.
    
    Parameters
    ----------
    x : ndarray
        Shape (num_points_y, num_points_x).
        The x coordinates of the measurement points.
    y : ndarray
        Shape (num_points_y, num_points_x).
        The y coordinates of the measurement points.
    specimen_mask : ndarray of bool
        Shape (num_points_y, num_points_x).
        A mask indicating the specimen region (True for points inside the specimen).
    boundary_conditions : BoundaryConditions
        The boundary conditions associated with the test data. 
        Defined as a dictionary with keys 'x' and 'y', each mapping to a list of 4 strings corresponding to the 4 edges of the specimen.
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
    # plot_virtual_fields_mesh(
    #     x,
    #     y,
    #     data_mesh=data_mesh_nodal_coord,
    #     specimen_mask=specimen_mask,
    #     plot_data_points=True,
    # )

    # Construct coarse virtual mesh (of user-defined size) by snapping a regular grid onto the data point element edges
    vf_mesh_nodal_coord = _generate_vf_mesh_nodal_coord(data_mesh_nodal_coord,mesh_size)

    # Debug: plot virtual fields mesh and data mesh overlaid on data points 
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
    # CONVENTION (Providing physical y coords increase downwards):
    # left = min(x)
    # right = max(x)
    # lower = min(y)
    # upper = max(y)
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
            & (x_points >= x_min)
            & (x_points <= x_max)
            & (y_points >= y_min)
            & (y_points <= y_max)
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

        # 
        strain_displacement_matrix, shape_functions = (
            _compute_point_shape_and_strain_matrices(
                point_coordinates,
                element_node_coordinates,
                use_nlgeom=False,
            )
        )

        global_shape_function_matrix[datapoint_row, element_node_ids] = (
            shape_functions
        )

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

    constrained_strain_displacement_matrix = None
    strain_displacement_pseudoinverse = None






    # node_dof_ids: int32[(n_nodes, 2)] where columns are [ux_dof, uy_dof]
    # element_dof_ids: int32[(n_elements, 8)]
    # global_shape_function_matrix: float64[(n_specimen_points, n_nodes)]
    # global_strain_displacement_matrix: float64[(3*n_specimen_points, 2*n_nodes)]
    # constrained_strain_displacement_matrix: float64[(3*n_specimen_points, n_active_dofs)]
    # strain_displacement_pseudoinverse: float64[(n_active_dofs, 3*n_specimen_points)]




    return None






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




@dataclass(slots=True)
class GlobalVirtualFields:
    """Virtual strains and edge displacements generated from one sensitivity map."""

    virtual_strain: npt.NDArray[np.float64]
    edge_displacement: npt.NDArray[np.float64]
    full_displacement: npt.NDArray[np.float64]




def generate_vf_from_mesh(
    reference_map: npt.NDArray[np.float64],
    virtual_fields_mesh: VirtualFieldsMesh,
) -> GlobalVirtualFields:
    num_timesteps, _, size_y, size_x = reference_map.shape
    num_measured_points = int(virtual_fields_mesh.indices.size)
    num_dofs = int(virtual_fields_mesh.b_glob.shape[1])

    virtual_strain = np.full((num_timesteps, 3, size_y, size_x), np.nan, dtype=np.float64)
    edge_displacement = np.zeros((num_timesteps, 2, 4), dtype=np.float64)
    full_displacement = np.full((num_timesteps, 2, size_y, size_x), np.nan, dtype=np.float64)

    for timestep in range(num_timesteps):
        target_strain = np.concatenate(
            [
                reference_map[timestep, 0, :, :].flatten(order="F")[virtual_fields_mesh.indices],
                reference_map[timestep, 1, :, :].flatten(order="F")[virtual_fields_mesh.indices],
                reference_map[timestep, 2, :, :].flatten(order="F")[virtual_fields_mesh.indices],
            ]
        )
        target_strain = np.nan_to_num(target_strain, nan=0.0)

        virtual_displacement_vector = np.zeros(num_dofs, dtype=np.float64)
        virtual_displacement_vector[virtual_fields_mesh.act_dofs] = (
            virtual_fields_mesh.b_inv @ target_strain
        )
        virtual_displacement_vector = _apply_boundary_conditions(
            virtual_displacement_vector,
            virtual_fields_mesh.boundary_condition_settings,
            virtual_fields_mesh.virtual_elements,
        )

        reconstructed_virtual_strain = (
            virtual_fields_mesh.b_glob @ virtual_displacement_vector
        )

        for component in range(3):
            component_map = np.full(size_x * size_y, np.nan, dtype=np.float64)
            start = component * num_measured_points
            stop = (component + 1) * num_measured_points
            component_map[virtual_fields_mesh.indices] = reconstructed_virtual_strain[start:stop]
            virtual_strain[timestep, component, :, :] = component_map.reshape(
                (size_y, size_x),
                order="F",
            )

        x_displacement = virtual_fields_mesh.n_glob @ virtual_displacement_vector[0::2]
        y_displacement = virtual_fields_mesh.n_glob @ virtual_displacement_vector[1::2]

        flat_x = np.full(size_x * size_y, np.nan, dtype=np.float64)
        flat_y = np.full(size_x * size_y, np.nan, dtype=np.float64)
        flat_x[virtual_fields_mesh.indices] = x_displacement
        flat_y[virtual_fields_mesh.indices] = y_displacement
        full_displacement[timestep, 0, :, :] = flat_x.reshape((size_y, size_x), order="F")
        full_displacement[timestep, 1, :, :] = flat_y.reshape((size_y, size_x), order="F")

        edge_displacement[timestep, 0, 0] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[0, :]])
        edge_displacement[timestep, 0, 1] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[:, 0]])
        edge_displacement[timestep, 0, 2] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[-1, :]])
        edge_displacement[timestep, 0, 3] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[:, -1]])

        edge_displacement[timestep, 1, 0] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[0, :] + 1])
        edge_displacement[timestep, 1, 1] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[:, 0] + 1])
        edge_displacement[timestep, 1, 2] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[-1, :] + 1])
        edge_displacement[timestep, 1, 3] = np.mean(virtual_displacement_vector[2 * virtual_fields_mesh.virtual_elements[:, -1] + 1])

    return SensitivityBasedVirtualFields(
        virtual_strain=virtual_strain,
        edge_displacement=edge_displacement,
        full_displacement=full_displacement,
    )



def _apply_boundary_conditions(
    virtual_displacement: npt.NDArray[np.float64],
    settings: npt.NDArray[np.uint32],
    virtual_elements: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    updated = virtual_displacement.copy()

    for edge in range(4):
        if edge == 0:
            edge_nodes = virtual_elements[0, :]
            master_node = virtual_elements[0, 0]
            slave_nodes = edge_nodes[1:]
        elif edge == 1:
            edge_nodes = virtual_elements[:, 0]
            master_node = virtual_elements[0, 0]
            slave_nodes = edge_nodes[1:]
        elif edge == 2:
            edge_nodes = virtual_elements[-1, :]
            master_node = virtual_elements[-1, -1]
            slave_nodes = edge_nodes[:-1]
        else:
            edge_nodes = virtual_elements[:, -1]
            master_node = virtual_elements[-1, -1]
            slave_nodes = edge_nodes[:-1]

        edge_dofs_x = 2 * edge_nodes
        edge_dofs_y = edge_dofs_x + 1
        master_dof_x = 2 * master_node
        master_dof_y = master_dof_x + 1
        slave_dofs_x = 2 * slave_nodes
        slave_dofs_y = slave_dofs_x + 1

        if settings[0, edge] == 1:
            updated[edge_dofs_x] = 0.0
        elif settings[0, edge] == 2:
            updated[slave_dofs_x] = updated[master_dof_x]

        if settings[1, edge] == 1:
            updated[edge_dofs_y] = 0.0
        elif settings[1, edge] == 2:
            updated[slave_dofs_y] = updated[master_dof_y]

    return updated

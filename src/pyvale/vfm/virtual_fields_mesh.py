import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt



@dataclass(slots=True)
class VirtualFieldsMesh:
    """Virtual-field helper mesh and the matrices derived from it."""

    # vXcoords 
    x: npt.NDArray[np.float64]
    # vYcoords 
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
    # vCoordsGrid 
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
class DataPointMesh:
    """Fine mesh formed by the corners of the measured data-point areas."""
    nodal_coords_x: npt.NDArray[np.float64]   # shape (num_points_y + 1, num_points_x + 1)
    nodal_coords_y: npt.NDArray[np.float64]   # shape (num_points_y + 1, num_points_x + 1)



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
) -> DataPointMesh:
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

    nodal_coords_x = 0.25 * (
        x_ext[:-1, :-1]
        + x_ext[:-1, 1:]
        + x_ext[1:, :-1]
        + x_ext[1:, 1:]
    )
    nodal_coords_y = 0.25 * (
        y_ext[:-1, :-1]
        + y_ext[:-1, 1:]
        + y_ext[1:, :-1]
        + y_ext[1:, 1:]
    )

    return DataPointMesh(
        nodal_coords_x=nodal_coords_x,
        nodal_coords_y=nodal_coords_y,
    )


def plot_data_mesh(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    data_mesh: DataPointMesh,
    specimen_mask: npt.NDArray[np.bool_] | None = None,
):
    """Debug plot of the data-point mesh overlaid on the measurement points."""

    import matplotlib.pyplot as plt

    if specimen_mask is None:
        specimen_mask = np.ones(x.shape, dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 4))

    for row in range(data_mesh.nodal_coords_x.shape[0]):
        ax.plot(
            data_mesh.nodal_coords_x[row, :],
            data_mesh.nodal_coords_y[row, :],
            color="black",
            linewidth=0.8,
        )
    for col in range(data_mesh.nodal_coords_x.shape[1]):
        ax.plot(
            data_mesh.nodal_coords_x[:, col],
            data_mesh.nodal_coords_y[:, col],
            color="black",
            linewidth=0.8,
        )

    ax.scatter(
        x[specimen_mask],
        y[specimen_mask],
        s=8,
        marker="x",
        color="red",
        linewidths=0.5,
    )

    ax.set_title("Data-point mesh")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")


    # ax.invert_yaxis()

    fig.tight_layout()
    plt.show()




def _generate_vf_mesh_nodal_coord(
    data_mesh:DataPointMesh,
    mesh_size:npt.NDArray[np.uint32],  # column_count, row_count
) -> VirtualFieldsMesh:
    """Snap user-defined virtual mesh onto the measured x/y grid lines."""
    
    if mesh_size.shape != (2,):
        raise ValueError("mesh_size must contain [column_count, row_count].")
    if mesh_size[0] < 1 or mesh_size[1] < 1:
        raise ValueError("mesh_size must be at least [1, 1].")
    
    # Check data mesh nodal y coordinates are constant along rows (within 0.1% of mean row value)
    if not np.all(np.isclose(data_mesh.nodal_coords_y.mean(axis=1), data_mesh.nodal_coords_y[:, 0], rtol=0.001)):
        raise ValueError("Data mesh nodal y coordinates are not constant along rows. Current implementation assumes they should be.")
    # Check data mesh nodal x coordinates are constant along columns (within 0.1% of mean row value)
    if not np.all(np.isclose(data_mesh.nodal_coords_x.mean(axis=0), data_mesh.nodal_coords_x[0, :], rtol=0.001)):
        raise ValueError("Data mesh nodal x coordinates are not constant along columns. Current implementation assumes they should be.")

    # Extract the structured 1D nodal coordinate lines from the data mesh
    data_mesh_nodal_coord_x_1d = data_mesh.nodal_coords_x[0, :]
    data_mesh_nodal_coord_y_1d = data_mesh.nodal_coords_y[:, 0]

    # Initialise vf mesh nodal coords by linearly spacing along specimen dimensions, ensuring one on each edge
    vf_mesh_nodal_coord_x_1d=np.linspace(data_mesh_nodal_coord_x_1d[0],data_mesh_nodal_coord_x_1d[-1],mesh_size[0]+1)
    vf_mesh_nodal_coord_y_1d=np.linspace(data_mesh_nodal_coord_y_1d[0],data_mesh_nodal_coord_y_1d[-1],mesh_size[1]+1)

    # Compute x distance from vf mesh nodes to data mesh nodes
    x_distances = np.abs(
        vf_mesh_nodal_coord_x_1d[:, np.newaxis]
        - data_mesh_nodal_coord_x_1d[np.newaxis, :]
    )

    # Closest data mesh nodal x coordinate idx for each vf mesh node
    closest_x_idx = np.argmin(x_distances, axis=1)

    # Compute y distance from vf mesh nodes to data mesh nodes
    y_distances = np.abs(
        vf_mesh_nodal_coord_y_1d[:, np.newaxis]
        - data_mesh_nodal_coord_y_1d[np.newaxis, :]
    )

    # Closest data mesh nodal y coordinate idx for each vf mesh node
    closest_y_idx = np.argmin(y_distances, axis=1)

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

    # Expand vf mesh nodal coords to 2D grid
    vf_mesh_nodal_coord_x, vf_mesh_nodal_coord_y = np.meshgrid(
        vf_mesh_nodal_coord_x_1d,
        vf_mesh_nodal_coord_y_1d,
    )
    
    return VirtualFieldsMesh(
        nodal_coords_x=vf_mesh_nodal_coord_x,
        nodal_coords_y=vf_mesh_nodal_coord_y,
    )

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
        The number of virtual elements in the x and y directions, respectively.   

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
    
    # Construct fine mesh around data points
    data_mesh=_generate_data_mesh_nodal_coord(x,y)

    # Debug: plot data mesh overlaid on data points 
    # plot_data_mesh(x, y, data_mesh, specimen_mask)

    vf_mesh = _generate_vf_mesh_nodal_coord(data_mesh,mesh_size)


    num_virtual_elements = mesh_x.size * mesh_y.size

    virtual_elements = np.arange(num_virtual_elements).reshape(
        (mesh_y.size, mesh_x.size), order="F"
    )

    top_left = virtual_elements[0:-1, 0:-1].flatten(order="F")
    bottom_left = virtual_elements[1:, 0:-1].flatten(order="F")
    bottom_right = virtual_elements[1:, 1:].flatten(order="F")
    top_right = virtual_elements[0:-1, 1:].flatten(order="F")

    virtual_element_connectivity = np.stack(
        (top_left, bottom_left, bottom_right, top_right),
        axis=0
    ).astype(np.uint32)

    grid_x, grid_y = np.meshgrid(x, y)

    mesh_grid_x, mesh_grid_y = np.meshgrid(mesh_x, mesh_y)
    mesh_grid_x = mesh_grid_x.flatten(order="F")
    mesh_grid_y = mesh_grid_y.flatten(order="F")

    elem_x = mesh_grid_x[virtual_element_connectivity]
    elem_y = mesh_grid_y[virtual_element_connectivity]

    xmin = elem_x.min(axis=0)
    xmax = elem_x.max(axis=0)
    ymin = elem_y.min(axis=0)
    ymax = elem_y.max(axis=0)

    x_points = grid_x.flatten(order="F")[specimen_mask]
    y_points = grid_y.flatten(order="F")[specimen_mask]

    inside = (
        (x_points[:, None] >= xmin[None, :]) &
        (x_points[:, None] <= xmax[None, :]) &
        (y_points[:, None] >= ymin[None, :]) &
        (y_points[:, None] <= ymax[None, :])
    )

    virtual_element_point_mapping = inside.argmax(axis=1)

    num_measured_points = specimen_mask.size
    degrees_of_freedom = 2 * num_virtual_elements

    b_glob = np.zeros((3 * num_measured_points, degrees_of_freedom))
    n_glob = np.zeros((num_measured_points, num_virtual_elements))

    transform_matrix = np.array([
        [-1,  1, -1,  1],
        [-1, -1,  1,  1],
        [ 1, -1, -1,  1]
    ])

    for p in range(num_measured_points):

        elem_index = virtual_element_point_mapping[p]
        connected = virtual_element_connectivity[:, elem_index]

        x_dofs = 2 * connected
        y_dofs = 2 * connected + 1
        p_dofs = np.vstack((x_dofs, y_dofs)).reshape(-1, order="F")

        x_point = x_points[p]
        y_point = y_points[p]

        rows = connected % mesh_y.size
        cols = connected // mesh_y.size

        coords = np.column_stack((mesh_x[cols], mesh_y[rows]))

        a_matrix = transform_matrix.dot(coords)

        d_vector = np.array([
            4 * x_point - np.sum(coords[:, 0]),
            4 * y_point - np.sum(coords[:, 1])
        ])

        d1, d2 = d_vector
        a1, a2 = a_matrix[0]
        b1, b2 = a_matrix[1]
        c1, c2 = a_matrix[2]

        ab = a1*b2 - a2*b1
        ac = a1*c2 - a2*c1
        ad = a1*d2 - a2*d1
        cb = c1*b2 - c2*b1
        da = d1*a2 - d2*a1
        dc = d1*c2 - d2*c1
        ba = b1*a2 - b2*a1
        db = d1*b2 - d2*b1
        bd = b1*d2 - b2*d1
        bc = b1*c2 - b2*c1

        if (a1*a2*ab*ac != 0) or (a1 == 0 and a2*c1 != 0) or (a2 == 0 and a1*b2 != 0):
            xi_candidates = np.roots([ab, cb+da, dc])
            xi = xi_candidates[np.abs(xi_candidates) < 1][0]
            eta = (ad + ba*xi) / ac
        elif a1*a2 != 0 and ab == 0:
            xi = (a1*dc) / (b1*ac + a1*ad)
            eta = ad / ac
        elif a1*a2 != 0 and ac == 0:
            xi = ad / ab
            eta = (a1*db) / (c1*ab + a1*ad)
        else:
            xi = dc / (a1*d2 + bc)
            eta = bd / (a2*d1 + bc)

        n = 0.25 * np.array([
            (1-xi)*(1+eta),
            (1-xi)*(1-eta),
            (1+xi)*(1-eta),
            (1+xi)*(1+eta)
        ])

        dn_dxi = np.array([
            [-0.25*(1+eta),  0.25*(1-xi)],
            [-0.25*(1-eta), -0.25*(1-xi)],
            [ 0.25*(1-eta), -0.25*(1+xi)],
            [ 0.25*(1+eta),  0.25*(1+xi)]
        ])

        jacobian = dn_dxi.T.dot(coords)
        dn_dx = np.linalg.solve(jacobian.T, dn_dxi.T).T

        b = np.array([
            [dn_dx[0,0], 0, dn_dx[1,0], 0, dn_dx[2,0], 0, dn_dx[3,0], 0],
            [0, dn_dx[0,1], 0, dn_dx[1,1], 0, dn_dx[2,1], 0, dn_dx[3,1]],
            [dn_dx[0,1], dn_dx[0,0], dn_dx[1,1], dn_dx[1,0],
             dn_dx[2,1], dn_dx[2,0], dn_dx[3,1], dn_dx[3,0]]
        ])

        b_glob[p, p_dofs] = b[0]
        b_glob[p + num_measured_points, p_dofs] = b[1]
        b_glob[p + 2*num_measured_points, p_dofs] = b[2]

        n_glob[p, connected] = n


    # boundary condition stuff
    b_bar = b_glob.copy()

    # TODO: should types be uint?
    bc_fixed = np.array([], dtype=np.int64)
    bc_slaves = np.array([], dtype=np.int64)
    bc_masters = np.array([], dtype=np.int64)

    for edge in range(4):
        # Get edge DOFs
        if edge == 0:  # Top edge
            edge_dofs_x = 2 * virtual_elements[0, :]
            edge_dofs_y = edge_dofs_x + 1
            master_dofs = np.array([2 * virtual_elements[0, 0],
                                    2 * virtual_elements[0, 0] + 1])
            slave_dofs = np.vstack((edge_dofs_x[1:], edge_dofs_y[1:]))
        elif edge == 1:  # Left edge
            edge_dofs_x = (2 * virtual_elements[:, 0]).T
            edge_dofs_y = edge_dofs_x + 1
            master_dofs = np.array([2 * virtual_elements[0, 0],
                                    2 * virtual_elements[0, 0] + 1])
            slave_dofs = np.vstack((edge_dofs_x[1:], edge_dofs_y[1:]))
        elif edge == 2:  # Bottom edge
            edge_dofs_x = 2 * virtual_elements[-1, :]
            edge_dofs_y = edge_dofs_x + 1
            master_dofs = np.array([2 * virtual_elements[-1, -1],
                                    2 * virtual_elements[-1, -1] + 1])
            slave_dofs = np.vstack((edge_dofs_x[:-1], edge_dofs_y[:-1]))
        else:  # Right edge
            edge_dofs_x = (2 * virtual_elements[:, -1]).T
            edge_dofs_y = edge_dofs_x + 1
            master_dofs = np.array([2 * virtual_elements[-1, -1],
                                    2 * virtual_elements[-1, -1] + 1])
            slave_dofs = np.vstack((edge_dofs_x[:-1], edge_dofs_y[:-1]))

        # X-direction BCs
        if boundary_conditions[0, edge] == 1:  # fixed
            bc_fixed = np.concatenate((bc_fixed, edge_dofs_x))
        elif boundary_conditions[0, edge] == 2:  # constant
            b_bar[:, master_dofs[0]] += np.sum(b_bar[:, slave_dofs[0, :]], axis=1)
            bc_slaves = np.concatenate((bc_slaves, slave_dofs[0, :]))
            bc_masters = np.concatenate((bc_masters, [master_dofs[0]]))

        # Y-direction BCs
        if boundary_conditions[1, edge] == 1:  # fixed
            bc_fixed = np.concatenate((bc_fixed, edge_dofs_y))
        elif boundary_conditions[1, edge] == 2:  # constant
            b_bar[:, master_dofs[1]] += np.sum(b_bar[:, slave_dofs[1, :]], axis=1)
            bc_slaves = np.concatenate((bc_slaves, slave_dofs[1, :]))
            bc_masters = np.concatenate((bc_masters, [master_dofs[1]]))

    # Remove duplicates
    bc_fixed = np.unique(bc_fixed)
    bc_slaves = np.unique(bc_slaves)
    bc_masters = np.unique(bc_masters)

    # Check for conflicts
    conflict = np.intersect1d(bc_masters, bc_fixed)
    if conflict.size > 0:
        raise ValueError(
            "Incompatible Boundary Conditions, adjacent boundary "
            "conditions cannot be both fixed/uniform"
        )

    # Remove BCs from b_bar
    remove_cols = np.unique(np.concatenate((bc_fixed, bc_slaves)))
    b_bar = np.delete(b_bar, remove_cols.astype(int), axis=1)

    # Active DOFs
    total_dofs = 2 * virtual_elements.size
    act_dofs = np.setdiff1d(np.arange(total_dofs), remove_cols)

    # Calculate pseudo-inverse of modified global strain-displacement matrix
    b_inv = np.linalg.pinv(b_bar)  # Bbar is the modified global matrix

    # Compute DOFs for x and y directions
    xDofGrid = virtual_elements * 2      # 0-based nodes: x DOFs
    yDofGrid = virtual_elements * 2 + 1  # y DOFs come right after x

    # Initialize all DOFs as free
    free_dof = np.zeros((xDofGrid.size, yDofGrid.size), dtype=np.uint32)
    free_dof = np.hstack((xDofGrid, yDofGrid)).flatten(order="F")

    # Loop over the four edges
    for e in range(4):
        # Only consider edges that are fixed or constant
        if boundary_conditions[0, e] == 1 or boundary_conditions[0, e] == 2:
            if e == 0:  # Top e (first row)
                not_free_dof_x = xDofGrid[0, :]
                not_free_dof_y = yDofGrid[0, :]
            elif e == 1:  # Left e (first column)
                not_free_dof_x = xDofGrid[:, 0]
                not_free_dof_y = yDofGrid[:, 0]
            elif e == 2:  # Bottom e (last row)
                not_free_dof_x = xDofGrid[-1, :]
                not_free_dof_y = yDofGrid[-1, :]
            else:  # Right edge (last column)
                not_free_dof_x = xDofGrid[:, -1]
                not_free_dof_y = yDofGrid[:, -1]

            # Remove the DOFs on this edge from the free DOFs
            # Use np.isin to mimic MATLAB's ismember
            not_free_dof = np.concatenate((not_free_dof_x, not_free_dof_y))  # 1D array of all DOFs to remove
            mask = ~np.isin(free_dof, not_free_dof).flatten(order="F")
            free_dof = free_dof[mask]
    
    return VirtualFieldsMesh(
        mesh_grid_x,
        mesh_grid_y,
        b_glob,
        b_inv,
        act_dofs,
        virtual_element_connectivity,
        virtual_elements,
        boundary_conditions,
        specimen_mask,
        n_glob,
        virtual_element_point_mapping,
        free_dof
    )






def plot_virtual_fields_mesh(
    data_x: npt.NDArray[np.float64],
    data_y: npt.NDArray[np.float64],
    specimen_mask: npt.NDArray[np.bool_],
    virtual_fields_mesh: VirtualFieldsMesh,
    output_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Plot measured points, data-element grid lines, and the virtual mesh."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    x_coords = np.nanmean(data_x, axis=0)
    y_coords = np.nanmean(data_y, axis=1)

    data_grid_x, data_grid_y = np.meshgrid(
        _compute_data_element_edges(x_coords),
        _compute_data_element_edges(y_coords),
    )
    virtual_grid_shape = virtual_fields_mesh.virtual_elements.shape
    virtual_grid_x = virtual_fields_mesh.x.reshape(virtual_grid_shape, order="F")
    virtual_grid_y = virtual_fields_mesh.y.reshape(virtual_grid_shape, order="F")

    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_grid_lines(ax, data_grid_x, data_grid_y, color="#49dbe6", linewidth=0.7)
    _plot_grid_lines(ax, virtual_grid_x, virtual_grid_y, color="black", linewidth=0.9)

    ax.scatter(
        data_x[specimen_mask],
        data_y[specimen_mask],
        s=5,
        marker="x",
        linewidths=0.4,
        color="red",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            markersize=6,
            markeredgewidth=0.8,
            color="red",
            label="Data points",
        ),
        Line2D([0], [0], color="#49dbe6", linewidth=1.0, label="Data elements"),
        Line2D([0], [0], color="black", linewidth=1.0, label="Virtual elements"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    ax.set_title("SBVF Virtual Mesh")
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
    plt.close(fig)
    return saved_path


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

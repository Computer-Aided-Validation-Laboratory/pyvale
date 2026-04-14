import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


# TODO: are these appropriate names?
class EBoundaryConditionSetting(enum.Enum):
    Free = enum.auto()
    Fixed = enum.auto()
    Constant = enum.auto()


# TODO: maybe rename to stuff like min y edge, max x edge etc
# Using edge numbering convention from globaloptions.m
class EEdge(enum.Enum):
    Top = 0
    Bottom = 2
    Left = 1
    Right = 3


# TODO: assuming x and y are the only dofs for now, may need to be expanded
# for non linear geometries
@dataclass(slots=True)
class BoundaryConditionSettings():
    x: dict[EEdge, EBoundaryConditionSetting]
    y: dict[EEdge, EBoundaryConditionSetting]


# TODO: discuss type decisions e.g. using uint32s
# TODO: do we need y decrease flag
# TODO: naming
# TODO: should we return 1d mesh or the meshgrid?
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
    indices: npt.NDArray[np.uint32]
    # NGlob 
    # TODO: I think this is some kind of global shape function, probably worth a rename
    n_glob: npt.NDArray[np.float64]
    # YDownDecreaseFlag 
    # ptElemAss
    virtual_element_point_mapping: npt.NDArray[np.uint32]
    # freeDof
    # TODO: rename to free_degrees_of_freedom
    free_dof: npt.NDArray[np.int64]


def _compute_data_element_edges(
    coords: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build element-edge coordinates from a 1D centroid coordinate vector."""

    if coords.ndim != 1:
        raise ValueError("Expected a 1D coordinate vector.")
    if coords.size < 2:
        raise ValueError("Need at least two coordinates to build element edges.")

    edges = np.empty(coords.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (coords[:-1] + coords[1:])
    edges[0] = coords[0] - 0.5 * (coords[1] - coords[0])
    edges[-1] = coords[-1] + 0.5 * (coords[-1] - coords[-2])
    return edges


# x and y and DIC centroids
# Expect x and y to be 1d coordinate arrays without nans
# Assuming that 0,0 in index space is top left in coord space
# TODO: add return type
# TODO: indices coming from matlab test data will be 1 index rather than zero indexed
def generate_virtual_fields_mesh(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    indices: npt.NDArray[np.uint32],
    settings: npt.NDArray[np.uint32],
    mesh_size: npt.NDArray[np.uint32] | None = None,
    # nan_mask: npt.NDArray[np.float64], # TODO: is this needed?
):
    """Build the reduced virtual-fields mesh used by the SBVF routines."""
    if mesh_size is None:
        mesh_size = np.array([15, 15], dtype=np.uint32)

    mesh_x, mesh_y = generate_mesh(x, y, mesh_size)

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

    x_points = grid_x.flatten(order="F")[indices]
    y_points = grid_y.flatten(order="F")[indices]

    inside = (
        (x_points[:, None] >= xmin[None, :]) &
        (x_points[:, None] <= xmax[None, :]) &
        (y_points[:, None] >= ymin[None, :]) &
        (y_points[:, None] <= ymax[None, :])
    )

    virtual_element_point_mapping = inside.argmax(axis=1)

    num_measured_points = indices.size
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
        if settings[0, edge] == 1:  # fixed
            bc_fixed = np.concatenate((bc_fixed, edge_dofs_x))
        elif settings[0, edge] == 2:  # constant
            b_bar[:, master_dofs[0]] += np.sum(b_bar[:, slave_dofs[0, :]], axis=1)
            bc_slaves = np.concatenate((bc_slaves, slave_dofs[0, :]))
            bc_masters = np.concatenate((bc_masters, [master_dofs[0]]))

        # Y-direction BCs
        if settings[1, edge] == 1:  # fixed
            bc_fixed = np.concatenate((bc_fixed, edge_dofs_y))
        elif settings[1, edge] == 2:  # constant
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
        if settings[0, e] == 1 or settings[0, e] == 2:
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
        settings,
        indices,
        n_glob,
        virtual_element_point_mapping,
        free_dof
    )





# TODO: add return type
# Generate a grid from DIC centroids and create a mesh which conforms to those
# grid lines
def generate_mesh(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mesh_size: npt.NDArray[np.uint32]
):
    """Snap a coarse virtual mesh onto the measured x/y grid lines."""
    grid_x = _compute_data_element_edges(x)
    grid_y = _compute_data_element_edges(y)

    mesh_size_x = mesh_size[0]
    mesh_size_y = mesh_size[1]

    mesh_x = np.zeros(mesh_size_x + 1)
    mesh_y = np.zeros(mesh_size_y + 1)

    mesh_nodes_x = np.linspace(grid_x[0], grid_x[-1], mesh_size_x + 1)
    mesh_nodes_y = np.linspace(grid_y[0], grid_y[-1], mesh_size_y + 1)

    closest_grid_points_x = (
        np.abs(mesh_nodes_x[1:-1, np.newaxis] - grid_x).argmin(axis=1)
    )

    closest_grid_points_y = (
        np.abs(mesh_nodes_y[1:-1, np.newaxis] - grid_y).argmin(axis=1)
    )

    mesh_x[0] = grid_x[0]
    mesh_y[0] = grid_y[0]

    mesh_x[-1] = grid_x[-1]
    mesh_y[-1] = grid_y[-1]

    mesh_x[1:-1] = grid_x[closest_grid_points_x]
    mesh_y[1:-1] = grid_y[closest_grid_points_y]

    return (mesh_x, mesh_y)


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

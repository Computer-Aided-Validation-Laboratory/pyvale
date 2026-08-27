#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

"""
Generic tools for creating SimData objects based on analytic functions for the
underlying physical fields. Useful for testing pyvale.
"""

from dataclasses import dataclass
import numpy as np
import sympy
import pyvale.dataio as io
from pyvale.verif.analyticmeshgen import (
    rectangle_mesh_2d,
    box_mesh_3d,
    fill_dims_2d,
    fill_dims_3d,
)


@dataclass(slots=True)
class AnalyticData2D:
    """Dataclass for describing a 2D analytic test case for pyvale sensor
    simulation. Includes information about the geometry, the mesh and the
    analytic functions used to generate the field data.
    """

    length_x: float = 10.0
    """Length of the test case geometry in the X direction in length units.
    Defaults to 10.0.
    """

    length_y: float = 7.5
    """Length of the test case geometry in the Y direction in length units.
    Defaults to 7.5.
    """

    num_elem_x: int = 4
    """Number of elements in the mesh in the X direction. Defaults to 4.
    """

    num_elem_y: int = 3
    """Number of elements in the mesh in the Y direction. Defaults to 3.
    """

    time_steps: np.ndarray | None = None
    """1D array of time steps for the analytic test case. Defaults to None which
    is for a test case that only has spatially varying functions.
    """

    field_keys: tuple[str, ...] = ("scalar",)
    """Keys used to describe the field of interest. For a scalar field there is
    only a single key. For a vector field 2 keys are required in 2D (xx,yy). For
    a tensor field 3 keys are required for 2D (xx,yy,xy). Defaults to a single
    key for a scalar field: ("scalar",).
    """

    funcs_x: dict[str, sympy.Expr] | None = None
    """Analytic functions describing the field variation as a function of the x
    coordinate. This tuple should have the same number of functions as the
    number of field keys. Analytic functions in x, y and t are multiplied
    together so setting a function to a constant of 1 will have no effect.
    """
    funcs_y: dict[str, sympy.Expr] | None = None
    """Analytic functions describing the field variation as a function of the y
    coordinate. This tuple should have the same number of functions as the
    number of field keys. Analytic functions in x, y and t are multiplied
    together so setting a function to a constant of 1 will have no effect.
    """

    funcs_t: dict[str, sympy.Expr] | None = None
    """Analytic functions describing the field variation as a function of time
    This tuple should have the same number of functions as the number of field
    keys. Analytic functions in x, y and t are multiplied together so setting a
    function to a constant of 1 will have no effect.
    """

    symbols: tuple[sympy.Symbol, ...] = (
        sympy.Symbol("y"),
        sympy.Symbol("x"),
        sympy.Symbol("t"),
    )
    """Sympy symbols describing the relevant dimensions of the problem. For 2D
    spatial dimensions default to x and y and time is denoted t. Note that these
    are the symbols used to describe the analytic field functions.
    """

    offset_space_x: dict[str, float] | None = None
    """Constants which are added to the physical field functions in each spatial
    dimensions.
    """

    offset_space_y: dict[str, float] | None = None
    """Constants which are added to the physical field functions in each spatial
    dimensions.
    """

    offset_time: dict[str, float] | None = None
    """Constant which is added to the physical field function in time.
    """

    nodes_per_elem: int = 4
    """Number of nodes per element. Currently only rectangular meshes with
    4 nodes per element are supported. Defaults to 4.
    """

    def __post_init__(self) -> None:
        if self.funcs_x is None:
            self.funcs_x = {}

        if self.funcs_y is None:
            self.funcs_y = {}

        if self.funcs_t is None:
            self.funcs_t = {}

        if self.offset_space_x is None:
            self.offset_space_x = {}

        if self.offset_space_y is None:
            self.offset_space_y = {}

        if self.offset_time is None:
            self.offset_time = {}

        # Set everything to have no effect if the key is not found. That way
        # we can iterate over keys
        for kk in self.field_keys:
            if kk not in self.funcs_x:
                self.funcs_x[kk] = 1.0

            if kk not in self.funcs_y:
                self.funcs_y[kk] = 1.0

            if kk not in self.funcs_t:
                self.funcs_t[kk] = 1.0

            if kk not in self.offset_space_x:
                self.offset_space_x[kk] = 0.0

            if kk not in self.offset_space_y:
                self.offset_space_y[kk] = 0.0

            if kk not in self.offset_time:
                self.offset_time[kk] = 0.0


@dataclass(slots=True)
class AnalyticData3D:
    """Dataclass for describing a 3D analytic test case for pyvale sensor
    simulation. Includes information about the geometry, the mesh and the
    analytic functions used to generate the field data.
    """

    length_x: float = 10.0
    """Length of the test case geometry in the X direction."""

    length_y: float = 7.5
    """Length of the test case geometry in the Y direction."""

    length_z: float = 5.0
    """Length of the test case geometry in the Z direction."""

    num_elem_x: int = 4
    """Number of elements in the mesh in the X direction."""

    num_elem_y: int = 3
    """Number of elements in the mesh in the Y direction."""

    num_elem_z: int = 2
    """Number of elements in the mesh in the Z direction."""

    time_steps: np.ndarray | None = None
    """1D array of time steps for the analytic test case."""

    field_keys: tuple[str, ...] = ("scalar",)
    """Keys used to describe the field of interest."""

    funcs_x: dict[str, sympy.Expr] | None = None
    """Analytic functions in x."""

    funcs_y: dict[str, sympy.Expr] | None = None
    """Analytic functions in y."""

    funcs_z: dict[str, sympy.Expr] | None = None
    """Analytic functions in z."""

    funcs_t: dict[str, sympy.Expr] | None = None
    """Analytic functions in t."""

    symbols: tuple[sympy.Symbol, ...] = (
        sympy.Symbol("z"),
        sympy.Symbol("y"),
        sympy.Symbol("x"),
        sympy.Symbol("t"),
    )
    """Sympy symbols describing the relevant dimensions (z, y, x, t)."""

    offset_space_x: dict[str, float] | None = None
    offset_space_y: dict[str, float] | None = None
    offset_space_z: dict[str, float] | None = None
    offset_time: dict[str, float] | None = None

    nodes_per_elem: int = 8
    """Number of nodes per element (8 for HEX8)."""

    def __post_init__(self) -> None:
        if self.funcs_x is None:
            self.funcs_x = {}
        if self.funcs_y is None:
            self.funcs_y = {}
        if self.funcs_z is None:
            self.funcs_z = {}
        if self.funcs_t is None:
            self.funcs_t = {}
        if self.offset_space_x is None:
            self.offset_space_x = {}
        if self.offset_space_y is None:
            self.offset_space_y = {}
        if self.offset_space_z is None:
            self.offset_space_z = {}
        if self.offset_time is None:
            self.offset_time = {}

        for kk in self.field_keys:
            if kk not in self.funcs_x:
                self.funcs_x[kk] = 1.0
            if kk not in self.funcs_y:
                self.funcs_y[kk] = 1.0
            if kk not in self.funcs_z:
                self.funcs_z[kk] = 1.0
            if kk not in self.funcs_t:
                self.funcs_t[kk] = 1.0
            if kk not in self.offset_space_x:
                self.offset_space_x[kk] = 0.0
            if kk not in self.offset_space_y:
                self.offset_space_y[kk] = 0.0
            if kk not in self.offset_space_z:
                self.offset_space_z[kk] = 0.0
            if kk not in self.offset_time:
                self.offset_time[kk] = 0.0


class AnalyticSimDataGen:
    """Class for generating analytic field data as a `SimData` object to test
    the sensor simulation functionality of pyvale. Provides tools to evaluate
    the analytic field functions at a given spatial coordinate/time to check
    against pyvale interpolation functions. Supports both 2D and 3D cases.
    """

    __slots__ = (
        "case_data",
        "coords",
        "connect",
        "field_sym_funcs",
        "field_lam_funcs",
        "field_eval",
    )

    def __init__(self, case_data: AnalyticData2D | AnalyticData3D) -> None:
        """
        Parameters
        ----------
        case_data : AnalyticData2D | AnalyticData3D
            Data class containing the parameters required to create the analytic
            mesh and the underlying physical field functions.
        """
        self.case_data = case_data
        if isinstance(case_data, AnalyticData3D):
            (self.coords, self.connect) = box_mesh_3d(
                case_data.length_x,
                case_data.length_y,
                case_data.length_z,
                case_data.num_elem_x,
                case_data.num_elem_y,
                case_data.num_elem_z,
            )
        else:
            (self.coords, self.connect) = rectangle_mesh_2d(
                case_data.length_x,
                case_data.length_y,
                case_data.num_elem_x,
                case_data.num_elem_y,
            )

        self.field_sym_funcs = {}
        self.field_lam_funcs = {}
        for kk in case_data.field_keys:
            if isinstance(case_data, AnalyticData3D):
                self.field_sym_funcs[kk] = (
                    (case_data.funcs_x[kk] + case_data.offset_space_x[kk])
                    * (case_data.funcs_y[kk] + case_data.offset_space_y[kk])
                    * (case_data.funcs_z[kk] + case_data.offset_space_z[kk])
                    * (case_data.funcs_t[kk] + case_data.offset_time[kk])
                )
            else:
                self.field_sym_funcs[kk] = (
                    (case_data.funcs_x[kk] + case_data.offset_space_x[kk])
                    * (case_data.funcs_y[kk] + case_data.offset_space_y[kk])
                    * (case_data.funcs_t[kk] + case_data.offset_time[kk])
                )

            self.field_lam_funcs[kk] = sympy.lambdify(
                case_data.symbols, self.field_sym_funcs[kk], "numpy"
            )
        self.field_eval = dict()


    def _get_eval_args(
        self,
        coords: np.ndarray,
        time_steps: np.ndarray | None,
    ) -> list[np.ndarray]:
        """Maps coordinates and time steps to lambdified function arguments in
        the order specified by case_data.symbols.
        """
        if time_steps is None:
            time_steps = self.case_data.time_steps

        if isinstance(self.case_data, AnalyticData3D):
            x_eval, y_eval, z_eval, t_eval = fill_dims_3d(
                coords[:, 0], coords[:, 1], coords[:, 2], time_steps
            )
            arg_map = {"x": x_eval, "y": y_eval, "z": z_eval, "t": t_eval}
        else:
            x_eval, y_eval, t_eval = fill_dims_2d(
                coords[:, 0], coords[:, 1], time_steps
            )
            arg_map = {"x": x_eval, "y": y_eval, "t": t_eval}

        return [arg_map[sym.name] for sym in self.case_data.symbols]

    def evaluate_field_truth(
        self,
        field_key: str,
        coords: np.ndarray,
        time_steps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculates the 'truth' from the analytical functions describing the
        physical fields at the specified coordinates and time steps.

        Parameters
        ----------
        field_key : str
            Key for the underlying physical field.
        coords : np.ndarray
            Coordinates at which to evaluate the analytic physical field. shape
            =(n_coords, coord[x,y,z])
        time_steps : np.ndarray | None, optional
            Time steps at which to evaluate the physical field, by default None.
            If this is none the evaluation time steps are assumed to match the
            nominal time steps.

        Returns
        -------
        np.ndarray
            Array of analytic field evaluations with shape = (n_coords,
            n_time_steps)
        """
        args = self._get_eval_args(coords, time_steps)
        field_vals = self.field_lam_funcs[field_key](*args)
        return field_vals

    def evaluate_all_fields_truth(
        self,
        coords: np.ndarray,
        time_steps: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluates all analytic physical fields at the specified coordinates
        and time steps.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates at which to evaluate the analytic physical field. shape
            =(n_coords, coord[x,y,z])
        time_steps : np.ndarray | None, optional
            Time steps at which to evaluate the physical field, by default None.
            If this is none the evaluation time steps are assumed to match the
            nominal time steps.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary keyed by field name giving numpy array with shape =
            (n_coords, n_timesteps)
        """
        args = self._get_eval_args(coords, time_steps)
        eval_comps = dict()
        for kk in self.case_data.field_keys:
            eval_comps[kk] = self.field_lam_funcs[kk](*args)
        return eval_comps

    def evaluate_field_at_nodes(self, field_key: str) -> np.ndarray:
        """Evaluates the underlying physical field at the node locations and
        nominal time steps.

        Parameters
        ----------
        field_key : str
            String key for the field to be evaluated.

        Returns
        -------
        np.ndarray
            Array of field evaluations with shape=(n_nodes, n_timesteps)
        """
        args = self._get_eval_args(self.coords, self.case_data.time_steps)
        self.field_eval[field_key] = self.field_lam_funcs[field_key](*args)
        return self.field_eval[field_key]

    def evaluate_all_fields_at_nodes(self) -> dict[str, np.ndarray]:
        """Evaluates all physical fields at the node locations and nominal time
        steps.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary keyed by field name giving numpy array with shape =
            (n_coords, n_timesteps)
        """
        args = self._get_eval_args(self.coords, self.case_data.time_steps)
        eval_comps = dict()
        for kk in self.case_data.field_keys:
            eval_comps[kk] = self.field_lam_funcs[kk](*args)
        self.field_eval = eval_comps
        return self.field_eval

    def generate_sim_data(self) -> io.SimData:
        """Creates a SimData object using the analytic case geometry, mesh
        parameters and the underlying physical fields.

        Returns
        -------
        io.SimData
            SimData object built from the analytic case data.
        """
        sim_data = io.SimData()
        sim_data.num_spat_dims = (
            3 if isinstance(self.case_data, AnalyticData3D) else 2
        )
        sim_data.time = self.case_data.time_steps
        sim_data.coords = self.coords
        sim_data.connect = {"connect1": self.connect}

        if not self.field_eval:
            self.evaluate_all_fields_at_nodes()
        sim_data.node_vars = self.field_eval

        return sim_data

    def integrate_symbolic(
        self,
        field_key: str,
        bounds_x: tuple[float, float],
        bounds_y: tuple[float, float],
        bounds_z: tuple[float, float] | None = None,
        bounds_t: tuple[float, float] | None = None,
    ) -> float:
        """Computes exact symbolic integral of the specified field over spatial
        and temporal intervals using SymPy.

        Parameters
        ----------
        field_key : str
            String key for the field to be integrated.
        bounds_x : tuple[float, float]
            Integration bounds (x_min, x_max).
        bounds_y : tuple[float, float]
            Integration bounds (y_min, y_max).
        bounds_z : tuple[float, float] | None, optional
            Integration bounds (z_min, z_max) for 3D cases.
        bounds_t : tuple[float, float] | None, optional
            Integration bounds (t_min, t_max) in time.

        Returns
        -------
        float
            Exact symbolic integral value evaluated to float.
        """
        expr = self.field_sym_funcs[field_key]
        sym_map = {sym.name: sym for sym in self.case_data.symbols}

        if bounds_x[0] == bounds_x[1]:
            expr = expr.subs(sym_map["x"], bounds_x[0])
        else:
            expr = sympy.integrate(
                expr, (sym_map["x"], bounds_x[0], bounds_x[1])
            )

        if bounds_y[0] == bounds_y[1]:
            expr = expr.subs(sym_map["y"], bounds_y[0])
        else:
            expr = sympy.integrate(
                expr, (sym_map["y"], bounds_y[0], bounds_y[1])
            )

        if bounds_z is not None and "z" in sym_map:
            if bounds_z[0] == bounds_z[1]:
                expr = expr.subs(sym_map["z"], bounds_z[0])
            else:
                expr = sympy.integrate(
                    expr, (sym_map["z"], bounds_z[0], bounds_z[1])
                )

        if bounds_t is not None and "t" in sym_map:
            if bounds_t[0] == bounds_t[1]:
                expr = expr.subs(sym_map["t"], bounds_t[0])
            else:
                expr = sympy.integrate(
                    expr, (sym_map["t"], bounds_t[0], bounds_t[1])
                )

        return float(expr)


    def get_visualisation_grid(self,
                               field_key: str | None = None,
                               time_step: int = -1
                               ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Creates a visualisation grid for plotting heatmaps of the specified
        analytic field using matplotlib.

        Parameters
        ----------
        field_key : str | None, optional
            String key for the field to be visualised, by default None. If None
            then the first field key is used.
        time_step : int, optional
            Time step at which to extract the field to be plotted, by default -1

        Returns
        -------
        tuple[np.ndarray,np.ndarray,np.ndarray]
            Tuple containing the 2D grid of x coordinates, grid of y coordinates
            and a grid of field evaluations.
        """
        if field_key is None:
            field_key = self.case_data.field_keys[0]

        grid_shape = (self.case_data.num_elem_y+1,
                      self.case_data.num_elem_x+1)

        grid_x = np.atleast_2d(self.coords[:,0]).T.reshape(grid_shape)
        grid_y = np.atleast_2d(self.coords[:,1]).T.reshape(grid_shape)

        if not self.field_eval:
            self.evaluate_all_fields_at_nodes()

        scalar_grid = np.reshape(
            self.field_eval[field_key][:, time_step], grid_shape
        )

        return (grid_x,grid_y,scalar_grid)







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
import mooseherder as mh
from pyvale.analyticmeshgen import rectangle_mesh_2d, fill_dims_2d


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

    field_keys: tuple[str,...] = ('scalar',)
    """Keys used to describe the field of interest. For a scalar field there is
    only a single key. For a vector field 2 keys are required in 2D (xx,yy). For
    a tensor field 3 keys are required for 2D (xx,yy,xy). Defaults to a single
    key for a scalar field: ("scalar",).
    """

    funcs_x: tuple[sympy.Expr,...] | None = None
    """Analytic functions describing the field variation as a function of the x
    coordinate. This tuple should have the same number of functions as the
    number of field keys. Analytic functions in x, y and t are multiplied
    together so setting a function to a constant of 1 will have no effect.
    """
    funcs_y: tuple[sympy.Expr,...] | None = None
    """Analytic functions describing the field variation as a function of the y
    coordinate. This tuple should have the same number of functions as the
    number of field keys. Analytic functions in x, y and t are multiplied
    together so setting a function to a constant of 1 will have no effect.
    """

    funcs_t: tuple[sympy.Expr,...] | None = None
    """Analytic functions describing the field variation as a function of time
    This tuple should have the same number of functions as the number of field
    keys. Analytic functions in x, y and t are multiplied together so setting a
    function to a constant of 1 will have no effect.
    """

    symbols: tuple[sympy.Symbol,...] = (sympy.Symbol("y"),
                                        sympy.Symbol("x"),
                                        sympy.Symbol("t"))
    """Sympy symbols describing the relevant dimensions of the problem. For 2D
    spatial dimensions default to x and y and time is denoted t. Note that these
    are the symbols used to describe the analytic field functions.
    """

    offsets_space: tuple[float,...] = (0.0,)
    """Constants which are added to the physical field functions in each spatial
    dimensions.
    """

    offsets_time: tuple[float,...] = (0.0,)
    """Constant which is added to the physical field function in time.
    """

    nodes_per_elem: int = 4
    """Number of nodes per element. Currently only rectangular meshes and with
    4 nodes per element are supported. Defaults to 4.
    """


class AnalyticSimDataGen:
    """Class for generating analytic field data as a `SimData` object to test
    the sensor simulation functionality of pyvale. Provides tools to evaluate
    the analytic field functions at a given spatial coordinate/time to check
    against pyvale interpolation functions. Currently only support 2D cases.
    """

    __slots__ = ("case_data","coords","connect","field_sym_funcs",
                 "field_lam_funcs","field_eval")

    def __init__(self, case_data: AnalyticData2D
                 ) -> None:
        """
        Parameters
        ----------
        case_data : AnalyticCaseData2D
            Data class containing the parameters required to create the analytic
            mesh and the underlying physical field functions.
        """
        self.case_data = case_data
        (self.coords,self.connect) = rectangle_mesh_2d(case_data.length_x,
                                                         case_data.length_y,
                                                         case_data.num_elem_x,
                                                         case_data.num_elem_y)

        self.field_sym_funcs = dict()
        self.field_lam_funcs = dict()
        for ii,kk in enumerate(case_data.field_keys):
            self.field_sym_funcs[kk] = ((case_data.funcs_x[ii] *
                                          case_data.funcs_y[ii] +
                                          case_data.offsets_space[ii]) *
                                        (case_data.funcs_t[ii] +
                                         case_data.offsets_time[ii]))

            self.field_lam_funcs[kk] = sympy.lambdify(case_data.symbols,
                                                self.field_sym_funcs[kk],
                                                'numpy')
        self.field_eval = dict()


    def evaluate_field_truth(self,
                       field_key: str,
                       coords: np.ndarray,
                       time_steps: np.ndarray | None = None) -> np.ndarray:
        """Calculates the 'truth' from the analytical functions describing the
        physical fields at the specified coordinates and time steps.

        Parameters
        ----------
        field_key : str
            Key for the underlying physical field.
        coords : np.ndarray
            Coordinates at which to evaluate the analytic physical field. shape
            =(n_coords,coord[x,y,z])
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
        if time_steps is None:
            time_steps = self.case_data.time_steps

        (x_eval,y_eval,t_eval) = fill_dims_2d(coords[:,0],
                                           coords[:,1],
                                           time_steps)

        field_vals = self.field_lam_funcs[field_key](y_eval,
                                                      x_eval,
                                                      t_eval)
        return field_vals


    def evaluate_all_fields_truth(self,
                       coords: np.ndarray,
                       time_steps: np.ndarray | None = None
                       ) -> dict[str,np.ndarray]:
        """Evaluates all analytic physical fields at the specified coordinates
        and time steps.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates at which to evaluate the analytic physical field. shape
            =(n_coords,coord[x,y,z])
        time_steps : np.ndarray | None, optional
            Time steps at which to evaluate the physical field, by default None.
            If this is none the evaluation time steps are assumed to match the
            nominal time steps.

        Returns
        -------
        dict[str,np.ndarray]
            Dictionary keyed by the field name giving a numpy array with shape =
            (n_coords,n_timesteps)
        """
        if time_steps is None:
            time_steps = self.case_data.time_steps

        (x_eval,y_eval,t_eval) = fill_dims_2d(coords[:,0],
                                            coords[:,1],
                                            time_steps)

        eval_comps = dict()
        for kk in  self.case_data.field_keys:
            eval_comps[kk] = self.field_lam_funcs[kk](y_eval,
                                                        x_eval,
                                                        t_eval)
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
            Array of field evaluations with shape=(n_nodes,n_timesteps)
        """
        (x_eval,y_eval,t_eval) = fill_dims_2d(self.coords[:,0],
                                           self.coords[:,1],
                                           self.case_data.time_steps)

        self.field_eval[field_key] = self.field_lam_funcs[field_key](y_eval,
                                                                        x_eval,
                                                                        t_eval)
        return self.field_eval[field_key]

    def evaluate_all_fields_at_nodes(self) -> dict[str,np.ndarray]:
        """Evaluates all physical fields at the node locations and nominal time
        steps.

        Returns
        -------
        dict[str,np.ndarray]
            Dictionary keyed by the field name giving a numpy array with shape =
            (n_coords,n_timesteps)
        """
        (x_eval,y_eval,t_eval) = fill_dims_2d(self.coords[:,0],
                                           self.coords[:,1],
                                           self.case_data.time_steps)
        eval_comps = dict()
        for kk in  self.case_data.field_keys:
            eval_comps[kk] = self.field_lam_funcs[kk](y_eval,
                                                        x_eval,
                                                        t_eval)
        self.field_eval = eval_comps
        return self.field_eval


    def generate_sim_data(self) -> mh.SimData:
        """Creates a SimData object using the analytic case geometry, mesh
        parameters and the underlying physical fields.

        Returns
        -------
        mh.SimData
            SimData object built from the analytic case data.
        """
        sim_data = mh.SimData()
        sim_data.num_spat_dims = 2
        sim_data.time = self.case_data.time_steps
        sim_data.coords = self.coords
        sim_data.connect = {'connect1': self.connect}

        if not self.field_eval:
            self.evaluate_all_fields_at_nodes()
        sim_data.node_vars = self.field_eval

        return sim_data


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

        scalar_grid = np.reshape(self.field_eval[field_key][:,time_step],grid_shape)

        return (grid_x,grid_y,scalar_grid)







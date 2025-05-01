#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

from dataclasses import dataclass
import numpy as np
import sympy
import mooseherder as mh
from pyvale.analyticmeshgen import rectangle_mesh_2d, fill_dims_2d


@dataclass(slots=True)
class AnalyticCaseData2D:
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
    spatial dimensions default to x and y and time is denoted t.
    """

    offsets_space: tuple[float,...] = (0.0,)
    """_summary_
    """

    offsets_time: tuple[float,...] = (0.0,)
    """_summary_
    """

    nodes_per_elem: int = 4
    """_summary_
    """


class AnalyticSimDataGenerator:
    """Class for generating analytic field data as a `SimData` object to test
    the sensor simulation functionality of pyvale. Provides tools to evaluate
    the analytic field functions at a given spatial coordinate and time to check
    against pyvale interpolation functions.
    """

    __slots__ = ("_case_data","_coords","_connect")

    def __init__(self, case_data: AnalyticCaseData2D
                 ) -> None:
        """_summary_

        Parameters
        ----------
        case_data : AnalyticCaseData2D
            _description_
        """
        self._case_data = case_data
        (self._coords,self._connect) = rectangle_mesh_2d(case_data.length_x,
                                                         case_data.length_y,
                                                         case_data.num_elem_x,
                                                         case_data.num_elem_y)

        self._field_sym_funcs = dict()
        self._field_lam_funcs = dict()
        for ii,kk in enumerate(case_data.field_keys):
            self._field_sym_funcs[kk] = ((case_data.funcs_x[ii] *
                                          case_data.funcs_y[ii] +
                                          case_data.offsets_space[ii]) *
                                        (case_data.funcs_t[ii] +
                                         case_data.offsets_time[ii]))

            self._field_lam_funcs[kk] = sympy.lambdify(case_data.symbols,
                                                self._field_sym_funcs[kk],
                                                'numpy')
        self._field_eval = dict()


    def evaluate_field_truth(self,
                       field_key: str,
                       coords: np.ndarray,
                       time_steps: np.ndarray | None = None) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        field_key : str
            _description_
        coords : np.ndarray
            _description_
        time_steps : np.ndarray | None, optional
            _description_, by default None

        Returns
        -------
        np.ndarray
            _description_
        """
        if time_steps is None:
            time_steps = self._case_data.time_steps

        (x_eval,y_eval,t_eval) = fill_dims_2d(coords[:,0],
                                           coords[:,1],
                                           time_steps)

        field_vals = self._field_lam_funcs[field_key](y_eval,
                                                      x_eval,
                                                      t_eval)
        return field_vals


    def evaluate_all_fields_truth(self,
                       coords: np.ndarray,
                       time_steps: np.ndarray | None = None) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        coords : np.ndarray
            _description_
        time_steps : np.ndarray | None, optional
            _description_, by default None

        Returns
        -------
        np.ndarray
            _description_
        """
        if time_steps is None:
            time_steps = self._case_data.time_steps

        (x_eval,y_eval,t_eval) = fill_dims_2d(coords[:,0],
                                            coords[:,1],
                                            time_steps)

        eval_comps = dict()
        for kk in  self._case_data.field_keys:
            eval_comps[kk] = self._field_lam_funcs[kk](y_eval,
                                                        x_eval,
                                                        t_eval)
        return eval_comps


    def evaluate_field_at_nodes(self, field_key: str) -> np.ndarray:
        (x_eval,y_eval,t_eval) = fill_dims_2d(self._coords[:,0],
                                           self._coords[:,1],
                                           self._case_data.time_steps)
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        self._field_eval[field_key] = self._field_lam_funcs[field_key](y_eval,
                                                                        x_eval,
                                                                        t_eval)
        return self._field_eval[field_key]

    def evaluate_all_fields_at_nodes(self) -> dict[str,np.ndarray]:
        """_summary_

        Returns
        -------
        dict[str,np.ndarray]
            _description_
        """
        (x_eval,y_eval,t_eval) = fill_dims_2d(self._coords[:,0],
                                           self._coords[:,1],
                                           self._case_data.time_steps)
        eval_comps = dict()
        for kk in  self._case_data.field_keys:
            eval_comps[kk] = self._field_lam_funcs[kk](y_eval,
                                                        x_eval,
                                                        t_eval)
        self._field_eval = eval_comps
        return self._field_eval


    def generate_sim_data(self) -> mh.SimData:
        """_summary_

        Returns
        -------
        mh.SimData
            _description_
        """
        sim_data = mh.SimData()
        sim_data.num_spat_dims = 2
        sim_data.time = self._case_data.time_steps
        sim_data.coords = self._coords
        sim_data.connect = {'connect1': self._connect}

        if not self._field_eval:
            self.evaluate_all_fields_at_nodes()
        sim_data.node_vars = self._field_eval

        return sim_data


    def get_visualisation_grid(self,
                               field_key: str | None = None,
                               time_step: int = -1
                               ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        """_summary_

        Parameters
        ----------
        field_key : str | None, optional
            _description_, by default None
        time_step : int, optional
            _description_, by default -1

        Returns
        -------
        tuple[np.ndarray,np.ndarray,np.ndarray]
            _description_
        """
        if field_key is None:
            field_key = self._case_data.field_keys[0]

        grid_shape = (self._case_data.num_elem_y+1,
                      self._case_data.num_elem_x+1)

        grid_x = np.atleast_2d(self._coords[:,0]).T.reshape(grid_shape)
        grid_y = np.atleast_2d(self._coords[:,1]).T.reshape(grid_shape)

        if not self._field_eval:
            self.evaluate_all_fields_at_nodes()

        scalar_grid = np.reshape(self._field_eval[field_key][:,time_step],grid_shape)

        return (grid_x,grid_y,scalar_grid)







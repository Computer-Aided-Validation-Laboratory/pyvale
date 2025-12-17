# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from dataclasses import dataclass
from collections.abc import Iterable
import numpy as np


@dataclass(slots=True)
class SimData:
    """Data class for simulation output. Allows for structured meshes with
    connectivity tables or for point clouds."""

    num_spat_dims: int = 3
    """ Number of spatial dimensions in the simulation, required to determine
    element types given that all coords are padded to [x,y,z]. Allows for 2D and
    1D simulations using any combination of the [x,y,z] axes.
    """

    time: np.ndarray | None = None
    """ Vector of time steps with dimensions [t]. Defaults to None.
    """

    coords: np.ndarray | None = None
    """ Array of nodal coordinates in N by 3 where N is the number of nodes
        columns are [x,y,z] coordinates and rows are the nth node. Defaults to
        None.
    """

    connect: dict[str,np.ndarray] | None = None
    """ Element connectivity table:
        key = "connectX" where X is the subdomain e.g. connect1
        Element table given as E by n_e rows where E is the number of elements
        in the given subdomain. n_e is the number of nodes per element.
        Defaults to None.
    """

    side_sets: dict[tuple[str,str],np.ndarray] | None = None
    """ Sidesets by name and associated node and element numbers.
        key = (name, "node" or "elem") e.g. ("bottom","node") will return node
        numbers associated with associated with sideset called "bottom" as a
        numpy array with n_s entries where n_s is the number of nodes in the
        sideset.
        Defaults to None.
    """

    node_vars: dict[str,np.ndarray] | None = None
    """ Nodal variable by name.
        key = "name" e.g. "disp_x" or "temp"
        Gives the nodal variable as a numpy array, N by t where N is the number
        of nodes and t is the number of time steps. Note that element variables
        can be stored as nodal depending on output options or material output
        order selected.
        Defaults to None.
    """

    elem_vars: dict[tuple[str,int],np.ndarray] | None = None
    """ Element variables by name and block.
        key = (name, block num)
        Gives the element variable as a numpy array, E by t where E is the
        number of elements and t is the number of time steps. Note that element
        variables might exist as nodal variables only depending on output
        options and specified material output order.
        Defaults to None.
    """

    glob_vars: dict[str,np.ndarray] | None = None
    """ Global variables by name. Global variable include postprocessors and
        extracted reactions at boundaries.
        key = name (as specified in input file post-processor), e.g. "react_y"
        Gives a numpy array with t entries corresponding to the number of time
        steps in the simulation.
        Defaults to None.
    """

@dataclass(slots=True)
class SimLoadConfig:
    """Used to specify names of variables to be read into the SimData class.
       This class allows the user to only extract the required variables by
       name.
    """
    time: bool = True
    coords: bool = True
    connect: bool = True
    sidesets: Iterable[str] | None = None
    node_vars: Iterable[str] | None = None
    elem_vars: Iterable[tuple[str,int]] | None = None
    glob_vars: Iterable[str] | None = None
    time_inds: Iterable[int] | slice | None = None


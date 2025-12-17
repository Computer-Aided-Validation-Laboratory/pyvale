# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

# ===============================================================================
# EXODUS READER
# Used to read output *.e from MOOSE simulations.

# There are several different cases that lead to different MOOSE output formats.

# 1) Outputs can have 2 or 3 spatial dimensions for nodal DOFs
#     (e.g. disp_x, disp_y and possibly disp_z)
# 2) Element output may or may not be present (e.g. stress/strain)
#     2.1) Element outputs might appear as nodal variables if
#          material_output_order = FIRST or greater
#     2.2) Element output is split by block if material_output_order = CONSTANT
# 4) Sub-domains may or may not be present but see 2.1 above for exception

# Authors: Lloyd Fletcher, Rory Spencer
# ===============================================================================

from pathlib import Path
import netCDF4 as nc
import numpy as np
from pyvale.mooseherder.simdata import SimData, SimLoadConfig
from pyvale.mooseherder.outputloader import IOutputLoader


class ExodusLoader(IOutputLoader):
    """Class to read exodus files output by MOOSE using the netCDF package.
    This class handles extracting the data from the exodus file and creates
    a SimData object with the required data. Most used cases are covered with
    by creating an ExodusReader and then calling either read_sim_data() or
    read_all_sim_data() specified at the bottom of the class.
    """

    __slots__ = ("_exodus_path","_data")

    def __init__(self, output_file: Path) -> None:
        """
        Parameters
        ----------
        output_file : Path
            Construct class by reading the exodus file using the
            netCDF package. The exodus file must exist.

        Raises
        ------
        FileNotFoundError
            The specified exodus file does not exist.
        """

        if not output_file.is_file() and output_file.suffix != '.e':
            raise FileNotFoundError('Exodus file not found at specified path')

        self._exodus_path = output_file
        self._data = nc.Dataset(str(self._exodus_path))


    def get_names(self, key: str | None) -> np.ndarray | None:
        """get_names: Extract a list of variable names from the dataset. Useful
        for getting node/element/sideset/global variables names.

        Parameters
        ----------
        key : str | None
            string key used to extract a list of names from
            the dataset e.g. 'node_var_names'. If key is None returns None.


        Returns
        -------
        np.ndarray | None
            numpy array of strings representing the names
            that correspond to the variables in the dataset. Returns None
            if the specified key does not exist in the dataset.

        """
        if key not in self._data.variables or key is None:
            return None

        return nc.chartostring(np.array(self._data.variables[key]))


    def get_var(self, key: str, time_inds: np.ndarray | None = None
                ) -> np.ndarray:
        """get_var: Extract a numeric variable from the dataset.

        Parameters
        ----------
        key : str
            key corresponding to the variable in the dataset. e.g.
            'time_whole'
        key: str :

        time_inds: np.ndarray | None :
             (Default value = None)

        Returns
        -------
        np.ndarray
            numpy numeric array containing the variable data.

        """
        if key not in self._data.variables:
            return np.array([])

        data = np.array(self._data.variables[key]).T

        if time_inds is None:
            return data

        return data[:,time_inds]


    def get_key(self,
                name: str,
                all_names: np.ndarray,
                key_tag: str) -> str | None:
        """get_key: builds the key required to extract a given variable from
        the exodus dataset.

        Parameters
        ----------
        all_names : np.ndarray
            all possible name keys extracted using the
            get names function.
        name : str
            the specific name key that the user wants to extract
        key_tag : str
            the string tag that is prepended to get the variable
            from the dataset.
        name: str :

        all_names: np.ndarray :

        key_tag: str :


        Returns
        -------
        str | None
            the string key in the dataset to get the variable

        """
        inds = np.where(all_names == name)[0]
        if inds.shape[0] == 0:
            return None

        key = f'{key_tag}{inds[0]+1:d}'
        return key


    def get_connectivity_names(self) -> np.ndarray:
        """get_connectivity_names: gets the connectivity names in the exodus
        dataset. These are of the form 'connect1', 'connect2' etc.

        Parameters
        ----------

        Returns
        -------
        np.ndarray
            array of element connectivity keys as strings of the
            form connectX where X is an integer of 1 or greater e.g.
            connect1.

        """
        names = np.array([])
        for bb in range(self.get_num_elem_blocks()):
            key = f'connect{bb+1:d}'
            if key in self._data.variables:
                names = np.append(names,key)

        return names


    def get_connectivity(self) -> dict[str,np.ndarray]:
        """get_connectivity: returns the connectivity table as a dictionary
        keyed with the name 'connectX' and the table itseld as numpy array.

        Parameters
        ----------

        Returns
        -------
        dict[str,np.ndarray]
            dictionary containing the element
            connectivity tables based on keys related to the subdomain e.g.
            key 'connect1' returns the element connectivity table for
            subdomain 1. The table has dimensions N by n_e where N is the
            total number of nodes in the subdomain and n_e is the number
            of nodes per element.

        """
        connect = dict({})
        for key in self.get_connectivity_names():
            connect[key] = self.get_var(key)

        return connect


    def get_sideset_names(self) -> np.ndarray | None:
        """get_sideset_names: returns the sideset names as a numpy array of
        strings.

        Parameters
        ----------

        Returns
        -------
        np.ndarray | None
            numpy array of strings corresponding to the
            sideset names specified in the simulation. Returns None if no
            sideset names are found.

        """
        return self.get_names('ss_names')


    def get_sidesets(self, names: np.ndarray | None
                     ) -> dict[tuple[str,str], np.ndarray] | None:
        """get_sidesets: returns the sidesets as a dictionary keyed by a tuple
        of ('sideset_name', 'node' | 'elem'). Gives either the list of node
        numbers or element numbers based on the specified key.

        Parameters
        ----------
        names : np.ndarray | None
            numpy array of strings specifying the
            sideset names to extract from the dataset. If None return None.:

        Returns
        -------
        dict[tuple[str,str], np.ndarray] | None
            dictionary of sideset
            nodes and element sets by name. The key is a tuple with the
            first string being the sideset name and the second being either
            'node' or 'elem'. Returns None if no sidesets found.

        """
        all_names = self.get_sideset_names()

        if names is None or all_names is None:
            return None

        node_key_tag = 'node_ns'
        elem_key_tag = 'elem_ss'

        side_sets = dict({})
        for nn in names: # type: ignore
            node_key = self.get_key(nn,all_names,node_key_tag)
            elem_key = self.get_key(nn,all_names,elem_key_tag)

            if node_key is None:
                side_sets[(nn,'node')] = None
            else:
                side_sets[(nn,'node')] = self.get_var(node_key)

            if elem_key is None:
                side_sets[(nn,'elem')] = None
            else:
                side_sets[(nn,'elem')] = self.get_var(elem_key)

        return side_sets


    def get_all_sidesets(self) -> dict[tuple[str,str], np.ndarray] | None:
        """get_all_sidesets: returns all sidesets as a dictionary keyed by a tuple
        of ('sideset_name', 'node' | 'elem'). Gives either the list of node
        numbers or element numbers based on the specified key.

        Parameters
        ----------

        Returns
        -------
        dict[tuple[str,str], np.ndarray] | None
            dictionary of sideset
            nodes and element sets by name. The key is a tuple with the
            first string being the sideset name and the second being either
            'node' or 'elem'. Returns None if no sidesets found.

        """

        return self.get_sidesets(self.get_sideset_names())


    def get_node_var_names(self) -> np.ndarray | None:
        """get_node_var_names: gets the nodal variable names as a numpy array
        of strings e.g. np.array(['disp_x','disp_y'])

        Parameters
        ----------

        Returns
        -------
        np.ndarray | None
            numpy array of strings containing the nodal
            variable names. Returns None if no nodal variables are found.

        """
        return self.get_names('name_nod_var')


    def get_node_vars(self,
                      names: np.ndarray | None,
                      time_inds: np.ndarray | None  = None
                      ) -> dict[str,np.ndarray] | None:
        """get_node_vars: gets the specified nodal variables as a dictionary
        keyed by the variable name (e.g. 'disp_x') where the nodal variable is
        given as a numpy array of dimensions NxT where N is the number of nodes
        and T is the number of time steps in the simulation.

        Parameters
        ----------
        names : np.ndarray | None
            numpy array of strings that are the
            variables to be extracted from the exodus dataset.
        time_inds: np.ndarray | None :
             (Default value = None)

        Returns
        -------
        dict[str,np.ndarray] | None
            dictionary of requested nodal
            variables. Keys are nodal variable names e.g. 'disp_x' and the
            variable data is given as a numpy array. returns None if no
            nodal variables are found.

        """
        if names is None:
            return None

        all_names = self.get_node_var_names()
        key_tag = 'vals_nod_var'
        vars = dict({})

        for nn in names: # type: ignore
            inds = np.where(all_names == nn)[0]
            key = f'{key_tag}{inds[0]+1:d}'
            vars[nn] = self.get_var(key,time_inds)

        return vars


    def get_all_node_vars(self) -> dict[str, np.ndarray] | None:
        """get_all_node_vars: as get_node_vars but returns all nodal variables
        found in the dataset. Gets all specified nodal variables as a dictionary
        keyed by the variable name (e.g. 'disp_x') where the nodal variable is
        given as a numpy array of dimensions NxT where N is the number of nodes
        and T is the number of time steps in the simulation.

        Parameters
        ----------

        Returns
        -------
        dict[str, np.ndarray] | None
            dictionary of requested nodal
            variables. Keys are nodal variable names e.g. 'disp_x' and the
            variable data is given as a numpy array. returns None if no
            nodal variables are found.

        """
        return self.get_node_vars(self.get_node_var_names())


    def get_elem_var_names(self) -> np.ndarray | None:
        """get_elem_var_names: gets the element variable names as a numpy array
        of strings if they exist. Note that there are several cases where the
        element variables may be interpolated to nodes and stored as nodal data

        Parameters
        ----------

        Returns
        -------
        np.ndarray | None
            element variable names as a numpy array of
            strings. An example variable name is 'strain_xx'. Returns None
            if no element variable names exist in the dataset.

        """
        return self.get_names('name_elem_var')


    def get_num_elem_blocks(self) -> int:
        """get_num_elem_blocks: gets the number of element blocks (i.e.
        sub-domains) in the simulation. These are used to partition the element
        data.

        Parameters
        ----------

        Returns
        -------
        int
            number of element blocks/sub-domains in the simulation.

        """
        return self.get_names('eb_names').shape[0] # type: ignore


    def get_elem_var_names_and_blocks(self) -> list[tuple[str,int]] | None:
        """get_elem_var_names_and_blocks: returns a list of all possible
        combinations of element variables names and block numbers present in
        the dataset.

        Parameters
        ----------

        Returns
        -------
        list[tuple[str,int]] | None
            list of tuples containing the element
            variable names and block numbers. Returns None if there are no
            element variable name or element blocks.

        """
        if self.get_elem_var_names() is None or self.get_num_elem_blocks() is None:
            return None

        blocks = [ii+1 for ii in range(self.get_num_elem_blocks())] # type: ignore
        names_blocks = list([])

        for nn in self.get_elem_var_names(): # type: ignore
            for bb in blocks:
                names_blocks.append((str(nn),bb))

        return names_blocks


    def get_elem_vars(self,
                      names_blocks: list[tuple[str,int]] | None,
                      time_inds: np.ndarray | None = None
                      ) -> dict[tuple[str,int],np.ndarray] | None:
        """get_elem_vars: gets the element variables as a dictionary keyed by
        tuples which containg the element variable name and the block number.
        For example: ('strain_xx',1). The element data is given as a numpy
        array with dimensions E_bxT where E_b is the number of element in the
        block and T is the number of time steps.

        Parameters
        ----------
        names_blocks : list[tuple[str,int]] | None :
            list of tuples containing the combination of element variables names
            and blocks to be extracted from the dataset.
        time_inds: np.ndarray | None :
             (Default value = None)

        Returns
        -------
        dict[tuple[str,int],np.ndarray] | None
            contains the variables requested keyed using the input names_blocks
            with the data given as a numpy array.

        """
        all_names = self.get_elem_var_names()

        if all_names is None or names_blocks is None:
            return None

        key_tag = 'vals_elem_var'

        vars = dict({})
        for nn in names_blocks:
            key = self.get_key(nn[0],all_names,key_tag) + f'eb{nn[1]:d}' # type: ignore
            vars[nn] = self.get_var(key,time_inds)

        return vars


    def get_all_elem_vars(self) -> dict[tuple[str,int], np.ndarray] | None:
        """get_all_elem_vars: gets all element variables as a dictionary keyed by
        tuples which containg the element variable name and the block number.
        For example: ('strain_xx',1). The element data is given as a numpy
        array with dimensions E_bxT where E_b is the number of element in the
        block and T is the number of time steps.

        Parameters
        ----------

        Returns
        -------
        dict[tuple[str,int], np.ndarray] | None
            contains the variables requested keyed using the input names_blocks
            with the data given as a numpy array.
        """

        return self.get_elem_vars(self.get_elem_var_names_and_blocks())


    def get_glob_var_names(self) -> np.ndarray | None:
        """get_glob_var_names: gets the names of all global variables in the
        dataset. Global variables include the output of all MOOSE post-
        processors.

        Parameters
        ----------

        Returns
        -------
        np.ndarray | None
            numpy array containing the global variable
            names as strings.

        """
        return self.get_names('name_glo_var')


    def get_glob_vars(self,
                      names: np.ndarray | None,
                      time_inds: np.ndarray | None  = None
                      ) -> dict[str, np.ndarray] | None:
        """get_glob_vars: gets the specified global variables as a dictionary
        keyed by the variable name specified in the MOOSE input file. The data
        is given as a numpy array of T dimensions where T is the number of time
        steps.

        Parameters
        ----------
        names : np.ndarray | None
            numpy array of strings specifying the
            global variable names to extract from the dataset. If this is
            None then return None.
        time_inds: np.ndarray | None :
             (Default value = None)

        Returns
        -------
        dict[str, np.ndarray] | None
            dictionary keyed with the global
            variable names requested giving the data as a numpy array.

        """
        all_names = self.get_glob_var_names()

        if all_names is None or names is None:
            return None

        key = 'vals_glo_var'

        glob_vars = dict({})
        for nn in names: # type: ignore
            inds = np.where(all_names == nn)[0]
            if time_inds is None:
                glob_vars[nn] = np.array(self._data.variables[key][:,inds[0]])
            else:
                data = np.array(self._data.variables[key][:,inds[0]])
                data = data[time_inds]
                glob_vars[nn] = data

        return glob_vars


    def get_all_glob_vars(self) -> dict[str, np.ndarray] | None:
        """get_all_glob_vars: gets all global variables as a dictionary
        keyed by the variable name specified in the MOOSE input file. The data
        is given as a numpy array of T dimensions where T is the number of time
        steps.

        Parameters
        ----------

        Returns
        -------
        dict[str, np.ndarray] | None
            dictionary keyed with all global
            variable names giving the data as numpy arrays.

        """
        return self.get_glob_vars(self.get_glob_var_names())


    def get_coords(self) -> tuple[np.ndarray,int]:
        """Gets the nodal coordinates in each spatial dimension setting any
        undefined dimensions to zeros.

        Parameters
        ----------

        Returns
        -------
        np.array
            returns the nodal coordinates as an array with shape
            (N,3) where N is the number of nodes and the three columns
            are the (x,y,z) spatial dimensions.

        Raises
        ------
        RuntimeError
            no spatial dimensions found.

        """
        # If the problem is not 3D any of these could not exist
        x = self.get_var('coordx')
        y = self.get_var('coordy')
        z = self.get_var('coordz')

        # Problem has to be at least 1D in space if not raise an error
        num_coords = np.max(np.array([x.shape[0],y.shape[0],z.shape[0]]))
        if num_coords == 0:
            raise RuntimeError("No spatial coordinate dimensions detected, problem must be at least 1D.")

        # Any dimensions that do not exist are assumed to be zeros
        x = self._expand_coord(x,num_coords)
        y = self._expand_coord(y,num_coords)
        z = self._expand_coord(z,num_coords)

        return (np.vstack((x,y,z)).T,num_coords)


    def _expand_coord(self, coord: np.ndarray, dim: int) -> np.ndarray:
        """Helper function to create an array of zeros to pad any spatial
        dimensions that are not defined for the simulation.

        Parameters
        ----------
        coord : np.array
            the coordinate array.
        dim : int
            the size of the vector of zeros to create if coord is
            empty.

        Returns
        -------
        np.array
            returns a vector of zeros with shape (dim,) if the
            input array is empty, otherwise return the input coord array.

        """
        if coord.shape[0] == 0:
            return np.zeros([dim,])

        return coord


    def get_time(self, time_inds: np.ndarray | None = None) -> np.ndarray:
        """Get a vector of simulation time steps.

        Parameters
        ----------
        time_inds: np.ndarray | None :
             (Default value = None)

        Returns
        -------
        np.array
            returns an array with shape (T,) where T is the number
            of time steps and the values of the elements are the simulation
            time and each time step.

        """
        time_steps = np.array([]
                              )
        if 'time_whole' in self._data.variables:
            time_steps = np.array(self._data.variables['time_whole'])

            if time_inds is not None:
                time_steps = time_steps[time_inds]

        return time_steps


    def print_vars(self) -> None:
        """Prints all variable strings in the exodus file to console."""
        for vv in self._data.variables:
            print(vv)

    def get_read_config(self) -> SimLoadConfig:
        """get_read_config: constructs a SimReadConfig object by extracting
        all the variable names found in the exodus dataset. Useful for creating
        a mostly populated SimReadConfig and removing variables that are
        unwanted.

        Parameters
        ----------

        Returns
        -------
        SimReadConfig
            data class containing names of variables to be
            extracted from the exodus dataset. See mooseherder.simdata.

        """
        read_config = SimLoadConfig()

        read_config.sidesets = self.get_sideset_names()
        read_config.node_vars = self.get_node_var_names()
        read_config.elem_vars = self.get_elem_var_names_and_blocks()
        read_config.glob_vars = self.get_glob_var_names()

        return read_config


    def load_sim_data(self,
                      read_config: SimLoadConfig) -> SimData:
        """read_sim_data: reads the simulation data based on the specified
        SimReadConfig object.

        Parameters
        ----------
        read_config : SimReadConfig
            data class containing the names of
            the variables that are to be extracted from the exodus dataset.
        read_config: SimReadConfig :


        Returns
        -------
        SimData
            data class containing data from the simulation.

        """
        data = SimData()

        if read_config.time:
            data.time = self.get_time(read_config.time_inds)
        if read_config.coords:
            (data.coords,data.num_spat_dims) = self.get_coords()
        if read_config.connect:
            data.connect = self.get_connectivity()

        data.side_sets = self.get_sidesets(read_config.sidesets)

        data.node_vars = self.get_node_vars(read_config.node_vars,
                                            read_config.time_inds)
        data.elem_vars = self.get_elem_vars(read_config.elem_vars,
                                            read_config.time_inds)
        data.glob_vars = self.get_glob_vars(read_config.glob_vars,
                                            read_config.time_inds)

        return data


    def load_all_sim_data(self) -> SimData:
        """read_all_sim_data: gets all simulation data from the exodus dataset.

        Parameters
        ----------

        Returns
        -------
        SimData
            data class containing the data from the simulation.

        """
        data = SimData()

        data.time = self.get_time()
        (data.coords,data.num_spat_dims) = self.get_coords()
        data.connect = self.get_connectivity()
        data.side_sets = self.get_all_sidesets()
        data.node_vars = self.get_all_node_vars()
        data.elem_vars = self.get_all_elem_vars()
        data.glob_vars = self.get_all_glob_vars()

        return data


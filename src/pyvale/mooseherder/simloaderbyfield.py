# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from pathlib import Path
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
from pyvale.mooseherder.outputloader import IOutputLoader
from pyvale.mooseherder.simdata import SimData, SimLoadConfig
from pyvale.mooseherder.simloadtools import (str_to_path,
                                          load_array,
                                          load_connectivity,
                                          load_field_files,
                                          check_sim_data_consistency,
                                          load_glob_vars,
                                          inv_group_dict)
from pyvale.mooseherder.simloadopts import SimLoadOpts
from pyvale.mooseherder.exceptions import SimLoadErr


class SimLoaderByField(IOutputLoader):
    """Class for loading simulation data (i.e. a `SimData` object) from a series
    of plain text delimited files or binary numpy npy files.

    Implements the `IOutputLoader` interface.
    """

    __slots__ = ("_coords","_time_steps","_fields_dir","_file_patterns",
                 "_field_slices","_load_opts","_connect","_glob_file",
                 "_glob_slices")

    def __init__(self,
                 load_dir: Path,
                 coords_file: str | Path | None,
                 time_step_file: str | Path | None,
                 node_field_files: dict[str,str] | None,
                 connect_files: str | list[str] | None = None,
                 glob_file: str | None = None,
                 glob_slices: dict[str,slice] | None = None,
                 load_opts: SimLoadOpts | None = None) -> None:
        """
        Parameters
        ----------
        load_dir : Path
            Directory to load the simulation data files from.
        coords_file : str | Path | None
            String or full path specifying the coordinates file. If None then
            no coordinates are loaded and they can be manually specified in the
            SimData object.
        time_step_file : str | Path | None
            String or full path to the file containing the simulation time 
            steps. If None then no time step file is loaded and the time steps
            can be manually specified in the SimData object.
        node_field_files : dict[str,str] | None
            Dicitionary keyed by the node field variable name where the value is
            the file name for that field variable to be found in the load 
            directory. If None then no nodal field variables are loaded.
        connect_files : str | list[str] | None, optional
            Wildcard pattern specifying how to identify connectivity files in 
            the load directory or list of strings for the connectivity files, 
            by default None. If None then no connectivity tables are loaded.
        glob_file : str | None, optional
            File name for the global variables file in the load directory, by 
            default None. If None then global variables are not loaded.
        glob_slices : dict[str,slice] | None, optional
            Dictionary keyed with the global variable names with slices 
            specifying which columns to extract for the given global variable, 
            by default None. If None then no global variables are loaded.
        load_opts : SimLoadOpts | None, optional
            Options for loading the simulation data including the number of 
            threads for using multi-processing to load field files, by default 
            None. If None then a default load options dataclass is created.

        Raises
        ------
        SimLoadErr
            The specified load directory is not a directory.
        """
        self._load_dir = load_dir

        self._glob_file = glob_file
        self._glob_slices = glob_slices

        self._load_opts = load_opts

        self._coords = None
        self._time_steps = None
        self._connect = None
        self._node_file_pattern = None
        self._node_slices = None

        if not load_dir.is_dir():
            raise SimLoadErr(f"Load directory: {load_dir.resolve}, is not a "
                + "directory.")

        if coords_file is not None:
            coords_path = str_to_path(load_dir,coords_file)
            self._coords = load_array(coords_path,
                                      load_opts.coord_header,
                                      load_opts.delimiter)

        if time_step_file is not None:
            time_step_path = str_to_path(load_dir,time_step_file)
            self._time_steps = load_array(time_step_path,
                                          load_opts.time_header,
                                          load_opts.delimiter)

            # Fix for column of nans from reading a 1 column csv
            if self._time_steps.ndim != 1:
                self._time_steps = self._time_steps[:,0]

        if connect_files is not None:
            self._connect = load_connectivity(load_dir,
                                              connect_files,
                                              load_opts)


        
        if node_field_files is not None:
            # We are loading by field so only need empty slicesx
            self._node_slices = {kk: slice(None) for kk in node_field_files}

            # We invert the keys and values of this dictionary grouping
            # duplicate keys as values - that way we can loop over this and use
            # the value lists to index into the slices opening a file with a
            # given pattern a single time.
            self._node_file_pattern = inv_group_dict(node_field_files)


    # NOTE: interface function
    def load_sim_data(self, load_config: SimLoadConfig) -> SimData:
        """Loads the simulation data object based on the specified config.

        Parameters
        ----------
        load_config : SimLoadConfig
            Configuration specifying which parts of the SimData object to load.

        Returns
        -------
        SimData
            The SimData object assembled from loading files from disk.
        """
        #-----------------------------------------------------------------------
        # 1. Create SimData object to populate
        sim_data = SimData()

        if load_config.coords:
            sim_data.coords = self._coords

        if load_config.time:
            sim_data.time = self._time_steps

        if load_config.connect:
            sim_data.connect = self._connect

        #-----------------------------------------------------------------------
        # 2. Load global variables file
        if self._glob_file is not None and self._glob_slices is not None:
            sim_data.glob_vars = load_glob_vars(self._load_dir/self._glob_file,
                                                self._glob_slices,
                                                self._load_opts)

        #-----------------------------------------------------------------------
        # 3. Load node field variables by field
        if self._node_file_pattern is not None:
            node_vars = {}
            for file_pattern,field_keys in self._node_file_pattern.items():

                slices_to_ext = {}
                for kk in field_keys:
                    slices_to_ext[kk] = self._node_slices[kk]

                this_node_vars = load_field_files(
                    self._load_dir,
                    file_pattern,
                    slices_to_ext,
                    self._load_opts.node_field_header,
                    self._load_opts.delimiter,
                    load_config.time_inds,
                    self._load_opts.workers,
                )

                node_vars.update(this_node_vars)

            # Needed to get around extra axis issue for components in load func
            for nn in node_vars:
                node_vars[nn] = np.squeeze(node_vars[nn])

            sim_data.node_vars = node_vars

            check_sim_data_consistency(sim_data)

        return sim_data


    # NOTE: interface function
    def load_all_sim_data(self) -> SimData:
        """Loads all simulation data into a SimData object.

        Returns
        -------
        SimData
            The SimData object assembled from loading files from disk.
        """
        # Default load config reads all available data
        load_config = SimLoadConfig()
        return self.load_sim_data(load_config)

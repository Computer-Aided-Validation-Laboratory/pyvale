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
from pyvale.mooseherder.simloader import (str_to_path,
                                          load_array,
                                          load_connectivity,
                                          load_field_files,
                                          check_sim_data_consistency,
                                          load_glob_vars)
from pyvale.mooseherder.simloadopts import SimLoadOpts
from pyvale.mooseherder.exceptions import SimLoadErr


class SimLoaderByTime(IOutputLoader):
    """Class for loading simulation data (i.e. a `SimData` object) from a series
    of plain text delimited files or binary numpy npy files.

    Implements the `IOutputLoader` interface.
    """

    __slots__ = ("_coords","_time_steps","_load_dir","_file_patterns",
                 "_field_slices","_load_opts","_connect","_glob_file",
                 "_glob_slices")

    def __init__(self,
                 load_dir: Path,
                 coords_file: str | Path | None,
                 time_step_file: str | Path  | None,
                 node_files: str | None,
                 node_slices: dict[str, slice],
                 connect_files: str | list[str] | None = None,
                 glob_file: str | None = None,
                 glob_slices: dict[str,slice] | None = None,
                 load_opts: SimLoadOpts | None = None) -> None:

        self._load_dir = load_dir
        
        self._files_pattern = node_files
        self._field_slices = node_slices
        
        self._glob_file = glob_file
        self._glob_slices = glob_slices

        self._load_opts = load_opts

        self._coords = None
        self._time_steps = None
        self._connect = None

        if not load_dir.is_dir():
            raise SimLoadErr(f"Load directory: {load_dir.resolve}, is not a "
                + "directory.")
      
        if coords_file is not None:
            coord_path = str_to_path(load_dir,coords_file)
            self._coords = load_array(coord_path,
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


    # NOTE: interface function
    def load_sim_data(self, load_config: SimLoadConfig) -> SimData:

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
        # 3. Load node field variables as a time series
        sim_data.node_vars = load_field_files(self._load_dir,
                                     self._files_pattern,
                                     self._field_slices,
                                     self._load_opts.node_field_header,
                                     load_config.time_inds,
                                     self._load_opts)

        #-----------------------------------------------------------------------
        # 4. Perform consistency checks on nodes variables 
        check_sim_data_consistency(sim_data)

        return sim_data


    # NOTE: interface function
    def load_all_sim_data(self) -> SimData:

        # Default load config reads all available data
        load_config = SimLoadConfig()
        return self.load_sim_data(load_config)

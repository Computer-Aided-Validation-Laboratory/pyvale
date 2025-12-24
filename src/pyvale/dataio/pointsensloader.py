# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ==============================================================================

from pathlib import Path, PurePath
import numpy as np

from pyvale.dataio.exploader import IExpLoader
from pyvale.dataio.expdata import ExpData
from pyvale.dataio.loadtools import (load_array)
from pyvale.dataio.loadopts import LoadOpts
from pyvale.dataio.exceptions import ExpLoadErr


# NOTE: Problems
# - What labels do the sensor have? We just need to remember which is which
#   - Should we grab the labels from the header? 
#       - YES, need option to grab labels from header

class PointSensLoader(IExpLoader):
    __slots__ = ("_load_files","_sens_array_key","_sens_cols","sens_labels"
                 "_time_slice","_time_col","_coord_file","_coord_slice",
                 "_load_opts")

    def __init__(self,
                 load_files: list[Path] | Path,
                 sens_cols: np.ndarray | list[int],
                 sens_labels: str | list[str] = "S",
                 load_opts: LoadOpts | None = None,
                 time_col: int | None = None,
                 time_slice: slice | None = None,
                 coord_file: Path | None = None,
                 coord_slice: slice | None = None,   
                 ) -> None:

        if isinstance(load_files,PurePath):
            load_files = [load_files,]

        if isinstance(sens_labels,list):
            if len(sens_labels) != len(sens_cols):
                raise ExpLoadErr(
                    f"Number of sensor labels: {len(sens_labels)=}, must match "
                    + "the number of columns of sensor data to extract:" 
                    + f"{len(sens_cols)=}."
                )
        else:
            tag = sens_labels
            sens_labels = [f"{tag}{ii}" for ii in range(len(sens_cols))]

        if len(sens_labels) != len(set(sens_labels)):
            raise ExpLoadErr(
                "Sensor labels must be unique, duplicate sensor labels detected"
            )
                 
        if time_slice is None:
            time_slice = slice(None)

        if coord_slice is None:
            coord_slice = slice(None)
        
        if load_opts is None:
            load_opts = LoadOpts()            

        self._load_files = load_files
        self._sens_cols = sens_cols
        self._sens_labels = sens_labels
        self._time_slice = time_slice
        self._time_col = time_col
        self._coord_file = coord_file
        self._coord_slice = coord_slice
        self._load_opts = load_opts
        
    def load_data(self) -> ExpData:

        sens_arrays = []
        time_arrays = []
        for ff in self._load_files:
            data_array = load_array(ff,
                                    self._load_opts.header_rows,
                                    self._load_opts.delimiter)

            # Flips and copies to shape=(n_sensors,n_times)
            sens_arrays.append(
                data_array[self._time_slice, self._sens_cols].T.copy()
            )

            if self._time_col is not None:
                time_arrays.append(
                    data_array[self._time_slice,self._time_col].copy()
                )

        sens_array = np.hstack(sens_arrays)

        if self._time_col is None: 
            times = None
        else:
            times = np.hstack(time_arrays) 

        if self._coord_file is None:
            coords = None
        else: 
            coords = load_array(self._coord_file,
                                self._load_opts.header_rows,
                                self._load_opts.delimiter)
            coords = coords[self._coord_slice]

            if coords.shape[0] != len(self._sens_cols):
                raise ExpLoadErr(
                    f"Number of loaded sensor coordinates (rows), "
                    + f"{coords.shape[0]}, " 
                    + f"in file {self._coord_file.resolve()} does not match " 
                    + f"the number of extracted traces in sens_cols: "
                    + f"{len(self._sens_cols)}" )

        sens_label_to_ind = {}
        ind_to_sens_label = {}
        for ii,ss in enumerate(self._sens_labels):
            sens_label_to_ind[ss] = ii
            ind_to_sens_label[ii] = ss

        return ExpData(
            fields=sens_array,
            sens_label_to_ind = sens_label_to_ind,
            ind_to_sens_label = ind_to_sens_label,
            coords=coords,
            times=times,
        )

        

        

        

        



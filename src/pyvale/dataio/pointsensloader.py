# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ==============================================================================

from pathlib import Path
import numpy as np

from pyvale.dataio.exploader import IExpLoader
from pyvale.dataio.expdata import ExpData
from pyvale.dataio.loadtools import (load_array)
from pyvale.dataio.loadopts import LoadOpts
from pyvale.dataio.exceptions import ExpLoadErr


class PointSensLoader(IExpLoader):
    __slots__ = ("_load_file","_sens_key","_sens_cols","_time_slice",
                 "_time_col","_coord_file","_coord_slice","_load_opts")

    def __init__(self,
                 load_file: Path,
                 sens_key: str,
                 sens_cols: list[int],
                 load_opts: LoadOpts | None = None,
                 time_col: int | None = None,
                 time_slice: slice | None = None,
                 coord_file: Path | None = None,
                 coord_slice: slice | None = None,   
                 ) -> None:
                 
        if time_slice is None:
            time_slice = slice(None)

        if coord_slice is None:
            coord_slice = slice(None)
        
        if load_opts is None:
            load_opts = LoadOpts()            

        self._load_file = load_file
        self._sens_key = sens_key
        self._sens_cols = sens_cols
        self._time_slice = time_slice
        self._time_col = time_col
        self._coord_file = coord_file
        self._coord_slice = coord_slice
        self._load_opts = load_opts

    def get_sens_key(self) -> str:
        return self._sens_key
        
    def load_data(self) -> ExpData:

        if not load_file.is_file():
            raise ExpLoadErr(f"{load_file.resolve()}, is not a file.")

        data_array = load_array(file_path,
                                self._load_opts.header_rows,
                                self._load_opts.delimiter)

        # Flips and copies to shape=(n_sensors,n_times)
        sens_array = data_array[self._time_slice, self._sens_cols].T.copy()

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

        if self._time_col is None:
            times = None
        else:
            times = data_array[self._time_slice,self._time_col].copy()

        del data_array

        return ExpData(
            fields={self._sens_key:sens_array,},
            coords={self._sens_key:coords,},
            times={self._sens_key:times,},
        )

        

        

        

        



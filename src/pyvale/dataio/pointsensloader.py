# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ==============================================================================

from pathlib import Path, PurePath
from collections.abc import Sequence
import numpy as np

from pyvale.dataio.exploader import IExpLoader
from pyvale.dataio.expdata import ExpData
from pyvale.dataio.loadtools import load_array
from pyvale.dataio.loadopts import ExpLoadOpts
from pyvale.dataio.exceptions import ExpLoadErr


class PointSensLoader(IExpLoader):
    """Loader for point sensor experimental data from delimited text/CSV files.

    Supports extracting arbitrary sensor columns, time series, and spatial
    coordinates across single or multiple files.
    """

    __slots__ = (
        "_load_files",
        "_sens_cols",
        "_sens_labels",
        "_time_slice",
        "_time_col",
        "_coord_file",
        "_coord_slice",
        "_load_opts",
    )

    def __init__(
        self,
        load_files: Sequence[Path] | Path,
        sens_cols: np.ndarray | Sequence[int],
        sens_labels: str | Sequence[str] = "S",
        load_opts: ExpLoadOpts | None = None,
        time_col: int | None = None,
        time_slice: slice | None = None,
        coord_file: Path | None = None,
        coord_slice: slice | None = None,
    ) -> None:
        if isinstance(load_files, PurePath):
            load_files = [Path(load_files)]
        else:
            load_files = [Path(f) for f in load_files]

        if isinstance(sens_labels, (list, tuple)):
            if len(sens_labels) != len(sens_cols):
                raise ExpLoadErr(
                    f"Number of sensor labels: {len(sens_labels)=} must match "
                    f"columns to extract: {len(sens_cols)=}."
                )
            labels_list = list(sens_labels)
        else:
            tag = str(sens_labels)
            labels_list = [f"{tag}{ii}" for ii in range(len(sens_cols))]

        if len(labels_list) != len(set(labels_list)):
            raise ExpLoadErr(
                "Sensor labels must be unique, duplicate labels detected."
            )

        if time_slice is None:
            time_slice = slice(None)

        if coord_slice is None:
            coord_slice = slice(None)

        if load_opts is None:
            load_opts = ExpLoadOpts()

        self._load_files = load_files
        self._sens_cols = list(sens_cols)
        self._sens_labels = labels_list
        self._time_slice = time_slice
        self._time_col = time_col
        self._coord_file = coord_file
        self._coord_slice = coord_slice
        self._load_opts = load_opts

    def load_data(self) -> ExpData:
        """Loads experimental data into an ExpData container."""
        sens_arrays = []
        time_arrays = []
        for ff in self._load_files:
            data_array = load_array(
                ff,
                self._load_opts.header_rows,
                self._load_opts.delimiter,
            )

            # Flips and copies to shape=(n_sensors, n_times)
            sens_arrays.append(
                data_array[self._time_slice, self._sens_cols].T.copy()
            )

            if self._time_col is not None:
                time_arrays.append(
                    data_array[self._time_slice, self._time_col].copy()
                )

        sens_array = np.hstack(sens_arrays)

        if self._time_col is None:
            times = None
        else:
            times = np.hstack(time_arrays)

        if self._coord_file is None:
            coords = None
        else:
            coords = load_array(
                self._coord_file,
                self._load_opts.header_rows,
                self._load_opts.delimiter,
            )
            coords = coords[self._coord_slice]

            if coords.shape[0] != len(self._sens_cols):
                raise ExpLoadErr(
                    f"Loaded coordinate rows ({coords.shape[0]}) in "
                    f"{self._coord_file.resolve()} does not match number of "
                    f"sensor columns ({len(self._sens_cols)})."
                )

        sens_label_to_ind = {s: i for i, s in enumerate(self._sens_labels)}
        ind_to_sens_label = {i: s for i, s in enumerate(self._sens_labels)}

        return ExpData(
            fields=sens_array,
            sens_label_to_ind=sens_label_to_ind,
            ind_to_sens_label=ind_to_sens_label,
            coords=coords,
            times=times,
        )

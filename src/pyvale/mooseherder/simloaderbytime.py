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
from pyvale.mooseherder.simloadopts import SimLoadOpts
from pyvale.mooseherder.exceptions import SimLoadErr


class SimLoaderByTime(IOutputLoader):
    """Class for loading simulation data (i.e. a `SimData` object) from a series
    of plain text delimited files or binary numpy npy files.

    Implements the `IOutputLoader` interface.
    """

    __slots__ = ("_coords","_time_steps","_fields_dir","_file_patterns",
                 "_field_slices","_load_opts","_connect","_glob_file",
                 "_glob_slices")

    def __init__(self,
                 fields_dir: Path,
                 coords: Path | np.ndarray | None,
                 time_steps: Path | np.ndarray | None,
                 node_files: str | dict[str,str],
                 node_slices: dict[str, slice|None] | set[str],
                 connect_dir: Path | None = None,
                 connect_files: str | list[str] | None = None,
                 glob_file: Path | None = None,
                 glob_slices: dict[str,slice] | None = None,
                 load_opts: SimLoadOpts | None = None) -> None:
        """
        Parameters
        ----------
        fields_dir : Path
            Directory containing the nodal physics fields to load.
        coords : Path | np.ndarray | None
            Nodal coordinates of the simulation data which can be given as a
            Path to load a plain text on numpy binary file or the coordinates
            can be provided directly as a numpy array (see the `SimData` object
            for format). If None then the `coords` will be None in the `SimData`
            object.
        time_steps : Path | np.ndarray | None
            Time step vector for the simulation data which can be given as a
            Path to load a plain txt or numpy binary file or the time steps can
            be provided directly as a numpy array (see the `SimData` object for
            format). If None then the `time_steps` will be None in the `SimData`
            object.
        node_files : str | dict[str,str]

        node_slices : dict[str, slice | None] | set[str]
            _description_
        connect_dir : Path | None, optional
            _description_, by default None
        connect_files : str | list[str] | None, optional
            _description_, by default None
        glob_file : Path | None, optional
            _description_, by default None
        glob_slices : dict[str,slice] | None, optional
            _description_, by default None
        load_opts : SimTxtLoadOpts | None, optional
            _description_, by default None

        Raises
        ------
        SimLoadErr
            TODO
        """
        self._glob_file = glob_file
        self._glob_slices = glob_slices
        self._load_opts = load_opts

        if coords is None:
            self._coords = None
        else:
            self._coords = _load_or_set_var(coords,
                                            load_opts.coord_header,
                                            load_opts.delimiter)

        if time_steps is None:
            self._time_steps = None
        else:
            self._time_steps = _load_or_set_var(time_steps,
                                                load_opts.time_header,
                                                load_opts.delimiter)
            # Fix for column of nans from reading a 1 column csv
            if self._time_steps.ndim != 1:
                self._time_steps = self._time_steps[:,0]

        self._fields_dir = fields_dir

        # If the node_slices is a set then we turn it into a dictionary with
        # empty slice - this is the case where we load 'by field' and don't
        # need to slice out columns.
        if isinstance(node_slices,set):
            self._field_slices = {kk: slice(None) for kk in node_slices}
        else:
            self._field_slices = node_slices

        if isinstance(node_files,dict):
            if node_files.keys() != self._field_slices.keys():
                raise SimLoadErr("Keys of the file pattern and field" +
                                 " slice dictionaries do not match.")

            # We invert the keys and values of this dictionary grouping
            # duplicate keys as values - that way we can loop over this and use
            # the value lists to index into the slices opening a file with a
            # given pattern a single time.
            self._files_pattern = _inv_group_dict(node_files)

        else:
            self._files_pattern = node_files

        if connect_dir is not None:
            if connect_files is None:
                raise SimLoadErr("Connectivity file pattern must be specified" +
                    " alongside the connectivity path, e.g. str(connect*.csv)")

            self._connect = self._load_connectivity(connect_dir,connect_files)


    def _load_connectivity(self,
                           connect_dir: Path,
                           connect_pattern: str | list[str],
                           ) -> dict[str,np.ndarray]:
        """_summary_

        Parameters
        ----------
        connect_dir : Path
            _description_
        connect_pattern : str | list[str]
            _description_

        Returns
        -------
        dict[str,np.ndarray]
            _description_

        Raises
        ------
        SimLoadErr
            _description_
        """
        self._connect = {}

        connect_files= []
        if isinstance(connect_pattern,str):
            connect_files = list(connect_dir.glob(connect_pattern))
        elif isinstance(connect_pattern,list):
            for ff in connect_pattern:
                connect_files.append(connect_dir / ff)
        else:
            raise SimLoadErr("Connectivity file pattern must be a string" +
                             " or a  list.")

        for ff in connect_files:
            file_key = ff.stem
            self._connect[file_key] = _load_nparray(
                ff,
                self._load_opts.connect_header,
                self._load_opts.delimiter
            )

        return self._connect

    # NOTE: interface function
    def load_sim_data(self, load_config: SimLoadConfig) -> SimData:
        """_summary_

        Parameters
        ----------
        load_config : SimLoadConfig
            _description_

        Returns
        -------
        SimData
            _description_

        Raises
        ------
        SimLoadErr
            _description_
        SimLoadErr
            _description_
        SimLoadErr
            _description_
        SimLoadErr
            _description_
        SimLoadErr
            _description_
        """
        sim_data = SimData(coords = self._coords,
                           connect = self._connect,
                           time=self._time_steps)

        if isinstance(self._files_pattern,str):
            # Load all fields from a single time series of files
            node_vars = load_data_files(self._fields_dir,
                                        self._files_pattern,
                                        self._field_slices,
                                        self._load_opts.node_field_header,
                                        load_config.time_inds,
                                        self._load_opts)

        elif isinstance(self._files_pattern,dict):
            # Load each node variable in any number of files
            node_vars = {}
            for file_pattern,field_keys in self._files_pattern.items():

                slices_to_ext = {}
                for kk in field_keys:
                    slices_to_ext[kk] = self._field_slices[kk]

                this_node_vars = load_data_files(self._fields_dir,
                                                 file_pattern,
                                                 slices_to_ext,
                                                 self._load_opts.node_field_header,
                                                 None,
                                                 self._load_opts)

                node_vars.update(this_node_vars)

            # Needed to get around extra axis issue for components in load func
            for nn in node_vars:
                node_vars[nn] = np.squeeze(node_vars[nn])


        if self._glob_file is not None and self._glob_slices is not None:

            glob_file = self._fields_dir/self._glob_file
            if not glob_file.is_file():
                raise SimLoadErr(f"Global variables file:'{glob_file.resolve()}'"
                                  + "does not exist.")

            glob_data = _load_nparray(glob_file,
                                      self._load_opts.glob_header,
                                      self._load_opts.delimiter)

            glob_vars = {}
            for kk in self._glob_slices:
                glob_vars[kk] = glob_data[:,self._glob_slices[kk]]

            sim_data.glob_vars = glob_vars


        # Check that the number of nodes and time steps is consistent
        nodes_num = 0
        time_steps_num = 0
        for ii,nn in enumerate(node_vars):
            if ii == 0:
                nodes_num = node_vars[nn].shape[0]
                time_steps_num = node_vars[nn].shape[1]
            else:
                if nodes_num != node_vars[nn].shape[0]:
                    raise SimLoadErr("Number of nodes is not consistent" \
                        " between field variables.")

                if time_steps_num != node_vars[nn].shape[1]:
                    raise SimLoadErr("Number of time steps is not " \
                        "consistent between field variables.")


        # Check number of coords match the nodal fields
        if load_config.coords and self._coords is not None:
            if self._coords.shape[0] != nodes_num:
                raise SimLoadErr(f"Number of coords: '{self._coords.shape[0]}'"
                    + f" does not match field variables: '{nodes_num}'")

        # Check number of time steps match the nodal fields
        if load_config.time and self._time_steps is not None:
            if self._time_steps.shape[0] != time_steps_num:
                raise SimLoadErr("Number of time steps: "
                    + f"'{self._coords.shape[1]}'"
                    + f" does not match field variables: '{time_steps_num}'")

        sim_data.node_vars = node_vars

        return sim_data


    # NOTE: interface function
    def load_all_sim_data(self) -> SimData:
        """_summary_

        Returns
        -------
        SimData
            _description_
        """
        # Default load config reads all available data
        load_config = SimLoadConfig()
        return self.load_sim_data(load_config)



def load_data_files(fields_dir: Path,
                    files_pattern: str,
                    field_slices: dict[str,slice|None],
                    header: int | None,
                    frames: slice | None = None,
                    load_opts: SimTxtLoadOpts | None = None
                    ) -> dict[str,np.ndarray]:
    """_summary_

    Parameters
    ----------
    fields_dir : Path
        _description_
    files_pattern : str
        _description_
    field_slices : dict[str,slice | None]
        _description_
    header : int | None
        _description_
    frames : slice | None, optional
        _description_, by default None
    load_opts : SimTxtLoadOpts | None, optional
        _description_, by default None

    Returns
    -------
    dict[str,np.ndarray]
        _description_

    Raises
    ------
    FileNotFoundError
        _description_
    """
    if not fields_dir.is_dir():
        raise FileNotFoundError(f"Text data path '{fields_dir}' does not exist.")

    if load_opts is None:
        load_opts = SimTxtLoadOpts()

    data_files = list(fields_dir.glob(files_pattern))
    data_files = sorted(data_files)
    if not data_files:
        raise FileNotFoundError("No text files found that match the specified" +
            f" file pattern: '{files_pattern}'.")

    if frames is not None:
        data_files = data_files[frames]

    # Handle the case of the value being `None` as an empty slice to extract all
    for ff in field_slices:
        if field_slices[ff] is None:
            field_slices[ff] = slice(None)


    # We load the first csv to find out what shape of data we are expecting
    data = _load_nparray(data_files[0], header, load_opts.delimiter)

    # Using the first csv we initialise all our numpy arrays to the correct
    # shape to hold our data as shape=(num_frames,num_points,slice.len)
    field_data: dict[str,np.ndarray] = {}
    for ff in field_slices:

        # shape=(num_points,slice.len)
        field_temp = data[:,field_slices[ff]]
        # shape=(num_points,num_frames,slice.len)
        field_data[ff] = np.zeros((data.shape[0],
                                len(data_files),
                                field_temp.shape[1]))
        field_data[ff][:,0,:] = field_temp

        #print(f"key={ff} , {field_data.shape=}")

    # We have loaded the first data frame so we can remove it now, then we will
    # loop over all the others and load them
    data_files.pop(0)

    if load_opts.threads_num is not None:
        assert load_opts.threads_num > 0, "Number of threads must be greater than 0."

        with Pool(load_opts.threads_num) as pool:
            processes_with_id = []

            for ii,ff in enumerate(data_files):
                args = (ff,
                        field_slices,
                        header,
                        load_opts.delimiter)

                process = pool.apply_async(_load_one_array, args=args)
                # NOTE: ii+1 here because we already loaded the first txt file
                processes_with_id.append({"process": process,
                                          "frame": ii+1})

            for pp in processes_with_id:
                frame_data = pp["process"].get()

                for kk in field_slices:
                    field_data[kk][:,pp["frame"],:] = frame_data[kk]


    else:
        for ii,ff in enumerate(data_files):
            # print(f"Loading experiment data file: {ii+1}. From path:")
            # print(f"{ff}\n")

            data = _load_nparray(ff,
                                 header=header,
                                 delimiter=load_opts.delimiter)

            for kk in field_slices:
                # shape=(num_frames,num_points,slice.len)
                # NOTE: ii+1 here because we already loaded the first txt file
                field_data[kk][:,ii+1,:] = data[:,field_slices[kk]]


    # Needed for the case where we have one field for each key instead of
    # combining components, when we combine components we would have a disp
    # array with a third axis of components. When we split we have a disp_x
    # etc arrays. So we squeeze out the component axis.
    if field_temp.shape[1] == 1:
        for kk in field_data:
            field_data[kk] = np.squeeze(field_data[kk])

    return field_data # dict[str,np.ndarray]


def _load_one_array(path: Path,
                    field_slices: dict[str,slice],
                    header: int | None,
                    delimiter: str,
                    ) -> dict[str,np.ndarray]:
    """_summary_

    Parameters
    ----------
    path : Path
        _description_
    field_slices : dict[str,slice]
        _description_
    header : int | None
        _description_
    delimiter : str
        _description_

    Returns
    -------
    dict[str,np.ndarray]
        _description_
    """

    data = _load_nparray(path,header,delimiter)

    sim_data: dict[str,np.ndarray] = {}

    for ff in field_slices:
        # shape=(num_points,slice.len)
        sim_data[ff] = data[:,field_slices[ff]]

    return sim_data


def _load_nparray(file_path: Path,
                  header: int | None,
                  delimiter: str) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    file_path : Path
        _description_
    header : int | None
        _description_
    delimiter : str
        _description_

    Returns
    -------
    np.ndarray
        _description_

    Raises
    ------
    FileNotFoundError
        _description_
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File: '{file_path.resolve()}' does not exist.")

    if file_path.suffix == ".npy":
        return np.load(file_path)

    return _load_txt_file(file_path,header,delimiter)


def _load_txt_file(file_path: Path,
                   header: int | None,
                   delimiter: str) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    file_path : Path
        _description_
    header : int | None
        _description_
    delimiter : str
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    data = pd.read_csv(file_path,sep=delimiter,header=header)
    return data.to_numpy()


def _load_or_set_var(var_in: Path | np.ndarray,
                     header: int | None,
                     delimiter: str) -> np.ndarray:
    """Helper function

    Parameters
    ----------
    var_in : Path | np.ndarray
        Path or nump
    header : int | None
        _description_
    delimiter : str
        _description_

    Returns
    -------
    np.ndarray
        _description_

    Raises
    ------
    TypeError
        _description_
    """
    if isinstance(var_in, Path):
        return _load_nparray(var_in,header,delimiter)
    elif isinstance(var_in, np.ndarray):
        return var_in
    else:
        raise TypeError("Variable must be a pathlib.Path or a numpy.ndarray.")


def _inv_group_dict(dict_com: dict[str,str]) -> dict[str, str]:
    """Helper function to switch keys and values in a dictionary, i.e. invert
    the dictionary such that keys become values and values become keys.

    Parameters
    ----------
    dict_com : dict[str,str]
        Input dictionary to be inverted with keys and values of strings.

    Returns
    -------
    dict[str, str]
        Inverted dictionary where the keys and values are switched compared to
        the input dictionary.
    """
    # Invert keys and group values in the common dictionary
    dict_com_inv = {}
    for kk_new, vv_new in dict_com.items():
        if vv_new not in dict_com_inv:
            dict_com_inv[vv_new] = []
        dict_com_inv[vv_new].append(kk_new)

    return dict_com_inv


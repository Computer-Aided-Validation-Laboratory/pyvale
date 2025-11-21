# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from pathlib import Path
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
from pyvale.mooseherder.simdata import SimData
from pyvale.mooseherder.simloadopts import SimLoadOpts


def str_to_path(default_path: Path, file: str | Path) -> Path:

    if isinstance(file, Path):
        return file

    return default_path/file 


def load_field_files(fields_dir: Path,
                     files_pattern: str,
                     field_slices: dict[str,slice|None],
                     header: int | None,
                     frames: slice | None = None,
                     load_opts: SimLoadOpts | None = None
                     ) -> dict[str,np.ndarray]:

    if not fields_dir.is_dir():
        raise FileNotFoundError(f"Text data path '{fields_dir}' does not exist.")

    if load_opts is None:
        load_opts = SimLoadOpts()

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
    data = load_array(data_files[0], header, load_opts.delimiter)

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

                process = pool.apply_async(load_field_dict, args=args)
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

            data = load_array(ff,
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


def load_field_dict(path: Path,
                   field_slices: dict[str,slice],
                   header: int | None,
                   delimiter: str,
                   ) -> dict[str,np.ndarray]:

    data = load_array(path,header,delimiter)

    sim_data: dict[str,np.ndarray] = {}

    for ff in field_slices:
        # shape=(num_points,slice.len)
        sim_data[ff] = data[:,field_slices[ff]]

    return sim_data


def load_array(file_path: Path,
               header: int | None,
               delimiter: str) -> np.ndarray:
    if not file_path.is_file():
        raise FileNotFoundError(f"File: '{file_path.resolve()}' is not a file.")

    if file_path.suffix == ".npy":
        return np.load(file_path)

    return _load_txt_file(file_path,header,delimiter)


def load_txt_file(file_path: Path,
                   header: int | None,
                   delimiter: str) -> np.ndarray:
    data = pd.read_csv(file_path,sep=delimiter,header=header)
    return data.to_numpy()


def load_connectivity(connect_dir: Path,
                      connect_pattern: str | list[str],
                      load_opts: SimLoadOpts,
                      ) -> dict[str,np.ndarray]:

        connect = {}

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
            connect[file_key] = load_array(
                ff,
                load_opts.connect_header,
                load_opts.delimiter
            )

        return connect


def load_glob_vars(glob_file: Path,
                   glob_slices: dict[str,slice],
                   load_opts: SimLoadOpts) -> dict[str,np.ndarray]:

        if not glob_file.is_file():
            raise SimLoadErr(f"Global variables file:'{glob_file.resolve()}'"
                              + "does not exist.")

        glob_data = load_array(glob_file,
                               load_opts.glob_header,
                               load_opts.delimiter)

        glob_vars = {}
        for kk in glob_slices:
            glob_vars[kk] = np.squeeze(glob_data[:,glob_slices[kk]])

        return glob_vars


def check_sim_data_consistency(sim_data: SimData) -> None:

    # Check that the number of nodes and time steps is consistent over all 
    # node variables in the dictionary
    nodes_num = 0
    time_steps_num = 0
    for ii,nn in enumerate(sim_data.node_vars):
        if ii == 0:
            nodes_num = sim_data.node_vars[nn].shape[0]
            time_steps_num = sim_data.node_vars[nn].shape[1]
        else:
            if nodes_num != sim_data.node_vars[nn].shape[0]:
                raise SimLoadErr("Number of nodes is not consistent" \
                    " between field variables.")

            if time_steps_num != sim_data.node_vars[nn].shape[1]:
                raise SimLoadErr("Number of time steps is not " \
                    "consistent between field variables.")


    # Check number of coords match the nodal fields
    if sim_data.coords is not None:
        if sim_data.coords.shape[0] != nodes_num:
            raise SimLoadErr(
                f"Number of coords: '{sim_data.coords.shape[0]}'"
                + f" in '.coords' does not match field variables: '{nodes_num}'"
            )

    # Check number of time steps match the nodal fields
    if sim_data.time is not None:
        if sim_data.time.shape[0] != time_steps_num:
            raise SimLoadErr(
                f"Number of time steps in '.time': '{sim_data.time.shape[0]}'"
                + f" does not match field variables: '{time_steps_num}'"
            )        

    # Check global variables are consistent with time steps
    if sim_data.glob_vars is not None:
        for kk in sim_data.glob_vars:
            glob_time_steps = np.max(sim_data.glob_vars[kk].shape)
            if glob_time_steps != sim_data.time.shape[0]:
                raise SimLoadErr(
                    f"Number of time steps: {sim_data.time.shape[0]} in '.time'"
                    +f"does not match '.glob_var[{kk}]' = {glob_time_steps}"
                ) 
    


def inv_group_dict(dict_com: dict[str,str]) -> dict[str, str]:
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



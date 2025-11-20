# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from pathlib import Path
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
from pyvale.mooseherder import SimLoadOpts


def load_data_files(fields_dir: Path,
                    files_pattern: str,
                    field_slices: dict[str,slice|None],
                    header: int | None,
                    frames: slice | None = None,
                    load_opts: SimLoadOpts | None = None
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


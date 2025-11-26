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
from pyvale.mooseherder.exceptions import SimLoadErr

def str_to_path(default_path: Path, file: str | Path) -> Path:
    """Appends a string filename to a path or just returns a path to the file.
    Also, checks the file exists.

    Parameters
    ----------
    default_path : Path
        The default path to be used if the file is a string.
    file : str | Path
        The file as either a string name or a full path to the file.

    Returns
    -------
    Path
        Full path to the file.

    Raises
    ------
    FileNotFoundError
        The assemebled full path is not a file.
    """

    if isinstance(file, Path):
        file_path = file
    else:
        file_path = default_path/file

    if not file_path.is_file():
        raise FileNotFoundError(f"{file_path.resolve()} is not a file.")

    return file_path


def load_field_files(fields_dir: Path,
                     files_pattern: str,
                     field_slices: dict[str,slice|None],
                     header: int | None,
                     delimiter: str = ',',
                     frames: slice | None = None,
                     workers: int | None = None,
                     ) -> dict[str,np.ndarray]:
    """Loads a series of physics field files into a dictionary keyed by the
    field names given in the field slices with values that are numpy arrays.

    Parameters
    ----------
    fields_dir : Path
        Path to the directory in which the field files are contained.
    files_pattern : str
        Wildcard string pattern used to identify the field files to load. For
        example: 'node_field_*'' or 'node_field_frame*''.
    field_slices : dict[str,slice | None]
        Dictionary keyed with the field names with slices that identify which
        column contains the field data. If the slice is None then the whole
        file is assumed to be one field where the second dimension is time.
    header : int | None
        Header rows to skip for loading plain text files, starts at 0 to skip
        the first row. None does not skip any rows.
    frames : slice | None, optional
        _description_, by default None
    load_opts : SimLoadOpts | None, optional
        Options for loading the data including parallelisation and delimiter for
        plain text files, by default None

    Returns
    -------
    dict[str,np.ndarray]
        The field data as numpy arrays keyed with the field name e.g. 'disp_x'.

    Raises
    ------
    FileNotFoundError
        The specified directory to load from is not a directory or there are no
        files found in the directory using the given wildcard matching pattern.
    """
    if not fields_dir.is_dir():
        raise FileNotFoundError(f"Data path '{fields_dir}' is not a directory.")

    data_files = list(fields_dir.glob(files_pattern))
    data_files = sorted(data_files)
    if not data_files:
        raise FileNotFoundError("No files found that match the specified" +
            f" file pattern: '{files_pattern}'.")

    if frames is not None:
        data_files = data_files[frames]

    # Handle the case of the value being `None` as an empty slice to extract all
    for ff in field_slices:
        if field_slices[ff] is None:
            field_slices[ff] = slice(None)

    # We load the first csv to find out what shape of data we are expecting
    data = load_array(data_files[0], header, delimiter)

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

    if workers is not None:
        assert workers > 0, "Number of threads must be greater than 0."

        with Pool(workers) as pool:
            processes_with_id = []

            for ii,ff in enumerate(data_files):
                args = (ff,
                        field_slices,
                        header,
                        delimiter)

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
                              delimiter=delimiter)

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
    """Loads a single field file into a dictionary keyed by the field name.

    Parameters
    ----------
    path : Path
        Full file path to load the field array from can be plain txt or numpy
        npy binary format.
    field_slices : dict[str,slice]
        Slices to extract the field data from the array
    header : int | None
        Number of header rows for plain text files to skip starting at 0. If
        None then now headers rows are skipped.
    delimiter : str
        Delimiter used for plain text files.

    Returns
    -------
    dict[str,np.ndarray]
        The field data from the single file as a dictionary keyed by the field
        name.
    """
    data = load_array(path,header,delimiter)

    sim_data: dict[str,np.ndarray] = {}

    for ff in field_slices:
        # shape=(num_points,slice.len)
        sim_data[ff] = data[:,field_slices[ff]]

    return sim_data


def load_array(file_path: Path,
               header: int | None,
               delimiter: str) -> np.ndarray:
    """Loads a single array from disk as either plain text or in numpy npy
    binary format.

    Parameters
    ----------
    file_path : Path
        Full path to the file including the extension. Note that for numpy
        binary format the .npy extension must be used all other extensions will
        be treated as plain text.
    header : int | None
        Number of header rows for plain text files to skip starting at 0. If
        None then now headers rows are skipped.
    delimiter : str
        Delimiter used for plain text files.

    Returns
    -------
    np.ndarray
        The data array loaded from disk.

    Raises
    ------
    FileNotFoundError
        The specified path is not a file.
    """

    if not file_path.is_file():
        raise FileNotFoundError(f"File: '{file_path.resolve()}' is not a file.")

    if file_path.suffix == ".npy":
        return np.load(file_path)

    return load_txt_file(file_path,header,delimiter)


def load_txt_file(file_path: Path,
                  header: int | None,
                  delimiter: str) -> np.ndarray:
    """Wrapper function that loads a delimited plain text file as a numpy array.
    Allows for simple substitution of different text file loading backends.

    Parameters
    ----------
    file_path : Path
        Full path to the plain text file.
    header : int | None
        Number of header rows to skip, starts at 0 skipping the first row. If
        None then no header rows are skipped.
    delimiter : str
        _description_

    Returns
    -------
    np.ndarray
        Numpy array loaded from the specified plain text file.
    """
    data = pd.read_csv(file_path,sep=delimiter,header=header)
    return data.to_numpy()


def load_connectivity(connect_dir: Path,
                      connect_pattern: str | list[str],
                      load_opts: SimLoadOpts,
                      ) -> dict[str,np.ndarray]:
    """Loads the connectivity tables for all meshes in a given simulation into
    a dictionary keyed by the mesh name. The keys default to "connectX" where
    X is an integer. Note that files with a .npy extension will be loaded as 
    numpy binary arrays and any other extension is treated as delimited plain
    text.

    Parameters
    ----------
    connect_dir : Path
        Directory containing the connectivity table files with one file per mesh
        in the simulation.
    connect_pattern : str | list[str]
        Wildcard pattern used to identify connectivity files in the directory or
        list of file names in the given directory to load
    load_opts : SimLoadOpts
        Options for loading the simulation data including header information 
        for the connectivity file.

    Returns
    -------
    dict[str,np.ndarray]
        Dicitionary of connectivity tables for meshes in the simulation.

    Raises
    ------
    SimLoadErr
        The connectivity wildcard pattern is not a string or list of strings.
    """
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
    """Loads the global variables from disk into a dictionary. Examples of 
    global simulation variables include: the maximum temperature or a reaction
    force.

    Parameters
    ----------
    glob_file : Path
        Full path to the file containing the global variable data. Can be either
        numpy binary with a .npy extension of plain text with any other
        extension.
    glob_slices : dict[str,slice]
        Dictionary specifying which columns should be sliced to extract the 
        named global variable.
    load_opts : SimLoadOpts
        Options for loading the simulation data.

    Returns
    -------
    dict[str,np.ndarray]
        Dictionary containing the global variables keyed using the same keys
        as provided in the 'glob_slices' dictionary above. 

    Raises
    ------
    SimLoadErr
        The specified path is not a file.
    """

    if not glob_file.is_file():
        raise SimLoadErr(f"Global variables file:'{glob_file.resolve()}'"
                         + " is not a file.")

    glob_data = load_array(glob_file,
                            load_opts.glob_header,
                            load_opts.delimiter)

    glob_vars = {}
    for kk in glob_slices:
        glob_vars[kk] = np.squeeze(glob_data[:,glob_slices[kk]])

    return glob_vars


def check_sim_data_consistency(sim_data: SimData) -> None:
    """Checks consistency of nodal field variables with the rest of the SimData
    structure.

    Parameters
    ----------
    sim_data : SimData
        The SimData object to check the nodal field variables for consistency.

    Raises
    ------
    SimLoadErr
        The nodal fields in the node_vars dictionary are dimensionally 
        inconsistent with themselves, the coordinates or the time steps.
    """
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



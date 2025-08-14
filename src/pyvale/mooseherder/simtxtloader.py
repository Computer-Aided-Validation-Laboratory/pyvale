# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from dataclasses import dataclass
from pathlib import Path
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
from pyvale.mooseherder.outputloader import OutputLoader
from pyvale.mooseherder.simdata import SimData, SimLoadConfig

#-------------------------------------------------------------------------------
class SimLoadErr(Exception):
    pass


#-------------------------------------------------------------------------------
@dataclass(slots=True)
class SimTxtLoadOpts:
    delimiter: str = ","
    #skip_header: int = 1
    threads_num: int | None = None


#-------------------------------------------------------------------------------
# TODO:
# - field outputs in format: rows=n_pts, cols=fields
#   - check n_pts matches num coords
#   - check n_files matches num time steps
# - how to deal with glob vars?
#
# TODO: future
# - Harmonise field slices and SimReadConfig
# - Support the case of one field per file, currently supports one time
#   step per file
# - Loading global variables

class SimTxtLoader(OutputLoader):

    __slots__ = ("_coords","_time_steps","_files_path","_file_patterns",
                 "_field_slices","_load_config")

    def __init__(self,
                 files_path: Path,
                 files_pattern: str | dict[str,str],
                 field_slices: dict[str,slice],
                 coords: Path | np.ndarray | None,
                 time_steps: Path | np.ndarray | None,
                 load_config: SimTxtLoadOpts | None = None) -> None:

        if coords is None:
            self._coords = None
        else:
            self._coords = _load_or_set_var(coords,load_config.delimiter)

        if time_steps is None:
            self._time_steps = None
        else:
            self._time_steps = _load_or_set_var(time_steps,
                                                load_config.delimiter)

        self._files_path = files_path

        if isinstance(files_pattern,dict):
            if files_pattern.keys() != field_slices.keys():
                raise SimLoadErr("Keys of the file pattern and field" \
                " slice dictionaries do not match.")

            # We invert the keys and values of this dictionary grouping
            # duplicate keys as values - that way we can loop over this and use
            # the value lists to index into the slices opening a file with a
            # given pattern a single time.
            self._files_pattern = _inv_group_dict(files_pattern)

        else:
            self._files_pattern = files_pattern

        self._field_slices = field_slices
        self._load_config = load_config


    # NOTE: interface function
    def load_sim_data(self, load_config: SimLoadConfig) -> SimData:

        if isinstance(self._files_pattern,str):
            # Load all fields from a single time series fo files
            node_vars = load_txt_data(self._files_path,
                                      self._files_pattern,
                                      self._field_slices,
                                      load_config.time_inds,
                                      self._load_config)
        else:
            node_vars = {}
            for file_pattern,field_keys in self._file_patterns:

                slices_to_ext = {}
                for kk in field_keys:
                    slices_to_ext[kk] = self._field_slices[kk]

                this_node_vars = load_txt_data(self._files_path,
                                               file_pattern,
                                               slices_to_ext,
                                               load_config.time_inds,
                                               self._load_config)

                node_vars.update(this_node_vars)

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



        sim_data = SimData()

        if load_config.coords and self._coords is not None:
            if self._coords.shape[0] != nodes_num:
                raise SimLoadErr(f"Number of coords: '{self._coords.shape[0]}'"
                    + f" does not match field variables: '{nodes_num}'")

            sim_data.coords = self._coords


        if load_config.time and self._time_steps is not None:
            if self._coords.shape[0] != time_steps_num:
                raise SimLoadErr("Number of time steps: "
                    + f"'{self._coords.shape[1]}'"
                    + f" does not match field variables: '{time_steps_num}'")

            sim_data.time = self._time_steps

        sim_data.node_vars = node_vars

        return sim_data


    # NOTE: interface function
    def load_all_sim_data(self) -> SimData:
        load_config = SimLoadConfig()
        return self.load_sim_data(load_config)



#-------------------------------------------------------------------------------
def load_txt_data(files_path: Path,
                  files_pattern: str,
                  field_slices: dict[str,slice],
                  frames: slice | None = None,
                  load_opts: SimTxtLoadOpts | None = None
                  ) -> dict[str,np.ndarray]:

    if not files_path.is_dir():
        raise FileNotFoundError(f"Text data path '{files_path}' does not exist.")

    if load_opts is None:
        load_opts = SimTxtLoadOpts()

    txt_files = list(files_path.glob(files_pattern))
    txt_files = sorted(txt_files)
    if not txt_files:
        raise FileNotFoundError("No text files found that match the specified" +
            f" file pattern: '{files_pattern}'.")

    # print(80*"-")
    # print("Debug load_exp_data:")
    # print(f"{csv_files[0]=}")
    # print(f"{csv_files[1]=}")
    # print(f"{csv_files[-1]=}")
    # print()
    # if frames is not None:
    #     slice_frames = csv_files[frames]
    #     print(f"{slice_frames[0]=}")
    #     print(f"{slice_frames[-1]=}")
    # print(80*"-")

    if frames is not None:
        txt_files = txt_files[frames]

    # We load the first csv to find out what shape of data we are expecting
    data = _load_txt_file(txt_files[0],delimiter=load_opts.delimiter)

    # Using the first csv we initialise all our numpy arrays to the correct
    # shape to hold our data as shape=(num_frames,num_points,slice.len)
    field_data: dict[str,np.ndarray] = {}
    for ff in field_slices:
        # shape=(num_points,slice.len)
        field_temp = data[:,field_slices[ff]]
        # shape=(num_points,num_frames,slice.len)
        field_data[ff] = np.zeros((data.shape[0],
                                len(txt_files),
                                field_temp.shape[1]))
        field_data[ff][:,0,:] = field_temp

        #print(f"key={ff} , {field_data.shape=}")

    # We have loaded the first data frame so we can remove it now, then we will
    # loop over all the others and load them
    txt_files.pop(0)

    if load_opts.threads_num is not None:
        assert load_opts.threads_num > 0, "Number of threads must be greater than 0."

        with Pool(load_opts.threads_num) as pool:
            processes_with_id = []

            for ii,ff in enumerate(txt_files):
                args = (ff,
                        field_slices,
                        load_opts.delimiter)

                process = pool.apply_async(_load_one_txt, args=args)
                # NOTE: ii+1 here because we already loaded the first txt file
                processes_with_id.append({"process": process,
                                          "frame": ii+1})

            for pp in processes_with_id:
                frame_data = pp["process"].get()

                for kk in field_slices:
                    field_data[kk][:,pp["frame"],:] = frame_data[kk]

    else:
        for ii,ff in enumerate(txt_files):
            # print(f"Loading experiment data file: {ii+1}. From path:")
            # print(f"{ff}\n")

            data = _load_txt_file(ff,delimiter=load_opts.delimiter)

            for kk in field_slices:
                # shape=(num_frames,num_points,slice.len)
                # NOTE: ii+1 here because we already loaded the first txt file
                field_data[kk][:,ii+1,:] = data[:,field_slices[kk]]

    return field_data # dict[str,np.ndarray]


def _load_one_txt(path: Path,
                  field_slices: dict[str,slice],
                  delimiter: str = ",",
                  ) -> tuple[dict[str,np.ndarray]]:
    """Wrapper function for parallelisation with multi-processing.

    Parameters
    ----------
    path : Path
        _description_
    field_slices : dict[str,slice]
        _description_
    delimiter : str, optional
        _description_, by default ","

    Returns
    -------
    tuple[dict[str,np.ndarray]]
        _description_
    """

    data = _load_txt_file(path,delimiter)

    txt_data: dict[str,np.ndarray] = {}
    for ff in field_slices:
        # shape=(num_points,slice.len)
        txt_data[ff] = data[:,field_slices[ff]]

    return txt_data



def _load_txt_file(file_path: Path, delimiter: str = ",") -> np.ndarray:
    """Wrapper function for allowing different text file reading capabilities
    to a numpy array. Currently uses pandas as it is the most robust.

    Parameters
    ----------
    file_path : Path
        _description_
    delimiter : str, optional
        _description_, by default ","

    Returns
    -------
    np.ndarray
        _description_
    """
    data = pd.read_csv(file_path,sep=delimiter)
    return data.to_numpy()


def _load_or_set_var(var_in: Path | np.ndarray,
                     delimiter: str = ",") -> np.ndarray:

    if isinstance(var_in, Path):
        if not var_in.is_file():
            raise FileNotFoundError(f"Path '{var_in}' is not a file.")

        if var_in.suffix == ".npy":
            data = np.load(var_in)
        else:
            data = _load_txt_file(var_in,delimiter)

        return data

    elif isinstance(data, np.ndarray):
        return var_in
    else:
        raise TypeError("Variable must be a pathlib.Path or a numpy.ndarray.")


def _inv_group_dict(dict_com: dict[str,str]) -> dict[str, str]:

    # Invert keys and group values in the common dictionary
    dict_com_inv = {}
    for kk_new, vv_new in dict_com.items():
        if vv_new not in dict_com_inv:
            dict_com_inv[vv_new] = []
        dict_com_inv[vv_new].append(kk_new)

    return dict_com_inv


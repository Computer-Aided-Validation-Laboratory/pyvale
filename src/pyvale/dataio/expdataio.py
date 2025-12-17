# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ==============================================================================

from dataclasses import dataclass
from pathlib import Path
from multiprocessing.pool import Pool
import numpy as np


@dataclass(slots=True)
class ExpDataLoadOpts:
    file_ext: str = ".csv"
    delimiter: str = ","
    skip_header: int = 1
    threads_num: int | None = None


@dataclass(slots=True)
class ExpData:
    coords: dict[str,np.ndarray] | None = None
    time: dict[str,np.ndarray] | None = None
    fields: dict[str,np.ndarray] | None = None


def load_exp_data(data_path: Path,
                  field_slices: dict[str,slice],
                  frames: slice | None = None,
                  load_opts: ExpDataLoadOpts | None = None
                  ) -> dict[str,np.ndarray]:

    if not data_path.is_dir():
        raise FileNotFoundError("Data path does not exist.")

    if load_opts is None:
        load_opts = ExpDataLoadOpts()

    csv_files = list(data_path.glob("*" + load_opts.file_ext))
    csv_files = sorted(csv_files)

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
        csv_files = csv_files[frames]

    # We load the first csv to find out what shape of data we are expecting
    data = pd.read_csv(csv_files[0])
    data = data.to_numpy()

    # Using the first csv we initialise all our numpy arrays to the correct
    # shape to hold our data as shape=(num_frames,num_points,slice.len)
    field_data: dict[str,np.ndarray] = {}
    for ff in field_slices:
        # shape=(num_points,slice.len)
        field_temp = data[:,field_slices[ff]]
        # shape=(num_points,num_frames,slice.len)
        field_data[ff] = np.zeros((data.shape[0],
                                len(csv_files),
                                field_temp.shape[1]))
        field_data[ff][:,0,:] = field_temp

        #print(f"key={ff} , {field_data.shape=}")

    # if coord_slices is not None:
    #     coord_data: dict[str,np.ndarray] = {}
    #     for cc in coord_slices:
    #         # shape=(num_points,slice.len)
    #         coord_temp = data[:,field_slices[cc]]
    #         # shape=(num_points,num_frames,slice.len)
    #         coord_data[cc] = np.zeros((data.shape[0],
    #                             len(csv_files),
    #                             coord_temp.shape[1]))
    #         coord_data[cc][:,0,:] = coord_temp


    # We have loaded the first data frame so we can remove it now, then we will
    # loop over all the others and load them
    csv_files.pop(0)

    if load_opts.threads_num is not None:
        assert load_opts.threads_num > 0, ("Number of threads must be greater 
            + "than 0.")

        with Pool(load_opts.threads_num) as pool:
            processes_with_id = []

            for ii,ff in enumerate(csv_files):
                args = (ff,
                        field_slices)

                process = pool.apply_async(_load_one_exp, args=args)
                processes_with_id.append({"process": process,
                                          "frame": ii+1,
                                          "file": ff})

            for pp in processes_with_id:
                frame_data = pp["process"].get()

                for kk in field_slices:
                    field_data[kk][:,pp["frame"],:] = frame_data[kk]

    else:
        for ii,ff in enumerate(csv_files):
            # print(f"Loading experiment data file: {ii+1}. From path:")
            # print(f"{ff}\n")

            data = pd.read_csv(ff)
            data = data.to_numpy()

            for kk in field_slices:
                # shape=(num_frames,num_points,slice.len)
                field_data[kk][:,ii+1,:] = data[:,field_slices[kk]]

    return field_data # dict[str,np.ndarray]

def _load_one_exp(path: Path,
                  field_slices: dict[str,slice],
                  ) -> tuple[dict[str,np.ndarray]]:

    data = pd.read_csv(path)
    data = data.to_numpy()

    exp_data: dict[str,np.ndarray] = {}
    for ff in field_slices:
        # shape=(num_points,slice.len)
        exp_data[ff] = data[:,field_slices[ff]]

    # if coord_slices is not None:
    #     coord_data = {}
    #     for cc in coord_slices:
    #         coord_data[cc] = data[:]
    # else:
    #     coord_data = None

    return exp_data

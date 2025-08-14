
#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

from pathlib import Path
from dataclasses import dataclass
import enum
import numpy as np
import pyvale.mooseherder as mh



class ESaveArray(enum.Enum):
    NPY = enum.auto()
    TXT = enum.auto()
    BOTH = enum.auto()


def save_nparray(save_file: Path,
                 data: np.ndarray,
                 save_format: ESaveArray,
                 txt_header: str = "",
                 txt_delimiter: str = ",",
                 txt_ext: str = ".csv"
                 ) -> None:

    if not save_file.parent.exists():
        raise FileExistsError(f"Parent directory: {save_file.parent.resolve()},"
                               + " to save numpy array does not exist.")

    if save_format == ESaveArray.TXT or save_format == ESaveArray.BOTH:
        np.savetxt(
            save_file.with_suffix(txt_ext),
            data,
            delimiter=txt_delimiter,
            header=txt_header,
            comments="", # Removes '#' in header
        )

    if save_format == ESaveArray.TXT or save_format == ESaveArray.BOTH:
        np.save(save_file.with_suffix(".npy"),data)


class ESaveFieldOpt(enum.Enum):
    BY_TIME = enum.auto()
    BY_FIELD = enum.auto()
    BOTH = enum.auto()


@dataclass(slots=True)
class SimDataSaveOpts:
    fields_save_by: ESaveFieldOpt = ESaveFieldOpt.BY_TIME
    array_format: ESaveArray = ESaveArray.TXT
    sim_tag: str = ""

    coords_name: str = "coords"
    connect_name: str = "connect"
    time_name: str = "time"
    glob_name: str = "glob"
    node_field_name: str = "node_field"
    elem_field_name: str = "elem_field"

    def get_coord_name(self) -> str:
        if not self.sim_tag:
            return self.coords_name

        return f"{self.sim_tag}_{self.coords_name}"

    def get_connect_name_by_key(self, key: str) -> str:
        if not self.sim_tag:
            return key

        return f"{self.sim_tag}_{key}"

    def get_connect_name_by_block(self, block: int) -> str:
        if not self.sim_tag:
            return f"{self.connect_name}{block}"

        return f"{self.sim_tag}_{self.connect_name}{block}"

    def get_time_name(self) -> str:
        if not self.sim_tag:
            return self.time_name

        return f"{self.sim_tag}_{self.time_name}"

    def get_glob_name(self) -> str:
        if not self.sim_tag:
            return self.glob_name

        return f"{self.sim_tag}_{self.glob_name}"

    def get_node_field_name(self) -> str:
        if not self.sim_tag:
            return self.node_field_name

        return f"{self.sim_tag}_{self.node_field_name}"

    def get_elem_field_name(self, block: int) -> str:
        if not self.sim_tag:
            return f"{self.elem_field_name}_block{block}"

        return f"{self.sim_tag}_{self.elem_field_name}_block{block}"


def save_sim_data_to_arrays(output_path: Path,
                           sim_data: mh.SimData,
                           save_opts: SimDataSaveOpts | None = None) -> None:
    if not output_path.is_dir():
        raise FileExistsError("")

    if save_opts is None:
        save_opts = SimDataSaveOpts()

    if sim_data.coords is not None:
        save_nparray(output_path / save_opts.get_coord_name(),
                    sim_data.coords,
                    save_format= save_opts.array_format,
                    txt_header="coord_x,coord_y,coord_z")

    if sim_data.connect is not None:
        for ii,cc in enumerate(sim_data.connect):
            save_nparray(output_path / save_opts.get_connect_name_by_key(cc),
                        sim_data.connect[cc],
                        save_format= save_opts.array_format,
                        txt_header="")


    if sim_data.time is not None:
        save_nparray(output_path / save_opts.get_time_name(),
                     sim_data.time,
                     save_format=save_opts.array_format,
                     txt_header="time,")

    if sim_data.glob_vars is not None:
        glob_keys = list(sim_data.glob_vars.keys())
        glob_header = ",".join(glob_keys)
        times_num = sim_data.time.shape[0]

        glob_data = np.zeros((times_num,len(glob_keys)))
        for ii,gg in enumerate(glob_keys):
            glob_data[:,ii] = sim_data.glob_vars[gg]


        save_nparray(output_path / save_opts.get_glob_name(),
                     glob_data,
                     save_format=save_opts.array_format,
                     txt_header=glob_header)


    if sim_data.node_vars is not None:
        node_keys = list(sim_data.node_vars.keys())
        node_header = ",".join(node_keys)

        if (save_opts.fields_save_by == ESaveFieldOpt.BY_FIELD or
            save_opts.fields_save_by == ESaveFieldOpt.BOTH):

            for nn in sim_data.node_vars:
                save_file = save_opts.get_node_field_name() + f"_{nn}"
                save_nparray(output_path / save_file,
                            sim_data.node_vars[nn],
                            save_format=save_opts.array_format)

        if (save_opts.fields_save_by == ESaveFieldOpt.BY_TIME or
            save_opts.fields_save_by == ESaveFieldOpt.BOTH):

            nodes_num = sim_data.coords.shape[0]
            times_num = sim_data.time.shape[0]
            width = len(str(times_num))

            for tt in range(times_num):
                frame_data = np.zeros((nodes_num,len(node_keys)),
                                      dtype=np.float64)
                for ii,nn in enumerate(sim_data.node_vars):
                    frame_data[:,ii] = sim_data.node_vars[nn][:,tt]

                frame_str = str(tt).zfill(width)

                save_file = (save_opts.get_node_field_name()
                             + f"_frame{frame_str}")
                save_nparray(output_path / save_file,
                            frame_data,
                            save_format=save_opts.array_format,
                            txt_header=node_header)

    if sim_data.elem_vars is not None:

        if (save_opts.fields_save_by == ESaveFieldOpt.BY_FIELD or
            save_opts.fields_save_by == ESaveFieldOpt.BOTH):

            for ee in sim_data.elem_vars:
                save_file = (save_opts.get_elem_field_name(ee[1]) + f"_{ee[0]}")
                save_nparray(output_path / save_file,
                            sim_data.elem_vars[ee],
                            save_format=save_opts.array_format)

        if (save_opts.fields_save_by == ESaveFieldOpt.BY_TIME or
            save_opts.fields_save_by == ESaveFieldOpt.BOTH):

            elem_vars = {}
            elem_keys = []
            for (ff,bb), data in sim_data.elem_vars.items():

                if bb not in elem_vars:
                    elem_vars[bb] = {}

                if ff not in elem_keys:
                    elem_keys.append(ff)

                elem_vars[bb][ff] = data

            elem_header = ",".join(elem_keys)

            times_num = sim_data.time.shape[0]
            fields_num = len(elem_keys)
            width = len(str(times_num))

            elem_vars_by_time = {}
            for tt in range(times_num):
                for bb in elem_vars:

                    elems_num = sim_data.connect[f"connect{bb}"].shape[1]
                    this_field = np.zeros((times_num,elems_num,fields_num)
                                          ,dtype=np.float64)

                    if bb not in elem_vars_by_time:
                        elem_vars_by_time[bb] = {}

                    for ff in bb:
                        ii = elem_keys.index(ff)

                        this_field[tt,ii,:] = elem_vars[bb][ff][:,tt]

                    elem_vars_by_time[bb] = this_field


            for tt in range(times_num):
                for bb in elem_vars_by_time:

                    save_file = (save_opts.get_elem_field_name(bb)
                        + f"_frame{tt}.csv")

                    save_nparray(output_path / save_file,
                                 elem_vars_by_time[bb],
                                 save_opts.array_format,
                                 txt_header = elem_header)

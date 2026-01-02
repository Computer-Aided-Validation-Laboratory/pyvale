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
    """Enumeration setting the file type to save arrays as either numpy,
    delimited plain text or both.
    """
    NPY = enum.auto()
    TXT = enum.auto()
    BOTH = enum.auto()


def save_array(save_file: Path,
               data: np.ndarray,
               save_format: ESaveArray,
               txt_header: str = "",
               txt_delimiter: str = ",",
               txt_ext: str = ".csv"
               ) -> None:
    """Wrapper function to save a numpy array to disk in binary npy, delimited
    plain text or both formats.

    Parameters
    ----------
    save_file : Path
        Path including file name to save the numpy arrays to.
    data : np.ndarray
        Array to save to disk.
    save_format : ESaveArray
        Enumeration specifying to save the array in binary numpy, delimited
        plain text or both formats.
    txt_header : str, optional
        String specifying the headers for text files, by default "".
    txt_delimiter : str, optional
        Delimiter for text file array output, by default ","
    txt_ext : str, optional
        Extension for text file output, by default ".csv"

    Raises
    ------
    FileExistsError
        The parent directory where the array files to be saved does not exist.
    """
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
    """Enumeration specifying how to save physics fields as:
    - 'BY_TIME': One array per time step where the first dimension is the nodal
        coordinates and the second dimension is the field component.
    - 'BY_FIELD': A single array per nodal field where the first dimension is
        the coordinate and the second dimension is the time step.
    - 'BOTH': Save in both formats.
    """
    BY_TIME = enum.auto()
    BY_FIELD = enum.auto()
    BOTH = enum.auto()


@dataclass(slots=True)
class SimDataSaveOpts:
    """Options for saving sim data objects to disk.
    """

    fields_save_by: ESaveFieldOpt = ESaveFieldOpt.BY_TIME
    """Enumeration specifying the data structure for the physics fields.
    """

    array_format: ESaveArray = ESaveArray.TXT
    """Enumeration specifying the file format to save the output files in.
    """

    sim_tag: str = ""
    """String tag that will appear as a prefix to all saved output file names.
    """

    coords_name: str = "coords"
    """String that will be used after the 'sim_tag' prefix for the coordinates
    file.
    """

    connect_name: str = "connect"
    """String that will be used after the 'sim_tag' prefix for the connectivity
    table file names. Note that there will be one connectivity table per mesh
    and there will be labelled 'connect_nameX' where X is an integer.
    """

    time_name: str = "time"
    """String that will be used after the 'sim_tag' prefix for the time step
    data file.
    """

    glob_name: str = "glob"
    """String that will be used after the 'sim_tag' prefix for the global
    variable output file.
    """

    node_field_name: str = "node_field"
    """String that will be used after the 'sim_tag' prefix for the output node
    field variable files.
    """

    elem_field_name: str = "elem_field"
    """String that will be used after the 'sim_tag' prefix for the output
    element field variable files.
    """

    def get_coord_name(self) -> str:
        """Assembles the file name for the coordinates. If the 'sim_tag' prefix
        is empty it just returns the specified string name for the coordinate
        file.

        Returns
        -------
        str
            Assemebled filename for the coordinates.
        """
        if not self.sim_tag:
            return self.coords_name

        return f"{self.sim_tag}_{self.coords_name}"

    def get_connect_name_by_key(self, key: str) -> str:
        """Assembles the connectivity file name using the connectivity
        dictionary key taken from the `SimData` object.

        Parameters
        ----------
        key : str
            String key from the connectivity dictionary in the `SimData` object.

        Returns
        -------
        str
            Assembled file name for the specified connectivity table.
        """
        if not self.sim_tag:
            return key

        return f"{self.sim_tag}_{key}"

    def get_connect_name_by_block(self, block: int) -> str:
        """Assembles the connectivity file name using the specified block and
        connectvity name.

        Parameters
        ----------
        block : int
            Integer to identify the connectivity table.

        Returns
        -------
        str
            Assembled file name for the specified connectivity table.
        """
        if not self.sim_tag:
            return f"{self.connect_name}{block}"

        return f"{self.sim_tag}_{self.connect_name}{block}"

    def get_time_name(self) -> str:
        """Assembles the file name for the time steps.

        Returns
        -------
        str
            Assembled file name for the simulation time steps.
        """
        if not self.sim_tag:
            return self.time_name

        return f"{self.sim_tag}_{self.time_name}"

    def get_glob_name(self) -> str:
        """Assembles the file name for the global output variables.

        Returns
        -------
        str
            Assembled file name for the global simulation variables.
        """
        if not self.sim_tag:
            return self.glob_name

        return f"{self.sim_tag}_{self.glob_name}"

    def get_node_field_name(self) -> str:
        """Assembles the file name for nodal field variables.

        Returns
        -------
        str
            Assembled file name for nodal field variables.
        """
        if not self.sim_tag:
            return self.node_field_name

        return f"{self.sim_tag}_{self.node_field_name}"

    def get_elem_field_name(self, block: int) -> str:
        """Assembles the file name for an element field variable

        Parameters
        ----------
        block : int
            Block identifying which connectivity table the elements belong to.

        Returns
        -------
        str
            Assembled file name for the element field variable.
        """
        if not self.sim_tag:
            return f"{self.elem_field_name}_block{block}"

        return f"{self.sim_tag}_{self.elem_field_name}_block{block}"


def save_sim_data_to_arrays(output_path: Path,
                           sim_data: mh.SimData,
                           save_opts: SimDataSaveOpts | None = None) -> None:
    """Saves the simulation data to a series of output files in delimited plain
    text and/or binary numpy arrays.

    Parameters
    ----------
    output_path : Path
        Path to the directory where the simulation files will be saved.
    sim_data : mh.SimData
        Simulation data object containing the data to save to disk.
    save_opts : SimDataSaveOpts | None, optional
        Options for how the simulation data should be saved, by default None.

    Raises
    ------
    FileExistsError
        The specified output Path is not a directory.
    """
    if not output_path.is_dir():
        raise FileExistsError(f"Output directory: {output_path.resolve()}"
            + ", is not a directory.")

    if save_opts is None:
        save_opts = SimDataSaveOpts()

    if sim_data.coords is not None:
        save_array(output_path / save_opts.get_coord_name(),
                    sim_data.coords,
                    save_format= save_opts.array_format,
                    txt_header="coord_x,coord_y,coord_z")

    if sim_data.connect is not None:
        for ii,cc in enumerate(sim_data.connect):
            save_array(output_path / save_opts.get_connect_name_by_key(cc),
                        sim_data.connect[cc],
                        save_format= save_opts.array_format,
                        txt_header="")


    if sim_data.time is not None:
        save_array(output_path / save_opts.get_time_name(),
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


        save_array(output_path / save_opts.get_glob_name(),
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
                save_array(output_path / save_file,
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
                save_array(output_path / save_file,
                            frame_data,
                            save_format=save_opts.array_format,
                            txt_header=node_header)

    if sim_data.elem_vars is not None:

        if (save_opts.fields_save_by == ESaveFieldOpt.BY_FIELD or
            save_opts.fields_save_by == ESaveFieldOpt.BOTH):

            for ee in sim_data.elem_vars:
                save_file = (save_opts.get_elem_field_name(ee[1]) + f"_{ee[0]}")
                save_array(output_path / save_file,
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

                    save_array(output_path / save_file,
                                 elem_vars_by_time[bb],
                                 save_opts.array_format,
                                 txt_header = elem_header)

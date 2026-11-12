# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

# ===============================================================================
# EXODUS READER
# Used to read output *.e from MOOSE simulations.

# There are several different case_input_params that lead to different MOOSE output formats.

# 1) Outputs can have 2 or 3 spatial dimensions for nodal DOFs
#     (e.g. disp_x, disp_y and possibly disp_z)
# 2) Element output may or may not be present (e.g. stress/strain)
#     2.1) Element outputs might appear as nodal variables if
#          material_output_order = FIRST or greater
#     2.2) Element output is split by block if material_output_order = CONSTANT
# 4) Sub-domains may or may not be present but see 2.1 above for exception

# Authors: Lloyd Fletcher, Rory Spencer
# ===============================================================================

from collections.abc import Iterable
from pathlib import Path

import netCDF4 as nc
import numpy as np

from pyvale.dataio.meshconv import enforce_mesh_convention
from pyvale.dataio.simdata import SimData, SimLoadConfig
from pyvale.mooseherder.outputloader import IOutputLoader


class ExodusLoader(IOutputLoader):
    """Read MOOSE Exodus output into typed in-memory simulation containers.

    Notes
    -----
    Use as a context manager to close the underlying NetCDF dataset on normal
    exit or exceptions. Direct callers must call close(), which is idempotent.
    Returned SimData arrays remain usable after closure. Nodal arrays have axes
    (node, time); element arrays have axes (element, time) within each block.
    Field units are those of the source simulation; this loader does not rescale
    them. Geometry/topology conventions are normalised when requested.
    """

    __slots__ = ("_data", "_enforce_convention", "_exodus_path")

    def __init__(self, output_file: Path, enforce_convention: bool = True) -> None:
        """Open an Exodus output and select mesh-convention handling.

        Parameters
        ----------
        output_file : Path
            Existing Exodus output file to read.
        enforce_convention : bool, optional
            Normalise loaded mesh connectivity to PyVale's conventions. Defaults to
            True. This does not convert coordinate units.

        Raises
        ------
        FileNotFoundError
            The specified output cannot be found.
        OSError
            NetCDF cannot open the file as a supported dataset.
        """

        if not output_file.is_file():
            raise FileNotFoundError("Exodus file not found at specified path")

        self._exodus_path = output_file
        self._data = nc.Dataset(str(self._exodus_path))
        self._enforce_convention = enforce_convention

    def close(self) -> None:
        """Release the backing NetCDF dataset if it is still open.

        Notes
        -----
        This operation is idempotent. Arrays already returned in SimData remain
        valid after closure. The inherited IOutputLoader context manager calls
        this method automatically on normal exit and on exceptions.
        """
        if self._data.isopen():
            self._data.close()

    def get_names(self, key: str | None) -> np.ndarray | None:
        """Read decoded variable, boundary or block names from a NetCDF variable.

        Parameters
        ----------
        key : str | None
            NetCDF character-array key, such as "name_nod_var". None selects nothing.

        Returns
        -------
        np.ndarray | None
            One-dimensional string array, or None when the key is absent or None.
        """
        if key not in self._data.variables or key is None:
            return None

        # netCDF4's chartostring triggers NumPy's deprecated in-place shape
        # setting, so the character-to-string conversion is done here with a
        # view and decode instead.
        raw = np.array(self._data.variables[key][:])
        chars = raw.view(f"S{raw.shape[-1]}").reshape(raw.shape[:-1])
        return np.char.decode(chars, "utf-8")

    def get_var(
        self,
        key: str,
        time_inds: np.ndarray | list[int] | tuple[int, ...] | slice | None = None,
    ) -> np.ndarray:
        """Read a numeric NetCDF variable and transpose its stored axes.

        Parameters
        ----------
        key : str
            NetCDF variable key, such as "vals_nod_var1" or "connect1".
        time_inds : np.ndarray | list[int] | tuple[int, ...] | slice | None, optional
            Time columns selected after transposition for two-dimensional variables.
            None reads all entries. Use get_time for the one-dimensional time vector.

        Returns
        -------
        np.ndarray
            Independent array with reversed stored axes. Nodal and element fields
            use (node, time) and (element, time). Missing keys give an empty array.

        Raises
        ------
        IndexError
            Time indices are invalid or supplied for a variable without a time axis.
        """
        if key not in self._data.variables:
            return np.array([])

        data = np.array(self._data.variables[key][:]).T

        if time_inds is None:
            return data

        return data[:, time_inds]

    def get_key(self, name: str, all_names: np.ndarray, key_tag: str) -> str | None:
        """Find the one-based NetCDF variable key for a requested field name.

        Parameters
        ----------
        name : str
            Requested field or boundary name.
        all_names : np.ndarray
            Ordered names stored in the corresponding NetCDF names array.
        key_tag : str
            Prefix such as "vals_nod_var" to which the one-based position is appended.

        Returns
        -------
        str | None
            NetCDF key, or None when name is not in all_names.
        """
        inds = np.where(all_names == name)[0]
        if inds.shape[0] == 0:
            return None

        key = f"{key_tag}{inds[0] + 1:d}"
        return key

    def get_connectivity_names(self) -> np.ndarray:
        """List connectivity variables for the available element blocks.

        Returns
        -------
        np.ndarray
            String keys such as "connect1", ordered by one-based block position.
        """
        names = np.array([])
        for bb in range(self.get_num_elem_blocks()):
            key = f"connect{bb + 1:d}"
            if key in self._data.variables:
                names = np.append(names, key)

        return names

    def get_connectivity(self) -> dict[str, np.ndarray]:
        """Read the raw Exodus connectivity arrays for each element block.

        Returns
        -------
        dict[str, np.ndarray]
            Keys such as "connect1" map to transposed raw arrays with shape
            (nodes_per_element, elements_in_block). Node numbers remain one-based.

        Notes
        -----
        load_sim_data and load_all_sim_data normalise connectivity when convention
        handling is enabled. Those SimData tables instead use one element per row
        and zero-based node indices. This low-level accessor does not normalise.
        """
        connect = {}
        for key in self.get_connectivity_names():
            connect[key] = self.get_var(key)

        return connect

    def get_sideset_names(self) -> np.ndarray | None:
        """Read the named boundaries stored in the Exodus file.

        Returns
        -------
        np.ndarray | None
            Boundary name strings, or None when no boundary names are stored.
        """
        return self.get_names("ss_names")

    def get_sidesets(
        self, names: Iterable[str] | None
    ) -> dict[tuple[str, str], np.ndarray | None] | None:
        """Read node and element numbers for selected named boundaries.

        Parameters
        ----------
        names : Iterable[str] | None
            Boundary names to load. None selects no boundaries.

        Returns
        -------
        dict[tuple[str, str], np.ndarray | None] | None
            (boundary_name, "node") and (boundary_name, "elem") keys map to
            one-dimensional arrays of one-based Exodus numbers. A missing entry
            maps to None; the whole result is None if names or available names
            are absent. These numbers are not normalised by mesh-convention handling.
        """
        all_names = self.get_sideset_names()

        if names is None or all_names is None:
            return None

        node_key_tag = "node_ns"
        elem_key_tag = "elem_ss"

        side_sets = {}
        for nn in names:  # type: ignore
            node_key = self.get_key(nn, all_names, node_key_tag)
            elem_key = self.get_key(nn, all_names, elem_key_tag)

            if node_key is None:
                side_sets[(nn, "node")] = None
            else:
                side_sets[(nn, "node")] = self.get_var(node_key)

            if elem_key is None:
                side_sets[(nn, "elem")] = None
            else:
                side_sets[(nn, "elem")] = self.get_var(elem_key)

        return side_sets

    def get_all_sidesets(self) -> dict[tuple[str, str], np.ndarray | None] | None:
        """Read node and element numbers for every named boundary.

        Returns
        -------
        dict[tuple[str, str], np.ndarray | None] | None
            Boundary entries as returned by get_sidesets, retaining one-based
            Exodus numbering. None when the file has no boundary names.
        """

        return self.get_sidesets(self.get_sideset_names())

    def get_node_var_names(self) -> np.ndarray | None:
        """Read names of all fields stored at mesh nodes.

        Returns
        -------
        np.ndarray | None
            Names such as "disp_x" and "temperature", or None if absent.
        """
        return self.get_names("name_nod_var")

    def get_node_vars(
        self,
        names: Iterable[str] | None,
        time_inds: np.ndarray | list[int] | tuple[int, ...] | slice | None = None,
    ) -> dict[str, np.ndarray] | None:
        """Load selected nodal fields in node-by-time order.

        Parameters
        ----------
        names : Iterable[str] | None
            Existing nodal field names. None selects no nodal fields.
        time_inds : np.ndarray | list[int] | tuple[int, ...] | slice | None, optional
            Saved time indices or slice; None loads all saved times.

        Returns
        -------
        dict[str, np.ndarray] | None
            Field names mapped to arrays with shape (node_count, selected_times).
            None when names is None. Values retain the simulation's native units.

        Raises
        ------
        IndexError
            A requested name or time index is not present.
        """
        if names is None:
            return None

        all_names = self.get_node_var_names()
        key_tag = "vals_nod_var"
        node_vars = {}

        for nn in names:  # type: ignore
            inds = np.where(all_names == nn)[0]
            key = f"{key_tag}{inds[0] + 1:d}"
            node_vars[nn] = self.get_var(key, time_inds)

        return node_vars

    def get_all_node_vars(self) -> dict[str, np.ndarray] | None:
        """Load every nodal field at every saved time.

        Returns
        -------
        dict[str, np.ndarray] | None
            Field names mapped to (node_count, time_count) arrays in native units,
            or None when nodal fields are absent.
        """
        return self.get_node_vars(self.get_node_var_names())

    def get_elem_var_names(self) -> np.ndarray | None:
        """Read names of fields stored per element rather than per node.

        Returns
        -------
        np.ndarray | None
            Element field names, or None when absent. Material output settings may
            place quantities such as strain in node_vars instead of elem_vars.
        """
        return self.get_names("name_elem_var")

    def get_num_elem_blocks(self) -> int:
        """Count named element blocks (mesh subdomains).

        Returns
        -------
        int
            Number of entries in the Exodus element-block names array.
        """
        return self.get_names("eb_names").shape[0]  # type: ignore

    def get_elem_var_names_and_blocks(self) -> list[tuple[str, int]] | None:
        """Enumerate combinations of element field names and block positions.

        Returns
        -------
        list[tuple[str, int]] | None
            (field_name, one_based_block_position) combinations, or None when
            element fields are absent. A combination without stored values yields
            an empty array when loaded.
        """
        if self.get_elem_var_names() is None or self.get_num_elem_blocks() is None:
            return None

        blocks = [ii + 1 for ii in range(self.get_num_elem_blocks())]  # type: ignore
        names_blocks = []

        for nn in self.get_elem_var_names():  # type: ignore
            for bb in blocks:
                names_blocks.append((str(nn), bb))

        return names_blocks

    def get_elem_vars(
        self,
        names_blocks: list[tuple[str, int]] | None,
        time_inds: np.ndarray | list[int] | tuple[int, ...] | slice | None = None,
    ) -> dict[tuple[str, int], np.ndarray] | None:
        """Load selected element fields by name and one-based block position.

        Parameters
        ----------
        names_blocks : list[tuple[str, int]] | None
            Field/block keys such as ("stress_yy", 1). None selects no element fields.
        time_inds : np.ndarray | list[int] | tuple[int, ...] | slice | None, optional
            Saved time indices or slice; None loads all saved times.

        Returns
        -------
        dict[tuple[str, int], np.ndarray] | None
            Keys mapped to (elements_in_block, selected_times) arrays in native
            units. Missing stored combinations give empty arrays. None when no
            names are selected or the file contains no element fields.

        Raises
        ------
        TypeError
            A requested field name cannot be resolved.
        IndexError
            A time index is invalid.
        """
        all_names = self.get_elem_var_names()

        if all_names is None or names_blocks is None:
            return None

        key_tag = "vals_elem_var"

        elem_vars = {}
        for nn in names_blocks:
            key = self.get_key(nn[0], all_names, key_tag) + f"eb{nn[1]:d}"  # type: ignore
            elem_vars[nn] = self.get_var(key, time_inds)

        return elem_vars

    def get_all_elem_vars(self) -> dict[tuple[str, int], np.ndarray] | None:
        """Load every element field/block combination at every saved time.

        Returns
        -------
        dict[tuple[str, int], np.ndarray] | None
            Field/block keys mapped to (elements_in_block, time_count) arrays in
            native units, or None when element fields are absent.
        """

        return self.get_elem_vars(self.get_elem_var_names_and_blocks())

    def get_glob_var_names(self) -> np.ndarray | None:
        """Read global variable and postprocessor names.

        Returns
        -------
        np.ndarray | None
            Names such as "react_y_top", or None when global fields are absent.
        """
        return self.get_names("name_glo_var")

    def get_glob_vars(
        self,
        names: Iterable[str] | None,
        time_inds: np.ndarray | list[int] | tuple[int, ...] | slice | None = None,
    ) -> dict[str, np.ndarray] | None:
        """Load selected global fields, including reaction postprocessors.

        Parameters
        ----------
        names : Iterable[str] | None
            Existing global field names. None selects no global fields.
        time_inds : np.ndarray | list[int] | tuple[int, ...] | slice | None, optional
            Saved time indices or slice; None loads all saved times.

        Returns
        -------
        dict[str, np.ndarray] | None
            Field names mapped to one-dimensional arrays of selected time values
            in native units. None when no names are selected or globals are absent.

        Raises
        ------
        IndexError
            A requested name or time index is not present.
        """
        all_names = self.get_glob_var_names()

        if all_names is None or names is None:
            return None

        key = "vals_glo_var"

        glob_vars = {}
        for nn in names:  # type: ignore
            inds = np.where(all_names == nn)[0]
            if time_inds is None:
                glob_vars[nn] = np.array(self._data.variables[key][:, inds[0]])
            else:
                data = np.array(self._data.variables[key][:, inds[0]])
                data = data[time_inds]
                glob_vars[nn] = data

        return glob_vars

    def get_all_glob_vars(self) -> dict[str, np.ndarray] | None:
        """Load every global field at every saved time.

        Returns
        -------
        dict[str, np.ndarray] | None
            Names mapped to (time_count,) arrays in native units, or None when
            global variables are absent.
        """
        return self.get_glob_vars(self.get_glob_var_names())

    def get_coords(self) -> tuple[np.ndarray, int]:
        """Read reference nodal coordinates, padding missing axes with zeros.

        Returns
        -------
        coords : np.ndarray
            Shape (node_count, 3), columns x/y/z, in the simulation's length units.
        num_nodes : int
            Number of nodes. This is not the number of active spatial dimensions.

        Raises
        ------
        RuntimeError
            No spatial coordinate values are stored.
        """
        # If the problem is not 3D any of these could not exist
        x = self.get_var("coordx")
        y = self.get_var("coordy")
        z = self.get_var("coordz")

        # Problem has to be at least 1D in space if not raise an error
        num_coords = np.max(np.array([x.shape[0], y.shape[0], z.shape[0]]))
        if num_coords == 0:
            raise RuntimeError(
                "No spatial coordinate dimensions detected, problem must be at least 1D."
            )

        # Any dimensions that do not exist are assumed to be zeros
        x = self._expand_coord(x, num_coords)
        y = self._expand_coord(y, num_coords)
        z = self._expand_coord(z, num_coords)

        return (np.vstack((x, y, z)).T, num_coords)

    def _expand_coord(self, coord: np.ndarray, dim: int) -> np.ndarray:
        """Pad a missing coordinate axis without modifying a present axis.

        Parameters
        ----------
        coord : np.ndarray
            One-dimensional coordinate vector, possibly empty.
        dim : int
            Node count used for the replacement vector when coord is empty.

        Returns
        -------
        np.ndarray
            The original vector when populated, otherwise a zero vector of shape (dim,).
        """
        if coord.shape[0] == 0:
            return np.zeros(
                [
                    dim,
                ]
            )

        return coord

    def get_time(
        self, time_inds: np.ndarray | list[int] | tuple[int, ...] | slice | None = None
    ) -> np.ndarray:
        """Read the saved simulation times in their original units.

        Parameters
        ----------
        time_inds : np.ndarray | list[int] | tuple[int, ...] | slice | None, optional
            Saved time indices or slice; None loads all saved times.

        Returns
        -------
        np.ndarray
            Shape (selected_times,), or an empty array when time_whole is absent.

        Raises
        ------
        IndexError
            A requested time index is outside the stored time range.
        """
        time_steps = np.array([])
        if "time_whole" in self._data.variables:
            time_steps = np.array(self._data.variables["time_whole"][:])

            if time_inds is not None:
                time_steps = time_steps[time_inds]

        return time_steps

    def print_vars(self) -> None:
        """Print every raw NetCDF variable key to standard output.

        Notes
        -----
        These are storage keys such as vals_nod_var1, not physical field names.
        Use get_node_var_names, get_elem_var_names and get_glob_var_names for those.
        """
        for vv in self._data.variables:
            print(vv)

    def get_read_config(self) -> SimLoadConfig:
        """Build a load configuration containing all available variable names.

        Returns
        -------
        SimLoadConfig
            Configuration for time, geometry, side sets and all available nodal,
            element and global variables. Callers may remove unwanted fields before
            passing it to load_sim_data.
        """
        read_config = SimLoadConfig()

        read_config.sidesets = self.get_sideset_names()
        read_config.node_vars = self.get_node_var_names()
        read_config.elem_vars = self.get_elem_var_names_and_blocks()
        read_config.glob_vars = self.get_glob_var_names()

        return read_config

    def load_sim_data(self, read_config: SimLoadConfig) -> SimData:
        """Load selected fields and geometry into an independent SimData container.

        Parameters
        ----------
        read_config : SimLoadConfig
            Field names, geometry/connectivity selections and time indices to load.
            Its enforce_convention flag controls mesh normalisation for this call.

        Returns
        -------
        SimData
            Selected arrays in the source simulation's units. Nodal fields have axes
            (node, time); element fields have axes (element, time) per block. The arrays
            are in memory and remain valid after closing the loader.

        Raises
        ------
        RuntimeError
            The underlying NetCDF dataset has already been closed.
        """
        data = SimData()

        if read_config.time:
            data.time = self.get_time(read_config.time_inds)
        if read_config.coords:
            coords, _ = self.get_coords()
            data.coords = coords
            num_spat_dims = 0
            if self.get_var("coordx").shape[0] > 0:
                num_spat_dims += 1
            if self.get_var("coordy").shape[0] > 0:
                num_spat_dims += 1
            if self.get_var("coordz").shape[0] > 0:
                num_spat_dims += 1
            data.num_spat_dims = num_spat_dims
        if read_config.connect:
            data.connect = self.get_connectivity()

        data.side_sets = self.get_sidesets(read_config.sidesets)

        data.node_vars = self.get_node_vars(
            read_config.node_vars, read_config.time_inds
        )
        data.elem_vars = self.get_elem_vars(
            read_config.elem_vars, read_config.time_inds
        )
        data.glob_vars = self.get_glob_vars(
            read_config.glob_vars, read_config.time_inds
        )

        if (
            read_config.enforce_convention
            and data.connect is not None
            and data.coords is not None
        ):
            data = enforce_mesh_convention(data)

        data.refresh_mesh_type()

        return data

    def load_all_sim_data(self) -> SimData:
        """Load all available Exodus fields and geometry.

        Returns
        -------
        SimData
            Available nodal, element and global arrays, geometry, side sets and times.
            The constructor's enforce_convention setting controls mesh normalisation.
            Loaded arrays remain usable after the loader is closed.

        Raises
        ------
        RuntimeError
            The underlying NetCDF dataset has already been closed.
        """
        read_config = self.get_read_config()
        read_config.enforce_convention = self._enforce_convention
        return self.load_sim_data(read_config)

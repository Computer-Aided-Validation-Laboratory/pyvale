# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Load mesh coordinates and connectivity into simulation data."""

from pathlib import Path

from pyvale.dataio.exceptions import SimLoadErr
from pyvale.dataio.loadopts import SimLoadOpts
from pyvale.dataio.loadtools import (
    load_array,
    load_connectivity,
    str_to_path,
)
from pyvale.dataio.meshconv import enforce_mesh_convention
from pyvale.dataio.simdata import SimData


class MeshLoader:
    """Load coordinates and connectivity without simulation fields.

    Parameters
    ----------
    load_dir : Path
        Directory containing the mesh files.
    coords_file : str | Path
        Coordinate file name relative to ``load_dir`` or its complete path.
    connect_files : str | list[str]
        Connectivity file pattern or an explicit list of file names.
    load_opts : SimLoadOpts | None, optional
        Text and array loading options. Default options are created when this
        argument is ``None``.
    enforce_convention : bool, optional
        Convert supported legacy connectivity into the common PyVale mesh
        convention, by default ``True``.
    """

    __slots__ = (
        "_connect_files",
        "_coords_file",
        "_enforce_convention",
        "_load_dir",
        "_load_opts",
    )

    def __init__(
        self,
        load_dir: Path,
        coords_file: str | Path,
        connect_files: str | list[str],
        load_opts: SimLoadOpts | None = None,
        enforce_convention: bool = True,
    ) -> None:
        if not load_dir.is_dir():
            raise SimLoadErr(
                f"Load directory '{load_dir.resolve()}' is not a directory."
            )

        self._load_dir = load_dir
        self._coords_file = coords_file
        self._connect_files = connect_files
        self._load_opts = SimLoadOpts() if load_opts is None else load_opts
        self._enforce_convention = enforce_convention

    def load_mesh(self) -> SimData:
        """Load and return mesh coordinates and connectivity.

        Returns
        -------
        SimData
            Simulation data containing only coordinates and connectivity.
        """
        coords_path = str_to_path(self._load_dir, self._coords_file)
        coords = load_array(
            coords_path,
            self._load_opts.coord_header,
            self._load_opts.delimiter,
        )
        connect = load_connectivity(
            self._load_dir,
            self._connect_files,
            self._load_opts,
        )
        mesh = SimData(coords=coords, connect=connect)

        if self._enforce_convention:
            mesh = enforce_mesh_convention(mesh)

        mesh.refresh_mesh_type()
        return mesh


__all__ = ["MeshLoader"]

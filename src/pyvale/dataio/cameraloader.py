"""Load processed camera fields from column or matrix frames."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from pyvale.dataio.exceptions import ExpLoadErr
from pyvale.dataio.expdata import ExpData
from pyvale.dataio.exploader import IExpLoader
from pyvale.dataio.loadopts import ExpLoadOpts
from pyvale.dataio.loadtools import load_array


class CameraLoaderColumn(IExpLoader):
    """Load one table per frame using fixed reference coordinates.

    Supply zero-based column indices and frame paths in timestamp order.
    Point IDs, when supplied, may be strings and define row correspondence.
    """

    def __init__(
        self,
        frame_files: Sequence[Path],
        *,
        coord_cols: tuple[int, int] | tuple[int, int, int],
        component_cols: Mapping[str, int],
        point_id_col: int | None = None,
        times: np.ndarray | None = None,
        load_opts: ExpLoadOpts | None = None,
    ) -> None:
        self._frame_files = tuple(frame_files)
        self._coord_cols = coord_cols
        self._component_cols = dict(component_cols)
        self._point_id_col = point_id_col
        self._times = times
        self._load_opts = load_opts if load_opts is not None else ExpLoadOpts()

    def load_data(self) -> ExpData:
        if len(self._coord_cols) not in (2, 3):
            raise ExpLoadErr("Supply two or three coordinate columns.")

        component_labels = tuple(self._component_cols)
        times = _validate_frames(len(self._frame_files), component_labels, self._times)
        component_cols = tuple(self._component_cols.values())
        required_cols = self._coord_cols + component_cols
        if self._point_id_col is not None:
            required_cols += (self._point_id_col,)

        for frame_index, path in enumerate(self._frame_files):
            table = _read_array(path, self._load_opts)
            if min(required_cols) < 0 or max(required_cols) >= table.shape[1]:
                raise ExpLoadErr(f"Column index out of range in camera frame '{path}'.")

            point_indices: dict[str, int] = {}
            if self._point_id_col is None:
                point_indices = {str(index): index for index in range(table.shape[0])}
            else:
                point_ids = table[:, self._point_id_col]
                for index, point_id in enumerate(point_ids):
                    if point_id != point_id or str(point_id).strip() == "":
                        raise ExpLoadErr(f"Missing point ID in camera frame '{path}'.")
                    point_indices[str(point_id)] = index
                if len(point_indices) != table.shape[0]:
                    raise ExpLoadErr(f"Duplicate point IDs in camera frame '{path}'.")

            if frame_index == 0:
                reference_indices = point_indices
                fields = np.empty(
                    (table.shape[0], len(component_labels), len(self._frame_files))
                )
            if point_indices.keys() != reference_indices.keys():
                raise ExpLoadErr(
                    f"Point IDs or row count differ in camera frame '{path}'."
                )

            row_order = [point_indices[label] for label in reference_indices]
            try:
                frame_coords = table[np.ix_(row_order, self._coord_cols)].astype(float)
                fields[:, :, frame_index] = table[np.ix_(row_order, component_cols)]
            except (TypeError, ValueError) as exc:
                raise ExpLoadErr(
                    f"Invalid numeric values in camera frame '{path}'."
                ) from exc

            if frame_index == 0:
                coords = np.zeros((table.shape[0], 3))
                coords[:, :len(self._coord_cols)] = frame_coords
            if not np.all(np.isfinite(frame_coords)) or not np.allclose(
                frame_coords, coords[:, :len(self._coord_cols)], rtol=0.0, atol=0.0
            ):
                raise ExpLoadErr(
                    f"Invalid or changed reference coordinates in '{path}'."
                )

        return ExpData(
            fields=fields,
            sens_label_to_ind=reference_indices,
            ind_to_sens_label={
                index: label for label, index in reference_indices.items()
            },
            coords=coords,
            times=times,
            component_labels=component_labels,
        )


class CameraLoaderMatrix(IExpLoader):
    """Load one spatial matrix per component per frame in row-major order.

    Supply static x, y and optionally z coordinate matrices in the same grid
    layout as the measurements. Missing measurements may be stored as NaN.
    """

    def __init__(
        self,
        frame_files: Sequence[Mapping[str, Path]],
        *,
        coord_files: tuple[Path, Path] | tuple[Path, Path, Path],
        times: np.ndarray | None = None,
        load_opts: ExpLoadOpts | None = None,
    ) -> None:
        self._frame_files = tuple(dict(frame) for frame in frame_files)
        self._coord_files = coord_files
        self._times = times
        self._load_opts = (
            load_opts if load_opts is not None else ExpLoadOpts(header_rows=None)
        )

    def load_data(self) -> ExpData:
        component_labels = tuple(self._frame_files[0]) if self._frame_files else ()
        times = _validate_frames(len(self._frame_files), component_labels, self._times)
        if len(self._coord_files) not in (2, 3):
            raise ExpLoadErr("Supply two or three coordinate matrices.")

        coordinate_grids = [
            _read_array(path, self._load_opts, numeric=True)
            for path in self._coord_files
        ]
        grid_shape = coordinate_grids[0].shape
        coords = np.zeros((coordinate_grids[0].size, 3))
        for axis, grid in enumerate(coordinate_grids):
            if grid.shape != grid_shape or not np.all(np.isfinite(grid)):
                path = self._coord_files[axis]
                raise ExpLoadErr(f"Invalid coordinate grid '{path}'.")
            coords[:, axis] = grid.ravel()

        fields = np.empty(
            (coords.shape[0], len(component_labels), len(self._frame_files))
        )
        for frame_index, frame in enumerate(self._frame_files):
            if frame.keys() != self._frame_files[0].keys():
                raise ExpLoadErr(
                    f"Matrix frame {frame_index} has different components."
                )
            for component_index, label in enumerate(component_labels):
                path = frame[label]
                grid = _read_array(path, self._load_opts, numeric=True)
                if grid.shape != grid_shape:
                    raise ExpLoadErr(
                        f"Matrix '{path}' has shape {grid.shape}, "
                        f"expected {grid_shape}."
                    )
                fields[:, component_index, frame_index] = grid.ravel()

        return ExpData(
            fields=fields,
            sens_label_to_ind={str(index): index for index in range(coords.shape[0])},
            ind_to_sens_label={index: str(index) for index in range(coords.shape[0])},
            coords=coords,
            times=times,
            component_labels=component_labels,
            grid_shape=grid_shape,
        )


def _validate_frames(
    frame_count: int, component_labels: tuple[str, ...], times: np.ndarray | None
) -> np.ndarray | None:
    if not frame_count or not component_labels:
        raise ExpLoadErr("Supply at least one camera frame and one component.")
    if times is not None:
        times = np.array(times, dtype=float, copy=True)
        if times.shape != (frame_count,) or not np.all(np.isfinite(times)):
            raise ExpLoadErr("Supply one finite timestamp per camera frame.")
    return times


def _read_array(
    path: Path, load_opts: ExpLoadOpts, *, numeric: bool = False
) -> np.ndarray:
    try:
        array = load_array(path, load_opts.header_rows, load_opts.delimiter)
        if numeric:
            array = array.astype(float, copy=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ExpLoadErr(f"Could not read camera data from '{path}': {exc}") from exc
    if array.ndim != 2 or array.size == 0:
        raise ExpLoadErr(f"Expected a non-empty two-dimensional array in '{path}'.")
    return array

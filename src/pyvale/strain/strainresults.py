# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


from dataclasses import dataclass
from dataclasses import fields
import numpy as np
from pyvale.dic.dicenum import ELayout

@dataclass(slots=True)
class StrainResults:
    """
    Data container for Strain analysis results. This dataclass stores the strain 
    window coordinates, 2D deformation gradient and strain values.

    Attributes
    ----------

    window_x : np.ndarray
        The x-coordinates of the strain window centre. shape=(img_num,y,x)
    window_y : np.ndarray
        The y-coordinates of the strain window centre. shape=(img_num,y,x)
    x_mm : np.ndarray | None
        X-coordinate of the strain window centre in physical units relative to cam0.
    y_mm : np.ndarray | None
        Y-coordinate of the strain window centre in physical units relative to cam0.
    z_mm : np.ndarray | None
        Z-coordinate of the strain window centre in physical units relative to cam0.
    def_00 : np.ndarray
        The xx component (1 + ∂u/∂x) of the deformation gradient matrix. shape=(img_num, y, x)
    def_01 : np.ndarray
        The xy component (∂u/∂y) of the deformation gradient matrix. shape=(img_num, y, x)
    def_10 : np.ndarray
        The yx component (∂v/∂x) of the deformation gradient matrix. shape=(img_num, y, x)
    def_11 : np.ndarray
        The yy component (1 + ∂v/∂y) of the deformation gradient matrix. shape=(img_num, y, x)
    def_20 : np.ndarray
        The zx component (∂w/∂x) of the deformation gradient matrix. shape=(img_num, y, x)
    def_21 : np.ndarray
        The zy component (∂w/∂y) of the deformation gradient matrix. shape=(img_num, y, x)
    eps_xx : np.ndarray
        The xx component of the surface strain tensor. shape=(img_num, y, x)
    eps_xy : np.ndarray
        The xy component of the surface strain tensor. shape=(img_num, y, x)
    eps_yx : np.ndarray
        The yx component of the surface strain tensor. shape=(img_num, y, x)
    eps_yy : np.ndarray
        The yy component of the surface strain tensor. shape=(img_num, y, x)
    eps1 : np.ndarray | None
        The larger in-plane principal strain, ordered so eps1 >= eps2.
        shape=(img_num, y, x). None when importing legacy files.
    eps2 : np.ndarray | None
        The smaller in-plane principal strain, ordered so eps1 >= eps2.
        shape=(img_num, y, x). None when importing legacy files.
    filenames : list[str]
        name of Strain result files that have been found
    """

    window_x: np.ndarray
    window_y: np.ndarray
    def_00: np.ndarray
    def_01: np.ndarray
    def_10: np.ndarray
    def_11: np.ndarray
    def_20: np.ndarray
    def_21: np.ndarray
    eps_xx: np.ndarray
    eps_xy: np.ndarray
    eps_yx: np.ndarray
    eps_yy: np.ndarray
    filenames: list[str]
    eps1: np.ndarray | None = None
    eps2: np.ndarray | None = None
    x_mm: np.ndarray | None = None
    y_mm: np.ndarray | None = None
    z_mm: np.ndarray | None = None


def convert_dic_layout(data, layout: ELayout | str):
    """Convert a DIC result object to the requested column or matrix layout."""
    layout = ELayout(layout)
    is_column = data.ss_x.ndim == 1
    want_column = layout is ELayout.COLUMN
    if is_column == want_column:
        return data

    nframes = data.u_px.shape[0]
    if want_column:
        data.ss_x = np.asarray(data.ss_x).reshape(-1)
        data.ss_y = np.asarray(data.ss_y).reshape(-1)
        data_fields = ["u_px", "v_px", "mag_px", "converged", "cost", "ftol", "xtol", "niter"]
        stereo_fields = []
        if data.stereo is not None:
            stereo_fields = [field.name for field in fields(data.stereo)]
        for name in data_fields:
            owner = data
            value = getattr(owner, name, None)
            if value is not None and np.ndim(value) > 1:
                setattr(owner, name, np.asarray(value).reshape(nframes, -1))
        if data.stereo is not None:
            for name in stereo_fields:
                value = getattr(data.stereo, name)
                if value is not None and np.ndim(value) > 1:
                    setattr(data.stereo, name, np.asarray(value).reshape(nframes, -1))
        return data

    x_unique = np.unique(data.ss_x)
    y_unique = np.unique(data.ss_y)
    x_indices = np.searchsorted(x_unique, data.ss_x)
    y_indices = np.searchsorted(y_unique, data.ss_y)
    shape = (nframes, len(y_unique), len(x_unique))
    data.ss_x, data.ss_y = np.meshgrid(x_unique, y_unique)

    def grid(value):
        output = np.full(shape, np.nan, dtype=np.float64)
        output[:, y_indices, x_indices] = np.asarray(value).reshape(nframes, -1)
        return output

    data_fields = ["u_px", "v_px", "mag_px", "converged", "cost", "ftol", "xtol", "niter"]
    for name in data_fields:
        setattr(data, name, grid(getattr(data, name)))
    if data.stereo is not None:
        for field in fields(data.stereo):
            name = field.name
            value = getattr(data.stereo, name)
            if value is not None:
                setattr(data.stereo, name, grid(value))
    return data


def from_in_memory(raw, filenames: list[str] | None = None) -> StrainResults:
    """Adapt the owning C++ strain result buffers to ``StrainResults``."""
    nimg = int(raw.num_frames)
    if filenames is None:
        filenames = [f"strain_data_{i:04d}" for i in range(nimg)]
    elif len(filenames) != nimg:
        raise ValueError(
            f"Expected {nimg} filenames for in-memory strain results, got {len(filenames)}"
        )
    ny = int(raw.num_ss_y)
    nx = int(raw.num_ss_x)
    nwindows = ny * nx

    def frames(values):
        return np.asarray(values).reshape(nimg, ny, nx)

    deformation = np.asarray(raw.F()).reshape(nimg, ny, nx, 6)
    strain = np.asarray(raw.strain()).reshape(nimg, ny, nx, 4)
    principal = np.asarray(raw.principal_strain()).reshape(nimg, ny, nx, 2)

    return StrainResults(
        window_x=frames(raw.x()),
        window_y=frames(raw.y()),
        def_00=deformation[..., 0],
        def_01=deformation[..., 1],
        def_10=deformation[..., 2],
        def_11=deformation[..., 3],
        def_20=deformation[..., 4],
        def_21=deformation[..., 5],
        eps_xx=strain[..., 0],
        eps_xy=strain[..., 1],
        eps_yx=strain[..., 2],
        eps_yy=strain[..., 3],
        eps1=principal[..., 0],
        eps2=principal[..., 1],
        x_mm=frames(raw.x_mm()),
        y_mm=frames(raw.y_mm()),
        z_mm=frames(raw.z_mm()),
        filenames=list(filenames),
    )

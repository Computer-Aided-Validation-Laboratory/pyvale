# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


from dataclasses import dataclass
import numpy as np
from pyvale.dic.dicenum import ELayout


@dataclass(slots=True)
class StereoResults:
    """
    Data container for stereo DIC results
    """
    
    disparity_u_px: np.ndarray
    """Horizontal left-to-right stereo disparity in pixels. shape=(img_num,y,x)"""

    disparity_v_px: np.ndarray
    """Vertical left-to-right stereo disparity in pixels. shape=(img_num,y,x)"""

    u_mm: np.ndarray
    """Horizontal displacement in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    v_mm: np.ndarray
    """Vertical displacement in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    w_mm: np.ndarray
    """Axial displacement in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    x_mm: np.ndarray
    """X-coordinate in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    y_mm: np.ndarray
    """Y-coordinate in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""
    
    z_mm: np.ndarray
    """Z-coordinate in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""
    
    mag_px: np.ndarray | None = None
    """Stereo disparity magnitude in pixels, typically computed as sqrt(disparity_u_px^2 + disparity_v_px^2). shape=(img_num,y,x)"""

    converged: np.ndarray | None = None
    """boolean value for whether the subset has converged or not. shape=(img_num,y,x)"""

    cost: np.ndarray | None = None
    """Final cost or residual value from the correlation between subset in left and right image as calculated using ZNCC. shape=(img_num,y,x)"""

    ftol: np.ndarray | None = None
    """Final `ftol` value from the optimization routine, indicating function tolerance. shape=(img_num,y,x)"""
    
    xtol: np.ndarray | None = None
    """Final `xtol` value from the optimization routine, indicating solution tolerance. shape=(img_num,y,x)"""
    
    niter: np.ndarray | None = None
    """Number of iterations taken to converge for each subset point. shape=(img_num,y,x)"""

    epi_dist_px: np.ndarray | None = None
    """Perpendicular distance from the undistorted right subset centre to the
    epipolar line of the undistorted left centre, in pixels. Missing or invalid
    matches are NaN. shape=(img_num,y,x), or (img_num,subset) for column layout.
    """

@dataclass(slots=True)
class Results:
    """
    Data container for DIC analysis results.

    This dataclass stores the displacements, convergence info, and correlation data
    associated with a DIC computation.
    """

    ss_x: np.ndarray
    """The x-coordinates of the subset centers (in pixels). shape=(img_num,y,x)"""
    ss_y: np.ndarray
    """The y-coordinates of the subset centers (in pixels). shape=(img_num,y,x)"""

    u_px: np.ndarray
    """Horizontal displacements in pixels at each subset location. shape=(img_num,y,x)"""

    v_px: np.ndarray
    """Vertical displacements in pixels at each subset location. shape=(img_num,y,x)"""

    u_mm: np.ndarray | None = None
    """Horizontal displacement in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    v_mm: np.ndarray | None = None
    """Vertical displacement in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""
    
    x_mm: np.ndarray | None = None
    """X-coordinate in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    y_mm: np.ndarray | None = None
    """Y-coordinate in physical units of mm relative in cam0 world coordinate system. shape=(img_num,y,x)"""

    mag_px: np.ndarray | None = None
    """Displacement magnitude in mm at each subset location, typically computed as sqrt(u^2 + v^2). shape=(img_num,y,x)"""

    converged: np.ndarray | None = None
    """boolean value for whether the subset has converged or not. shape=(img_num,y,x)"""

    cost: np.ndarray | None = None
    """Final cost or residual value from the correlation optimization as calculated using ZNCC. shape=(img_num,y,x)"""

    ftol: np.ndarray | None = None
    """Final `ftol` value from the optimization routine, indicating function tolerance. shape=(img_num,y,x)"""

    xtol: np.ndarray | None = None
    """Final `xtol` value from the optimization routine, indicating solution tolerance. shape=(img_num,y,x)"""

    niter: np.ndarray | None = None
    """Number of iterations taken to converge for each subset point. shape=(img_num,y,x)"""

    filenames: list[str] | None = None
    """name of DIC result files that have been found"""

    stereo: StereoResults | None = None
    """Optional field to store stereo DIC results if available."""


def from_in_memory(raw,
    layout: ELayout | str = ELayout.COLUMN,
                   filenames: list[str] | None = None) -> Results:

    """Adapt a C++ ``InMemoryResults`` object to the public result dataclasses.

    Numeric floating-point fields remain NumPy views over the owning C++ result
    object. The convergence flags are converted from uint8 to NumPy bool.
    """

    layout = ELayout(layout)
    if layout not in {ELayout.COLUMN, ELayout.MATRIX}:
        raise ValueError("layout must be 'COLUMN' or 'MATRIX'")

    nimg = int(raw.num_frames)
    if filenames is None:
        filenames = [f"dic_data_{frame:04d}" for frame in range(nimg)]
    elif len(filenames) != nimg:
        raise ValueError(
            f"Expected {nimg} filenames for in-memory results, got {len(filenames)}"
        )

    ss_x = np.asarray(raw.ss_x())
    ss_y = np.asarray(raw.ss_y())

    if layout == ELayout.COLUMN:
        ss_x_out = ss_x
        ss_y_out = ss_y

        def frames(values):
            return np.asarray(values).reshape(nimg, -1)
    else:
        # C++ stores only the valid ROI subsets. Recreate the same compact
        # rectangular matrix layout as import_2d/import_3d.
        x_unique = np.unique(ss_x)
        y_unique = np.unique(ss_y)
        x_indices = np.searchsorted(x_unique, ss_x)
        y_indices = np.searchsorted(y_unique, ss_y)
        shape = (nimg, len(y_unique), len(x_unique))
        ss_x_out, ss_y_out = np.meshgrid(x_unique, y_unique)

        def frames(values):
            grid = np.full(shape, np.nan)
            grid[:, y_indices, x_indices] = np.asarray(values).reshape(nimg, -1)
            return grid

    result = Results(
        ss_x=ss_x_out,
        ss_y=ss_y_out,
        u_px=frames(raw.u()),
        v_px=frames(raw.v()),
        mag_px=frames(raw.mag()),
        converged=frames(raw.converged()).astype(bool),
        cost=frames(raw.cost()),
        ftol=frames(raw.ftol()),
        xtol=frames(raw.xtol()),
        niter=frames(raw.niter()),
        filenames=list(filenames),
    )

    if raw.stereo:
        result.stereo = StereoResults(
            disparity_u_px=frames(raw.disparity_u()),
            disparity_v_px=frames(raw.disparity_v()),
            mag_px=frames(raw.disparity_mag()),
            u_mm=frames(raw.u_mm()),
            v_mm=frames(raw.v_mm()),
            w_mm=frames(raw.w_mm()),
            x_mm=frames(raw.x_mm()),
            y_mm=frames(raw.y_mm()),
            z_mm=frames(raw.z_mm()),
            converged=frames(raw.stereo_converged()).astype(bool),
            cost=frames(raw.stereo_cost()),
            ftol=frames(raw.stereo_ftol()),
            xtol=frames(raw.stereo_xtol()),
            niter=frames(raw.stereo_niter()),
            epi_dist_px=frames(raw.epi_dist_px()),
        )

    return result

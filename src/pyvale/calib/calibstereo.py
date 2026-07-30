# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

"""Stereo camera calibration routines for dot-target image pairs."""


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize._lsq import common
import cv2
import math
from pathlib import Path
from typing import Literal
from enum import Enum

import pyvale.calib.calibcpp as calibcpp
from pyvale.calib.calibdataclass import Calib, CamIntrinsics
import pyvale.common_cpp.common_cpp as common_cpp
import pyvale.common_py.util as common_py_util

class ReprojError(str, Enum):
    """Available reprojection error formulations for calibration refinement."""

    RMSE = "RMSE"
    MEAN = "MEAN"
    MSE = "MSE"

def calibrate_stereo(dots_cam0: list[np.ndarray] | np.ndarray, 
                     dots_cam1: list[np.ndarray] | np.ndarray, 
                     grid: list[np.ndarray] | np.ndarray, 
                     img_dims: list[int] | np.ndarray, 
                     filenames: list[str] | list[Path] | None = None,
                     optimize_distortion: bool = True,
                     precision: float = 0.001,
                     max_iter: int = 40,
                     num_threads: int | None = None,
                     error_formulation: Literal["RMSE", "MEAN", "MSE"] = "RMSE"
                     ) -> tuple[Calib, np.ndarray, np.ndarray]:
    """Estimate stereo camera calibration parameters from matched dot targets.

    The function starts with OpenCV single-camera and stereo calibration to get
    an initial estimate, then passes the flattened parameters to the C++ bundle
    adjustment routine for refinement. The returned calibration is stored in
    Pyvale dataclasses and uses millimetres for translation and degrees for the
    stereo rotation angles.

    Parameters
    ----------
    dots_cam0, dots_cam1 : list[np.ndarray] or np.ndarray
        Matched 2D image coordinates for camera 0 and camera 1. Each image pair
        must contain the same number of points in the same order.
    grid : list[np.ndarray] or np.ndarray
        Corresponding 3D calibration target coordinates for each image pair.
        The first dimension must match the number of image pairs.
    img_dims : list[int] or np.ndarray
        Image dimensions as ``[width, height]`` in pixels.
    filenames : list[str] or list[pathlib.Path] or None, optional
        Optional names for the calibration images. This is currently only
        checked for length consistency when provided.
    optimize_distortion : bool, optional
        If ``True``, refine radial and tangential distortion coefficients. If
        ``False``, distortion coefficients are set to zero before refinement.
    precision : float, optional
        Convergence tolerance passed to the C++ optimizer.
    max_iter : int, optional
        Maximum number of C++ refinement iterations.
    num_threads : int or None, optional
        Number of OpenMP threads to use in the C++ optimizer. If ``None``, the
        current runtime default is used.
    error_formulation : {"RMSE", "MEAN", "MSE"}, optional
        Error metric used by the C++ calibration optimizer.

    Returns
    -------
    tuple[Calib, np.ndarray, np.ndarray]
        The refined stereo calibration, followed by per-point reprojection
        errors for camera 0 and camera 1.

    Raises
    ------
    TypeError
        If the camera point arrays and grid are not provided using compatible
        container types, or if ``optimize_distortion`` is not boolean.
    ValueError
        If image-pair counts, point counts, shapes, filenames, image dimensions,
        or the error formulation are invalid.
    """

    # Type checks
    if type(dots_cam0) != type(dots_cam1):
        raise TypeError(f"dots_cam0 and dots_cam1 must be the same type, got {type(dots_cam0)} and {type(dots_cam1)}")
    if type(dots_cam0) != type(grid):
        raise TypeError(f"dots and grid must be the same type, got {type(dots_cam0)} and {type(grid)}")

    # shape check if np.ndarray
    if isinstance(dots_cam0, np.ndarray):
        if dots_cam0.shape != dots_cam1.shape:
            raise ValueError(f"dots_cam0 and dots_cam1 ndarray shapes do not match: {dots_cam0.shape} vs {dots_cam1.shape}")
        if dots_cam0.shape[0] != grid.shape[0]:
            raise ValueError(f"dots and grid have mismatched first dimension: {dots_cam0.shape[0]} vs {grid.shape[0]}")

    # length check if list
    if len(dots_cam0) != len(dots_cam1):
        raise ValueError(f"dots_cam0 and dots_cam1 are different lengths: {len(dots_cam0)} vs {len(dots_cam1)}")
    if len(dots_cam0) != len(grid):
        raise ValueError(f"dots_cam0 and grid are different lengths: {len(dots_cam0)} vs {len(grid)}")

    # check elements size of dots/grid
    if isinstance(dots_cam0, list):
        for i, (d0, d1, g) in enumerate(zip(dots_cam0, dots_cam1, grid)):
            if d0.shape != d1.shape:
                raise ValueError(f"Shape mismatch at index {i}: dots_cam0 {d0.shape} vs dots_cam1 {d1.shape}")
            if d0.shape[0] != g.shape[0]:
                raise ValueError(f"Point count mismatch at index {i}: dots {d0.shape[0]} vs grid {g.shape[0]}")

    # --- filenames check (only if provided) ---
    if filenames is not None and len(filenames) != len(dots_cam0):
        raise ValueError(f"filenames length={len(filenames)} does not match with dots length={len(dots_cam0)}")

    # image dimensions check
    if len(img_dims) != 2:
        raise ValueError(f"img_dims should have 2 elements (width, height), got {len(img_dims)}")
    if any(d <= 0 for d in img_dims):
        raise ValueError(f"img_dims must be positive, got {img_dims}")

    if not isinstance(optimize_distortion, (bool, np.bool_)):
        raise TypeError("optimize_distortion must be a boolean")

    num_file_pairs = len(dots_cam0)

    
    flat_dots_cam0 = np.concatenate(dots_cam0,axis=0).astype(np.float32).ravel().tolist()
    flat_dots_cam1 = np.concatenate(dots_cam1,axis=0).astype(np.float32).ravel().tolist()
    flat_grid = np.concatenate(grid, axis=0).astype(np.float32).ravel().tolist()
    lengths = np.array([arr.shape[0] for arr in dots_cam1],dtype=np.int32).tolist()


    common_py_util.info(f"Performing Initial calibration guess...")

    # initial parameter guess with fixed distortion parameters
    flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    _, K0, D0, rvecs0, tvecs0 = cv2.calibrateCamera(grid, dots_cam0, img_dims, None, None, flags=flags)
    _, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(grid, dots_cam1, img_dims, None, None, flags=flags)

    # stereo calibration with variable distortion parameters. Zhang method.
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)
    ret, K0_stereo, D0_stereo, K1_stereo, D1_stereo, R_stereo, T_stereo, E, F = cv2.stereoCalibrate(
        grid, dots_cam0, dots_cam1,
        K0, D0, K1, D1,
        img_dims,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        criteria=criteria
    )


    common_py_util.info(f"Initial calibration guess completed.")

    # Compute consistent cam0 poses using refined intrinsics  
    rvecs0_consistent = []
    tvecs0_consistent = []
    for i in range(len(grid)):
        ret, rvec, tvec = cv2.solvePnP(grid[i], dots_cam0[i],K0_stereo, D0_stereo,flags=cv2.SOLVEPNP_ITERATIVE)
        rvecs0_consistent.append(rvec)
        tvecs0_consistent.append(tvec)

    # check reprojection following refinement
    for i in range(len(rvecs0)):
        R0, _ = cv2.Rodrigues(rvecs0_consistent[i])
        R1_expected = R_stereo @ R0
        T1_expected = R_stereo @ tvecs0_consistent[i] + T_stereo

        proj, _ = cv2.projectPoints(grid[i], cv2.Rodrigues(R1_expected)[0], T1_expected, K1_stereo, D1_stereo)
        err = np.mean(np.linalg.norm(proj.squeeze() - dots_cam1[i].squeeze(), axis=1))
        #print(f"Refinement: Image {i} cam1 reprojection error: {err:.4f} px")

    # visualize_initial_projection_no_images(
    #     grid,
    #     dots_cam0, dots_cam1,
    #     K0, D0, rvecs0, tvecs0,
    #     K1, D1, rvecs1, tvecs1,
    #     img_dims
    # )
    #


    # get into correct format for C++
    rvec_stereo, _ = cv2.Rodrigues(R_stereo)

    # distortion
    D0 = D0_stereo.flatten()
    D1 = D1_stereo.flatten()
    if not optimize_distortion:
        D0 = np.zeros_like(D0)
        D1 = np.zeros_like(D1)

    # intrinsic cam matrix
    fx0, fy0, fs0, cx0, cy0 = K0_stereo[0, 0], K0_stereo[1, 1], K0_stereo[0,1], K0_stereo[0, 2], K0_stereo[1, 2]
    fx1, fy1, fs1, cx1, cy1 = K1_stereo[0, 0], K1_stereo[1, 1], K1_stereo[0,1], K1_stereo[0, 2], K1_stereo[1, 2]

    # Initial poses from intrinsics_cam0
    initial_poses_cam0 = []
    for i in range(num_file_pairs):
            initial_poses_cam0.extend(rvecs0_consistent[i].flatten())
            initial_poses_cam0.extend(tvecs0_consistent[i].flatten())


    # full list of initial parameters
    initial_params = np.hstack([fx0, fy0, fs0, cx0, cy0, D0,
                                fx1, fy1, fs1, cx1, cy1, D1,
                                rvec_stereo.flatten(), T_stereo.flatten(),
                                initial_poses_cam0])

    flat_initial_params = initial_params.ravel().tolist()


    error_formulation_enum = ReprojError(error_formulation)
    error_formulation_cpp  = getattr(calibcpp.ReprojError, error_formulation_enum.name)

    #set the number of OMP threads
    if num_threads is not None:
        common_cpp.set_num_threads(num_threads)


    common_py_util.info("Starting stereo calibration bundle adjustment...")
    result_cpp = calibcpp.calibrate_stereo(flat_initial_params,
                                           flat_dots_cam0,
                                           flat_dots_cam1,
                                           flat_grid,
                                           lengths,
                                           img_dims[0],
                                           img_dims[1],
                                           num_file_pairs,
                                           bool(optimize_distortion),
                                           precision, 
                                           max_iter,
                                           error_formulation_cpp)

    calib_cpp = result_cpp.calib
    calib = Calib(
        cam0=CamIntrinsics(calib_cpp.cam0.fx, calib_cpp.cam0.fy, calib_cpp.cam0.fs,
                            calib_cpp.cam0.cx, calib_cpp.cam0.cy,
                            np.asarray(calib_cpp.cam0.distortion, dtype=np.float64)),
        cam1=CamIntrinsics(calib_cpp.cam1.fx, calib_cpp.cam1.fy, calib_cpp.cam1.fs,
                            calib_cpp.cam1.cx, calib_cpp.cam1.cy,
                            np.asarray(calib_cpp.cam1.distortion, dtype=np.float64)),
        translation=np.asarray(calib_cpp.translation, dtype=np.float64),
        rotation=np.asarray(np.rad2deg(calib_cpp.rotation), dtype=np.float64),
    )

    errors0 = np.asarray(result_cpp.errors_cam0, dtype=np.float64)
    errors1 = np.asarray(result_cpp.errors_cam1, dtype=np.float64)

    return calib, errors0, errors1





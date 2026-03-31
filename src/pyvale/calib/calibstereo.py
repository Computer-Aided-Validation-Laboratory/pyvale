# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


import matplotlib.pyplot as plt
import numpy as np
import cv2


import pyvale.calib.calibcpp as calibcpp


def stereo_calibration(dots_cam0, dots_cam1, grid, img_dims, method: str="bundle_adjustment") -> None:

    # check dots are the same length
    if len(dots_cam0) != len(dots_cam1):
        ValueError(f"ERROR: dots_cam0 and dots_cam1 are different lengths:"
                   f" - length of dot_cam0: {len(dots_cam0)}"
                   f" - length of dot_cam1: {len(dots_cam1)}")

    # check dots and grid are the same length
    if len(dots_cam0) != len(grid):
        ValueError(f"ERROR: dots_cam0 and grid are different lengths:"
                   f" - length of dot_cam0: {len(dots_cam0)}"
                   f" - length of grid:     {len(grid)}")

    num_file_pairs = len(dots_cam0)

    if method=="bundle_adjustment":
        bundle(dots_cam0, dots_cam1, grid, img_dims, num_file_pairs)
    elif method=="zhang":
        zhang(dots_cam0, dots_cam1, grid, img_dims, num_file_pairs)
    elif method=="cpp":
        cpp(dots_cam0, dots_cam1, grid, img_dims, num_file_pairs) 
    else:    
        raise ValueError(f"ERROR: Unknown calibration method: {method}. "
                   f"Allowed options: 'bundle', 'zhang', 'cpp'")




def cpp(dots_cam0, dots_cam1, grid, img_dims, num_file_pairs):

    flat_dots_cam0 = np.concatenate(dots_cam0,axis=0).astype(np.float32).ravel().tolist()
    flat_dots_cam1 = np.concatenate(dots_cam1,axis=0).astype(np.float32).ravel().tolist()
    flat_grid = np.concatenate(grid, axis=0).astype(np.float32).ravel().tolist()
    lengths = np.array([arr.shape[0] for arr in dots_cam1],dtype=np.int32).tolist()

    # initial parameter guess
    flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    _, K0, D0, rvecs0, tvecs0 = cv2.calibrateCamera(grid, dots_cam0, img_dims, None, None, flags=flags)
    _, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(grid, dots_cam1, img_dims, None, None, flags=flags)

    ret, K0_stereo, D0_stereo, K1_stereo, D1_stereo, R_stereo, T_stereo, E, F = cv2.stereoCalibrate(
        grid, dots_cam0, dots_cam1,
        K0, D0, K1, D1,
        img_dims,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    rvec_stereo, _ = cv2.Rodrigues(R_stereo)

    D0 = D0.flatten()
    D1 = D1.flatten()

    fx0, fy0, cx0, cy0 = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]
    fx1, fy1, cx1, cy1 = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]


    # Initial poses from intrinsics_cam0
    initial_poses_cam0 = []
    for i in range(num_file_pairs):
            initial_poses_cam0.extend(rvecs0[i].flatten())
            initial_poses_cam0.extend(tvecs0[i].flatten())


    # full list of initial parameters
    initial_params = np.hstack([fx0, fy0, cx0, cy0, D0,
                                fx1, fy1, cx1, cy1, D1,
                                rvec_stereo.flatten(), T_stereo.flatten(),
                                initial_poses_cam0])


    flat_initial_params = initial_params.ravel().tolist()

    calibcpp.stereo_calibration(flat_initial_params,flat_dots_cam0, flat_dots_cam1, flat_grid, 
                                lengths, img_dims[0], img_dims[1], num_file_pairs)




def zhang(dots_cam0, dots_cam1, grid, img_dims, num_file_pairs):

    print(f"Running calibration with {len(grid)} valid image pairs...")

    # Left and Right cam calib
    _, K0, D0, rvec0, tvec0 = cv2.calibrateCamera(grid, dots_cam0, img_dims, None, None)
    _, K1, D1, rvec1, tvec1 = cv2.calibrateCamera(grid, dots_cam1, img_dims, None, None)

    error0 = []
    error1 = []

    for i, objp in enumerate(grid):

        # Projected points
        projected_points_opt0, _ = cv2.projectPoints(objp, rvec0[i], tvec0[i], K0, D0)
        projected_points_opt1, _ = cv2.projectPoints(objp, rvec1[i], tvec1[i], K1, D1)

        # Ensure points are Nx2 arrays
        projected_points_opt0 = projected_points_opt0.reshape(-1, 2)
        projected_points_opt1 = projected_points_opt1.reshape(-1, 2)

        dots_cam0_i = dots_cam0[i].reshape(-1, 2)
        dots_cam1_i = dots_cam1[i].reshape(-1, 2)

        # Compute RMS reprojection error
        diff0 = np.sqrt(np.sum((dots_cam0_i - projected_points_opt0)**2, axis=1))
        diff1 = np.sqrt(np.sum((dots_cam1_i - projected_points_opt1)**2, axis=1))

        error0 = np.mean(diff0)
        error1 = np.mean(diff1)
        print("ERROR", error0, error1)

    print(f"Mean left RMS error: {np.mean(error0):.4f} px")
    print(f"Mean right RMS error: {np.mean(error1):.4f} px")

    # stereo calib
    ret, K0_opt, D0_opt, K1_opt, D1_opt, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=grid,
        imagePoints1=dots_cam0,
        imagePoints2=dots_cam1,
        cameraMatrix1=K0,
        distCoeffs1=D0,
        cameraMatrix2=K1,
        distCoeffs2=D1,
        imageSize=img_dims,
        flags=0
    )

    print("\n--- Calibration Results ---")
    print("Calibration RMS error:", ret)
    print('\nLeft Camera Matrix:\n', K0_opt)
    print('Left Distortion Coefficients:\n', D0_opt)
    print('\nRight Camera Matrix:\n', K1_opt)
    print('Right Distortion Coefficients:\n', D1_opt)
    print('\nRotation Matrix (R):\n', R)
    print('Translation Vector (T):\n', T)

    error0 = []
    error1 = []


    _, K0_test, D0_test, rvec0_opt, tvec0_opt = cv2.calibrateCamera(
        objectPoints=grid,
        imagePoints=dots_cam0,
        imageSize=img_dims,
        cameraMatrix=K0_opt,
        distCoeffs=D0_opt,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    _, K1_test, D1_test, rvec1_opt, tvec1_opt = cv2.calibrateCamera(
        objectPoints=grid,
        imagePoints=dots_cam1,
        imageSize=img_dims,
        cameraMatrix=K1_opt,
        distCoeffs=D1_opt,
        flags=cv2.CALIB_FIX_INTRINSIC
    )


    for i, objp in enumerate(grid):

        # Projected points
        projected_points_opt0, _ = cv2.projectPoints(objp, rvec0[i], tvec0[i], K0_opt, D0_opt)
        projected_points_opt1, _ = cv2.projectPoints(objp, rvec1[i], tvec1[i], K1_opt, D1_opt)

        # Ensure points are Nx2 arrays
        projected_points_opt0 = projected_points_opt0.reshape(-1, 2)
        projected_points_opt1 = projected_points_opt1.reshape(-1, 2)

        dots_cam0_i = dots_cam0[i].reshape(-1, 2)
        dots_cam1_i = dots_cam1[i].reshape(-1, 2)

        # Compute RMS reprojection error
        diff0 = np.sqrt(np.sum((dots_cam0_i - projected_points_opt0)**2, axis=1))
        diff1 = np.sqrt(np.sum((dots_cam1_i - projected_points_opt1)**2, axis=1))

        error0 = np.mean(diff0)
        error1 = np.mean(diff1)
        print("ERROR", error0, error1)

        fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        ax[0].scatter(dots_cam0_i[:, 0], dots_cam0_i[:, 1], label='Observed', c='blue')
        ax[0].scatter(projected_points_opt0[:, 0], projected_points_opt0[:, 1], label='Projected', c='red', marker='x')
        ax[1].scatter(dots_cam1_i[:, 0], dots_cam1_i[:, 1], label='Observed', c='blue')
        ax[1].scatter(projected_points_opt1[:, 0], projected_points_opt1[:, 1], label='Projected', c='red', marker='x')
        plt.gca().invert_yaxis()  # Optional: match image coordinates
        plt.ticklabel_format(style='plain')
        plt.grid(True)
        plt.show()

    print(f"Mean left RMS error: {np.mean(error0):.4f} px")
    print(f"Mean right RMS error: {np.mean(error1):.4f} px")

    # Save as .npy (NumPy binary)
    # np.save('stereo_calibration.npy', {
    #     'ret': ret,
    #     'Kl': Kl,
    #     'Dl': Dl,
    #     'Kr': Kr,
    #     'Dr': Dr,
    #     'R': R,
    #     'T': T,
    #     'E': E,
    #     'F': F
    # })

    # # Save as .yaml (human-readable)
    # calib_data = {
    #     'ret': float(ret),
    #     'Kl': Kl.tolist(),
    #     'Dl': Dl.tolist(),
    #     'Kr': Kr.tolist(),
    #     'Dr': Dr.tolist(),
    #     'R': R.tolist(),
    #     'T': T.tolist(),
    #     'E': E.tolist(),
    #     'F': F.tolist()
    # }

    # with open('stereo_calibration.yaml', 'w') as f:
    #     yaml.dump(calib_data, f)

def bundle(dots_cam0, dots_cam1, grid, img_dims, num_file_pairs):

    flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    _, K0, D0, rvecs0, tvecs0 = cv2.calibrateCamera(grid, dots_cam0, img_dims, None, None, flags=flags)
    _, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(grid, dots_cam1, img_dims, None, None, flags=flags)

    ret, K0_stereo, D0_stereo, K1_stereo, D1_stereo, R_stereo, T_stereo, E, F = cv2.stereoCalibrate(
        grid, dots_cam0, dots_cam1,
        K0, D0, K1, D1,
        img_dims,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    rvec_stereo, _ = cv2.Rodrigues(R_stereo)

    D0 = D0.flatten()
    D1 = D1.flatten()

    fx0, fy0, cx0, cy0 = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]
    fx1, fy1, cx1, cy1 = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]


    # Initial poses from intrinsics_cam0
    initial_poses_cam0 = []
    for i in range(num_file_pairs):
            initial_poses_cam0.extend(rvecs0[i].flatten())
            initial_poses_cam0.extend(tvecs0[i].flatten())


    # full list of initial parameters
    initial_params = np.hstack([fx0, fy0, cx0, cy0, D0,
                                fx1, fy1, cx1, cy1, D1,
                                rvec_stereo.flatten(), T_stereo.flatten(),
                                initial_poses_cam0])

    result = least_squares(
        bundle_adjustment_error,
        initial_params,
        args=(grid, dots_cam0, dots_cam1, num_file_pairs),
        verbose=2,
        max_nfev=500,  # Increased iterations for complex optimization
        # x_scale=scales,
        # bounds=(lower_bounds, upper_bounds),
        ftol=1e-10,     # Tighter tolerance for better accuracy
        xtol=None
    )

     # --- Step 7: Extract results ---
    opt = result.x
    fx0, fy0, cx0, cy0 = opt[0:4]
    D0_opt = opt[4:9]
    fx1, fy1, cx1, cy1 = opt[9:13]
    D1_opt = opt[13:18]
    rvec_stereo = opt[18:21]
    tvec_stereo = opt[21:24]
    base = 24 + 0 * 6
    rvec0 = opt[base:base+3]
    tvec0 = opt[base+3:base+6]

    K0_opt = np.array([[fx0, 0, cx0],
                    [0, fy0, cy0],
                    [0,  0,   1]])
    K1_opt = np.array([[fx1, 0, cx1],
                    [0, fy1, cy1],
                    [0,  0,   1]])

    print("\n--- Optimized Left Camera Intrinsics ---")
    print("K0:\n", K0_opt)
    print("Distortion:", D0_opt)

    print("\n--- Optimized Right Camera Intrinsics ---")
    print("K1:\n", K1_opt)
    print("Distortion:", D1_opt)

    print("\n--- Stereo Transform (Right from Left) ---")
    print("Rotation Vector:", rvec_stereo)
    print("Translation Vector:", tvec_stereo)

    # ADD THIS: Calculate right camera pose from stereo transform
    R_stereo, _ = cv2.Rodrigues(rvec_stereo)
    R0, _ = cv2.Rodrigues(rvec0)
    T0 = tvec0.reshape(3, 1)

    # Right camera pose
    R1 = R_stereo @ R0
    T1 = R_stereo @ T0 + tvec_stereo.reshape(3, 1)
    rvec1, _ = cv2.Rodrigues(R1)
    tvec1 = T1.flatten()  # Make sure it's 1D for cv2.projectPoints


    # Compute right camera pose from stereo transform
    R_stereo, _ = cv2.Rodrigues(rvec_stereo)
    R0, _ = cv2.Rodrigues(rvec0)
    T0 = tvec0.reshape(3, 1)

    R1 = R_stereo @ R0
    T1 = R_stereo @ T0 + tvec_stereo.reshape(3, 1)
    rvec1, _ = cv2.Rodrigues(R1)
    tvec1 = T1.flatten()

    # Loop over all image pairs
    for i in range(num_file_pairs):
        rvec_i = opt[base + i*6 : base + i*6 + 3]
        tvec_i = opt[base + i*6 + 3 : base + i*6 + 6]

        # Project points to cam0
        proj0, _ = cv2.projectPoints(grid[i], rvec_i, tvec_i, K0_opt, D0_opt)
        proj0 = proj0.reshape(-1, 2)

        # Compose pose for cam1
        R_i, _ = cv2.Rodrigues(rvec_i)
        R1_i = R_stereo @ R_i
        T1_i = R_stereo @ tvec_i.reshape(3, 1) + tvec_stereo.reshape(3, 1)
        rvec1_i, _ = cv2.Rodrigues(R1_i)
        tvec1_i = T1_i.flatten()

        # Project points to cam1
        proj1, _ = cv2.projectPoints(grid[i], rvec1_i, tvec1_i, K1_opt, D1_opt)
        proj1 = proj1.reshape(-1, 2)

        # Observed points
        obs0 = dots_cam0[i].reshape(-1, 2)
        obs1 = dots_cam1[i].reshape(-1, 2)

        print(np.sqrt((obs0 - proj0)**2))
        print(np.sqrt((obs1 - proj1)**2))

        # RMS error
        err0 = np.sqrt(np.sum((obs0 - proj0)**2, axis=1)).mean()
        err1 = np.sqrt(np.sum((obs1 - proj1)**2, axis=1)).mean()
        print(f"Image {i}: RMS Error cam0 = {err0:.3f}, cam1 = {err1:.3f}")

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].scatter(obs0[:, 0], obs0[:, 1], c='blue', label='Observed')
        ax[0].scatter(proj0[:, 0], proj0[:, 1], c='red', marker='x', label='Projected')
        ax[0].set_title(f'Camera 0 - Image {i}')
        ax[0].invert_yaxis()
        ax[0].legend()
        ax[0].grid(True)

        ax[1].scatter(obs1[:, 0], obs1[:, 1], c='blue', label='Observed')
        ax[1].scatter(proj1[:, 0], proj1[:, 1], c='red', marker='x', label='Projected')
        ax[1].set_title(f'Camera 1 - Image {i}')
        ax[1].invert_yaxis()
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

def bundle_adjustment_error(params, gridpoints, dots_cam0, dots_cam1, num_img):

    # --- Extract intrinsics ---
    fx0, fy0, cx0, cy0 = params[0:4]
    D0 = params[4:9]
    fx1, fy1, cx1, cy1 = params[9:13]
    D1 = params[13:18] 

    # Stereo tranlation and rotation
    rvec_stereo = params[18:21]
    tvec_stereo = params[21:24]
    R_stereo, _ = cv2.Rodrigues(rvec_stereo)

    # Camera matrices
    K0 = np.array([[fx0, 0, cx0],
                [0, fy0, cy0],
                [0,  0,   1]])
    K1 = np.array([[fx1, 0, cx1],
                [0, fy1, cy1],
                [0,  0,   1]])

    pose0_start = 24
    residuals = []

    for i in range(num_img):
        
        # Cam0 Pose
        rvec0 = params[pose0_start + i*6 : pose0_start + i*6 + 3]
        tvec0 = params[pose0_start + i*6 + 3 : pose0_start + i*6 + 6]
        R0, _ = cv2.Rodrigues(rvec0)
        T0 = tvec0.reshape(3, 1)

        # Cam1 pose (derived from cam0 + stereo)
        R1 = R_stereo @ R0
        T1 = R_stereo @ T0 + tvec_stereo.reshape(3, 1)
        rvec1, _ = cv2.Rodrigues(R1)
        tvec1 = T1

        # Projected points
        proj0, _ = cv2.projectPoints(gridpoints[i], rvec0, tvec0, K0, D0)
        proj1, _ = cv2.projectPoints(gridpoints[i], rvec1, tvec1, K1, D1)
        proj0 = proj0.reshape(-1, 2)
        proj1 = proj1.reshape(-1, 2)
        
        # residual
        res0 = (proj0 - dots_cam0[i]).flatten()
        res1 = (proj1 - dots_cam1[i]).flatten()
        residuals.extend(res0)
        residuals.extend(res1)

    return np.array(residuals)



# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import cv2
import numpy as np
import re
import os
import yaml
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from scipy.optimize import least_squares

import pyvale.calib.calibcpp as calibcpp

def dot_detection(cam0: Path | list[Path] | np.ndarray | str,
                  cam1: Path | list[Path] | np.ndarray | str,
                  grid_height: int, grid_width: int,
                  grid_spacing: float,
                  visualisation: bool=False) -> tuple[list, list, list, np.ndarray]:


    files_cam0 = []
    files_cam1 = []

    # Check cam0 and cam1 are same type
    if type(cam0) is not type(cam1):
        raise ValueError(
            f"cam0 and cam1 are different dtypes: cam0={type(cam0)}, cam1={type(cam1)}"
        )

    # check np.ndarray dims agree
    if isinstance(cam0, np.ndarray):
        if cam0.shape != cam1.shape:
            raise ValueError(
                f"cam0 and cam1 are different numpy shapes: cam0.shape={cam0.shape}, cam1.shape={cam1.shape}"
            )

    # handle strings. convert to path for import
    elif isinstance(cam0, (str, Path)) and isinstance(cam1, (str, Path)):
        cam0 = Path(cam0)
        cam1 = Path(cam1)
        files_cam0, files_cam1 = get_file_list(cam0, cam1)

    # handle lists
    elif isinstance(cam0, list) and isinstance(cam1, list):
        cam0 = [Path(x) for x in cam0]
        cam1 = [Path(x) for x in cam1]

        if len(cam0) != len(cam1):
            raise ValueError("Number of images for camera 0 and camera 1 must be identical. \n"
                                f"number of images for cam0: {len(cam0)} \n"
                                f"number of images for cam1: {len(cam1)} \n")

        files_cam0 = cam0
        files_cam1 = cam1

    else:
        raise TypeError(f"Unsupported input type: cam0={type(cam0)}")


    # Generate full 3D grid
    fullgrid_3d = np.zeros((grid_width * grid_height, 3), np.float32)
    fullgrid_3d[:, :2] = np.mgrid[-2:grid_width-2, -2:grid_height-2].T.reshape(-1, 2)
    fullgrid_3d[:, :2] *= grid_spacing


    missing_idx = np.array([
        [2, grid_height-2-1],
        [2, grid_height-7],
        [9, grid_height-2-1],
    ])

    missing_grid = (missing_idx * grid_spacing - 2*grid_spacing)
    missing_grid = missing_grid.astype(np.float32)

    # Convert to flat indices
    missing_indices = [y * grid_width + x for (x, y) in missing_idx]
    mask = np.ones(len(fullgrid_3d), dtype=bool)
    mask[missing_indices] = False
    finalgrid_3d = fullgrid_3d[mask]
    finalgrid_2d = finalgrid_3d[:, :2]

    # create a light n dark blob detector
    detector_lght = create_blob_detector(light=True)
    detector_dark = create_blob_detector(light=False)

    # arrays that are going to contain matching points for each image
    gridpoints = []
    dots_cam0 = []
    dots_cam1 = []

    num_file_pairs = len(files_cam1)
    img_dims = np.zeros((2))

    for i in range(0, num_file_pairs):

        print("Running Dot detection on image pair: "
                f"{os.path.basename(files_cam0[i])}, "
                f"{os.path.basename(files_cam1[i])}")

        # read images
        img0 = cv2.imread(str(files_cam0[i]), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(str(files_cam1[i]), cv2.IMREAD_GRAYSCALE)


        if img0 is None or img1 is None:
            print(f"Skipping missing pair: {files_cam0[i]} {files_cam1[i]}")
            continue

        img_dims0 = (img0.shape[1], img0.shape[0])
        img_dims1 = (img1.shape[1], img1.shape[0])

        # check image dimensions agree
        if (img_dims0[0] != img_dims1[0]) or (img_dims0[1] != img_dims1[1]):
            print("image dimensions don't agree: "
                f" - dimensions of {files_cam0}: {img_dims0}"
                f" - dimensions of {files_cam1}: {img_dims1}"
                "Skipping image pair")

        img_dims = img_dims0


        # Detect LIGHT blobs
        keypoints_lght_cam0 = detector_lght.detect(img0)
        keypoints_lght_cam1 = detector_lght.detect(img1)

        # there should always be 3 points in keypoints_lght_cam0 and keypoints_lght_cam1
        if len(keypoints_lght_cam0) != 3 or len(keypoints_lght_cam1) != 3:
            print(f"Skipping pair due to insufficient light blobs.")
            print("left:", len(keypoints_lght_cam0))
            print("right:", len(keypoints_lght_cam1))
            num_file_pairs = num_file_pairs-1
            continue


        # Detect DARK blobs
        keypoints_dark_cam0 = detector_dark.detect(img0)
        keypoints_dark_cam1 = detector_dark.detect(img1)


        if visualisation:

            # Convert grayscale images to BGR for visualization
            img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
            img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

            # Make overlays for left and right images in color
            overlay_l = img0_color.copy()
            overlay_r = img1_color.copy()

            alpha = 0.5  # 0 = fully transparent, 1 = fully opaque
            r = 20      # radius for circles

            # Draw dark cam0 keypoints in red
            for kp in keypoints_dark_cam0:
                x, y = map(int, kp.pt)
                cv2.circle(overlay_l, (x, y), r, (0, 0, 255), thickness=-1)

            # Draw dark cam1 keypoints in red
            for kp in keypoints_dark_cam1:
                x, y = map(int, kp.pt)
                cv2.circle(overlay_r, (x, y), r, (0, 0, 255), thickness=-1)

            # Draw light cam0 keypoints in green
            for kp in keypoints_lght_cam0:
                x, y = map(int, kp.pt)
                cv2.circle(overlay_l, (x, y), r, (0, 255, 0), thickness=-1)

            # Draw light cam1 keypoints in green
            for kp in keypoints_lght_cam1:
                x, y = map(int, kp.pt)
                cv2.circle(overlay_r, (x, y), r, (0, 255, 0), thickness=-1)

            # Blend overlays with original color images
            im_with_keypoints_l = cv2.addWeighted(overlay_l, alpha, img0_color, 1 - alpha, 0)
            im_with_keypoints_r = cv2.addWeighted(overlay_r, alpha, img1_color, 1 - alpha, 0)

            # Combine side-by-side
            side_by_side = np.hstack((im_with_keypoints_l, im_with_keypoints_r))

            # Show result
            cv2.namedWindow("Stereo Keypoints", cv2.WINDOW_NORMAL)
            cv2.imshow("Stereo Keypoints", side_by_side)
            cv2.resizeWindow("Stereo Keypoints", 1200, 600)
            cv2.waitKey(1)
            # im_with_keypoints_l = cv2.drawKeypoints(img0, keypoints_dark_cam0, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # im_with_keypoints_r = cv2.drawKeypoints(img1, keypoints_dark_cam1, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # im_with_keypoints_l = cv2.drawKeypoints(im_with_keypoints_l, keypoints_lght_cam0, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # im_with_keypoints_r = cv2.drawKeypoints(im_with_keypoints_r, keypoints_lght_cam1, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # side_by_side = np.hstack((im_with_keypoints_l, im_with_keypoints_r))
            # cv2.namedWindow("Stereo Keypoints", cv2.WINDOW_NORMAL)
            # cv2.imshow("Stereo Keypoints", side_by_side)
            # cv2.waitKey(0)



        # there should always be 3 points in keypoints_lght_cam0 and keypoints_lght_cam1
        if len(keypoints_lght_cam0) != 3 or len(keypoints_lght_cam1) != 3:
            print(f"WARNING: Skipping pair due to insufficient light blobs."
                  f"left: {len(keypoints_lght_cam0)}"
                  f"right: {len(keypoints_lght_cam1)}")
            continue

        # Convert KeyPoints to NumPy arrays
        light_pts_cam0 = np.array([kp.pt for kp in keypoints_lght_cam0], dtype=np.float32)
        light_pts_cam1 = np.array([kp.pt for kp in keypoints_lght_cam1], dtype=np.float32)


        # print(light_pts_cam0)
        # print(light_pts_cam1)

        # Order points consistently based on right angle
        light_pts_cam0_ordered = order_triangle_points_by_angle(light_pts_cam0)
        light_pts_cam1_ordered = order_triangle_points_by_angle(light_pts_cam1)

        # get the translation matrix between the triangle that forms from the light blobs between the left and right images
        cam0togrid = cv2.getAffineTransform(light_pts_cam0_ordered, missing_grid[:,:])
        cam1togrid = cv2.getAffineTransform(light_pts_cam1_ordered, missing_grid[:,:])
        
        pts_cam0_raw = np.array([kp.pt for kp in keypoints_dark_cam0], dtype=np.float32)
        pts_cam1_raw = np.array([kp.pt for kp in keypoints_dark_cam1], dtype=np.float32)

        ######################################
        # map cam0 to grid. keep mutual points
        ######################################
        transformed_cam0 = (pts_cam0_raw @ cam0togrid[:, :2].T) + cam0togrid[:, 2]

        # get the nearest neighbours
        tree = cKDTree(finalgrid_2d)
        dist, indices = tree.query(transformed_cam0, distance_upper_bound=1.5)

        valid_mask = dist != np.inf
        valid_indices = indices[valid_mask]
        valid_dist = dist[valid_mask]
        valid_pts = pts_cam0_raw[valid_mask]
        valid_kps = np.array(keypoints_dark_cam0)[valid_mask]

        best_for_grid = {}
        best_kps = {}
        for pt, idx, d, kp in zip(valid_pts, valid_indices, valid_dist, valid_kps):
            if idx not in best_for_grid or d < best_for_grid[idx][1]:
                best_for_grid[idx] = (pt, d)
                best_kps[idx] = kp


        matched_cam0 = np.array([v[0] for v in best_for_grid.values()])
        matched_kps0 = [best_kps[i] for i in best_for_grid.keys()]
        matched_grid = finalgrid_2d[list(best_for_grid.keys())]


        #########################################################
        # map cam1 to grid. keep mutual points from prev matching
        #########################################################
        transformed_cam1 = (pts_cam1_raw @ cam1togrid[:, :2].T) + cam1togrid[:, 2]
        tree = cKDTree(matched_grid)
        dist, indices = tree.query(transformed_cam1, distance_upper_bound=1.5)

        valid_mask = dist != np.inf
        valid_indices = indices[valid_mask]
        valid_dist = dist[valid_mask]
        valid_pts = pts_cam1_raw[valid_mask]
        valid_kps = np.array(keypoints_dark_cam1)[valid_mask]

        best_for_grid = {}
        best_kps = {}
        for pt, idx, d, kp in zip(valid_pts, valid_indices, valid_dist, valid_kps):
            if idx not in best_for_grid or d < best_for_grid[idx][1]:
                best_for_grid[idx] = (pt, d)
                best_kps[idx] = kp

        unique_indices = sorted(best_for_grid.keys())
        matched_cam1 = np.array([best_for_grid[i][0] for i in unique_indices])
        matched_kps1 = np.array([best_kps[i] for i in unique_indices])
        matched_cam0 = matched_cam0[unique_indices]
        matched_grid = matched_grid[unique_indices]
        matched_kps0 = [matched_kps0[i] for i in unique_indices]

        # Add back in the light points
        matched_grid = np.append(matched_grid, missing_grid, axis=0)
        matched_cam0 = np.append(matched_cam0, light_pts_cam0_ordered, axis=0)
        matched_cam1 = np.append(matched_cam1, light_pts_cam1_ordered, axis=0)

        # ret, corners0 = cv2.findCirclesGrid(matched_cam0, (4,8), None, flags = cv2.CALIB_CB_SYMMETRIC_GRID)
        # ret, corners1 = cv2.findCirclesGrid(matched_cam0, (4,8), None, flags = cv2.CALIB_CB_SYMMETRIC_GRID)
        # print(corners0)
        # print(ret)


        #
        #
        # pts = matched_cam0.reshape(-1,1,2).astype(np.float32)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        # refined_cam0 = cv2.cornerSubPix(
        #     img0, pts,
        #     winSize=(50,50),     # local search window
        #     zeroZone=(-1,-1),  # no exclusion
        #     criteria=criteria
        # )
        # print(refined_cam0)
        #
        # np.savetxt("matched_cam0",matched_cam0)
        # np.savetxt("matched_cam1",matched_cam1)
        # np.savetxt("matched_grid",matched_grid)


        # print(matched_cam0.shape)
        # print(matched_cam1.shape)
        # print(matched_grid.shape)


        # np.savetxt("matched_cam0_old",matched_cam0)
        # np.savetxt("matched_cam1_old",matched_cam1)
        # np.savetxt("matched_grid",matched_grid)

        # H, _ = cv2.findHomography(matched_grid, matched_cam0, method=0)
        # M, _ = cv2.findHomography(matched_grid, matched_cam1, method=0)
        # matched_cam0 = cv2.perspectiveTransform(matched_grid.reshape(-1, 1, 2), H).reshape(-1, 2)
        # matched_cam1 = cv2.perspectiveTransform(matched_grid.reshape(-1, 1, 2), M).reshape(-1, 2)



        # np.savetxt("matched_cam0_new",updated_cam0_points)
        # np.savetxt("matched_cam1_new",updated_cam1_points)

        #########################################################
        # Debugging
        #########################################################
        # im_with_keypoints_cam0 = cv2.drawKeypoints(img0, keypoints_dark_cam0, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_cam1ICH_KEYPOINTS)
        # im_with_keypoints_cam1 = cv2.drawKeypoints(img1, keypoints_dark_cam1, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_cam1ICH_KEYPOINTS)
        # im_with_keypoints_cam0 = cv2.drawKeypoints(im_with_keypoints_cam0, keypoints_lght_cam0, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_cam1ICH_KEYPOINTS)
        # im_with_keypoints_cam1 = cv2.drawKeypoints(im_with_keypoints_cam1, keypoints_lght_cam1, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_cam1ICH_KEYPOINTS)
        # side_by_side = np.hstack((im_with_keypoints_cam0, im_with_keypoints_cam1))
        # cv2.namedWindow("Stereo Keypoints", cv2.WINDOW_NORMAL)
        # cv2.imshow("Stereo Keypoints", side_by_side)
        # cv2.waitKey(0)

        # # debugging plot
        # pts_cam0 = matched_cam0.reshape(-1, 2)
        # pts_cam1 = matched_cam1.reshape(-1, 2)
        # fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        #
        # # # Left image with detected circles
        # axes[0].imshow(img0, cmap='gray')
        # axes[0].plot(light_pts_cam0_ordered[0, 0], light_pts_cam0_ordered[0, 1], 'co', markersize=5)
        # axes[0].plot(light_pts_cam0_ordered[1, 0], light_pts_cam0_ordered[1, 1], 'yo', markersize=5)
        # axes[0].plot(light_pts_cam0_ordered[2, 0], light_pts_cam0_ordered[2, 1], 'mo', markersize=5)
        # axes[0].plot(pts_cam0[:, 0], pts_cam0[:, 1], 'ro', markersize=5)
        # # corners0 = corners0.reshape(-1,2)
        # # axes[0].plot(corners0[:, 0], corners0[:, 1], 'bo', markersize=2)
        # axes[0].set_title('Left Image with \n Detected Circles')
        #
        # # Right image with detected circles
        # axes[1].imshow(img1, cmap='gray')
        # axes[1].plot(light_pts_cam1_ordered[0, 0], light_pts_cam1_ordered[0, 1], 'co', markersize=5)
        # axes[1].plot(light_pts_cam1_ordered[1, 0], light_pts_cam1_ordered[1, 1], 'yo', markersize=5)
        # axes[1].plot(light_pts_cam1_ordered[2, 0], light_pts_cam1_ordered[2, 1], 'mo', markersize=5)
        # axes[1].plot(pts_cam1[:, 0], pts_cam1[:, 1], 'ro', markersize=5)
        # # corners1 = corners1.reshape(-1,2)
        # # axes[1].plot(corners1[:, 0], corners1[:, 1], 'bo', markersize=2)
        # axes[1].set_title('Right Image with \n Detected Circles')
        #
        # axes[2].plot(transformed_cam0[:, 0], transformed_cam0[:, 1], 'ro', markersize=5)
        # axes[2].plot(finalgrid_2d[:, 0], finalgrid_2d[:, 1], 'x', markersize=5)
        # axes[2].invert_yaxis()
        # axes[2].set_title('left circles mapped to \n to grid reference frame ')
        #
        # axes[3].plot(transformed_cam1[:, 0], transformed_cam1[:, 1], 'ro', markersize=5)
        # axes[3].plot(finalgrid_2d[:, 0], finalgrid_2d[:, 1], 'x', markersize=5)
        # axes[3].invert_yaxis()
        # axes[3].set_title('right cricles mapped to \n to grid reference frame ')
        #
        # # Save the figure to a temporary PNG file
        # # filename = f"output/frame_{i:03d}.png"
        # # plt.savefig(filename)
        # # plt.close(fig)
        # plt.show()
        #
        # # np.savetxt("matched_cam0.txt", matched_cam0, fmt='%.2f')
        # # np.savetxt("matched_cam1.txt", matched_cam1, fmt='%.2f')
        # # np.savetxt("matched_grid.txt", matched_grid, fmt='%.2f')

        # Append for calibration
        matched_grid = np.hstack((matched_grid, np.zeros((matched_grid.shape[0], 1), dtype=matched_grid.dtype)))
        gridpoints.append(matched_grid)
        dots_cam0.append(matched_cam0)
        dots_cam1.append(matched_cam1)


        print(f"Points found in cam0: {len(pts_cam0_raw)+len(light_pts_cam0)}, "
              f"cam1: {len(pts_cam1_raw)+len(light_pts_cam1)}, "
              f"mutual: {matched_grid.shape[0]}")
        print()

        # test_gridpoints = []
        # test_dots_cam0 = []
        # test_dots_cam1 = []
        # test_gridpoints.append(matched_grid)
        # test_dots_cam0.append(matched_cam0)
        # test_dots_cam1.append(matched_cam1)
        # _, Kl, Dl, rvecs_cam0, tvecs_cam0 = cv2.calibrateCamera(test_gridpoints, test_dots_cam0, img_size, None, None)
        # _, Kr, Dr, rvecs_cam1, tvecs_cam1 = cv2.calibrateCamera(test_gridpoints, test_dots_cam1, img_size, None, None)
        # print('\nLeft Camera Matrix:\n', Kl)
        # print('Left Distortion Coefficients:\n', Dl)
        # print('\nRight Camera Matrix:\n', Kr)
        # print('Right Distortion Coefficients:\n', Dr)
        # print()
    

    return dots_cam0, dots_cam1, gridpoints, img_dims

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

    flat_dots_cam0 = np.concatenate(dots_cam0,axis=0).astype(np.float64).ravel().tolist()
    flat_dots_cam1 = np.concatenate(dots_cam1,axis=0).astype(np.float64).ravel().tolist()
    flat_grid = np.concatenate(grid, axis=0).astype(np.int32).ravel().tolist()
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
    print(flat_initial_params)

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




def initial_reconstruction(dots_cam0, dots_cam1, grid, 
                           img_dims, num_file_pairs: int) -> tuple[dict,dict]:

    print(f"Running initial reconstruction with {len(grid)} valid image pairs...")


    # Initial guess at parameter values
    cx = img_dims[0] / 2
    cy = img_dims[1] / 2
    fx = fy = 1.2 * max(img_dims[0], img_dims[1])
    k1, k2, p1, p2, k3 = 0.0,0.0,0.0,0.0,0.0

    # stack parameters for each camera
    initial_params_cam0 = np.hstack([fx, fy, cx, cy, k1, k2, p1, p2, k3])
    initial_params_cam1 = np.hstack([fx, fy, cx, cy, k1, k2, p1, p2, k3])
    
    # adjustment to make distortion parameters less volatile in opt
    scales = np.ones_like(initial_params_cam0)
    scales[4:9] = 0.1

    # defualt bounds
    lower_bounds = -np.inf * np.ones_like(initial_params_cam0)
    upper_bounds =  np.inf * np.ones_like(initial_params_cam0)

    # keep all focal lengths positive
    lower_bounds[0] = 1.0  # fx
    lower_bounds[1] = 1.0  # fy
    lower_bounds[2] = 1.0  # fx
    lower_bounds[3] = 1.0  # cy


    intrinsics_cam0 = []
    intrinsics_cam1 = []

    for i in range(0,num_file_pairs):

        # Perform optimization on intrinsics for cam 0
        result_cam0 = least_squares(reprojection_intrinsics_error, initial_params_cam0,
                                    args=(grid[i], dots_cam0[i]),
                                    verbose=0, xtol=1e-10, ftol=1e-10, x_scale=scales,
                                    bounds=(lower_bounds, upper_bounds))
        
        # Perform optimization on intrinsics for cam 1
        result_cam1 = least_squares(reprojection_intrinsics_error, initial_params_cam1,
                                    args=(grid[i], dots_cam1[i]),
                                    verbose=0, xtol=1e-10, ftol=1e-10, x_scale=scales,
                                    bounds=(lower_bounds, upper_bounds))

        # update initial guess based on converged results from nonlinear opt
        # if result_cam0.success: initial_params_cam0 = result_cam0.x
        # if result_cam1.success: initial_params_cam1 = result_cam1.x

        # intrinsic parameters cam0
        fx, fy, cx, cy = result_cam0.x[0:4]
        k1, k2, p1, p2, k3 = result_cam0.x[4:9]
        K0_opt = np.array([[fx, 0, cx], [0, fy, cy],[0,  0,  1]])
        D0_opt = np.array([k1, k2, p1, p2, k3])

        # intrinsic parameters cam1
        fx, fy, cx, cy = result_cam1.x[0:4]
        k1, k2, p1, p2, k3 = result_cam1.x[4:9]
        K1_opt = np.array([[fx, 0, cx], [0, fy, cy], [0,  0,  1]])
        D1_opt = np.array([k1, k2, p1, p2, k3])

        # rotation and translation vector for cam0
        _, rvec_cam0, tvec_cam0 = cv2.solvePnP(grid[i], dots_cam0[i],
                                                        K0_opt, D0_opt, flags=cv2.SOLVEPNP_ITERATIVE) 

        # rotation and translation vector for cam1
        _, rvec_cam1, tvec_cam1 = cv2.solvePnP(grid[i], dots_cam1[i],
                                                        K1_opt, D1_opt, flags=cv2.SOLVEPNP_ITERATIVE)


        # Projected points for cam0
        projected_points_opt0, _ = cv2.projectPoints(grid[i], rvec_cam0, tvec_cam0,
                                                        K0_opt, D0_opt)


        # Projected points for cam1
        projected_points_opt1, _ = cv2.projectPoints(grid[i], rvec_cam1, tvec_cam1,
                                                        K1_opt, D1_opt)
        
        
        # Error for the new projected points for cam0
        
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
        print(K0_opt[0,0], K0_opt[0,2], K0_opt[1,1], K0_opt[1,2], D0_opt[0],D0_opt[1],D0_opt[2],D0_opt[3],D0_opt[4], error0, error1)

        # Error for the new projected points for cam1
        # projected_points_opt1 = projected_points_opt1.reshape(-1, 2)
        # err_cam1 = cv2.norm(dots_cam1[i], projected_points_opt1, cv2.NORM_L2)/len(dots_cam1[i])
        # print("ERROR cam1", err_cam1)

        intrinsics_cam0.append({
            "K": K0_opt,
            "D": D0_opt,
            "rvec": rvec_cam0,
            "tvec": tvec_cam0,
            "err": error0,
            "success": result_cam0.success
        })

        intrinsics_cam1.append({
            "K": K1_opt,
            "D": D1_opt,
            "rvec": rvec_cam1,
            "tvec": tvec_cam1,
            "err": error1,
            "success": result_cam1.success
        })

        # imgpoints = dots_cam0[i].reshape(-1, 2)
        # dots_cam0 = dots_cam0[i].reshape(-1, 2)
        # dots_cam1 = dots_cam1[i].reshape(-1, 2)
        #
        # fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        # ax[0].scatter(dots_cam0[:, 0], dots_cam0[:, 1], label='Observed', c='blue')
        # ax[0].scatter(projected_points_opt0[:, 0], projected_points_opt0[:, 1], label='Projected', c='red', marker='x')
        # ax[1].scatter(dots_cam1[:, 0], dots_cam1[:, 1], label='Observed', c='blue')
        # ax[1].scatter(projected_points_opt1[:, 0], projected_points_opt1[:, 1], label='Projected', c='red', marker='x')
        #
        # ax[0].invert_yaxis()  # Correct - call directly on the axis
        # ax[1].invert_yaxis()  # Correct - call directly on the axis
        # ax[0].grid(True)
        # ax[1].grid(True)
        #
        # # Optional: Add titles and legends for clarity
        # ax[0].set_title('Left Camera')
        # ax[0].legend()
        # ax[1].set_title('Right Camera') 
        # ax[1].legend()
        #
        # plt.show()

    return intrinsics_cam0, intrinsics_cam1





def reprojection_intrinsics_error(params, gridpoints, dots):

    fx, fy, cx, cy = params[0:4]
    k1, k2, p1, p2, k3 = params[4:9]

    # Camera matrix
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]])

    # Distortion coefficients
    D = np.array([k1, k2, p1, p2, k3])

    # Estimate pose from current intrinsics
    success, rvec, tvec = cv2.solvePnP(gridpoints, dots, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return np.full(dots.shape[0] * 2, 1e6)  # Large error if pose fails

    # Project points
    projected_dots, _ = cv2.projectPoints(gridpoints, rvec, tvec, K, D)
    projected_dots = projected_dots.reshape(-1, 2)

    # Residuals
    #residuals = np.linalg.norm(projected_points - imgpoints, axis=1)
    residuals = (projected_dots - dots).flatten()
    return residuals

    
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

def bundle_adjustment_error_test(params, grid_points, dots_cam0, dots_cam1, num_poses, K0, K1):
    idx = 0

    D0 = params[idx:idx+5].astype(np.float64)
    idx += 5
    D1 = params[idx:idx+5].astype(np.float64)
    idx += 5

    rvec_stereo = params[idx:idx+3].astype(np.float64)
    idx += 3
    tvec_stereo = params[idx:idx+3].astype(np.float64)
    idx += 3

    R_stereo, _ = cv2.Rodrigues(rvec_stereo)

    total_residuals = []

    for pose_idx in range(num_poses):
        rvec_target = params[idx:idx+3].astype(np.float64)
        tvec_target = params[idx+3:idx+6].astype(np.float64)
        idx += 6

        grid_3d = grid_points[pose_idx].astype(np.float64)
        obs_cam0 = dots_cam0[pose_idx].astype(np.float64)
        obs_cam1 = dots_cam1[pose_idx].astype(np.float64)

        proj_cam0, _ = cv2.projectPoints(grid_3d, rvec_target, tvec_target, K0, D0)
        proj_cam0 = proj_cam0.reshape(-1, 2)

        R_target, _ = cv2.Rodrigues(rvec_target)
        R_composed = R_stereo @ R_target
        t_composed = R_stereo @ tvec_target + tvec_stereo
        rvec_composed, _ = cv2.Rodrigues(R_composed)

        proj_cam1, _ = cv2.projectPoints(grid_3d, rvec_composed, t_composed, K1, D1)
        proj_cam1 = proj_cam1.reshape(-1, 2)

        residuals_cam0 = (obs_cam0 - proj_cam0).flatten()
        residuals_cam1 = (obs_cam1 - proj_cam1).flatten()

        total_residuals.extend(residuals_cam0)
        total_residuals.extend(residuals_cam1)

    return np.array(total_residuals)




def visualisation():

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    for i in range(0, 80):
        # For one image (e.g., first):
        rvec = rvecs_cam0[i]  # rotation vector of plate wrt left camera
        tvec = tvecs_cam0[i]  # translation vector of plate wrt left camera

        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)

        # Plate origin in camera coords is tvec (translation)
        print("Plate position relative to left camera:", tvec.ravel())

        # Distance from camera center to plate:
        distance = np.linalg.norm(tvec)
        print(f"Distance from left camera to plate: {distance:.2f} mm")

        points_in_camera0 = (R @ object_points_3d.T).T + tvec.ravel()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # print(points_in_camera0)
        # print(points_in_camera0[:,0])


        ax.set_xlim([70,-70])
        ax.set_ylim([-70,70])
        ax.set_zlim([5,300])
        ax.scatter(0,0,0)
        ax.scatter(-62.74,0.38,20.9)
        ax.scatter(points_in_camera0[:,1],points_in_camera0[:,0],points_in_camera0[:,2])
        filename = f"output/frame_{i:03d}.png"
        plt.savefig(filename)
        plt.close(fig)

def create_blob_detector(light: bool):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 10000
    params.filterByCircularity = True
    params.minCircularity = 0.86
    params.filterByColor = True
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.blobColor = 255 if light else 0
    return cv2.SimpleBlobDetector_create(params)

def get_file_list(path0: Path, path1: Path):

    sorted_cam0 = sorted(glob.glob(str(path0)))
    sorted_cam1 = sorted(glob.glob(str(path1)))

    if not sorted_cam0:
        raise FileNotFoundError(f"No cam0 found: {path0}")
    if not sorted_cam1:
        raise FileNotFoundError(f"No cam1 found: {path1}")
    

    if len(sorted_cam0) != len(sorted_cam1):
        print("\033[1mWARNING: Number of images for camera 0 and camera 1 are not identical: \033[0m \n"
                f" - number of images found for cam0: {len(sorted_cam0)} \n"
                f" - number of images found for cam1: {len(sorted_cam1)}")
        
        # need to exclude files that aren't mutual between the two lists
        base_cam0 = {get_base_name(f) for f in sorted_cam0}
        base_cam1 = {get_base_name(f) for f in sorted_cam1}
        common_bases = base_cam0.intersection(base_cam1)

        only_in_cam0 = base_cam0 - common_bases
        only_in_cam1 = base_cam1 - common_bases

        print()
        print("\033[1mWARNING: Files only in cam0 (without matching cam1):\033[0m")
        for base in sorted(only_in_cam0):
            unmatched_files = [f for f in sorted_cam0 if get_base_name(f) == base]
            for f in unmatched_files:
                print(f" - {f}")

        print()
        print("\033[1mWARNING: Files only in cam1 (without matching cam0):\033[0m")
        for base in sorted(only_in_cam1):
            unmatched_files = [f for f in sorted_cam1 if get_base_name(f) == base]
            for f in unmatched_files:
                print(f" - {f}")
        print()
        print("\033[1mWARNING: Excluding the above files from the calibration. \033[0m")
        print()

        filtered_cam0 = [f for f in sorted_cam0 if get_base_name(f) in common_bases]
        filtered_cam1 = [f for f in sorted_cam1 if get_base_name(f) in common_bases]

        files_cam0 = filtered_cam0
        files_cam1 = filtered_cam1

    else:
        files_cam0 = [Path(f) for f in sorted_cam0]
        files_cam1 = [Path(f) for f in sorted_cam1]


    return files_cam0, files_cam1


def get_base_name(filename):
    match = re.match(r"(.*)_\d\.tiff$", filename)
    return match.group(1) if match else None

def order_triangle_points_by_angle(pts):
    angles = []
    for i in range(3):
        ang = angle_between(pts[(i+1)%3], pts[i], pts[(i+2)%3])
        angles.append(ang)
    angles = np.array(angles)

    sorted_indices = np.argsort(-angles)
    ordered_pts = np.array([pts[i] for i in sorted_indices], dtype=np.float32)
    return ordered_pts

def angle_between(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)


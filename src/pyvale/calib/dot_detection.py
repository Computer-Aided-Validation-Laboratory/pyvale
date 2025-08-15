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
    fullgrid_3d[:, :2] = np.mgrid[0:grid_width, 0:grid_height].T.reshape(-1, 2)
    fullgrid_3d[:, :2] *= grid_spacing
    

    missing_idx = np.array([
        [2, grid_height-2-1],
        [2, grid_height-7],
        [9, grid_height-2-1],
    ])

    missing_grid = (missing_idx * grid_spacing)
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

        print("\033[1mRunning Dot detection on:\033[0m\n"
                f" - {files_cam0[i]} \n"
                f" - {files_cam1[i]}")

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
            print(f"Skipping pair due to insufficient light blobs.")
            print("left:", len(keypoints_lght_cam0))
            print("right:", len(keypoints_lght_cam1))
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

        best_for_grid = {}
        for pt, idx, d in zip(valid_pts, valid_indices, valid_dist):
            if idx not in best_for_grid or d < best_for_grid[idx][1]:
                best_for_grid[idx] = (pt, d)

        matched_cam0 = np.array([v[0] for v in best_for_grid.values()])
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

        best_for_grid = {}
        for pt, idx, d in zip(valid_pts, valid_indices, valid_dist):
            if idx not in best_for_grid or d < best_for_grid[idx][1]:
                best_for_grid[idx] = (pt, d)

        unique_indices = sorted(best_for_grid.keys())
        matched_cam1 = np.array([best_for_grid[i][0] for i in unique_indices])
        matched_cam0 = matched_cam0[unique_indices]
        matched_grid = matched_grid[unique_indices]

        # Add back in the light points
        matched_grid = np.append(matched_grid, missing_grid, axis=0)
        matched_cam0 = np.append(matched_cam0, light_pts_cam0_ordered, axis=0)
        matched_cam1 = np.append(matched_cam1, light_pts_cam1_ordered, axis=0)
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

        # # # debugging plot
        # pts_cam0 = matched_cam0.reshape(-1, 2)
        # pts_cam1 = matched_cam1.reshape(-1, 2)
        # fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        
        # # # Left image with detected circles
        # axes[0].imshow(img0, cmap='gray')
        # axes[0].plot(light_pts_cam0_ordered[0, 0], light_pts_cam0_ordered[0, 1], 'co', markersize=5)
        # axes[0].plot(light_pts_cam0_ordered[1, 0], light_pts_cam0_ordered[1, 1], 'yo', markersize=5)
        # axes[0].plot(light_pts_cam0_ordered[2, 0], light_pts_cam0_ordered[2, 1], 'mo', markersize=5)
        # axes[0].plot(pts_cam0[:, 0], pts_cam0[:, 1], 'ro', markersize=5)
        # axes[0].set_title('Left Image with \n Detected Circles')
        
        # # Right image with detected circles
        # axes[1].imshow(img1, cmap='gray')
        # axes[1].plot(light_pts_cam1_ordered[0, 0], light_pts_cam1_ordered[0, 1], 'co', markersize=5)
        # axes[1].plot(light_pts_cam1_ordered[1, 0], light_pts_cam1_ordered[1, 1], 'yo', markersize=5)
        # axes[1].plot(light_pts_cam1_ordered[2, 0], light_pts_cam1_ordered[2, 1], 'mo', markersize=5)
        # axes[1].plot(pts_cam1[:, 0], pts_cam1[:, 1], 'ro', markersize=5)
        # axes[1].set_title('Right Image with \n Detected Circles')
        
        # axes[2].plot(transformed_cam0[:, 0], transformed_cam0[:, 1], 'ro', markersize=5)
        # axes[2].plot(__finalgrid_2d[:, 0], __finalgrid_2d[:, 1], 'x', markersize=5)
        # axes[2].invert_yaxis()
        # axes[2].set_title('left circles mapped to \n to grid reference frame ')
        
        # axes[3].plot(transformed_cam1[:, 0], transformed_cam1[:, 1], 'ro', markersize=5)
        # axes[3].plot(__finalgrid_2d[:, 0], __finalgrid_2d[:, 1], 'x', markersize=5)
        # axes[3].invert_yaxis()
        # axes[3].set_title('right cricles mapped to \n to grid reference frame ')

        # # Save the figure to a temporary PNG file
        # filename = f"output/frame_{i:03d}.png"
        # plt.savefig(filename)
        # plt.close(fig)
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
    else:
        raise ValueError(f"ERROR: Unknown calibration method: {method}. "
                   f"Allowed options: 'bundle', 'zhang'")




def zhang(dots_cam0, dots_cam1, grid, img_dims, num_file_pairs):

    print(f"Running calibration with {len(grid)} valid image pairs...")

    # Left and Right cam calib
    _, Kl, Dl, rvecs_cam0, tvecs_cam0 = cv2.calibrateCamera(grid, dots_cam0, img_dims, None, None)
    _, Kr, Dr, rvecs_cam1, tvecs_cam1 = cv2.calibrateCamera(grid, dots_cam1, img_dims, None, None)

    # --- Step 3: Per-view reprojection errors ---
    errors_cam0 = []
    errors_cam1 = []

    for i, objp in enumerate(grid):


        # Left camera projection
        imgpts_cam0_proj, _ = cv2.projectPoints(objp, rvecs_cam0[i], tvecs_cam0[i], Kl, Dl)
        imgpts_cam0_proj = imgpts_cam0_proj.reshape(-1, 2)
        err_cam0 = cv2.norm(dots_cam0[i], imgpts_cam0_proj, cv2.NORM_L2)/len(dots_cam0[i])
        errors_cam0.append(err_cam0)

        np.savetxt("imgpts_cam0_proj", imgpts_cam0_proj)
        np.savetxt("dots_cam0", dots_cam0[i])

        # Right camera projection
        imgpts_cam1_proj, _ = cv2.projectPoints(objp, rvecs_cam1[i], tvecs_cam1[i], Kr, Dr)
        imgpts_cam1_proj = imgpts_cam1_proj.reshape(-1, 2)
        err_cam1 = cv2.norm(dots_cam1[i], imgpts_cam1_proj, cv2.NORM_L2)/len(dots_cam1[i])
        errors_cam1.append(err_cam1)

        np.savetxt("./output/imgpts_cam0_proj_"+str(i), imgpts_cam0_proj)
        np.savetxt("./output/imgpts_cam1_proj_"+str(i), imgpts_cam1_proj)
        np.savetxt("./output/dots_cam0_"+str(i), dots_cam0[i])
        np.savetxt("./output/dots_cam1_"+str(i), dots_cam1[i])

    # stereo calib
    ret, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=grid,
        imagePoints1=dots_cam0,
        imagePoints2=dots_cam1,
        cameraMatrix1=Kl,
        distCoeffs1=Dl,
        cameraMatrix2=Kr,
        distCoeffs2=Dr,
        imageSize=img_dims,
        flags=0
    )

    print(f"Stereo RMS error: {ret:.4f} px")
    print(f"Mean left RMS error: {np.mean(errors_cam0):.4f} px")
    print(f"Mean right RMS error: {np.mean(errors_cam1):.4f} px")

    print("\n--- Calibration Results ---")
    print("Calibration RMS error:", ret)
    print('\nLeft Camera Matrix:\n', Kl)
    print('Left Distortion Coefficients:\n', Dl)
    print('\nRight Camera Matrix:\n', Kr)
    print('Right Distortion Coefficients:\n', Dr)
    print('\nRotation Matrix (R):\n', R)
    print('Translation Vector (T):\n', T)

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
    
    # First need to get an initial reconstruction for each image pair before
    # we pass to bundle adjustment simultaneuos paratemeter non-liniear optimization

    # we'll take an average of the converged intrinsics for the seed
    intrin_cam0, intrin_cam1 = initial_reconstruction(dots_cam0,
                                                      dots_cam1, grid,
                                                      img_dims,
                                                      num_file_pairs)

    # camera matrix
    mean_K0 = np.mean([d["K"] for d in intrin_cam0 if d["success"]], axis=0)
    mean_K1 = np.mean([d["K"] for d in intrin_cam1 if d["success"]], axis=0)

    # Distortion
    mean_D0 = np.mean([d["D"] for d in intrin_cam0 if d["success"]], axis=0)
    mean_D1 = np.mean([d["D"] for d in intrin_cam1 if d["success"]], axis=0)

    fx0, fy0, cx0, cy0 = mean_K0[0, 0], mean_K0[1, 1], mean_K0[0, 2], mean_K0[1, 2]
    fx1, fy1, cx1, cy1 = mean_K1[0, 0], mean_K1[1, 1], mean_K1[0, 2], mean_K1[1, 2]
    mean_D0 = mean_D0.flatten()
    mean_D1 = mean_D1.flatten()

    # initial guess for translation and rotation between cameras
    mean_rvec0 = np.mean([d["rvec"] for d in intrin_cam0 if d["success"]], axis=0)
    mean_rvec1 = np.mean([d["rvec"] for d in intrin_cam1 if d["success"]], axis=0)
    R0, _ = cv2.Rodrigues(mean_rvec0)
    R1, _ = cv2.Rodrigues(mean_rvec1)
    R_rel = R1 @ R0.T
    rvec_stereo, _ = cv2.Rodrigues(R_rel)
    # rvec_stereo = np.zeros(3)


    mean_tvec0 = np.mean([d["tvec"] for d in intrin_cam0 if d["success"]], axis=0)
    mean_tvec1 = np.mean([d["tvec"] for d in intrin_cam1 if d["success"]], axis=0)
    tvec_stereo = mean_tvec1 - (R_rel @ mean_tvec0)
    # tvec_stereo = np.array([1.0, 0.0, 0.0])
    
    print(rvec_stereo)
    print(tvec_stereo)


    # Initial poses from intrinsics_cam0
    initial_poses_cam0 = []
    for d in intrin_cam0:
        if d["success"]:
            initial_poses_cam0.extend(d["rvec"].flatten())
            initial_poses_cam0.extend(d["tvec"].flatten())

    # full list of initial parameters
    initial_params = np.hstack([fx0, fy0, cx0, cy0, mean_D0,
                                fx1, fy1, cx1, cy1, mean_D1,
                                rvec_stereo.flatten(), tvec_stereo.flatten(),
                                initial_poses_cam0])


    # adjustment to make distortion parameters less volatile in opt
    scales = np.ones_like(initial_params)
    scales[4:9] = 0.1   # Left distortion
    scales[13:18] = 0.1 # Right distortion
    scales[18:21] = 0.1 # stereo rotation likely to be small

    # default bound values
    lower_bounds = -np.inf * np.ones_like(initial_params)
    upper_bounds =  np.inf * np.ones_like(initial_params)

    # keep all focal lengths positive
    lower_bounds[0] = 1.0   # fx0
    lower_bounds[1] = 1.0   # fy0
    lower_bounds[9] = 1.0   # fx1
    lower_bounds[10] = 1.0  # fy1

    # Filter successful images
    success_indices = [i for i, d in enumerate(intrin_cam0) if d["success"]]
    gridpoints_success = [grid[i] for i in success_indices]
    dots_cam0_success = [dots_cam0[i] for i in success_indices]
    dots_cam1_success = [dots_cam1[i] for i in success_indices]
    intrin_cam0_success = [intrin_cam0[i] for i in success_indices]
    intrin_cam1_success = [intrin_cam1[i] for i in success_indices]

    num_success = len(success_indices)
    print("num soccessful:", success_indices)


    # optimization
    result = least_squares(
        bundle_adjustment_error,
        initial_params,
        jac='2-point',
        args=(gridpoints_success, dots_cam0_success, dots_cam1_success, num_success),
        verbose=2,
        xtol=0.001,
        ftol=0.005,
        max_nfev=200,
        x_scale=scales,
        bounds=(lower_bounds, upper_bounds)
    )

    # --- Step 7: Extract results ---
    opt = result.x
    fx0, fy0, cx0, cy0 = opt[0:4]
    k0 = opt[4:9]
    fx1, fy1, cx1, cy1 = opt[9:13]
    k1 = opt[13:18]
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
    print("Distortion:", k0)

    print("\n--- Optimized Right Camera Intrinsics ---")
    print("K1:\n", K1_opt)
    print("Distortion:", k1)

    print("\n--- Stereo Transform (Right from Left) ---")
    print("Rotation Vector:", rvec_stereo)
    print("Translation Vector:", tvec_stereo)

    import matplotlib.pyplot as plt

    # ADD THIS: Calculate right camera pose from stereo transform
    R_stereo, _ = cv2.Rodrigues(rvec_stereo)
    R0, _ = cv2.Rodrigues(rvec0)
    T0 = tvec0.reshape(3, 1)

    # Right camera pose
    R1 = R_stereo @ R0
    T1 = R_stereo @ T0 + tvec_stereo.reshape(3, 1)
    rvec1, _ = cv2.Rodrigues(R1)
    tvec1 = T1.flatten()  # Make sure it's 1D for cv2.projectPoints

    # Projected points
    projected_points_opt0, _ = cv2.projectPoints(grid[0], rvec0, tvec0, K0_opt, k0)
    projected_points_opt1, _ = cv2.projectPoints(grid[0], rvec1, tvec1, K1_opt, k1)

    # get the detected dots
    dots_cam0 = dots_cam0[0].reshape(-1, 2)
    dots_cam1 = dots_cam1[0].reshape(-1, 2)

    # reshape for error calc
    projected_points_opt0 = projected_points_opt0.reshape(-1, 2)
    projected_points_opt1 = projected_points_opt1.reshape(-1, 2)
    
    # error
    err_cam0 = cv2.norm(dots_cam0[0], projected_points_opt0, cv2.NORM_L2)/len(dots_cam0)
    err_cam1 = cv2.norm(dots_cam1[0], projected_points_opt1, cv2.NORM_L2)/len(dots_cam1)
    print("ERROR", err_cam0)
    print("ERROR", err_cam1)


    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].scatter(dots_cam0[:, 0], dots_cam0[:, 1], label='Observed', c='blue')
    ax[0].scatter(projected_points_opt0[:, 0], projected_points_opt0[:, 1], label='Projected', c='red', marker='x')
    ax[1].scatter(dots_cam1[:, 0], dots_cam1[:, 1], label='Observed', c='blue')
    ax[1].scatter(projected_points_opt1[:, 0], projected_points_opt1[:, 1], label='Projected', c='red', marker='x')
    plt.gca().invert_yaxis()  # Optional: match image coordinates
    plt.ticklabel_format(style='plain')
    plt.grid(True)
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
                                    verbose=0, xtol=0.005, ftol=0.001, x_scale=scales,
                                    bounds=(lower_bounds, upper_bounds))
        
        # Perform optimization on intrinsics for cam 1
        result_cam1 = least_squares(reprojection_intrinsics_error, initial_params_cam1,
                                    args=(grid[i], dots_cam1[i]),
                                    verbose=0, xtol=0.005, ftol=0.001, x_scale=scales,
                                    bounds=(lower_bounds, upper_bounds))

        # update initial guess based on converged results from nonlinear opt
        if result_cam0.success: initial_params_cam0 = result_cam0.x
        if result_cam1.success: initial_params_cam1 = result_cam1.x

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
        projected_points_opt0 = projected_points_opt0.reshape(-1, 2)
        err_cam0 = cv2.norm(dots_cam0[i], projected_points_opt0, cv2.NORM_L2)/len(dots_cam0[i])
        print("ERROR cam0", err_cam0)

        # Error for the new projected points for cam1
        projected_points_opt1 = projected_points_opt1.reshape(-1, 2)
        err_cam1 = cv2.norm(dots_cam1[i], projected_points_opt1, cv2.NORM_L2)/len(dots_cam1[i])
        print("ERROR cam1", err_cam1)

        intrinsics_cam0.append({
            "K": K0_opt,
            "D": D0_opt,
            "rvec": rvec_cam0,
            "tvec": tvec_cam0,
            "err": err_cam0,
            "success": result_cam0.success
        })

        intrinsics_cam1.append({
            "K": K1_opt,
            "D": D1_opt,
            "rvec": rvec_cam1,
            "tvec": tvec_cam1,
            "err": err_cam1,
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
    residuals = (np.abs(projected_dots - dots)).flatten()
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

def bundle_adjustment_jac(params, gridpoints, dots_cam0, dots_cam1, num_img):
    """
    Analytical Jacobian for stereo bundle adjustment.
    Returns a 2*num_points*num_img x num_params Jacobian for least_squares.
    """
    # Extract intrinsics
    fx0, fy0, cx0, cy0 = params[0:4]
    D0 = params[4:9]
    fx1, fy1, cx1, cy1 = params[9:13]
    D1 = params[13:18]

    rvec_stereo = params[18:21]
    tvec_stereo = params[21:24]
    R_stereo, _ = cv2.Rodrigues(rvec_stereo)

    K0 = np.array([[fx0, 0, cx0],
                   [0, fy0, cy0],
                   [0,  0,   1]])
    K1 = np.array([[fx1, 0, cx1],
                   [0, fy1, cy1],
                   [0,  0,   1]])

    pose0_start = 24
    n_points = gridpoints[0].shape[0]
    n_residuals = 2 * n_points * num_img * 2  # 2 cameras
    n_params = params.size

    J = np.zeros((n_residuals, n_params))

    row = 0
    for i in range(num_img):
        # cam0
        rvec0 = params[pose0_start + i*6 : pose0_start + i*6 + 3]
        tvec0 = params[pose0_start + i*6 + 3 : pose0_start + i*6 + 6]

        # Project points and get Jacobian
        proj0, jac0 = cv2.projectPoints(gridpoints[i], rvec0, tvec0, K0, D0)
        proj0 = proj0.reshape(-1, 2)

        # Assign jac0 to the correct slice
        # cv2.projectPoints returns jac shape (2N, 12): [rvec(3), tvec(3), fx, fy, cx, cy, k1..k5]
        # Map them to our params ordering
        # rvec/tvec of cam0
        J[row:row+2*n_points, pose0_start + i*6 : pose0_start + i*6 + 6] = jac0[:, :6]
        # intrinsics cam0
        J[row:row+2*n_points, 0:4] = jac0[:, 6:10]  # fx, fy, cx, cy
        J[row:row+2*n_points, 4:9] = jac0[:, 10:15]  # D0
        row += 2*n_points

        # cam1 (derived from cam0 + stereo)
        R0, _ = cv2.Rodrigues(rvec0)
        T0 = tvec0.reshape(3,1)
        R1 = R_stereo @ R0
        T1 = R_stereo @ T0 + tvec_stereo.reshape(3,1)
        rvec1, _ = cv2.Rodrigues(R1)
        tvec1 = T1

        proj1, jac1 = cv2.projectPoints(gridpoints[i], rvec1, tvec1, K1, D1)
        proj1 = proj1.reshape(-1, 2)

        # cam1 jacobian mapping
        # rvec/tvec stereo
        J[row:row+2*n_points, 18:24] = jac1[:, :6]  # approximate: rotation+translation w.r.t stereo extrinsics
        # cam0 pose affects cam1 through stereo, so this is more complex
        # for simplicity, assign main contribution
        J[row:row+2*n_points, pose0_start + i*6 : pose0_start + i*6 + 6] = jac1[:, :6]
        # intrinsics cam1
        J[row:row+2*n_points, 9:13] = jac1[:, 6:10]
        J[row:row+2*n_points, 13:18] = jac1[:, 10:15]

        row += 2*n_points

    return J

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

        print(filtered_cam0)
        print(filtered_cam1)
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


# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import cv2
import numpy as np
import re
import os
from scipy.spatial import cKDTree
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

        # Append for calibration
        matched_grid = np.hstack((matched_grid, np.zeros((matched_grid.shape[0], 1), dtype=matched_grid.dtype)))
        gridpoints.append(matched_grid)
        dots_cam0.append(matched_cam0)
        dots_cam1.append(matched_cam1)


        print(f"Points found in cam0: {len(pts_cam0_raw)+len(light_pts_cam0)}, "
              f"cam1: {len(pts_cam1_raw)+len(light_pts_cam1)}, "
              f"mutual: {matched_grid.shape[0]}")
        print()

    return dots_cam0, dots_cam1, gridpoints, img_dims


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
        #print(K0_opt[0,0], K0_opt[0,2], K0_opt[1,1], K0_opt[1,2], D0_opt[0],D0_opt[1],D0_opt[2],D0_opt[3],D0_opt[4], error0, error1)

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


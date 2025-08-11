# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import cv2
import numpy as np
import os
import yaml
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pathlib import Path
import glob

class StereoCalibration:
    def __init__(self, 
                 cam0: Path | list[Path] | np.ndarray | str,
                 cam1: Path | list[Path] | np.ndarray | str,
                 grid_height: int, grid_width: int,
                 grid_spacing: float) -> None:

        self.cam0 = cam0
        self.cam1 = cam1
        self.files_cam0 = []
        self.files_cam1 = []
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_spacing = grid_spacing

        # Check cam0 and cam1 are same type
        if type(self.cam0) is not type(self.cam1):
            raise ValueError(
                f"cam0 and cam1 are different dtypes: cam0={type(self.cam0)}, cam1={type(self.cam1)}"
            )

        # check np.ndarray dims agree
        if isinstance(self.cam0, np.ndarray):
            if self.cam0.shape != self.cam1.shape:
                raise ValueError(
                    f"cam0 and cam1 are different numpy shapes: cam0.shape={self.cam0.shape}, cam1.shape={self.cam1.shape}"
                )

        # handle strings. convert to path for import
        elif isinstance(self.cam0, (str, Path)) and isinstance(self.cam1, (str, Path)):
            self.cam0 = Path(self.cam0)
            self.cam1 = Path(self.cam1)
            self.__load_from_glob(self.cam0, self.cam1)

        # handle lists
        elif isinstance(self.cam0, list) and isinstance(self.cam1, list):
            self.cam0 = [Path(x) for x in self.cam0]
            self.cam1 = [Path(x) for x in self.cam1]

            if len(self.cam0) != len(self.cam1):
                raise ValueError("Number of images for camera 0 and camera 1 must be identical")

            self.files_cam0 = [p.name for p in self.cam0]
            self.files_cam1 = [p.name for p in self.cam1]

        else:
            raise TypeError(f"Unsupported input type: cam0={type(self.cam0)}")


        print(self.files_cam0) 
        print(self.files_cam1) 

        # Generate full 3D grid
        object_points_3d = np.zeros((self.grid_width * self.grid_height, 3), np.float32)
        object_points_3d[:, :2] = np.mgrid[0:self.grid_width, 0:self.grid_height].T.reshape(-1, 2)
        object_points_3d[:, :2] *= self.grid_spacing

        # Known missing points (x, y in grid)
        missing_points_xy = np.array([
            [2, self.grid_height-2-1],
            [2, self.grid_height-7],
            [9, self.grid_height-2-1],
        ])

        missing_vals = (missing_points_xy * self.grid_spacing)
        missing_vals = missing_vals.astype(np.float32)
        print(missing_vals)

        # Convert to flat indices
        missing_indices = [y * self.grid_width + x for (x, y) in missing_points_xy]
        mask = np.ones(len(object_points_3d), dtype=bool)
        mask[missing_indices] = False
        filtered_points = object_points_3d[mask]
        filtered_points_2d = filtered_points[:, :2]

        self.gridpoints = None
        self.imgpoints_l = None
        self.imgpoints_r = None
        self.img_size = None
        self.img_width = None
        self.img_height = None
        self.num_img = len(self.files_cam0)

    def __load_from_glob(self, path0: Path, path1: Path):
        sorted_cam0 = sorted(glob.glob(str(path0)))
        sorted_cam1 = sorted(glob.glob(str(path1)))

        if not sorted_cam0:
            raise FileNotFoundError(f"No cam0 found: {path0}")
        if not sorted_cam1:
            raise FileNotFoundError(f"No cam1 found: {path1}")
        if len(sorted_cam0) != len(sorted_cam1):
            raise ValueError("Number of images for camera 0 and camera 1 must be identical")

        self.files_cam0 = [os.path.basename(f) for f in sorted_cam0]
        self.files_cam1 = [os.path.basename(f) for f in sorted_cam1]


    def __create_blob_detector(self, light: bool):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.85
        params.filterByColor = True
        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        params.blobColor = 255 if light else 0
        return cv2.SimpleBlobDetector_create(params)

    def __angle_between(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle)

    def __order_triangle_points_by_angle(self, pts):
        angles = []
        for i in range(3):
            ang = self.__angle_between(pts[(i+1)%3], pts[i], pts[(i+2)%3])
            angles.append(ang)
        angles = np.array(angles)
        sorted_indices = np.argsort(-angles)
        ordered_pts = np.array([pts[i] for i in sorted_indices], dtype=np.float32)

        return ordered_pts



    def dot_detection(self):

        detector_lght = self.__create_blob_detector(light=True)
        detector_dark = self.__create_blob_detector(light=False)

        # arrays that are going to contain matching points for each image
        self.gridpoints = []
        self.imgpoints_l = []
        self.imgpoints_r = []

        for i in range(0, self.num_img):

            print("Running Dot detection on: "
                  f"{self.files_cam0[i]}"
                  f"{self.files_cam1[i]}")

            # read images
            im_l = cv2.imread(self.files_cam0[i], cv2.IMREAD_GRAYSCALE)
            im_r = cv2.imread(self.files_cam1[i], cv2.IMREAD_GRAYSCALE)


            if im_l is None or im_r is None:
                print(f"Skipping missing pair: {self.files_cam0[i]} {self.files_cam1[i]}")
                continue

            img_size = (im_l.shape[1], im_l.shape[0])


            # Detect LIGHT blobs
            keypoints_lght_l = detector_lght.detect(im_l)
            keypoints_lght_r = detector_lght.detect(im_r)

            # Detect DARK blobs
            keypoints_dark_l = detector_dark.detect(im_l)
            keypoints_dark_r = detector_dark.detect(im_r)

            # debugging



            # there should always be 3 points in keypoints_lght_l and keypoints_lght_r
            if len(keypoints_lght_l) != 3 or len(keypoints_lght_r) != 3:
                print(f"Skipping pair due to insufficient light blobs.")
                print("left:", len(keypoints_lght_l))
                print("right:", len(keypoints_lght_r))
                continue

            # Convert KeyPoints to NumPy arrays
            pts_l = np.array([kp.pt for kp in keypoints_lght_l], dtype=np.float32)
            pts_r = np.array([kp.pt for kp in keypoints_lght_r], dtype=np.float32)

            # Order points consistently based on right angle
            pts_l_ordered = self.__order_triangle_points_by_angle(pts_l)
            pts_r_ordered = self.__order_triangle_points_by_angle(pts_r)
            # print(pts_l)
            # print(pts_r)

            # print(pts_l_ordered)
            # print(pts_r_ordered)

            # get the translation matrix between the triangle that forms from the light blobs between the left and right images
            H = cv2.getAffineTransform(pts_l_ordered, missing_vals[:,:])
            M = cv2.getAffineTransform(pts_r_ordered, missing_vals[:,:])


            pts_l_raw = np.array([kp.pt for kp in keypoints_dark_l], dtype=np.float32)
            pts_r_raw = np.array([kp.pt for kp in keypoints_dark_r], dtype=np.float32)


            # matching points from left image with regular grid
            transformed_l = (pts_l_raw @ H[:, :2].T) + H[:, 2]  # Shape: (N, 2)
            tree = cKDTree(filtered_points_2d)
            dist, indices = tree.query(transformed_l, distance_upper_bound=0.4)
            valid = dist != np.inf
            matched_l = pts_l_raw[valid]
            matched_f = filtered_points_2d[indices[valid]]


            # get matching points in right image with updated grid
            transformed_r = (pts_r_raw @ M[:, :2].T) + M[:, 2]  # Shape: (N, 2)
            tree = cKDTree(matched_f)
            dist, indices = tree.query(transformed_r, distance_upper_bound=0.4)
            valid = dist != np.inf
            matched_r = pts_r_raw[valid]
            matched_l = matched_l[indices[valid]]
            matched_f = matched_f[indices[valid]]

            # pts_l = matched_l.reshape(-1, 2)
            # pts_r = matched_r.reshape(-1, 2)


            # # debugging plot
            # fig, axes = plt.subplots(1, 4, figsize=(20, 6))

            # # Convert KeyPoints to NumPy arrays
            # light_l = np.array([kp.pt for kp in keypoints_lght_l], dtype=np.float32)
            # light_r = np.array([kp.pt for kp in keypoints_lght_r], dtype=np.float32)

            # # # Left image with detected circles
            # axes[0].imshow(im_l, cmap='gray')
            # axes[0].plot(light_l[:, 0], light_l[:, 1], 'gx', markersize=10)
            # axes[0].plot(pts_l[:, 0], pts_l[:, 1], 'ro', markersize=5)
            # axes[0].set_title('Left Image with \n Detected Circles')

            # # Right image with detected circles
            # axes[1].imshow(im_r, cmap='gray')
            # axes[1].plot(light_r[:, 0], light_r[:, 1], 'gx', markersize=10)
            # axes[1].plot(pts_r[:, 0], pts_r[:, 1], 'ro', markersize=5)
            # axes[1].set_title('Right Image with \n Detected Circles')

            # axes[2].plot(transformed_l[:, 0], transformed_l[:, 1], 'ro', markersize=5)
            # axes[2].plot(filtered_points_2d[:, 0], filtered_points_2d[:, 1], 'x', markersize=5)
            # axes[2].invert_yaxis()
            # axes[2].set_title('left circles mapped to \n to grid reference frame ')

            # axes[3].plot(transformed_r[:, 0], transformed_r[:, 1], 'ro', markersize=5)
            # axes[3].plot(filtered_points_2d[:, 0], filtered_points_2d[:, 1], 'x', markersize=5)
            # axes[3].invert_yaxis()
            # axes[3].set_title('right cricles mapped to \n to grid reference frame ')

            im_with_keypoints_l = cv2.drawKeypoints(im_l, pts_l, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            im_with_keypoints_r = cv2.drawKeypoints(im_r, pts_r, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            im_with_keypoints_l = cv2.drawKeypoints(im_with_keypoints_l, keypoints_lght_l, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            im_with_keypoints_r = cv2.drawKeypoints(im_with_keypoints_r, keypoints_lght_r, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            side_by_side = np.hstack((im_with_keypoints_l, im_with_keypoints_r))
            cv2.namedWindow("Stereo Keypoints", cv2.WINDOW_NORMAL)
            cv2.imshow("Stereo Keypoints", side_by_side)
            cv2.waitKey(0)

            # Save the figure to a temporary PNG file
            # filename = f"output/frame_{i:03d}.png"
            # plt.savefig(filename)
            # plt.close(fig)
            plt.show()
            #
            # # np.savetxt("matched_l.txt", matched_l, fmt='%.2f')
            # # np.savetxt("matched_r.txt", matched_r, fmt='%.2f')
            # # np.savetxt("matched_f.txt", matched_f, fmt='%.2f')

            # Append for calibration
            matched_f = np.hstack((matched_f, np.zeros((matched_f.shape[0], 1), dtype=matched_f.dtype)))
            self.gridpoints.append(matched_f)
            self.imgpoints_l.append(matched_l)
            self.imgpoints_r.append(matched_r)



    def stereo_calibration(self):
        print(f"Running calibration with {len(self.gridpoints)} valid image pairs...")

        # Left and Right cam calib
        ret_l, Kl, Dl, rvecs_l, tvecs_l = cv2.calibrateCamera(self.gridpoints, self.imgpoints_l, self.img_size, None, None)
        ret_r, Kr, Dr, rvecs_r, tvecs_r = cv2.calibrateCamera(self.gridpoints, self.imgpoints_r, self.img_size, None, None)

        # stereo calib
        ret, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
            objectPoints=self.gridpoints,
            imagePoints1=self.imgpoints_l,
            imagePoints2=self.imgpoints_r,
            cameraMatrix1=Kl,
            distCoeffs1=Dl,
            cameraMatrix2=Kr,
            distCoeffs2=Dr,
            imageSize=self.img_size,
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        # --- Step 3: Per-view reprojection errors ---
        errors_left = []
        errors_right = []

        for i, objp in enumerate(self.gridpoints):
            # Left camera projection
            imgpts_l_proj, _ = cv2.projectPoints(objp, rvecs_l[i], tvecs_l[i], Kl, Dl)
            imgpts_l_proj = imgpts_l_proj.reshape(-1, 2)
            err_l = cv2.norm(self.imgpoints_l[i], imgpts_l_proj, cv2.NORM_L2)/len(self.imgpoints_l)
            errors_left.append(err_l)

            # Right camera projection
            imgpts_r_proj, _ = cv2.projectPoints(objp, rvecs_r[i], tvecs_r[i], Kr, Dr)
            imgpts_r_proj = imgpts_r_proj.reshape(-1, 2)
            err_r = cv2.norm(self.imgpoints_r[i], imgpts_r_proj, cv2.NORM_L2)/len(self.imgpoints_r)
            errors_right.append(err_r)


        print(f"Stereo RMS error: {ret:.4f} px")
        print(f"Mean left RMS error: {np.mean(errors_left):.4f} px")
        print(f"Mean right RMS error: {np.mean(errors_right):.4f} px")
        ret, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
            objectPoints=self.gridpoints,
            imagePoints1=self.imgpoints_l,
            imagePoints2=self.imgpoints_r,
            cameraMatrix1=None,
            distCoeffs1=None,
            cameraMatrix2=None,
            distCoeffs2=None,
            imageSize=self.img_size,
            flags=cv2.CALIB_FIX_ASPECT_RATIO
        )

        print("\n--- Calibration Results ---")
        print("Calibration RMS error:", ret)
        print('\nLeft Camera Matrix:\n', Kl)
        print('Left Distortion Coefficients:\n', Dl)
        print('\nRight Camera Matrix:\n', Kr)
        print('Right Distortion Coefficients:\n', Dr)
        print('\nRotation Matrix (R):\n', R)
        print('Translation Vector (T):\n', T)

        # Save as .npy (NumPy binary)
        np.save('stereo_calibration.npy', {
            'ret': ret,
            'Kl': Kl,
            'Dl': Dl,
            'Kr': Kr,
            'Dr': Dr,
            'R': R,
            'T': T,
            'E': E,
            'F': F
        })

        # Save as .yaml (human-readable)
        calib_data = {
            'ret': float(ret),
            'Kl': Kl.tolist(),
            'Dl': Dl.tolist(),
            'Kr': Kr.tolist(),
            'Dr': Dr.tolist(),
            'R': R.tolist(),
            'T': T.tolist(),
            'E': E.tolist(),
            'F': F.tolist()
        }

        with open('stereo_calibration.yaml', 'w') as f:
            yaml.dump(calib_data, f)




    def setup_visualisation():

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        for i in range(0, 80):
            # For one image (e.g., first):
            rvec = rvecs_l[i]  # rotation vector of plate wrt left camera
            tvec = tvecs_l[i]  # translation vector of plate wrt left camera

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










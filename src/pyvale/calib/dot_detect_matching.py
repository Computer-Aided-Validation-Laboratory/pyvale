import cv2
import numpy as np
import os
import yaml
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# circle detection algorithm
def create_blob_detector(light):
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

# === Utility function to find right-angle vertex and reorder points ===
def angle_between(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

def order_triangle_points_by_angle_desc(pts):
    angles = []
    for i in range(3):
        ang = angle_between(pts[(i+1)%3], pts[i], pts[(i+2)%3])
        angles.append(ang)
    angles = np.array(angles)
    #print("Angles:", angles)
    
    sorted_indices = np.argsort(-angles)
    ordered_pts = np.array([pts[i] for i in sorted_indices], dtype=np.float32)
    
    #print("Sorted indices:", sorted_indices)
    return ordered_pts


detector_lght = create_blob_detector(light=True)
detector_dark = create_blob_detector(light=False)

grid_width = 12
grid_height = 9
square_size = 5.0 # in mm

# Generate full 3D grid
object_points_3d = np.zeros((grid_width * grid_height, 3), np.float32)
object_points_3d[:, :2] = np.mgrid[0:grid_width, 0:grid_height].T.reshape(-1, 2)
object_points_3d[:, :2] *= square_size

# Known missing points (x, y in grid)
missing_points_xy = np.array([
    [2, grid_height-2-1],
    [2, grid_height-7],
    [9, grid_height-2-1],
])

missing_vals = (missing_points_xy * square_size)
missing_vals = missing_vals.astype(np.float32)
print(missing_vals)

# Convert to flat indices
missing_indices = [y * grid_width + x for (x, y) in missing_points_xy]
mask = np.ones(len(object_points_3d), dtype=bool)
mask[missing_indices] = False
filtered_points = object_points_3d[mask]
filtered_points_2d = filtered_points[:, :2]

# arrays that are going to contain matching points for each image
objpoints = []
imgpoints_l = []
imgpoints_r = []

# testing blender images
# path = "../../../../datasets/stereobenchmarks/calibration/faceon_calibration/"
# prefix = "cal_"
# ext = ".tiff"
# prefill = 0

# testing matchID calibration images
path = "../../../../datasets/matchid_calib/DEMO_Calibration"
prefix = "Calibration_"
ext = ".tiff"
prefill = 4
#


path_l = os.path.join(path, f"{prefix}0000_0{ext}")
path_r = os.path.join(path, f"{prefix}0000_1{ext}")
im_l = cv2.imread(path_l, cv2.IMREAD_GRAYSCALE)
im_r = cv2.imread(path_r, cv2.IMREAD_GRAYSCALE)
img_size = (im_l.shape[1], im_l.shape[0])

for i in range(0, 80):
    num = str(i).zfill(prefill)
    path_l = os.path.join(path, f"{prefix}{num}_0{ext}")
    path_r = os.path.join(path, f"{prefix}{num}_1{ext}")

    #print(f"Processing {path_l}")

    im_l = cv2.imread(path_l, cv2.IMREAD_GRAYSCALE)
    im_r = cv2.imread(path_r, cv2.IMREAD_GRAYSCALE)
    if im_l is None or im_r is None:
        print(f"Skipping missing pair: {path_l} {path_r}")
        continue



    # Detect LIGHT blobs
    keypoints_lght_l = detector_lght.detect(im_l)
    keypoints_lght_r = detector_lght.detect(im_r)

    # Detect DARK blobs
    keypoints_dark_l = detector_dark.detect(im_l)
    keypoints_dark_r = detector_dark.detect(im_r)

    # debugging
    # im_with_keypoints_l = cv2.drawKeypoints(im_l, keypoints_dark_l, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # im_with_keypoints_r = cv2.drawKeypoints(im_r, keypoints_dark_r, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # im_with_keypoints_l = cv2.drawKeypoints(im_with_keypoints_l, keypoints_lght_l, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # im_with_keypoints_r = cv2.drawKeypoints(im_with_keypoints_r, keypoints_lght_r, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # side_by_side = np.hstack((im_with_keypoints_l, im_with_keypoints_r))
    # cv2.namedWindow("Stereo Keypoints", cv2.WINDOW_NORMAL)
    # cv2.imshow("Stereo Keypoints", side_by_side)
    # cv2.waitKey(0)


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
    pts_l_ordered = order_triangle_points_by_angle_desc(pts_l)
    pts_r_ordered = order_triangle_points_by_angle_desc(pts_r)
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
    dist, indices = tree.query(transformed_l, distance_upper_bound=1.5)
    valid = dist != np.inf
    matched_l = pts_l_raw[valid]
    matched_f = filtered_points_2d[indices[valid]]


    # get matching points in right image with updated grid
    transformed_r = (pts_r_raw @ M[:, :2].T) + M[:, 2]  # Shape: (N, 2)
    tree = cKDTree(matched_f)
    dist, indices = tree.query(transformed_r, distance_upper_bound=1.5)
    valid = dist != np.inf
    matched_r = pts_r_raw[valid]
    matched_l = matched_l[indices[valid]]
    matched_f = matched_f[indices[valid]]


    # print(matched_l.shape)
    # print(matched_r.shape)
    # print(matched_f.shape)
    #
    #
    print(matched_f.shape[0] / (9*12-4))

    pts_l = matched_l.reshape(-1, 2)
    pts_r = matched_r.reshape(-1, 2)


    # debugging plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    # Convert KeyPoints to NumPy arrays
    light_l = np.array([kp.pt for kp in keypoints_lght_l], dtype=np.float32)
    light_r = np.array([kp.pt for kp in keypoints_lght_r], dtype=np.float32)
    
    # # Left image with detected circles
    axes[0].imshow(im_l, cmap='gray')
    axes[0].plot(light_l[0, 0], light_l[0, 1], 'co', markersize=5)
    axes[0].plot(light_l[1, 0], light_l[1, 1], 'yo', markersize=5)
    axes[0].plot(light_l[2, 0], light_l[2, 1], 'mo', markersize=5)
    axes[0].plot(pts_l[:, 0], pts_l[:, 1], 'ro', markersize=5)
    axes[0].set_title('Left Image with \n Detected Circles')
    
    # Right image with detected circles
    axes[1].imshow(im_r, cmap='gray')
    axes[1].plot(light_r[0, 0], light_r[0, 1], 'co', markersize=5)
    axes[1].plot(light_r[1, 0], light_r[1, 1], 'yo', markersize=5)
    axes[1].plot(light_r[2, 0], light_r[2, 1], 'mo', markersize=5)
    axes[1].plot(pts_r[:, 0], pts_r[:, 1], 'ro', markersize=5)
    axes[1].set_title('Right Image with \n Detected Circles')
    
    axes[2].plot(transformed_l[:, 0], transformed_l[:, 1], 'ro', markersize=5)
    axes[2].plot(filtered_points_2d[:, 0], filtered_points_2d[:, 1], 'x', markersize=5)
    axes[2].invert_yaxis()
    axes[2].set_title('left circles mapped to \n to grid reference frame ')
    
    axes[3].plot(transformed_r[:, 0], transformed_r[:, 1], 'ro', markersize=5)
    axes[3].plot(filtered_points_2d[:, 0], filtered_points_2d[:, 1], 'x', markersize=5)
    axes[3].invert_yaxis()
    axes[3].set_title('right cricles mapped to \n to grid reference frame ')


    # Save the figure to a temporary PNG file
    filename = f"output/frame_{i:03d}.png"
    plt.savefig(filename)
    plt.close(fig)
    # plt.show()
    
    # np.savetxt("matched_l.txt", matched_l, fmt='%.2f')
    # np.savetxt("matched_r.txt", matched_r, fmt='%.2f')
    # np.savetxt("matched_f.txt", matched_f, fmt='%.2f')
    
    print("matched_f:",matched_f.shape)
    print("matched_l:",matched_l.shape)
    print("matched_r:",matched_r.shape)

    # Append for calibration
    matched_f = np.hstack((matched_f, np.zeros((matched_f.shape[0], 1), dtype=matched_f.dtype)))
    # print(matched_f)
    objpoints.append(matched_f)
    imgpoints_l.append(matched_l)
    imgpoints_r.append(matched_r)



print(f"Running calibration with {len(objpoints)} valid image pairs...")

# --- Step 1: Calibrate left and right cameras separately ---
ret_l, Kl, Dl, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, img_size, None, None)
ret_r, Kr, Dr, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, img_size, None, None)

# --- Step 2: Stereo calibration using fixed intrinsics ---
ret, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
    objectPoints=objpoints,
    imagePoints1=imgpoints_l,
    imagePoints2=imgpoints_r,
    cameraMatrix1=Kl,
    distCoeffs1=Dl,
    cameraMatrix2=Kr,
    distCoeffs2=Dr,
    imageSize=img_size,
    flags=cv2.CALIB_FIX_INTRINSIC
)

# --- Step 3: Per-view reprojection errors ---
errors_left = []
errors_right = []

for i, objp in enumerate(objpoints):
    # Left camera projection
    imgpts_l_proj, _ = cv2.projectPoints(objp, rvecs_l[i], tvecs_l[i], Kl, Dl)
    err_l = np.sqrt(np.mean(np.sum((imgpoints_l[i] - imgpts_l_proj.squeeze())**2, axis=1)))
    errors_left.append(err_l)

    # Right camera projection
    imgpts_r_proj, _ = cv2.projectPoints(objp, rvecs_r[i], tvecs_r[i], Kr, Dr)
    err_r = np.sqrt(np.mean(np.sum((imgpoints_r[i] - imgpts_r_proj.squeeze())**2, axis=1)))
    errors_right.append(err_r)

print(f"Stereo RMS error: {ret:.4f} px")
print(f"Mean left RMS error: {np.mean(errors_left):.4f} px")
print(f"Mean right RMS error: {np.mean(errors_right):.4f} px")
ret, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
    objectPoints=objpoints,
    imagePoints1=imgpoints_l,
    imagePoints2=imgpoints_r,
    cameraMatrix1=None,
    distCoeffs1=None,
    cameraMatrix2=None,
    distCoeffs2=None,
    imageSize=img_size,
    flags=cv2.CALIB_FIX_ASPECT_RATIO
)

# === Output Results ===
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

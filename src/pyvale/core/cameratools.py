"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import numpy as np
import copy
from scipy.spatial.transform import Rotation
from pyvale.core.cameradata import CameraData2D
from pyvale.core.sensordata import SensorData
from pyvale.core.cameradata import CameraData
from pyvale.core.blenderscene import BlenderScene

# NOTE: This module is a feature under developement.

def build_pixel_vec_px(cam_data: CameraData2D) -> tuple[np.ndarray,np.ndarray]:
    px_vec_x = np.arange(0,cam_data.num_pixels[0],1)
    px_vec_y = np.arange(0,cam_data.num_pixels[1],1)
    return (px_vec_x,px_vec_y)

def build_pixel_grid_px(cam_data: CameraData2D) -> tuple[np.ndarray,np.ndarray]:
    (px_vec_x,px_vec_y) = build_pixel_vec_px(cam_data)
    return np.meshgrid(px_vec_x,px_vec_y)

def vectorise_pixel_grid_px(cam_data: CameraData2D) -> tuple[np.ndarray,np.ndarray]:
    (px_grid_x,px_grid_y) = build_pixel_grid_px(cam_data)
    return (px_grid_x.flatten(),px_grid_y.flatten())


def build_pixel_vec_leng(cam_data: CameraData2D) -> tuple[np.ndarray,np.ndarray]:
    px_vec_x = np.arange(cam_data.leng_per_px/2,
                         cam_data.field_of_view_local[0],
                         cam_data.leng_per_px)
    px_vec_y = np.arange(cam_data.leng_per_px/2,
                         cam_data.field_of_view_local[1],
                         cam_data.leng_per_px)
    return (px_vec_x,px_vec_y)

def build_pixel_grid_leng(cam_data: CameraData2D) -> tuple[np.ndarray,np.ndarray]:
    (px_vec_x,px_vec_y) = build_pixel_vec_leng(cam_data)
    return np.meshgrid(px_vec_x,px_vec_y)

def vectorise_pixel_grid_leng(cam_data: CameraData2D) -> tuple[np.ndarray,np.ndarray]:
    (px_grid_x,px_grid_y) = build_pixel_grid_leng(cam_data)
    return (px_grid_x.flatten(),px_grid_y.flatten())

def calc_resolution_from_sim(num_px: np.ndarray,
                             coords: np.ndarray,
                             border_px: int,
                             view_plane: tuple[int,int] = (0,1),
                             ) -> float:

    coords_min = np.min(coords, axis=0)
    coords_max = np.max(coords, axis=0)
    field_of_view = np.abs(coords_max - coords_min)
    roi_px = np.array(num_px - 2*border_px,dtype=np.float64)

    resolution = np.zeros_like(view_plane,dtype=np.float64)
    for ii in view_plane:
        resolution[ii] = field_of_view[view_plane[ii]] / roi_px[ii]

    return np.max(resolution)


def calc_centre_from_sim(coords: np.ndarray,
                         view_axes: tuple[int,int] = (0,1)) -> np.ndarray:
    centre = np.mean(coords,axis=0)

    for ii,_ in enumerate(centre):
        if ii not in view_axes:
            centre[ii] = 0.0

    return centre


def build_sensor_data_from_camera(cam_data: CameraData2D) -> SensorData:
    pixels_vectorised = vectorise_pixel_grid_leng(cam_data)

    positions = np.zeros((pixels_vectorised[0].shape[0],3))
    for ii,vv in enumerate(cam_data.view_axes):
        positions[:,vv] = pixels_vectorised[ii] + cam_data.roi_shift_world[ii]

    if cam_data.angle is None:
        angle = None
    else:
        angle = (cam_data.angle,)

    sens_data = SensorData(positions=positions,
                           sample_times=cam_data.sample_times,
                           angles=angle)

    return sens_data


#-------------------------------------------------------------------------------
# NOTE: keep these functions!
# These functions work for 3D cameras calculating imaging dist and fov taking
# account of camera rotation by rotating the bounding box of the sim into cam
# coords

def fov_from_cam_rot(cam_rot: Rotation,
                     coords_world: np.ndarray) -> np.ndarray:
    (xx,yy,zz) = (0,1,2)

    cam_to_world_mat = cam_rot.as_matrix()
    world_to_cam_mat = np.linalg.inv(cam_to_world_mat)

    bb_min = np.min(coords_world,axis=0)
    bb_max = np.max(coords_world,axis=0)

    bound_box_world_vecs = np.array([[bb_min[xx],bb_min[yy],bb_max[zz]],
                                     [bb_max[xx],bb_min[yy],bb_max[zz]],
                                     [bb_max[xx],bb_max[yy],bb_max[zz]],
                                     [bb_min[xx],bb_min[yy],bb_max[zz]],
                                     [bb_min[xx],bb_min[yy],bb_min[zz]],
                                     [bb_max[xx],bb_min[yy],bb_min[zz]],
                                     [bb_max[xx],bb_max[yy],bb_min[zz]],
                                     [bb_min[xx],bb_min[yy],bb_min[zz]],])

    bound_box_cam_vecs = np.matmul(world_to_cam_mat,bound_box_world_vecs.T)
    boundbox_cam_leng = (np.max(bound_box_cam_vecs,axis=1)
                         - np.min(bound_box_cam_vecs,axis=1))

    return np.array((boundbox_cam_leng[xx],boundbox_cam_leng[yy]))


def image_dist_from_fov(num_pixels: np.ndarray,
                        pixel_size: np.ndarray,
                        focal_leng: float,
                        fov_leng: np.ndarray) -> np.ndarray:

    sensor_dims = num_pixels * pixel_size
    fov_angle = 2*np.arctan(sensor_dims/(2*focal_leng))
    image_dist = fov_leng/(2*np.tan(fov_angle/2))
    return image_dist

#-------------------------------------------------------------------------------
# Blender camera tools

def calculate_FOV(cam_data: CameraData) -> tuple[float, float]:
    """A method to calulate the camera's field of view in mm

    Parameters
    ----------
    cam_data : CameraData
        A dataclass containing the camera parameters

    Returns
    -------
    tuple[float, float]
         A tuple containing the field of view in mm in both x and y directions
    """
    FOV_x = (((cam_data.image_dist - cam_data.focal_length)
                / cam_data.focal_length) *
                (cam_data.pixels_size) *
                cam_data.pixels_num[0])[0]
    FOV_y = (cam_data.pixels_num[1] / cam_data.pixels_num[0]) * FOV_x
    FOV_mm = (FOV_x, FOV_y)
    return FOV_mm

def blender_FOV(cam_data: CameraData) -> tuple[float, float]:
    """A method to calculate the camera's field of view in mm using Blender's
    method. This method differs due to one simplification.

    Parameters
    ----------
    cam_data : CameraData
        A dataclass containing the camera parameters

    Returns
    -------
    tuple[float, float]
        A tuple containing the FOV in x and y directions
    """
    FOV_x = (cam_data.pixels_num[0] * cam_data.pixels_size[0] * cam_data.image_dist) / cam_data.focal_length
    FOV_y = (cam_data.pixels_num[1] / cam_data.pixels_num[0]) * FOV_x
    FOV_blender = (FOV_x, FOV_y)
    return FOV_blender

def angular_fov(cam_data: CameraData) -> float: # Not sure if this function is necessary
    """A method to calculate the angular field of view of a camera in degrees

    Parameters
    ----------
    cam_data : CameraData
        A dataclass containing the camera parameters

    Returns
    -------
    float
        The angular field of view in the x-direction in degrees
    """
    (FOV_x, _) = calculate_FOV(cam_data)
    working_dist = cam_data.pos_world[2] - cam_data.roi_cent_world[2]
    half_FOV = FOV_x / 2
    half_AFOV = np.arctan(half_FOV / working_dist)
    AFOV_x = np.degrees(half_AFOV) * 2
    return AFOV_x

def focal_length_from_resolution(pixels_size: np.ndarray,
                                 working_dist: float,
                                 resolution: float) -> float:
    """A method to calculate the required focal length to achieve a certain
    resolution. This is calculated given the pixel size and working distance.
    This method can be used for a 2D setup or for camera 0 for a stereo setup.

    Parameters
    ----------
    pixels_size : np.ndarray
        The camera pixel size in the x and y directions (in mm).
    working_dist : float
        The working distance of the camera to the sample.
    resolution : float
        The desired resolution in mm/px.

    Returns
    -------
    float
        The focal length required to obtain the desired image resolution.
    """
    focal_length = working_dist / ((resolution / pixels_size[0]))
    return focal_length

def blender_camera_from_resolution(pixels_num: np.ndarray,
                                   pixels_size: np.ndarray,
                                   working_dist: float,
                                   resolution: float) -> CameraData:
    """A convenience function to create a camera object in Blender from its pixels,
    the pixel size, the working distance and desired resolution.

    Parameters
    ----------
    pixels_num : np.ndarray
        The number of pixels in the camera, in the x and y directions.
    pixels_size : np.ndarray
        The camera pixels size in mm, in the x and y directions.
    working_dist : float
        The working distance of the camera.
    resolution : float
        The desired mm/px resolution

    Returns
    -------
    CameraData
        A dataclass containing the created camera's parameters.
    """
    focal_length = focal_length_from_resolution(pixels_size, working_dist, resolution)

    cam_data = CameraData(pixels_num=pixels_num,
                          pixels_size=pixels_size,
                          pos_world=(0, 0, working_dist),
                          rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                          roi_cent_world=(0, 0, 0),
                          focal_length=focal_length)
    cam = BlenderScene.add_camera(cam_data)
    return cam_data

def blender_symmetric_stereo(cam_data_0: CameraData, stereo_angle:float) -> CameraData:
    """A convenience function to set up a symmetric stereo camera system, given
    an initial CameraData dataclass and a stereo angle. This assumes the basic
    camera parameters are the same.

    Parameters
    ----------
    cam_data_0 : CameraData
        A dataclass containing the camera parameters for a single camera, which
        will be camera 0.
    stereo_angle : float
        The stereo angle between the two cameras.

    Returns
    -------
    CameraData
        A dataclass for the created camera, camera 1 in the stereo setup.
    """
    cam_data_1 = copy.deepcopy(cam_data_0)
    base = 2 * cam_data_0.pos_world[2] * np.tan(np.radians(stereo_angle) / 2)

    cam_data_0.pos_world[0] -= base / 2
    cam_data_1.pos_world[0] += base / 2

    cam_0_rot = (0, -np.radians(stereo_angle / 2), 0)
    cam_0_rot = Rotation.from_euler("xyz", cam_0_rot, degrees=False)
    cam_data_0.rot_world = cam_0_rot

    cam_1_rot = (0, np.radians(stereo_angle / 2), 0)
    cam_1_rot = Rotation.from_euler("xyz", cam_1_rot, degrees=False)
    cam_data_1.rot_world = cam_1_rot

    cam0 = BlenderScene.add_camera(cam_data_0)
    cam1 = BlenderScene.add_camera(cam_data_1)

    return cam_data_1


def blender_faceon_stereo(cam_data_0: CameraData, stereo_angle: float) -> CameraData:
    """A convenience function to set up a face-on stereo camera system, given
    an initial CameraData dataclass and a stereo angle. This assumes the basic
    camera parameters are the same.

    Parameters
    ----------
    cam_data_0 : CameraData
        A dataclass containing the camera parameters for a single camera, which
        will be camera 0.
    stereo_angle : float
        The stereo angle between the two cameras.

    Returns
    -------
    CameraData
        A dataclass for the created camera, camera 1 in the stereo setup.
    """
    cam_data_1 = copy.deepcopy(cam_data_0)
    base = cam_data_0.pos_world[2] * np.tan(np.radians(stereo_angle))
    cam_data_1.pos_world[0] += base

    rotation_angle = (0, np.radians(stereo_angle), 0)
    rotation_angle = Rotation.from_euler("xyz", rotation_angle, degrees=False)
    cam_data_1.rot_world = rotation_angle

    cam0 = BlenderScene.add_camera(cam_data_0)
    cam1 = BlenderScene.add_camera(cam_data_1)

    return cam_data_1
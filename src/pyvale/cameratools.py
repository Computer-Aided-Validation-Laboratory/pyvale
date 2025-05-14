# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: This module is a feature under developement.
"""

import warnings
from pathlib import Path
import numpy as np
from scipy.signal import convolve2d
import copy
from scipy.spatial.transform import Rotation
import matplotlib.image as mplim
from PIL import Image
from pyvale.cameradata2d import CameraData2D
from pyvale.sensordata import SensorData
from pyvale.cameradata import CameraData
from pyvale.camerastereo import CameraStereo


class CameraTools:
    @staticmethod
    def load_image(im_path: Path) -> np.ndarray:

        input_im = mplim.imread(im_path).astype(np.float64)
        # If we have RGB then get rid of it
        # TODO: make sure this is collapsing RGB to grey scale coorectly
        if input_im.ndim > 2:
            input_im = input_im[:,:,0]

        return input_im

    @staticmethod
    def save_image(save_file: Path,
                image: np.ndarray,
                n_bits: int = 16) -> None:

        # Need to flip image so coords are top left with Y down
        # TODO check this
        image = image[::-1,:]

        if n_bits > 8:
            im = Image.fromarray(image.astype(np.uint16))
        else:
            im = Image.fromarray(image.astype(np.uint8))

        im.save(save_file)

    @staticmethod
    def image_num_str(im_num: int, width: int , cam_num: int = -1) -> str:
        num_str = str(im_num)
        num_str = num_str.zfill(width)

        if cam_num >= 0:
            num_str = num_str+'_'+str(cam_num)

        return num_str

    @staticmethod
    def pixel_vec_px(pixels_count: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        px_vec_x = np.arange(0,pixels_count[0],1)
        px_vec_y = np.arange(0,pixels_count[1],1)
        return (px_vec_x,px_vec_y)
    @staticmethod
    def pixel_grid_px(pixels_count: np.ndarray
                            ) -> tuple[np.ndarray,np.ndarray]:
        (px_vec_x,px_vec_y) = CameraTools.pixel_vec_px(pixels_count)
        return np.meshgrid(px_vec_x,px_vec_y)
    @staticmethod
    def vectorise_pixel_grid_px(pixels_count: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        (px_grid_x,px_grid_y) = CameraTools.pixel_grid_px(pixels_count)
        return (px_grid_x.flatten(),px_grid_y.flatten())

    @staticmethod
    def subpixel_vec_px(pixels_count: np.ndarray,
                            subsample: int = 2) -> tuple[np.ndarray,np.ndarray]:
        px_vec_x = np.arange(1/(2*subsample),pixels_count[0],1/subsample)
        px_vec_y = np.arange(1/(2*subsample),pixels_count[1],1/subsample)
        return (px_vec_x,px_vec_y)

    @staticmethod
    def subpixel_grid_px(pixels_count: np.ndarray,
                            subsample: int = 2) -> tuple[np.ndarray,np.ndarray]:
        (px_vec_x,px_vec_y) = CameraTools.subpixel_vec_px(pixels_count,subsample)
        return np.meshgrid(px_vec_x,px_vec_y)

    @staticmethod
    def vectorise_subpixel_grid_px(pixels_count: np.ndarray,
                                subsample: int = 2) -> tuple[np.ndarray,np.ndarray]:
        (px_grid_x,px_grid_y) = CameraTools.subpixel_grid_px(pixels_count,subsample)
        return (px_grid_x.flatten(),px_grid_y.flatten())

    @staticmethod
    def pixel_vec_leng(field_of_view: np.ndarray,
                            leng_per_px: float) -> tuple[np.ndarray,np.ndarray]:
        px_vec_x = np.arange(leng_per_px/2,
                            field_of_view[0],
                            leng_per_px)
        px_vec_y = np.arange(leng_per_px/2,
                            field_of_view[1],
                            leng_per_px)
        return (px_vec_x,px_vec_y)

    @staticmethod
    def pixel_grid_leng(field_of_view: np.ndarray,
                            leng_per_px: float) -> tuple[np.ndarray,np.ndarray]:
        (px_vec_x,px_vec_y) = CameraTools.pixel_vec_leng(field_of_view,leng_per_px)
        return np.meshgrid(px_vec_x,px_vec_y)

    @staticmethod
    def vectorise_pixel_grid_leng(field_of_view: np.ndarray,
                                leng_per_px: float) -> tuple[np.ndarray,np.ndarray]:
        (px_grid_x,px_grid_y) = CameraTools.pixel_grid_leng(field_of_view,leng_per_px)
        return (px_grid_x.flatten(),px_grid_y.flatten())


    @staticmethod
    def subpixel_vec_leng(field_of_view: np.ndarray,
                          leng_per_px: float,
                          subsample: int = 2) -> tuple[np.ndarray,np.ndarray]:
        px_vec_x = np.arange(leng_per_px/(2*subsample),
                            field_of_view[0],
                            leng_per_px/subsample)
        px_vec_y = np.arange(leng_per_px/(2*subsample),
                            field_of_view[1],
                            leng_per_px/subsample)
        return (px_vec_x,px_vec_y)

    @staticmethod
    def subpixel_grid_leng(field_of_view: np.ndarray,
                                leng_per_px: float,
                                subsample: int = 2) -> tuple[np.ndarray,np.ndarray]:
        (px_vec_x,px_vec_y) = CameraTools.subpixel_vec_leng(
                                                    field_of_view,
                                                    leng_per_px,
                                                    subsample)
        return np.meshgrid(px_vec_x,px_vec_y)

    @staticmethod
    def vectorise_subpixel_grid_leng(field_of_view: np.ndarray,
                                    leng_per_px: float,
                                    subsample: int = 2) -> tuple[np.ndarray,np.ndarray]:
        (px_grid_x,px_grid_y) = CameraTools.subpixel_grid_leng(
                                                        field_of_view,
                                                        leng_per_px,
                                                        subsample)
        return (px_grid_x.flatten(),px_grid_y.flatten())

    @staticmethod
    def calc_resolution_from_sim_2d(pixels_count: np.ndarray,
                                    coords: np.ndarray,
                                    pixels_border: int,
                                    view_plane_axes: tuple[int,int] = (0,1),
                                    ) -> float:

        coords_min = np.min(coords, axis=0)
        coords_max = np.max(coords, axis=0)
        field_of_view = np.abs(coords_max - coords_min)
        roi_px = np.array(pixels_count - 2*pixels_border,dtype=np.float64)

        resolution = np.zeros_like(view_plane_axes,dtype=np.float64)
        for ii in view_plane_axes:
            resolution[ii] = field_of_view[view_plane_axes[ii]] / roi_px[ii]

        return np.max(resolution)

    @staticmethod
    def calc_roi_cent_from_sim_2d(coords: np.ndarray,) -> np.ndarray:
        return np.mean(coords,axis=0)

    @staticmethod
    def crop_image_rectangle(image: np.ndarray,
                             pixels_count: np.ndarray,
                             corner: tuple[int,int] = (0,0)
                             ) -> np.ndarray:

        crop_x = np.array((corner[0],pixels_count[0]),dtype=np.int32)
        crop_y = np.array((corner[1],pixels_count[1]),dtype=np.int32)

        if corner[0] < 0:
            crop_x[0] = 0
            warnings.warn("Crop edge outside image, setting to image edge.")

        if corner[1] < 0:
            crop_y[0] = 0
            warnings.warn("Crop edge outside image, setting to image edge.")

        if ((corner[0]+pixels_count[0]) > image.shape[1]):
            crop_x[1] = image.shape[0]
            warnings.warn("Crop edge outside image, setting to image edge.")

        if (corner[1]+pixels_count[1]) > image.shape[0]:
            crop_y[1] = image.shape[1]
            warnings.warn("Crop edge outside image, setting to image edge.")

        return image[crop_y[0]:crop_y[1],crop_x[0]:crop_x[1]]

    @staticmethod
    def average_subpixel_image(subpx_image: np.ndarray,
                           subsample: int) -> np.ndarray:
        if subsample <= 1:
            return subpx_image

        conv_mask = np.ones((subsample,subsample))/(subsample**2)
        subpx_image_conv = convolve2d(subpx_image,conv_mask,mode='same')
        avg_image = subpx_image_conv[round(subsample/2)-1::subsample,
                                    round(subsample/2)-1::subsample]
        return avg_image

    @staticmethod
    def build_sensor_data_from_camera_2d(cam_data: CameraData2D) -> SensorData:
        pixels_vectorised = CameraTools.vectorise_pixel_grid_leng(cam_data.field_of_view,
                                                    cam_data.leng_per_px)

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

    @staticmethod
    def fov_from_cam_rot_3d(cam_rot: Rotation,
                            coords_world: np.ndarray) -> np.ndarray:
        (xx,yy,zz) = (0,1,2)

        cam_to_world_mat = cam_rot.as_matrix()
        world_to_cam_mat = np.linalg.inv(cam_to_world_mat)

        bb_min = np.min(coords_world,axis=0)
        bb_max = np.max(coords_world,axis=0)

        bound_box_world_vecs = np.array([[bb_min[xx],bb_min[yy],bb_max[zz]],
                                         [bb_max[xx],bb_min[yy],bb_max[zz]],
                                         [bb_max[xx],bb_max[yy],bb_max[zz]],
                                         [bb_min[xx],bb_max[yy],bb_max[zz]],

                                         [bb_min[xx],bb_min[yy],bb_min[zz]],
                                         [bb_max[xx],bb_min[yy],bb_min[zz]],
                                         [bb_max[xx],bb_max[yy],bb_min[zz]],
                                         [bb_min[xx],bb_max[yy],bb_min[zz]],])

        bound_box_cam_vecs = np.matmul(world_to_cam_mat,bound_box_world_vecs.T)
        boundbox_cam_leng = (np.max(bound_box_cam_vecs,axis=1)
                            - np.min(bound_box_cam_vecs,axis=1))

        # print(80*"-")
        # print(f"{bb_min=}")
        # print(f"{bb_max=}")
        # print()
        # print("Cam to world mat:")
        # print(cam_to_world_mat)
        # print()
        # print("World to cam mat:")
        # print(world_to_cam_mat)
        # print(80*"-")

        return np.array((boundbox_cam_leng[xx],boundbox_cam_leng[yy]))

    @staticmethod
    def image_dist_from_fov_3d(pixel_num: np.ndarray,
                               pixel_size: np.ndarray,
                               focal_leng: float,
                               fov_leng: np.ndarray) -> np.ndarray:

        sensor_dims = pixel_num * pixel_size
        fov_angle = 2*np.arctan(sensor_dims/(2*focal_leng))
        image_dist = fov_leng/(2*np.tan(fov_angle/2))
        return image_dist

    @staticmethod
    def pos_fill_frame_from_rotation(coords_world: np.ndarray,
                                     pixel_num: np.ndarray,
                                     pixel_size: np.ndarray,
                                     focal_leng: float,
                                     cam_rot: Rotation,
                                     frame_fill: float = 1.0,
                                     ) -> tuple[np.ndarray,
                                                np.ndarray]:
        fov_leng = CameraTools.fov_from_cam_rot_3d(
            cam_rot=cam_rot,
            coords_world=coords_world,
        )

        # Scales the FOV by the given factor, greater than 1.0 will zoom out
        # making sure the mesh is wholly within the image
        fov_leng = frame_fill*fov_leng

        image_dist = CameraTools.image_dist_from_fov_3d(
            pixel_num=pixel_num,
            pixel_size=pixel_size,
            focal_leng=focal_leng,
            fov_leng=fov_leng,
        )

        roi_pos_world = (np.max(coords_world[:,:-1],axis=0)
                         + np.min(coords_world[:,:-1],axis=0))/2.0
        cam_z_dir_world = cam_rot.as_matrix()[:,-1]
        cam_pos_world = (roi_pos_world + np.max(image_dist)*cam_z_dir_world)

        print(80*"-")
        print(f"{fov_leng=}")
        print(f"{image_dist=}")
        print(80*"-")

        return (roi_pos_world,cam_pos_world)


    #---------------------------------------------------------------------------
    # Blender camera tools

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def calculate_mm_px_resolution(cam_data: CameraData) -> float:
        """Function to calculate the mm/px resolution of a camera

        Parameters
        ----------
        cam_data : CameraData
            A dataclass containing the camera parameters

        Returns
        -------
        float
            The mm/px resolution
        """
        FOV_mm = CameraTools.blender_FOV(cam_data)
        resolution = FOV_mm[0] / cam_data.pixels_num[0]
        return resolution

    @staticmethod
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

    @staticmethod
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
        focal_length = CameraTools.focal_length_from_resolution(pixels_size, working_dist, resolution)

        cam_data = CameraData(pixels_num=pixels_num,
                            pixels_size=pixels_size,
                            pos_world=(0, 0, working_dist),
                            rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                            roi_cent_world=(0, 0, 0),
                            focal_length=focal_length)
        return cam_data

    @staticmethod
    def symmetric_stereo_cameras(cam_data_0: CameraData,
                                stereo_angle:float) -> CameraStereo:
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
        CameraStereo
            An instance of the CameraStereo class. This class contains
            information about each of the cameras, as well as the extrinsic
            parameters between them.
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

        stereo_system = CameraStereo(cam_data_0, cam_data_1)

        return stereo_system

    @staticmethod
    def faceon_stereo_cameras(cam_data_0: CameraData,
                            stereo_angle: float) -> CameraStereo:
        # TODO: Correct docstring
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
        CameraStereo
            An instance of the CameraStereo class. This class contains
            information about each of the cameras, as well as the extrinsic
            parameters between them.
        """
        cam_data_1 = copy.deepcopy(cam_data_0)
        base = cam_data_0.pos_world[2] * np.tan(np.radians(stereo_angle))
        cam_data_1.pos_world[0] += base

        rotation_angle = (0, np.radians(stereo_angle), 0)
        rotation_angle = Rotation.from_euler("xyz", rotation_angle, degrees=False)
        cam_data_1.rot_world = rotation_angle

        stereo_system = CameraStereo(cam_data_0, cam_data_1)

        return stereo_system
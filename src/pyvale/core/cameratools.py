"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import warnings
from pathlib import Path
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.transform import Rotation
import matplotlib.image as mplim
from PIL import Image
from pyvale.core.cameradata2d import CameraData2D
from pyvale.core.sensordata import SensorData

# NOTE: This module is a feature under developement.

class CameraTools:
    #-------------------------------------------------------------------------------
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

    #-------------------------------------------------------------------------------
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

    #-------------------------------------------------------------------------------
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

    #-------------------------------------------------------------------------------
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

    #-------------------------------------------------------------------------------
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

    #-------------------------------------------------------------------------------
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

    #---------------------------------------------------------------------------
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
                                        [bb_min[xx],bb_min[yy],bb_max[zz]],
                                        [bb_min[xx],bb_min[yy],bb_min[zz]],
                                        [bb_max[xx],bb_min[yy],bb_min[zz]],
                                        [bb_max[xx],bb_max[yy],bb_min[zz]],
                                        [bb_min[xx],bb_min[yy],bb_min[zz]],])

        bound_box_cam_vecs = np.matmul(world_to_cam_mat,bound_box_world_vecs.T)
        boundbox_cam_leng = (np.max(bound_box_cam_vecs,axis=1)
                            - np.min(bound_box_cam_vecs,axis=1))

        return np.array((boundbox_cam_leng[xx],boundbox_cam_leng[yy]))

    @staticmethod
    def image_dist_from_fov_3d(num_pixels: np.ndarray,
                            pixel_size: np.ndarray,
                            focal_leng: float,
                            fov_leng: np.ndarray) -> np.ndarray:

        sensor_dims = num_pixels * pixel_size
        fov_angle = 2*np.arctan(sensor_dims/(2*focal_leng))
        image_dist = fov_leng/(2*np.tan(fov_angle/2))
        return image_dist

    #-------------------------------------------------------------------------------
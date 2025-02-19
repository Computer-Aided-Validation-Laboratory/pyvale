"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass, field
import time
from pathlib import Path
from multiprocessing.pool import ThreadPool, Pool

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import convolve2d
import numba

import mooseherder as mh
import pyvale


@dataclass(slots=True)
class CameraData:
    num_pixels: np.ndarray
    pixel_size: np.ndarray

    pos_world: np.ndarray
    rot_world: Rotation

    roi_center_world: np.ndarray

    focal_length: float = 50.0
    sub_samp: int = 2

    back_face_removal: bool = True

    sensor_size: np.ndarray = field(init=False)
    image_dims: np.ndarray = field(init=False)
    image_dist: float = field(init=False)
    cam_to_world_mat: np.ndarray = field(init=False)
    world_to_cam_mat: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.image_dist = np.linalg.norm(self.pos_world - self.roi_center_world)
        self.sensor_size = self.num_pixels*self.pixel_size
        self.image_dims = (self.image_dist
                           *self.sensor_size/self.focal_length)

        self.cam_to_world_mat = np.zeros((4,4))
        self.cam_to_world_mat[0:3,0:3] = self.rot_world.as_matrix()
        self.cam_to_world_mat[-1,-1] = 1.0
        self.cam_to_world_mat[0:3,-1] = self.pos_world
        self.world_to_cam_mat = np.linalg.inv(self.cam_to_world_mat)


class Rasteriser:
    @staticmethod
    def world_to_raster_coords(cam_data: CameraData,
                               coords_world: np.ndarray) -> np.ndarray:
        # Index notation for numpy array index interpretation
        xx: int = 0
        yy: int = 1
        zz: int = 2
        ww: int = 3

        # Project onto camera coords
        coords_raster = np.matmul(cam_data.world_to_cam_mat,coords_world)

        # NOTE: w is not 1 when the matrix is a perspective projection! It is only 1
        # here when we have an affine transformation
        coords_raster[xx,:] = coords_raster[xx,:] / coords_raster[ww,:]
        coords_raster[yy,:] = coords_raster[yy,:] / coords_raster[ww,:]
        coords_raster[zz,:] = coords_raster[zz,:] / coords_raster[ww,:]

        # Coords Image: Perspective divide
        coords_raster[xx,:] = (cam_data.image_dist * coords_raster[xx,:]
                            / -coords_raster[zz,:])
        coords_raster[yy,:] = (cam_data.image_dist * coords_raster[yy,:]
                            / -coords_raster[zz,:])

        # Coords NDC: Convert to normalised device coords in the range [-1,1]
        coords_raster[xx,:] = 2*coords_raster[xx,:] / cam_data.image_dims[xx]
        coords_raster[yy,:] = 2*coords_raster[yy,:] / cam_data.image_dims[yy]

        # Coords Raster: Covert to pixel (raster) coords
        # Shape = ([X,Y,Z],num_nodes)
        coords_raster[xx,:] = (coords_raster[xx,:] + 1)/2 * cam_data.num_pixels[xx]
        coords_raster[yy,:] = (1-coords_raster[yy,:])/2 * cam_data.num_pixels[yy]
        coords_raster[zz,:] = -coords_raster[zz,:]

        return coords_raster

    @staticmethod
    def create_transformed_elem_arrays(cam_data: CameraData,
                                       coords_world: np.ndarray,
                                       connectivity: np.ndarray,
                                       field_array: np.ndarray,
                                       ) -> tuple[np.ndarray,np.ndarray]:
        zz: int = 2

        # Convert world coords of all elements in the scene
        # shape=(coord[X,Y,Z],num_nodes)
        coords_raster = Rasteriser.world_to_raster_coords(cam_data,coords_world)

        # Convert to perspective correct hyperbolic interpolation for z interp
        # shape=(coord[X,Y,Z],num_nodes)
        coords_raster[zz,:] = 1/coords_raster[zz,:]

        # shape=(coord[X,Y,Z],node_per_elem,elem_num)
        elem_raster_coords = coords_raster[:,connectivity]
        # shape=(nodes_per_elem,coord[X,Y,Z],elem_num)
        elem_raster_coords = np.swapaxes(elem_raster_coords,0,1)

        # NOTE: we have already inverted the raster z coordinate above so to divide
        # by z here we need to multiply
        # shape=(n_nodes,num_time_steps)
        field_divide_z = (field_array.T * coords_raster[zz,:]).T
        # shape=(nodes_per_elem,num_elems,num_time_steps)
        field_divide_z = field_divide_z[connectivity,:]

        return (elem_raster_coords,field_divide_z)


    @staticmethod
    def back_face_removal_mask(cam_data: CameraData,
                            coords_world: np.ndarray,
                            connect: np.ndarray
                            ) -> np.ndarray:
        coords_cam = np.matmul(cam_data.world_to_cam_mat,coords_world)

        # shape=(coord[X,Y,Z,W],node_per_elem,elem_num)
        elem_cam_coords = coords_cam[:,connect]
        # shape=(nodes_per_elem,coord[X,Y,Z,W],elem_num)
        elem_cam_coords = np.swapaxes(elem_cam_coords,0,1)

        # Calculate the normal vectors for all of the elements, remove the w coord
        # shape=(coord[X,Y,Z],elem_num)
        elem_cam_edge0 = elem_cam_coords[1,:-1,:] - elem_cam_coords[0,:-1,:]
        elem_cam_edge1 = elem_cam_coords[2,:-1,:] - elem_cam_coords[0,:-1,:]
        elem_cam_normals = np.cross(elem_cam_edge0,elem_cam_edge1,
                                    axisa=0,axisb=0).T
        elem_cam_normals = elem_cam_normals / np.linalg.norm(elem_cam_normals,axis=0)

        cam_normal = np.array([0,0,1])
        # shape=(num_elems,)
        proj_elem_to_cam = np.dot(cam_normal,elem_cam_normals)

        # NOTE this should be a numerical precision tolerance (epsilon)
        back_face_mask = proj_elem_to_cam > 1e-8

        return back_face_mask

    @staticmethod
    def crop_and_bound_elements(cam_data: CameraData,
                                elem_raster_coords: np.ndarray,
                                ) -> tuple[np.ndarray,np.ndarray]:
        xx: int = 0
        yy: int = 1

        #shape=(coord[X,Y,Z],elem_num)
        elem_raster_coord_min = np.min(elem_raster_coords,axis=0)
        elem_raster_coord_max = np.max(elem_raster_coords,axis=0)

        # Check that min/max nodes are within the 4 edges of the camera image
        #shape=(4_edges_to_check,num_elems)
        crop_mask = np.zeros([4,elem_raster_coords.shape[-1]])
        crop_mask[0,elem_raster_coord_min[xx,:] <= (cam_data.num_pixels[xx]-1)] = 1
        crop_mask[1,elem_raster_coord_min[yy,:] <= (cam_data.num_pixels[yy]-1)] = 1
        crop_mask[2,elem_raster_coord_max[xx,:] >= 0] = 1
        crop_mask[3,elem_raster_coord_max[yy,:] >= 0] = 1
        crop_mask = np.sum(crop_mask,0) == 4

        # Get only the elements that are within the FOV
        # Mask the elem coords and the max and min elem coords for processing
        elem_raster_coord_min = elem_raster_coord_min[:,crop_mask]
        elem_raster_coord_max = elem_raster_coord_max[:,crop_mask]
        num_elems_in_image = elem_raster_coord_min.shape[1]

        # Find the indices of the bounding box that each element lies within on
        # the image, bounded by the upper and lower edges of the image
        elem_bound_boxes_inds = np.zeros([4,num_elems_in_image],dtype=np.int32)
        elem_bound_boxes_inds[0,:] = Rasteriser.elem_bound_box_low(
                                            elem_raster_coord_min[xx,:])
        elem_bound_boxes_inds[1,:] = Rasteriser.elem_bound_box_high(
                                            elem_raster_coord_max[xx,:],
                                            cam_data.num_pixels[xx]-1)
        elem_bound_boxes_inds[2,:] = Rasteriser.elem_bound_box_low(
                                            elem_raster_coord_min[yy,:])
        elem_bound_boxes_inds[3,:] = Rasteriser.elem_bound_box_high(
                                            elem_raster_coord_max[yy,:],
                                            cam_data.num_pixels[yy]-1)

        return (crop_mask,elem_bound_boxes_inds)

    @staticmethod
    def elem_bound_box_low(coord_min: np.ndarray) -> np.ndarray:
        bound_elem = np.floor(coord_min).astype(np.int32)
        bound_low = np.zeros_like(coord_min,dtype=np.int32)
        bound_mat = np.vstack((bound_elem,bound_low))
        return np.max(bound_mat,axis=0)

    @staticmethod
    def elem_bound_box_high(coord_max: np.ndarray,image_px: int) -> np.ndarray:
        bound_elem = np.ceil(coord_max).astype(np.int32)
        bound_high = image_px*np.ones_like(coord_max,dtype=np.int32)
        bound_mat = np.vstack((bound_elem,bound_high))
        bound = np.min(bound_mat,axis=0)
        return bound

    @staticmethod
    def raster_setup(cam_data: CameraData,
                     coords_world: np.ndarray,
                     connectivity: np.ndarray,
                     field_data: np.ndarray
                     ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:

        (elem_raster_coords,field_divide_z) = \
            Rasteriser.create_transformed_elem_arrays(cam_data,
                                                      coords_world,
                                                      connectivity,
                                                      field_data)

        #-----------------------------------------------------------------------
        # BACKFACE REMOVAL
        remove_backface = False
        if remove_backface:
            # shape=(num_elems,)
            back_face_mask = Rasteriser.back_face_removal_mask(cam_data,
                                                            coords_world,
                                                            connectivity)
            # Mask and remove w coord
            # shape=(nodes_per_elem,coord[X,Y,Z,W],num_elems_in_scene)
            elem_raster_coords = elem_raster_coords[:,:,back_face_mask]
            # shape=(nodes_per_elem,num_elems_in_scene,num_time_steps)
            field_divide_z = field_divide_z[:,back_face_mask,:]

        #-----------------------------------------------------------------------
        # CROPPING & BOUNDING BOX OPERATIONS
        (crop_mask,elem_bound_box_inds) = Rasteriser.crop_and_bound_elements(
            cam_data,
            elem_raster_coords
        )
        # Apply crop using mask and remove w coord
        # shape=(nodes_per_elem,coord[X,Y,Z],num_elems_in_scene)
        elem_raster_coords = elem_raster_coords[:,:-1,crop_mask]
        num_elems_in_image = elem_raster_coords.shape[-1]
        # shape=(nodes_per_elem,num_elems_in_scene,num_time_steps)
        field_divide_z = field_divide_z[:,crop_mask,:]

        #-----------------------------------------------------------------------
        # ELEMENT AREAS FOR INTERPOLATION
        elem_areas = edge_function(elem_raster_coords[0,:,:],
                                   elem_raster_coords[1,:,:],
                                   elem_raster_coords[2,:,:])

        return (elem_raster_coords,
                elem_bound_box_inds,
                elem_areas,
                field_divide_z)


    @staticmethod
    def raster_one_element(
                    cam_data: CameraData,
                    elem_raster_coords: np.ndarray,
                    elem_bound_box_inds: np.ndarray,
                    elem_area: float,
                    field_divide_z: np.ndarray
                    ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        # Index shorthand for interpreting numpy array dimensions
        zz: int = 2
        x_start: int = 0
        x_end: int = 1
        y_start: int = 2
        y_end: int = 3

        # Create the subpixel coords inside the bounding box to test with the
        # edge function. Use the pixel indices of the bounding box.
        bound_subpx_x = np.arange(elem_bound_box_inds[x_start],
                                  elem_bound_box_inds[x_end],
                                  1/cam_data.sub_samp) + 1/(2*cam_data.sub_samp)
        bound_subpx_y = np.arange(elem_bound_box_inds[y_start],
                                  elem_bound_box_inds[y_end],
                                  1/cam_data.sub_samp) + 1/(2*cam_data.sub_samp)
        (bound_subpx_grid_x,bound_subpx_grid_y) = np.meshgrid(bound_subpx_x,
                                                              bound_subpx_y)
        bound_coords_grid_shape = bound_subpx_grid_x.shape
        bound_subpx_coords_flat = np.vstack((bound_subpx_grid_x.flatten(),
                                             bound_subpx_grid_y.flatten()))

        # Create the subpixel indices for buffer slicing later
        subpx_inds_x = np.arange(cam_data.sub_samp*elem_bound_box_inds[x_start],
                                 cam_data.sub_samp*elem_bound_box_inds[x_end])
        subpx_inds_y = np.arange(cam_data.sub_samp*elem_bound_box_inds[y_start],
                                 cam_data.sub_samp*elem_bound_box_inds[y_end])
        (subpx_inds_grid_x,subpx_inds_grid_y) = np.meshgrid(subpx_inds_x,
                                                            subpx_inds_y)

        # We compute the edge function for all pixels in the box to determine if the
        # pixel is inside the element or not
        # NOTE: first axis of element_raster_coords is the node/vertex num.
        edge = np.zeros((3,bound_subpx_coords_flat.shape[1]))
        edge[0,:] = edge_function(elem_raster_coords[1,:],
                                  elem_raster_coords[2,:],
                                  bound_subpx_coords_flat)
        edge[1,:] = edge_function(elem_raster_coords[2,:],
                                  elem_raster_coords[0,:],
                                  bound_subpx_coords_flat)
        edge[2,:] = edge_function(elem_raster_coords[0,:],
                                elem_raster_coords[1,:],
                                bound_subpx_coords_flat)

        # Now we check where the edge function is above zero for all edges
        edge_check = np.zeros_like(edge,dtype=np.int8)
        edge_check[edge >= 0.0] = 1
        edge_check = np.sum(edge_check, axis=0)
        # Create a mask with the check, TODO check the 3 here for non triangles
        edge_mask_flat = edge_check == 3
        edge_mask_grid = np.reshape(edge_mask_flat,bound_coords_grid_shape)

        # Calculate the weights for the masked pixels
        edge_masked = edge[:,edge_mask_flat]
        interp_weights = edge_masked / elem_area

        # Compute the depth of all pixels using hyperbolic interp
        px_coord_z = 1/(elem_raster_coords[0,zz] * interp_weights[0,:]
                      + elem_raster_coords[1,zz] * interp_weights[1,:]
                      + elem_raster_coords[2,zz] * interp_weights[2,:])

        field_interp = ((field_divide_z[0] * interp_weights[0,:]
                       + field_divide_z[1] * interp_weights[1,:]
                       + field_divide_z[2] * interp_weights[2,:])
                       * px_coord_z)

        return (px_coord_z,
                field_interp,
                subpx_inds_grid_x[edge_mask_grid],
                subpx_inds_grid_y[edge_mask_grid])

    @staticmethod
    def raster_loop_sequential(cam_data: CameraData,
                               elem_raster_coords: np.ndarray,
                               elem_bound_box_inds: np.ndarray,
                               elem_areas: np.ndarray,
                               field_frame_divide_z: np.ndarray
                               ) -> tuple[np.ndarray,np.ndarray,int]:

        # NOTE: this version cannot be run in parallel as the depth buffer is
        # filled on the fly as we process each element. This version will be
        # more memory efficient as we do not need to store each rastered
        # element fragment in memory.

        depth_buffer = 1e6*np.ones(cam_data.sub_samp*cam_data.num_pixels).T
        image_buffer = np.full(cam_data.sub_samp*cam_data.num_pixels,0.0).T

        for ee in range(elem_raster_coords.shape[-1]):
            (px_coord_z,
            field_interp,
            subpx_inds_x_in,
            subpx_inds_y_in) = Rasteriser.raster_one_element(
                                                cam_data,
                                                elem_raster_coords[:,:,ee],
                                                elem_bound_box_inds[:,ee],
                                                elem_areas[ee],
                                                field_frame_divide_z[:,ee])


            #  Build a mask to replace the depth information if there is already an
            # element in front of the one we are rendering
            px_coord_z_depth_mask = (px_coord_z
                < depth_buffer[subpx_inds_y_in,subpx_inds_x_in])

            # Initialise the z coord to the value in the depth buffer
            px_coord_z_masked = depth_buffer[subpx_inds_y_in,subpx_inds_x_in]
            # Use the depth mask to overwrite the depth buffer values if points are in
            # front of the values in the depth buffer
            px_coord_z_masked[px_coord_z_depth_mask] = px_coord_z[px_coord_z_depth_mask]

            # Push the masked values into the depth buffer
            depth_buffer[subpx_inds_y_in,subpx_inds_x_in] = px_coord_z_masked

            # Mask the image buffer using the depth mask
            image_buffer_depth_masked = image_buffer[subpx_inds_y_in,subpx_inds_x_in]
            image_buffer_depth_masked[px_coord_z_depth_mask] = field_interp[px_coord_z_depth_mask]

            # Push the masked values into the image buffer
            image_buffer[subpx_inds_y_in,subpx_inds_x_in] = image_buffer_depth_masked

        #---------------------------------------------------------------------------
        # END RASTER LOOP
        depth_buffer = average_subpixel_image(depth_buffer,cam_data.sub_samp)
        image_buffer = average_subpixel_image(image_buffer,cam_data.sub_samp)

        return (image_buffer,depth_buffer,elem_raster_coords.shape[-1])

    @staticmethod
    def raster_loop(cam_data: CameraData,
                    elem_raster_coords: np.ndarray,
                    elem_bound_box_inds: np.ndarray,
                    elem_areas: np.ndarray,
                    field_frame_divide_z: np.ndarray
                    ) -> tuple[np.ndarray,np.ndarray,int]:

        num_elems_in_scene = elem_raster_coords.shape[-1]
        px_coord_z = [None]*num_elems_in_scene
        field_interp = [None]*num_elems_in_scene
        subpx_inds_x_in = [None]*num_elems_in_scene
        subpx_inds_y_in = [None]*num_elems_in_scene

        for ee in range(num_elems_in_scene):
            (px_coord_z[ee],
            field_interp[ee],
            subpx_inds_x_in[ee],
            subpx_inds_y_in[ee]) = Rasteriser.raster_one_element(
                                                cam_data,
                                                elem_raster_coords[:,:,ee],
                                                elem_bound_box_inds[:,ee],
                                                elem_areas[ee],
                                                field_frame_divide_z[:,ee])

        depth_buffer = 1e6*np.ones(cam_data.sub_samp*cam_data.num_pixels).T
        image_buffer = np.full(cam_data.sub_samp*cam_data.num_pixels,0.0).T

        # This loop cannot be parallelised as we need to know which element is
        # in front an push it into the image buffer
        for ee in range(num_elems_in_scene):
            #  Build a mask to replace the depth information if there is already an
            # element in front of the one we are rendering
            px_coord_z_depth_mask = (px_coord_z[ee]
                < depth_buffer[subpx_inds_y_in[ee],subpx_inds_x_in[ee]])

            # Initialise the z coord to the value in the depth buffer
            px_coord_z_masked = depth_buffer[subpx_inds_y_in[ee],subpx_inds_x_in[ee]]
            # Use the depth mask to overwrite the depth buffer values if points are in
            # front of the values in the depth buffer
            px_coord_z_masked[px_coord_z_depth_mask] = px_coord_z[ee][px_coord_z_depth_mask]

            # Push the masked values into the depth buffer
            depth_buffer[subpx_inds_y_in[ee],subpx_inds_x_in[ee]] = px_coord_z_masked

            # Mask the image buffer using the depth mask
            image_buffer_depth_masked = image_buffer[subpx_inds_y_in[ee],subpx_inds_x_in[ee]]
            image_buffer_depth_masked[px_coord_z_depth_mask] = field_interp[ee][px_coord_z_depth_mask]

            # Push the masked values into the image buffer
            image_buffer[subpx_inds_y_in[ee],subpx_inds_x_in[ee]] = image_buffer_depth_masked

        #---------------------------------------------------------------------------
        # END RASTER LOOP
        depth_buffer = average_subpixel_image(depth_buffer,cam_data.sub_samp)
        image_buffer = average_subpixel_image(image_buffer,cam_data.sub_samp)

        return (image_buffer,depth_buffer,num_elems_in_scene)

    @staticmethod
    def raster_loop_parallel(cam_data: CameraData,
                            elem_raster_coords: np.ndarray,
                            elem_bound_box_inds: np.ndarray,
                            elem_areas: np.ndarray,
                            field_frame_divide_z: np.ndarray,
                            num_para: int = 4
                            ) -> tuple[np.ndarray,np.ndarray,int]:

        with Pool(num_para) as pool:
            processes = list([])

            num_elems_in_scene = elem_raster_coords.shape[-1]
            fragments = [None]*num_elems_in_scene

            for ee in range(num_elems_in_scene):
                processes.append(pool.apply_async(
                    Rasteriser.raster_one_element,
                    args=(cam_data,
                          elem_raster_coords[:,:,ee],
                          elem_bound_box_inds[:,ee],
                          elem_areas[ee],
                          field_frame_divide_z[:,ee]
                          )))

            fragments = [pp.get() for pp in processes]

        # NOTE: fragements should be a list of tuples of numpy arrays returned
        # by the rasteriser

        # Tuple indices for variables in the fragments list
        px_co_z: int = 0
        field_int: int = 1
        inds_x: int = 2
        inds_y: int = 3

        depth_buffer = 1e6*np.ones(cam_data.sub_samp*cam_data.num_pixels).T
        image_buffer = np.full(cam_data.sub_samp*cam_data.num_pixels,0.0).T

        # This loop cannot be parallelised as we need to know which element is
        # in front an push it into the image buffer
        for ee in range(num_elems_in_scene):
            #  Build a mask to replace the depth information if there is already an
            # element in front of the one we are rendering
            px_coord_z_depth_mask = (fragments[ee][px_co_z]
                < depth_buffer[fragments[ee][inds_y],fragments[ee][inds_x]])

            # Initialise the z coord to the value in the depth buffer
            px_coord_z_masked = depth_buffer[fragments[ee][inds_y],fragments[ee][inds_x]]
            # Use the depth mask to overwrite the depth buffer values if points are in
            # front of the values in the depth buffer
            px_coord_z_masked[px_coord_z_depth_mask] = fragments[ee][px_co_z][px_coord_z_depth_mask]

            # Push the masked values into the depth buffer
            depth_buffer[fragments[ee][inds_y],fragments[ee][inds_x]] = px_coord_z_masked

            # Mask the image buffer using the depth mask
            image_buffer_depth_masked = image_buffer[fragments[ee][inds_y],fragments[ee][inds_x]]
            image_buffer_depth_masked[px_coord_z_depth_mask] = fragments[ee][field_int][px_coord_z_depth_mask]

            # Push the masked values into the image buffer
            image_buffer[fragments[ee][inds_y],fragments[ee][inds_x]] = image_buffer_depth_masked

        #---------------------------------------------------------------------------
        # END RASTER LOOP
        depth_buffer = average_subpixel_image(depth_buffer,cam_data.sub_samp)
        image_buffer = average_subpixel_image(image_buffer,cam_data.sub_samp)

        return (image_buffer,depth_buffer,num_elems_in_scene)


def edge_function(vert_a: np.ndarray,
                  vert_b: np.ndarray,
                  vert_c: np.ndarray) -> np.ndarray:
    xx: int = 0
    yy: int = 1
    edge_fun = ((vert_c[xx] - vert_a[xx]) * (vert_b[yy] - vert_a[yy])
              - (vert_c[yy] - vert_a[yy]) * (vert_b[xx] - vert_a[xx]))
    return edge_fun


def average_subpixel_image(subpx_image: np.ndarray,
                           subsample: int) -> np.ndarray:
    if subsample <= 1:
        return subpx_image

    conv_mask = np.ones((subsample,subsample))/(subsample**2)
    subpx_image_conv = convolve2d(subpx_image,conv_mask,mode='same')
    avg_image = subpx_image_conv[round(subsample/2)-1::subsample,
                                round(subsample/2)-1::subsample]
    return avg_image


#===============================================================================
# MAIN
def main() -> None:
    # 3D cylinder, mechanical, tets
    data_path = Path.home() / "pyvale" / "dev" / "lfdev" / "rastermeshbenchmarks"
    data_path = data_path / "case21_m5_out.e"

    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_keys = tuple(sim_data.node_vars.keys())
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0


    field_key = "disp_y"
    components = ("disp_x","disp_y","disp_z")
    (pv_grid,_) = pyvale.conv_simdata_to_pyvista(sim_data,
                                                components,
                                                spat_dim=3)
    pyvale.print_dimensions(sim_data)

    time_start_setup = time.perf_counter()

    pv_surf = pv_grid.extract_surface()
    faces = np.array(pv_surf.faces)

    first_elem_nodes_per_face = faces[0]
    nodes_per_face_vec = faces[0::(first_elem_nodes_per_face+1)]
    assert np.all(nodes_per_face_vec == first_elem_nodes_per_face), \
    "Not all elements in the simdata object have the same number of nodes per element"

    nodes_per_face = first_elem_nodes_per_face
    num_faces = int(faces.shape[0] / (nodes_per_face+1))

    # Reshape the faces table and slice off the first column which is just the
    # number of nodes per element and should be the same for all elements
    pv_connect = np.reshape(faces,(num_faces,nodes_per_face+1))
    pv_connect = pv_connect[:,1:].T
    pv_coords = np.array(pv_surf.points).T

    print()
    print(80*"-")
    print("EXTRACTED SURFACE MESH DATA")
    print(pv_surf)
    print()
    print("Attached array names:")
    print(pv_surf.array_names)
    print()
    print(f"{nodes_per_face=}")
    print(f"{num_faces=}")
    print()
    print("NOTE: shape needs to be coord/node_per_elem first.")
    print(f"{pv_coords.shape=}")
    print(f"{pv_connect.shape=}")
    print()
    print(f"{pv_surf[components[0]].shape=}")
    print()
    print(80*"-")
    print()

    #===========================================================================
    # Create Camera and World Parameters
    (xx,yy,zz,ww) = (0,1,2,3)

    #shape=(3,num_coords)
    coords_world = pv_coords
    coords_count = coords_world.shape[1]
    # shape=(4,num_nodes)
    coords_world= np.vstack((coords_world,np.ones((1,coords_count))))
    # shape=(nodes_per_elem,num_elems)
    connectivity = pv_connect
    elem_count = connectivity.shape[1]
    # shape=(num_nodes,num_time_steps)
    field_scalar = np.array(pv_surf[field_key])

    rot_axis: str = "x"
    phi_y_degs: float = -45
    theta_x_degs: float = -45
    psi_z_degs: float = 0.0

    phi_y_rads: float = phi_y_degs * np.pi/180.0
    theta_x_rads: float = theta_x_degs * np.pi/180.0

    # Set this to 0.0 to get some of the plate outside the FOV
    roi_pos_world = np.mean(sim_data.coords,axis=0)

    # Number of divisions (subsamples) for each pixel for anti-aliasing
    sub_samp: int = 2

    cam_type = "AV507"
    if cam_type == "AV507":
        cam_num_px = np.array([2464,2056],dtype=np.int32)
        pixel_size = np.array([3.45e-3,3.45e-3]) # in millimeters!
        focal_leng: float = 25.0

        imaging_rad: float = 150.0 # Not needed for camera data, just for cam pos below
    else:
        cam_num_px = np.array([510,260],dtype=np.int32)
        pixel_size = np.array([10.0e-3,10.0e-3])
        focal_leng: float = 25.0

        imaging_rad: float = 350.0 # Not needed for camera data, just for cam pos below

    if rot_axis == "y":
        cam_pos_world = np.array([roi_pos_world[xx] + imaging_rad*np.sin(phi_y_rads),
                                  roi_pos_world[yy],
                                  imaging_rad*np.cos(phi_y_rads)])

        cam_rot = Rotation.from_euler("zyx", [psi_z_degs, phi_y_degs, 0], degrees=True)
    elif rot_axis == "x":
        cam_pos_world = np.array([roi_pos_world[xx] ,
                                  roi_pos_world[yy] - imaging_rad*np.sin(theta_x_rads),
                                  imaging_rad*np.cos(theta_x_rads)])

        cam_rot = Rotation.from_euler("zyx", [psi_z_degs, 0, theta_x_degs], degrees=True)

    else:
        cam_pos_world = np.array([roi_pos_world[xx],
                                  roi_pos_world[yy],
                                  imaging_rad])
        cam_rot = Rotation.from_euler("zyx", [psi_z_degs, 0, 0], degrees=True)

    #---------------------------------------------------------------------------
    # RASTERISATION START
    cam_data = CameraData(num_pixels=cam_num_px,
                                pixel_size=pixel_size,
                                pos_world=cam_pos_world,
                                rot_world=cam_rot,
                                roi_center_world=roi_pos_world,
                                focal_length=focal_leng,
                                sub_samp=sub_samp)

    (elem_raster_coords,
    elem_bound_box_inds,
    elem_areas,
    field_divide_z) = Rasteriser.raster_setup(cam_data,
                                              coords_world,
                                              connectivity,
                                              field_scalar)

    # We only need to loop over elements and slice out and process the bound box
    frame = -1  # render the last frame
    field_frame_divide_z = field_divide_z[:,:,frame]

    time_end_setup = time.perf_counter()


    #---------------------------------------------------------------------------
    # RASTER LOOP START
    print()
    print(80*"=")
    print("RASTER ELEMENT LOOP START")
    print(80*"=")

    num_raster_loops: int = 1
    loop_times = []

    for ll in range(num_raster_loops):
        print(f"Running element raster loop {ll+1}")
        time_start_loop = time.perf_counter()
        (image_buffer,depth_buffer,num_elems_in_image) = Rasteriser.raster_loop(
            cam_data,
            elem_raster_coords,
            elem_bound_box_inds,
            elem_areas,
            field_frame_divide_z)
        time_end_loop = time.perf_counter()
        loop_times.append(time_end_loop - time_start_loop)

    print()
    print(80*"=")
    print("RASTER LOOP END")
    print(80*"=")
    print()
    print(80*"=")
    print("PERFORMANCE TIMERS")
    print(f"Total elements:    {connectivity.shape[1]}")
    print(f"Elements in image: {num_elems_in_image}")
    print()
    print(f"Setup time = {time_end_setup-time_start_setup} seconds")
    for ll in range(num_raster_loops):
        print(f"Raster loop {ll+1} time = {loop_times[ll]} seconds")
    print(80*"=")

    #===========================================================================
    # REGRESSION TESTING FOR REFACTOR
    save_regression_test_arrays = False
    check_regression_test_arrays = False

    test_path = Path.cwd() / "tests" / "regression"
    test_image = test_path / "image_buffer.npy"
    test_depth = test_path / "depth_buffer.npy"

    if save_regression_test_arrays:
        np.save(test_image,image_buffer)
        np.save(test_depth,depth_buffer)

    if check_regression_test_arrays:
        check_image = np.load(test_image)
        check_depth = np.load(test_depth)

        print(80*"/")
        print("REGRESSION TEST:")
        print(f"Image buffer check: {np.allclose(image_buffer,check_image)}")
        print(f"Depth buffer check: {np.allclose(depth_buffer,check_depth)}")
        print(80*"/")

        assert np.allclose(image_buffer,check_image), "Image buffer regression test FAILURE."
        assert np.allclose(depth_buffer,check_depth), "Depth buffer regression test FAILURE."

    #===========================================================================
    # PLOTTING
    plot_on = True
    depth_to_plot = np.copy(depth_buffer)
    depth_to_plot[depth_buffer > 10*cam_data.image_dist] = np.NaN
    image_to_plot = np.copy(image_buffer)
    image_to_plot[depth_buffer > 10*cam_data.image_dist] = np.NaN
    if plot_on:
        plot_opts = pyvale.PlotOptsGeneral()

        (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_square,
                                layout='constrained')
        fig.set_dpi(plot_opts.resolution)
        cset = plt.imshow(depth_to_plot,
                        cmap=plt.get_cmap(plot_opts.cmap_seq))
                        #origin='lower')
        ax.set_aspect('equal','box')
        fig.colorbar(cset)
        ax.set_title("Depth buffer",fontsize=plot_opts.font_head_size)
        ax.set_xlabel(r"x ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
        ax.set_ylabel(r"y ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

        (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_square,
                                layout='constrained')
        fig.set_dpi(plot_opts.resolution)
        cset = plt.imshow(image_to_plot,
                        cmap=plt.get_cmap(plot_opts.cmap_seq))
                        #origin='lower')
        ax.set_aspect('equal','box')
        fig.colorbar(cset)
        ax.set_title("Field Image",fontsize=plot_opts.font_head_size)
        ax.set_xlabel(r"x ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
        ax.set_ylabel(r"y ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

        plt.show()

if __name__ == "__main__":
    main()
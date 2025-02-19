"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from multiprocessing.pool import Pool
import numpy as np
from scipy.signal import convolve2d
from pyvale.core.cameradata import CameraData
import numba
from pyvale.core.cameratools import CameraTools

class RasteriserNP:
    @staticmethod
    def world_to_raster_coords(cam_data: CameraData,
                               coords_world: np.ndarray) -> np.ndarray:
        # coords_world.shape=(num_nodes,coord[X,Y,Z,W])

        # Project onto camera coords
        # coords_raster.shape=(num_nodes,coord[X,Y,Z,W])
        coords_raster = np.matmul(coords_world,cam_data.world_to_cam_mat.T)

        # NOTE: w is not 1 when the matrix is a perspective projection! It is only 1
        # here when we have an affine transformation
        coords_raster[:,0] = coords_raster[:,0] / coords_raster[:,3]
        coords_raster[:,1] = coords_raster[:,1] / coords_raster[:,3]
        coords_raster[:,2] = coords_raster[:,2] / coords_raster[:,3]

        # Coords Image: Perspective divide
        coords_raster[:,0] = (cam_data.image_dist * coords_raster[:,0]
                            / -coords_raster[:,2])
        coords_raster[:,1] = (cam_data.image_dist * coords_raster[:,1]
                            / -coords_raster[:,2])

        # Coords NDC: Convert to normalised device coords in the range [-1,1]
        coords_raster[:,0] = 2*coords_raster[:,0] / cam_data.image_dims[0]
        coords_raster[:,1] = 2*coords_raster[:,1] / cam_data.image_dims[1]

        # Coords Raster: Covert to pixel (raster) coords
        # Shape = ([X,Y,Z],num_nodes)
        coords_raster[:,0] = (coords_raster[:,0] + 1)/2 * cam_data.pixels_num[0]
        coords_raster[:,1] = (1-coords_raster[:,1])/2 * cam_data.pixels_num[1]
        coords_raster[:,2] = -coords_raster[:,2]

        return coords_raster

    @staticmethod
    def create_transformed_elem_arrays(cam_data: CameraData,
                                       coords_world: np.ndarray,
                                       connectivity: np.ndarray,
                                       field_array: np.ndarray,
                                       ) -> tuple[np.ndarray,np.ndarray]:

        # Convert world coords of all elements in the scene
        # shape=(num_nodes,coord[x,y,z,w])
        coords_raster = RasteriserNP.world_to_raster_coords(cam_data,coords_world)

        # Convert to perspective correct hyperbolic interpolation for z interp
        # shape=(num_nodes,coord[x,y,z,w])
        coords_raster[2,:] = 1/coords_raster[2,:]

        # shape=(num_elems,node_per_elem,coord[x,y,z,w])
        elem_raster_coords = np.ascontiguousarray(coords_raster[connectivity,:])

        # NOTE: we have already inverted the raster z coordinate above so to divide
        # by z here we need to multiply
        # shape=(n_nodes,num_time_steps)
        field_divide_z = (field_array.T*coords_raster[:,2]).T
        # shape=(num_elems,nodes_per_elem,num_time_steps)
        field_divide_z = np.ascontiguousarray(field_divide_z[connectivity,:])

        return (elem_raster_coords,field_divide_z)


    @staticmethod
    def back_face_removal_mask(cam_data: CameraData,
                            coords_world: np.ndarray,
                            connect: np.ndarray
                            ) -> np.ndarray:
        coords_cam = np.matmul(coords_world,cam_data.world_to_cam_mat.T)

        # shape=(num_elems,nodes_per_elem,coord[x,y,z,w])
        elem_cam_coords = coords_cam[connect,:]

        # Calculate the normal vectors for all of the elements, remove the w coord
        # shape=(num_elems,coord[x,y,z])
        elem_cam_edge0 = elem_cam_coords[:,1,:-1] - elem_cam_coords[:,0,:-1]
        elem_cam_edge1 = elem_cam_coords[:,2,:-1] - elem_cam_coords[:,0,:-1]
        elem_cam_normals = np.cross(elem_cam_edge0,elem_cam_edge1,
                                    axisa=1,axisb=1).T
        elem_cam_normals = elem_cam_normals / np.linalg.norm(elem_cam_normals,axis=0)

        cam_normal = np.array([0,0,1])
        # shape=(num_elems,)
        proj_elem_to_cam = np.dot(cam_normal,elem_cam_normals)

        # NOTE this should be a numerical precision tolerance (epsilon)
        back_face_mask = proj_elem_to_cam > 1e-6

        return back_face_mask

    @staticmethod
    def crop_and_bound_elements(cam_data: CameraData,
                                elem_raster_coords: np.ndarray,
                                ) -> tuple[np.ndarray,np.ndarray]:

        #shape=(num_elems,coord[x,y,z,w])
        elem_raster_coord_min = np.min(elem_raster_coords,axis=1)
        elem_raster_coord_max = np.max(elem_raster_coords,axis=1)

        # Check that min/max nodes are within the 4 edges of the camera image
        #shape=(4_edges_to_check,num_elems)
        crop_mask = np.zeros([elem_raster_coords.shape[0],4],dtype=np.int8)
        crop_mask[elem_raster_coord_min[:,0] <= (cam_data.pixels_num[0]-1), 0] = 1
        crop_mask[elem_raster_coord_min[:,1] <= (cam_data.pixels_num[1]-1), 1] = 1
        crop_mask[elem_raster_coord_max[:,0] >= 0, 2] = 1
        crop_mask[elem_raster_coord_max[:,1] >= 0, 3] = 1
        crop_mask = np.sum(crop_mask,axis=1) == 4

        # Get only the elements that are within the FOV
        # Mask the elem coords and the max and min elem coords for processing
        elem_raster_coord_min = elem_raster_coord_min[crop_mask,:]
        elem_raster_coord_max = elem_raster_coord_max[crop_mask,:]
        num_elems_in_image = elem_raster_coord_min.shape[0]


        # Find the indices of the bounding box that each element lies within on
        # the image, bounded by the upper and lower edges of the image
        elem_bound_boxes_inds = np.zeros([num_elems_in_image,4],dtype=np.int32)
        elem_bound_boxes_inds[:,0] = RasteriserNP.elem_bound_box_low(
                                            elem_raster_coord_min[:,0])
        elem_bound_boxes_inds[:,1] = RasteriserNP.elem_bound_box_high(
                                            elem_raster_coord_max[:,0],
                                            cam_data.pixels_num[0]-1)
        elem_bound_boxes_inds[:,2] = RasteriserNP.elem_bound_box_low(
                                            elem_raster_coord_min[:,1])
        elem_bound_boxes_inds[:,3] = RasteriserNP.elem_bound_box_high(
                                            elem_raster_coord_max[:,1],
                                            cam_data.pixels_num[1]-1)

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


        # elem_raster_coords.shape=(num_elems,nodes_per_elem,coord[x,y,z,w])
        # field_divide_z.shape=(num_elems,nodes_per_elem,num_time_steps)
        (elem_raster_coords,field_divide_z) = \
            RasteriserNP.create_transformed_elem_arrays(cam_data,
                                                        coords_world,
                                                        connectivity,
                                                        field_data)

        #-----------------------------------------------------------------------
        # BACKFACE REMOVAL
        if cam_data.back_face_removal:
            # shape=(num_elems,)
            back_face_mask = RasteriserNP.back_face_removal_mask(cam_data,
                                                                 coords_world,
                                                                 connectivity)
            # shape=(num_elems_in_scene,nodes_per_elem,coord[X,Y,Z,W])
            elem_raster_coords = np.ascontiguousarray(elem_raster_coords[back_face_mask,:,:])
            # shape=(num_elems_in_scene,nodes_per_elem,num_time_steps)
            field_divide_z = np.ascontiguousarray(field_divide_z[back_face_mask,:,:])

        #-----------------------------------------------------------------------
        # CROPPING & BOUNDING BOX OPERATIONS
        (crop_mask,elem_bound_box_inds) = RasteriserNP.crop_and_bound_elements(
            cam_data,
            elem_raster_coords
        )
        # Apply crop using mask and remove w coord
        # shape=(num_elems_in_scene,nodes_per_elem,coord[x,y,z])
        elem_raster_coords = np.ascontiguousarray(elem_raster_coords[crop_mask,:,:-1])
        # shape=(num_elems_in_scene,nodes_per_elem,num_time_steps)
        field_divide_z = np.ascontiguousarray(field_divide_z[crop_mask,:,:])

        #-----------------------------------------------------------------------
        # ELEMENT AREAS FOR INTERPOLATION
        elem_areas = edge_function_slice(elem_raster_coords[:,0,:],
                                         elem_raster_coords[:,1,:],
                                         elem_raster_coords[:,2,:])

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

        # Create the subpixel coords inside the bounding box to test with the
        # edge function. Use the pixel indices of the bounding box.
        bound_subpx_x = np.arange(elem_bound_box_inds[0],
                                  elem_bound_box_inds[1],
                                  1/cam_data.sub_samp) + 1/(2*cam_data.sub_samp)
        bound_subpx_y = np.arange(elem_bound_box_inds[2],
                                  elem_bound_box_inds[3],
                                  1/cam_data.sub_samp) + 1/(2*cam_data.sub_samp)
        (bound_subpx_grid_x,bound_subpx_grid_y) = np.meshgrid(bound_subpx_x,
                                                              bound_subpx_y)
        bound_coords_grid_shape = bound_subpx_grid_x.shape
        # shape=(coord[x,y],num_subpx_in_box)
        bound_subpx_coords_flat = np.vstack((bound_subpx_grid_x.flatten(),
                                             bound_subpx_grid_y.flatten()))

        # Create the subpixel indices for buffer slicing later
        subpx_inds_x = np.arange(cam_data.sub_samp*elem_bound_box_inds[0],
                                 cam_data.sub_samp*elem_bound_box_inds[1])
        subpx_inds_y = np.arange(cam_data.sub_samp*elem_bound_box_inds[2],
                                 cam_data.sub_samp*elem_bound_box_inds[3])
        (subpx_inds_grid_x,subpx_inds_grid_y) = np.meshgrid(subpx_inds_x,
                                                            subpx_inds_y)



        # We compute the edge function for all pixels in the box to determine if the
        # pixel is inside the element or not
        # NOTE: first axis of element_raster_coords is the node/vertex num.
        # shape=(num_elems_in_bound,nodes_per_elem)
        edge = np.zeros((3,bound_subpx_coords_flat.shape[1]),dtype=np.float64)
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
        # NOTE: second index on raster coords is Z
        px_coord_z = 1/(elem_raster_coords[0,2] * interp_weights[0,:]
                      + elem_raster_coords[1,2] * interp_weights[1,:]
                      + elem_raster_coords[2,2] * interp_weights[2,:])

        field_interp = ((field_divide_z[0] * interp_weights[0,:]
                       + field_divide_z[1] * interp_weights[1,:]
                       + field_divide_z[2] * interp_weights[2,:])
                       * px_coord_z)

        return (px_coord_z,
                field_interp,
                subpx_inds_grid_x[edge_mask_grid],
                subpx_inds_grid_y[edge_mask_grid])

    @staticmethod
    def raster_loop(cam_data: CameraData,
                    elem_raster_coords: np.ndarray,
                    elem_bound_box_inds: np.ndarray,
                    elem_areas: np.ndarray,
                    field_frame_divide_z: np.ndarray
                    ) -> tuple[np.ndarray,np.ndarray,int]:

        # NOTE: this version cannot be run in parallel as the depth buffer is
        # filled on the fly as we process each element. This version will be
        # more memory efficient as we do not need to store each rastered
        # element fragment in memory.

        depth_buffer = 1e6*np.ones(cam_data.sub_samp*cam_data.pixels_num).T
        image_buffer = np.full(cam_data.sub_samp*cam_data.pixels_num,0.0).T

        for ee in range(elem_raster_coords.shape[0]):
            (px_coord_z,
            field_interp,
            subpx_inds_x_in,
            subpx_inds_y_in) = RasteriserNP.raster_one_element(
                                                cam_data,
                                                elem_raster_coords[ee,:,:],
                                                elem_bound_box_inds[ee,:],
                                                elem_areas[ee],
                                                field_frame_divide_z[ee,:])


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
        depth_buffer = CameraTools.average_subpixel_image(depth_buffer,cam_data.sub_samp)
        image_buffer = CameraTools.average_subpixel_image(image_buffer,cam_data.sub_samp)

        return (image_buffer,depth_buffer,elem_raster_coords.shape[0])


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
                    RasteriserNP.raster_one_element,
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

        depth_buffer = 1e6*np.ones(cam_data.sub_samp*cam_data.pixels_num).T
        image_buffer = np.full(cam_data.sub_samp*cam_data.pixels_num,0.0).T

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
        depth_buffer = CameraTools.average_subpixel_image(depth_buffer,cam_data.sub_samp)
        image_buffer = CameraTools.average_subpixel_image(image_buffer,cam_data.sub_samp)

        return (image_buffer,depth_buffer,num_elems_in_scene)


@numba.jit(nopython=True)
def edge_function(vert_a: np.ndarray,
                  vert_b: np.ndarray,
                  vert_c: np.ndarray) -> np.ndarray:

    return  ((vert_c[0] - vert_a[0]) * (vert_b[1] - vert_a[1])
              - (vert_c[1] - vert_a[1]) * (vert_b[0] - vert_a[0]))

@numba.jit(nopython=True)
def edge_function_slice(vert_a: np.ndarray,
                        vert_b: np.ndarray,
                       vert_c: np.ndarray) -> np.ndarray:

    return  ((vert_c[:,0] - vert_a[:,0]) * (vert_b[:,1] - vert_a[:,1])
              - (vert_c[:,1] - vert_a[:,1]) * (vert_b[:,0] - vert_a[:,0]))



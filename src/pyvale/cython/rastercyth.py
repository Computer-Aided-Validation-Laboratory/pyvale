# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: this module is a feature under developement.
"""

import numpy as np
import cython
from cython.parallel import prange, parallel, threadid
from cython.cimports.libc.math import floor, ceil

from pyvale.rendermesh import RenderMeshData
from pyvale.cameradata import CameraData

# NOTE: This module is a feature under developement.

@cython.nogil
@cython.cfunc # python+C or cython.cfunc for C only
@cython.boundscheck(False) # Turn off array bounds checking
@cython.wraparound(False)  # Turn off negative indexing
@cython.cdivision(True)    # Turn off divide by zero check
@cython.inline
@cython.exceptval(check=False)
def range_len_double(start: cython.double,
                     stop: cython.double,
                     step: cython.double) -> cython.size_t:
    return int(ceil((stop - start) / step))


@cython.nogil
@cython.cfunc # python+C or cython.cfunc for C only
@cython.boundscheck(False) # Turn off array bounds checking
@cython.wraparound(False)  # Turn off negative indexing
@cython.cdivision(True)    # Turn off divide by zero check
@cython.inline
@cython.exceptval(check=False)
def vec_range_int(start: cython.int,
                  stop: cython.int,
                  step: cython.int,
                  vec_buffer: cython.long[:]) -> cython.long[:]:

    num_vals: cython.size_t = int(ceil((stop - start) / step))

    vec_buffer[0] = start
    ii: cython.size_t
    for ii in range(1,num_vals):
        vec_buffer[ii] = vec_buffer[ii-1] + step

    return vec_buffer[0:num_vals]


@cython.nogil
@cython.cfunc # python+C or cython.cfunc for C only
@cython.boundscheck(False) # Turn off array bounds checking
@cython.wraparound(False)  # Turn off negative indexing
@cython.cdivision(True)    # Turn off divide by zero check
@cython.inline
@cython.exceptval(check=False)
def vec_max_double(vals: cython.double[:]) -> cython.double:

    num_vals: cython.size_t = vals.shape[0]

    ii: cython.size_t = 0
    max_val: cython.double = vals[ii]

    for ii in range(1,num_vals):
        if vals[ii] > max_val:
            max_val = vals[ii]

    return max_val


@cython.nogil
@cython.cfunc # python+C or cython.cfunc for C only
@cython.boundscheck(False) # Turn off array bounds checking
@cython.wraparound(False)  # Turn off negative indexing
@cython.cdivision(True)    # Turn off divide by zero check
@cython.inline
@cython.exceptval(check=False)
def vec_min_double(vals: cython.double[:]) -> cython.double:

    num_vals: cython.size_t = vals.shape[0]

    ii: cython.size_t = 0
    min_val: cython.double = vals[ii]

    for ii in range(1,num_vals):
        if vals[ii] < min_val:
            min_val = vals[ii]

    return min_val


@cython.nogil
@cython.cfunc # python+C or cython.cfunc for C only
@cython.boundscheck(False) # Turn off array bounds checking
@cython.wraparound(False)  # Turn off negative indexing
@cython.cdivision(True)    # Turn off divide by zero check
@cython.inline
@cython.exceptval(check=False)
def vec_dot_double(vec0: cython.double[:], vec1: cython.double[:]
                   ) -> cython.double:
    vec0_len: cython.size_t = vec0.shape[0]
    vec1_len: cython.size_t = vec1.shape[0]
    if vec0_len != vec1_len:
        return 0.0

    ii: cython.size_t = 0
    dot: cython.double = 0.0
    for ii in range(vec0_len):
        dot += vec0[ii]*vec1[ii]

    return dot


@cython.nogil
@cython.cfunc # python+C or cython.cfunc for C only
@cython.boundscheck(False) # Turn off array bounds checking
@cython.wraparound(False)  # Turn off negative indexing
@cython.cdivision(True)    # Turn off divide by zero check
@cython.inline
@cython.exceptval(check=False)
def bound_index_min(min_val: cython.double) -> cython.int:
    min_ind: cython.int = int(floor(min_val))
    if min_ind < 0:
        min_ind = 0
    return min_ind


@cython.nogil
@cython.cfunc # python+C or cython.cfunc for C only
@cython.boundscheck(False) # Turn off array bounds checking
@cython.wraparound(False)  # Turn off negative indexing
@cython.cdivision(True)    # Turn off divide by zero check
@cython.inline
@cython.exceptval(check=False)
def bound_index_max(max_val: cython.double,
                    num_pixels: cython.int) -> cython.int:
    max_ind: cython.int = int(ceil(max_val))
    if max_ind > (num_pixels-1):
        max_ind = (num_pixels-1)
    return max_ind


@cython.nogil
@cython.cfunc # python+C or cython.cfunc for C only
@cython.boundscheck(False) # Turn off array bounds checking
@cython.wraparound(False)  # Turn off negative indexing
@cython.cdivision(True)    # Turn off divide by zero check
@cython.inline
@cython.exceptval(check=False)
def mult_mat44_by_vec3(mat44: cython.double[:,:], vec3_in: cython.double[:],
                       vec3_out: cython.double[:]) -> cython.double[:]:

    vec3_out[0] = (mat44[0,0]*vec3_in[0]
                      + mat44[0,1]*vec3_in[1]
                      + mat44[0,2]*vec3_in[2]
                      + mat44[0,3])
    vec3_out[1] = (mat44[1,0]*vec3_in[0]
                      + mat44[1,1]*vec3_in[1]
                      + mat44[1,2]*vec3_in[2]
                      + mat44[1,3])
    vec3_out[2] = (mat44[2,0]*vec3_in[0]
                      + mat44[2,1]*vec3_in[1]
                      + mat44[2,2]*vec3_in[2]
                      + mat44[2,3])
    vec3_out[3] = (mat44[3,0]*vec3_in[0]
                      + mat44[3,1]*vec3_in[1]
                      + mat44[3,2]*vec3_in[2]
                      + mat44[3,3])

    return vec3_out


@cython.nogil
@cython.cfunc # python+C or cython.cfunc for C only
@cython.boundscheck(False) # Turn off array bounds checking
@cython.wraparound(False)  # Turn off negative indexing
@cython.cdivision(True)    # Turn off divide by zero check
@cython.exceptval(check=False)
def world_to_raster_coords(coords_world: cython.double[:],
                           world_to_cam_mat: cython.double[:,:],
                           image_dist: cython.double,
                           image_dims: cython.double[:],
                           num_pixels: cython.int[:],
                           coords_raster: cython.double[:]
                           ) -> cython.double[:]:
    xx: cython.size_t = 0
    yy: cython.size_t = 1
    zz: cython.size_t = 2
    ww: cython.size_t = 3

    coords_raster = mult_mat44_by_vec3(world_to_cam_mat,
                                       coords_world,
                                       coords_raster)

    coords_raster[xx] = coords_raster[xx] / coords_raster[ww]
    coords_raster[yy] = coords_raster[yy] / coords_raster[ww]
    coords_raster[zz] = coords_raster[zz] / coords_raster[ww]

    coords_raster[xx] = (image_dist * coords_raster[xx]
                        / -coords_raster[zz])
    coords_raster[yy] = (image_dist * coords_raster[yy]
                        / -coords_raster[zz])

    coords_raster[xx] = 2*coords_raster[xx] / image_dims[xx]
    coords_raster[yy] = 2*coords_raster[yy] / image_dims[yy]

    coords_raster[xx] = (coords_raster[xx] + 1)/2 * num_pixels[xx]
    coords_raster[yy] = (1-coords_raster[yy])/2 * num_pixels[yy]
    coords_raster[zz] = -coords_raster[zz]

    return coords_raster

@cython.cfunc
@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.inline
@cython.exceptval(check=False)
def edge_function(vert_0: cython.double[:],
                  vert_1: cython.double[:],
                  vert_2: cython.double[:]) -> cython.double:
    edge_fun: cython.double = (
        (vert_2[0] - vert_0[0]) * (vert_1[1] - vert_0[1])
        - (vert_2[1] - vert_0[1]) * (vert_1[0] - vert_0[0]))
    return edge_fun


@cython.cfunc
@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.inline
@cython.exceptval(check=False)
def edge_function_pt(vert_0: cython.double[:],
                     vert_1: cython.double[:],
                     vert_2_x: cython.double,
                     vert_2_y: cython.double) -> cython.double:
    edge_fun: cython.double = (
        (vert_2_x - vert_0[0]) * (vert_1[1] - vert_0[1])
        - (vert_2_y - vert_0[1]) * (vert_1[0] - vert_0[0]))
    return edge_fun


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def average_image(image_subpx: cython.double[:,:],
                  sub_samp: cython.int,
                  ) -> cython.double[:,:]:

    if sub_samp <= 1:
        return np.asarray(image_subpx[:,:])

    px_num_y: cython.size_t = int(ceil(image_subpx.shape[0]/sub_samp))
    px_num_x: cython.size_t = int(ceil(image_subpx.shape[1]/sub_samp))

    image_buff_avg_np = np.full((px_num_y,px_num_x),0.0,dtype=np.float64)
    image_buff_avg: cython.double[:,:] = image_buff_avg_np

    num_subpx_y: cython.size_t = image_subpx.shape[0]
    num_subpx_x: cython.size_t = image_subpx.shape[1]
    subpx_per_px: cython.double = float(sub_samp*sub_samp)
    ss_size: cython.size_t = sub_samp

    num_px_y: cython.size_t = int(num_subpx_y/sub_samp)
    num_px_x: cython.size_t = int(num_subpx_x/sub_samp)

    px_sum: cython.double = 0.0

    ix: cython.size_t = 0
    iy: cython.size_t = 0
    sx: cython.size_t = 0
    sy: cython.size_t = 0

    for iy in range(num_px_y):
        for ix in range(num_px_x):
            px_sum = 0.0
            for sy in range(ss_size):
                for sx in range(ss_size):
                    px_sum += image_subpx[ss_size*iy+sy,ss_size*ix+sx]

            image_buff_avg[iy,ix] = px_sum / subpx_per_px

    return image_buff_avg


@cython.nogil
@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
def _average_image(image_buff_subpx_in: cython.double[:,:],
                   sub_samp: cython.int,
                   image_buff_avg_out: cython.double[:,:]
                   ) -> cython.int:

    num_subpx_y: cython.size_t = image_buff_subpx_in.shape[0]
    num_subpx_x: cython.size_t = image_buff_subpx_in.shape[1]
    subpx_per_px: cython.double = float(sub_samp*sub_samp)
    ss_size: cython.size_t = sub_samp

    num_px_y: cython.size_t = int(num_subpx_y/sub_samp)
    num_px_x: cython.size_t = int(num_subpx_x/sub_samp)

    px_sum: cython.double = 0.0

    ix: cython.size_t = 0
    iy: cython.size_t = 0
    sx: cython.size_t = 0
    sy: cython.size_t = 0

    for iy in range(num_px_y):
        for ix in range(num_px_x):
            px_sum = 0.0
            for sy in range(ss_size):
                for sx in range(ss_size):
                    px_sum += image_buff_subpx_in[ss_size*iy+sy,ss_size*ix+sx]

            image_buff_avg_out[iy,ix] = px_sum / subpx_per_px

    return 0


#///////////////////////////////////////////////////////////////////////////////
@cython.ccall # python+C or cython.cfunc for C only
@cython.boundscheck(False) # Turn off array bounds checking
@cython.wraparound(False)  # Turn off negative indexing
@cython.cdivision(True)    # Turn off divide by zero check
def raster_frame(coords: cython.double[:,:],
                 connect: cython.size_t[:,:],
                 fields_to_render: cython.double[:,:],
                 cam_data: CameraData,
                 ) -> tuple[np.ndarray,np.ndarray,int]:


    world_to_cam_mat: cython.double[:,:] = cam_data.world_to_cam_mat
    pixels_num: cython.int[:] = cam_data.pixels_num
    image_dims: cython.double[:] = cam_data.image_dims
    image_dist: cython.double = cam_data.image_dist
    sub_samp: cython.int = cam_data.sub_samp

    nodes_per_elem: cython.size_t = connect.shape[1]
    fields_num: cython.size_t = fields_to_render.shape[1]
    sub_pix_x: cython.int = pixels_num[0]*sub_samp
    sub_pix_y: cython.int = pixels_num[1]*sub_samp

    #---------------------------------------------------------------------------
    # Final image buffer memory allocation
    image_buff_avg_np = np.full((pixels_num[1],pixels_num[0],fields_num),0.0,dtype=np.float64)
    image_buff_avg: cython.double[:,:,:] = image_buff_avg_np

    depth_buff_avg_np = np.full((pixels_num[1],pixels_num[0]),0.0,dtype=np.float64)
    depth_buff_avg: cython.double[:,:] = depth_buff_avg_np

    #---------------------------------------------------------------------------
    # Per-thread scratch memory allocations
    depth_buffer_np = np.full((sub_pix_y,sub_pix_x),1.0e6,dtype=np.float64)
    depth_buff_subpx: cython.double[:,:] = depth_buffer_np

    image_buffer_np = np.full((sub_pix_y,sub_pix_x,fields_num),0.0,dtype=np.float64)
    image_buff_subpx: cython.double[:,:,:] = image_buffer_np

    # shape=(nodes_per_elem, coord[X,Y,Z,W])
    nodes_raster_np = np.empty((nodes_per_elem,4),dtype=np.float64)
    nodes_raster_buff: cython.double[:,:] = nodes_raster_np

    field_raster_np = np.empty((nodes_per_elem,),dtype=np.float64)
    field_raster_buff: cython.double[:] = field_raster_np

    px_coord_np = np.zeros((nodes_per_elem,),np.float64)
    px_coord_buff: cython.double[:] = px_coord_np

    weights_np = np.zeros((nodes_per_elem,),np.float64)
    weights_buff: cython.double[:] = weights_np
    #---------------------------------------------------------------------------

    elems_in_image: cython.size_t = _raster_frame(coords[:,:],
                                                connect[:,:],
                                                fields_to_render[:,:],
                                                world_to_cam_mat[:,:],
                                                pixels_num[:],
                                                image_dims[:],
                                                image_dist,
                                                sub_samp,
                                                image_buff_avg[:,:,:],
                                                depth_buff_avg[:,:],
                                                image_buff_subpx[:,:,:],
                                                depth_buff_subpx[:,:],
                                                nodes_raster_buff[:,:],
                                                field_raster_buff[:],
                                                px_coord_buff[:],
                                                weights_buff[:])

    return (image_buff_avg_np,depth_buff_avg_np,elems_in_image)


#///////////////////////////////////////////////////////////////////////////////
#@cython.nogil
@cython.cfunc # python+C or cython.cfunc for C only
@cython.boundscheck(False) # Turn off array bounds checking
@cython.wraparound(False)  # Turn off negative indexing
@cython.cdivision(True)    # Turn off divide by zero check
@cython.exceptval(check=False) # Turn off exceptions
def _raster_frame(coords: cython.double[:,:],
                 connect: cython.size_t[:,:],
                 fields_to_render: cython.double[:,:],
                 world_to_cam_mat: cython.double[:,:],
                 num_pixels: cython.int[:],
                 image_dims: cython.double[:],
                 image_dist: cython.double,
                 sub_samp: cython.int,
                 # From here these are memory buffers that will be written into
                 image_buff_avg: cython.double[:,:,:],
                 depth_buff_avg: cython.double[:,:],
                 image_buff_subpx: cython.double[:,:,:],
                 depth_buff_subpx: cython.double[:,:],
                 nodes_raster_buff: cython.double[:,:],
                 field_raster_buff: cython.double[:],
                 px_coord_buff: cython.double[:],
                 weights_buff: cython.double[:],
                 ) -> cython.size_t:
    """Rasters a single frame and all associated fields into the image and depth
    buffer provided as inputs to the function. This is a pure cython function
    with the GIL released for parallelisation. All fields (textures) are
    rendered in a sub-loop so that the depth buffer and inside/outside test is
    only performed once for all fields to be rendered.

    Parameters
    ----------
    coords : cython.double[:,:]
        Input. shape=(num_nodes,coords[x,y,z,w])
    connect : cython.size_t[:,:]
        Input. shape=(num_elems,nodes_per_elem)
    fields_to_render : cython.double[:,:]
        Input. shape=(num_nodes,num_fields)
    world_to_cam_mat : cython.double[:,:]
        Input. Homogeneous coordinate transformation matrix from world to camera
         coordinates. shape=(4,4).
    num_pixels : cython.int[:]
        Input. shape=(2 [num_px_x,num_px_y],)
    image_dims : cython.double[:]
        Input. shape=(2 [fov_size_x,fov_size_y],)
    image_dist : cython.double
        Input.
    sub_samp : cython.int
        Number of subsamples per pixel for anti-aliasing.
    image_buff_avg : cython.double[:,:,:]
        Output buffer. shape=(num_px_y,num_px_x,num_fields)
    depth_buff_avg : cython.double[:,:]
        Output buffer. shape=(num_px_y,num_px_x)
    image_buff_subpx : cython.double[:,:,:]
        Processing buffer (output). shape=(num_subpx_y,num_subpx_x,num_fields)
    depth_buff_subpx : cython.double[:,:]
        Processing buffer (output). shape=(num_subpx_y,num_subpx_x)
    nodes_raster_buff : cython.double[:,:]
        Processing buffer (output). shape=(nodes_per_elem, 4 coord[x,y,z,w])
    field_raster_buff : cython.double[:]
        Processing buffer (output). shape=(nodes_per_elem,)
    px_coord_buff : cython.double[:]
        Processing buffer (output). shape=(nodes_per_elem,)
    weights_buff : cython.double[:]
        Processing buffer (output). shape=(nodes_per_elem,)

    Returns
    -------
    cython.size_t
        Number of rendered elements after backface culling and cropping.
    """

    xx: cython.size_t = 0
    yy: cython.size_t = 1
    zz: cython.size_t = 2

    elem_count: cython.size_t = connect.shape[0]
    nodes_per_elem: cython.size_t = connect.shape[1]
    fields_num: cython.size_t = fields_to_render.shape[1]

    # tolerance for floating point zero dot product
    tol: cython.double = 1e-12

    #elem_count: cython.size_t = 1
    elems_in_image: cython.size_t = 0

    ee: cython.size_t = 0
    nn: cython.size_t = 0
    ii: cython.size_t = 0
    jj: cython.size_t = 0
    ww: cython.size_t = 0
    ff: cython.size_t = 0

    for ee in range(elem_count):

        for nn in range(nodes_per_elem):
            # shape=(nodes_per_elem, coord[X,Y,Z,W])
            nodes_raster_buff[nn,:] = world_to_raster_coords(coords[connect[ee,nn],:],
                                                        world_to_cam_mat,
                                                        image_dist,
                                                        image_dims,
                                                        num_pixels,
                                                        nodes_raster_buff[nn,:])


        elem_area: cython.double = edge_function(nodes_raster_buff[0,:],
                                                 nodes_raster_buff[1,:],
                                                 nodes_raster_buff[2,:])

        if elem_area < -tol: # Backface culling
            continue

        print(f"{nodes_raster_buff[0,0]},{nodes_raster_buff[0,1]},{nodes_raster_buff[0,2]}")
        print(f"{nodes_raster_buff[1,0]},{nodes_raster_buff[1,1]},{nodes_raster_buff[1,2]}")
        print(f"{nodes_raster_buff[2,0]},{nodes_raster_buff[2,1]},{nodes_raster_buff[2,2]}")
        print(f"{ee} ELEM AREA : {elem_area}")
        print()

        x_min: cython.double = vec_min_double(nodes_raster_buff[:,xx])
        x_max: cython.double = vec_max_double(nodes_raster_buff[:,xx])

        if ((x_min > num_pixels[xx]-1) or (x_max < 0)): # x crop
            continue

        y_min: cython.double = vec_min_double(nodes_raster_buff[:,yy])
        y_max: cython.double = vec_max_double(nodes_raster_buff[:,yy])

        if ((y_min > num_pixels[yy]-1) or (y_max < 0)): # y crop
            continue

        elems_in_image += 1

        xi_min: cython.size_t = bound_index_min(x_min)
        xi_max: cython.size_t = bound_index_max(x_max,num_pixels[xx])
        yi_min: cython.size_t = bound_index_min(y_min)
        yi_max: cython.size_t = bound_index_max(y_max,num_pixels[yy])

        for nn in range(nodes_per_elem):
            nodes_raster_buff[nn,zz] = 1/nodes_raster_buff[nn,zz]

        num_bound_x: cython.size_t = range_len_double(float(xi_min),
                                                      float(xi_max),
                                                      1.0/float(sub_samp))
        num_bound_y: cython.size_t = range_len_double(float(yi_min),
                                                      float(yi_max),
                                                      1.0/float(sub_samp))

        bound_coord_x: cython.double = float(xi_min) + 1.0/(2.0*float(sub_samp))
        bound_coord_y: cython.double = float(yi_min) + 1.0/(2.0*float(sub_samp))
        coord_step: cython.double = 1.0/float(sub_samp)
        bound_ind_x: cython.size_t = sub_samp*xi_min
        bound_ind_y: cython.size_t = sub_samp*yi_min


        for jj in range(num_bound_y):

            bound_coord_x = float(xi_min) + 1.0/(2.0*float(sub_samp))
            bound_ind_x: cython.size_t = sub_samp*xi_min

            for ii in range(num_bound_x):

                px_coord_buff[xx] = bound_coord_x
                px_coord_buff[yy] = bound_coord_y

                # Check the edge functions for each edge one at a time, as soon
                # as one is outside we don't need to do anymore work
                weights_buff[0] = edge_function(nodes_raster_buff[1,:],
                                                nodes_raster_buff[2,:],
                                                px_coord_buff)
                if (weights_buff[0] < -tol):
                    bound_coord_x += coord_step
                    bound_ind_x += 1
                    continue

                weights_buff[1] = edge_function(nodes_raster_buff[2,:],
                                                nodes_raster_buff[0,:],
                                                px_coord_buff)
                if (weights_buff[1] < -tol):
                    bound_coord_x += coord_step
                    bound_ind_x += 1
                    continue


                weights_buff[2] = edge_function(nodes_raster_buff[0,:],
                                                nodes_raster_buff[1,:],
                                                px_coord_buff)
                if (weights_buff[2] < -tol):
                    bound_coord_x += coord_step
                    bound_ind_x += 1
                    continue


                for ww in range(nodes_per_elem):
                    weights_buff[ww] = weights_buff[ww] / elem_area

                weight_dot_nodes: cython.double = vec_dot_double(
                                                    weights_buff,
                                                    nodes_raster_buff[:,zz])

                # Check the depth buffer, if the element is behind move on
                px_coord_z: cython.double = 1/weight_dot_nodes
                if px_coord_z >= depth_buff_subpx[bound_ind_y,bound_ind_x]:
                    continue

                # We only need one depth buffer for all fields
                depth_buff_subpx[bound_ind_y,bound_ind_x] = px_coord_z

                for ff in range(fields_num):
                    for nn in range(nodes_per_elem):

                        field_raster_buff[nn] = (fields_to_render[connect[ee,nn],ff]
                                                *nodes_raster_buff[nn,zz])

                    px_field: cython.double = (vec_dot_double(field_raster_buff,
                                                              weights_buff)
                                                              *px_coord_z)

                    image_buff_subpx[bound_ind_y,bound_ind_x,ff] = px_field

                # end for(x) - increment the x coords
                bound_coord_x += coord_step
                bound_ind_x += 1

            # end for(y) - increment the y coords
            bound_coord_y += coord_step
            bound_ind_y += 1

    _average_image(depth_buff_subpx,sub_samp,depth_buff_avg)

    for ff in range(fields_num):
        _average_image(image_buff_subpx[:,:,ff],
                        sub_samp,
                        image_buff_avg[:,:,ff])

    return elems_in_image

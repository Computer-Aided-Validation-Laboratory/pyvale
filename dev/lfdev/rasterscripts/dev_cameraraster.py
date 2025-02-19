"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import mooseherder as mh
import pyvale


def bound_box_low(coord_min: np.ndarray) -> np.ndarray:
    bound_elem = np.floor(coord_min).astype(np.int32)
    bound_low = np.zeros_like(coord_min,dtype=np.int32)
    bound_mat = np.vstack((bound_elem,bound_low))
    return np.max(bound_mat,axis=0)

def bound_box_high(coord_max: np.ndarray,
                    image_px: int) -> np.ndarray:
    bound_elem = np.ceil(coord_max).astype(np.int32)
    bound_high = image_px*np.ones_like(coord_max,dtype=np.int32)
    bound_mat = np.vstack((bound_elem,bound_high))
    bound = np.min(bound_mat,axis=0)
    return bound

def edge_function(vert_a: np.ndarray,
                  vert_b: np.ndarray,
                  vert_c: np.ndarray) -> np.ndarray:
    (xx,yy) = (0,1)
    edge_fun = ((vert_c[xx] - vert_a[xx]) * (vert_b[yy] - vert_a[yy])
              - (vert_c[yy] - vert_a[yy]) * (vert_b[xx] - vert_a[xx]))
    return edge_fun

def main() -> None:
    data_path = Path('src/pyvale/simcases/case24_out.e')
    #data_path = Path('src/pyvale/data/case13_out.e')
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_key = list(sim_data.node_vars.keys())[0]
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0
    pyvale.print_dimensions(sim_data)

    descriptor = pyvale.SensorDescriptorFactory.temperature_descriptor()

    field_key = 'temperature'
    t_field = pyvale.FieldScalar(sim_data,
                                 field_key=field_key,
                                 spat_dims=2)

    num_px = np.array((510,260))
    leng_per_px = pyvale.calc_resolution_from_sim_2d(num_px,
                                                  sim_data.coords,
                                                  pixels_border=5)
    roi_center_world = pyvale.calc_roi_cent_from_sim(sim_data.coords)

    cam_data = pyvale.CameraData2D(num_pixels=num_px,
                                   leng_per_px=leng_per_px,
                                   roi_center_world=roi_center_world)

    #===========================================================================
    # NOTE
    # - If user specified the roi location and the camera location we know the
    #   imaging distance
    # - We know the focal length and sensor size we can find the view angle
    #   - Any combination of 2 of focal length, sensor size and view angle
    # - The camera matrix is given as (column major): CameraToWorld
    #   [[x0, y0, z0, Tx],
    #    [x1, y1, z1, Ty],
    #    [x2, y2, z2, Tz],
    #    [0 ,  0 , 0, 1 ]]
    # Unit vector [x0,x1,x2] specified the x axis, [Tx,Ty,Tz] is the camera pos
    # in world coords.
    # The WorldToCamera matrix is the inverse of the above matrix!
    (xx,yy,zz,ww) = (0,1,2,3)

    coords_world = sim_data.coords.T #shape=(3,num_coords)
    coords_count = coords_world.shape[1]

    coords_world_with_w = np.vstack((coords_world,np.ones((1,coords_count))))
    connect = sim_data.connect["connect1"]
    field_scalar = sim_data.node_vars["temperature"]

    rot_axis: str = "n"
    phi_y_degs: float = 45
    theta_x_degs: float = 45

    phi_y_rads: float = phi_y_degs * np.pi/180.0
    theta_x_rads: float = theta_x_degs * np.pi/180.0

    # Set this to 250 to zoom in
    image_dist: float = 200.0
    #image_dist: float = 250.0

    # Set this to 0.0 to get some of the plate outside the FOV
    roi_pos_world = np.array([50.0,25.0,0.0])
    #roi_pos_world = np.array([0.0,0.0,0.0])

    if rot_axis == "y":
        cam_pos_world = np.array([roi_pos_world[xx] - image_dist*np.sin(phi_y_rads),
                                roi_pos_world[yy],
                                image_dist*np.cos(phi_y_rads)])

        cam_rot = Rotation.from_euler("zyx", [0, -phi_y_degs, 0], degrees=True)

    elif rot_axis == "x":
        cam_pos_world = np.array([roi_pos_world[xx] ,
                                  roi_pos_world[yy] + image_dist*np.sin(theta_x_rads),
                                  image_dist*np.cos(theta_x_rads)])

        cam_rot = Rotation.from_euler("zyx", [0, 0, -theta_x_degs], degrees=True)

    else:
        cam_pos_world = np.array([roi_pos_world[xx],
                                  roi_pos_world[yy],
                                  image_dist])
        cam_rot = Rotation.from_euler("zyx", [0, 0, 0], degrees=True)

    image_dist = np.linalg.norm(cam_pos_world - roi_pos_world)

    print()
    print(80*"-")
    print(f"{cam_pos_world=}")
    print(f"{image_dist=}")
    print(80*"-")

    cam_to_world_mat = np.zeros((4,4))
    cam_to_world_mat[0:3,0:3] = cam_rot.as_matrix()
    cam_to_world_mat[-1,-1] = 1.0
    cam_to_world_mat[0:3,-1] = cam_pos_world
    world_to_cam_mat = np.linalg.inv(cam_to_world_mat)

    print()
    print(80*"-")
    print("Camera to world matrix:")
    print(cam_to_world_mat)
    print()
    print("World to camera matrix:")
    print(world_to_cam_mat)
    print()
    print("Mesh data:")
    print(f"Total Nodes = {coords_world.shape[1]}")
    print(f"Total Elems = {connect.shape[1]}")
    print(f"Nodes per elem = {connect.shape[0]}")
    print()
    print(80*"-")

    resolution = 0.2 # mm/px - can be used to find the camera distance from ROI

    cam_num_px = np.array([510,260],dtype=np.int32)
    focal_leng = 25.0
    pixel_size = np.array([10.0e-3,10.0e-3])
    sensor_size = cam_num_px*pixel_size

    image_dims_from_res = resolution*cam_num_px
    image_dims = image_dist * sensor_size / focal_leng
    res_check = image_dims / cam_num_px

    left = -image_dims[0]/2
    right = image_dims[0]/2
    top = image_dims[1]/2
    bot = -image_dims[1]/2

    offset_lr = (right+left) / (right-left)
    offset_tb = (top+bot) / (top - bot)

    print()
    print(80*"-")
    print(f"{right-left=}")
    print(f"{right+left=}")
    print(f"{top-bot=}")
    print(f"{top+bot=}")
    print()
    print(f"{offset_lr=}")
    print(f"{offset_tb=}")
    print(80*"-")

    #///////////////////////////////////////////////////////////////////////////

    #---------------------------------------------------------------------------
    # FUNCTION: convert to raster coords

    # Project onto camera coords
    coords_cam = np.matmul(world_to_cam_mat,coords_world_with_w)

    #NOTE w is not 1 when the matrix is a perspective projection! It is only 1
    # here because this is an affine transformation.

    coords_cam[xx,:] = coords_cam[xx,:] / coords_cam[ww,:]
    coords_cam[yy,:] = coords_cam[yy,:] / coords_cam[ww,:]
    coords_cam[zz,:] = coords_cam[zz,:] / coords_cam[ww,:]

    # Perspective divide
    coords_image = np.zeros((2,coords_count))
    coords_image[xx,:] = image_dist * coords_cam[xx,:] / -coords_cam[zz,:]
    coords_image[yy,:] = image_dist * coords_cam[yy,:] / -coords_cam[zz,:]

    # Convert to normalised device coords in the range [-1,1]
    coords_ndc = np.zeros((2,coords_count))
    coords_ndc[xx,:] = 2*coords_image[xx,:] / image_dims[xx]
    coords_ndc[yy,:] = 2*coords_image[yy,:] / image_dims[yy]

    # Covert to pixel (raster) coords
    # Shape = ([X,Y,Z],num_nodes)
    coords_raster = np.zeros((3,coords_count))
    coords_raster[xx,:] = (coords_ndc[xx,:] + 1)/2 * cam_num_px[xx]
    coords_raster[yy,:] = (1-coords_ndc[yy,:])/2 * cam_num_px[yy]
    coords_raster[zz,:] = -coords_cam[zz,:]
    #---------------------------------------------------------------------------

    # Convert to perspective correct hyperbolic interpolation for z interp
    coords_raster[zz,:] = 1/coords_raster[zz,:]

    nodes_per_elem = connect.shape[0]
    num_elems = connect.shape[1]
    # NOTE: need the -1 here for zero indexing!
    # shape=(coord[X,Y,Z],node_per_elem,elem_num)
    elem_world_coords = coords_world[:,connect-1]
    # shape=(nodes_per_elem,coord[X,Y,Z],elem_num)
    elem_world_coords = np.swapaxes(elem_world_coords,0,1)

    # NOTE: need the -1 here for zero indexing as element nums start from 1!
    # shape=(coord[X,Y,Z],node_per_elem,elem_num)
    elem_raster_coords = coords_raster[:,connect-1]
    # shape=(nodes_per_elem,coord[X,Y,Z],elem_num)
    elem_raster_coords = np.swapaxes(elem_raster_coords,0,1)

    # NOTE: we have already inverted the raster z coordinate above so to divide
    # by z here we need to multiply
    # shape=(n_nodes,num_time_steps)
    field_divide_z = (field_scalar.T * coords_raster[zz,:]).T
    # shape=(nodes_per_elem,num_elems,num_time_steps)
    field_divide_z = field_divide_z[connect-1,:]

    #shape=(coord[X,Y,Z],elem_num)
    elem_raster_coord_min = np.min(elem_raster_coords,axis=0)
    elem_raster_coord_max = np.max(elem_raster_coords,axis=0)

    # Check that which nodes are within the 4 edges of the camera image
    #shape=(4_edges_to_check,num_elems)
    mask = np.zeros([4,num_elems])
    mask[0,elem_raster_coord_min[xx,:] <= (cam_num_px[xx]-1)] = 1
    mask[1,elem_raster_coord_min[yy,:] <= (cam_num_px[yy]-1)] = 1
    mask[2,elem_raster_coord_max[xx,:] >= 0] = 1
    mask[3,elem_raster_coord_max[yy,:] >= 0] = 1
    # If any of the 4 tests above fail then the element is outside the FOV
    mask = np.sum(mask,0) == 4

    # Get only the elements that are within the FOV
    # Mask the elem coords and the max and min elem coords for processing
    elem_raster_coord_min = elem_raster_coord_min[:,mask]
    elem_raster_coord_max = elem_raster_coord_max[:,mask]
    elem_raster_coords = elem_raster_coords[:,:,mask]
    num_elems_in_scene = elem_raster_coords.shape[2]
    # shape=(nodes_per_elem,elems_in_scene,num_time_steps)
    field_divide_z = field_divide_z[:,mask,:]

    print()
    print(80*"-")
    print("MASKING CHECKS:")
    print("Mask =")
    print(mask)
    print()
    print(f"Elems in mask =      {np.sum(np.sum(mask))}")
    print(f"Total elems =        {num_elems}")
    print(f"Num elems in scene = {num_elems_in_scene}")
    print()
    print(f"{elem_raster_coords.shape=}")
    print(f"{elem_raster_coord_min.shape=}")
    print(f"{elem_raster_coord_max.shape=}")
    print()
    print(f"{field_divide_z.shape=}")
    print(80*"-")

    # Pixel coords
    px_coord_x = np.arange(0,cam_num_px[xx]) + 0.5
    px_coord_y = np.arange(0,cam_num_px[yy]) + 0.5
    (px_coord_grid_x,px_coord_grid_y) = np.meshgrid(px_coord_x,px_coord_y)

    # Find the indices of the bounding box that each element lies within on the
    # image, bounded by the upper and lower edges of the image
    elem_bound_boxes_px_inds = np.zeros([4,num_elems_in_scene],dtype=np.int32)
    elem_bound_boxes_px_inds[0,:] = bound_box_low(elem_raster_coord_min[xx,:])
    elem_bound_boxes_px_inds[1,:] = bound_box_high(elem_raster_coord_max[xx,:],
                                                   cam_num_px[xx]-1)
    elem_bound_boxes_px_inds[2,:] = bound_box_low(elem_raster_coord_min[yy,:])
    elem_bound_boxes_px_inds[3,:] = bound_box_high(elem_raster_coord_max[yy,:],
                                                   cam_num_px[yy]-1)

    # NOTE: edge function
    # for 3 vectors in 2d A, B and C. The edge function is:
    # Where A and B are two edges of the element and C is the point to test
    # within the element.
    # (C[xx] - A[xx]) * (B[yy] - A[yy]) - (C[YY] - A[yy])*(B[xx] - A[xx])
    # If the edge function is greater than 1 then the point is within the elem
    # If A, B, C are the vertices of the element then the edge function is twice
    # the area of the element.

    (aa,bb,cc) = (0,1,2)
    elem_areas = edge_function(elem_raster_coords[aa,:,:],
                               elem_raster_coords[bb,:,:],
                               elem_raster_coords[cc,:,:])

    # RASTER
    #---------------------------------------------------------------------------
    # Option 1: I think we have to loop here because the number of pixels in the
    # bound boxes is not necessarily the same for each element.
    # Option 2: Ignore the bounding box and just test all pixels for every
    # element. Will consume more memory but might be faster
    # Option 3: If we have to loop then we should probably use Numba or Cython

    # Create a depth buffer and an image buffer
    depth_buffer = 1e6*np.ones(cam_num_px).T
    image_buffer = np.zeros(cam_num_px).T

    # We only need to loop over elements and slice out and process the bound box
    (x_start,x_end,y_start,y_end) = (0,1,2,3)
    (v0,v1,v2) = (0,1,2)
    ee = 0      # element number to test before loop
    frame = -1  # render the last frame
    field_frame_divide_z = field_divide_z[:,:,frame]
    loop_print = False

    print()
    print(80*"=")
    print("RASTER ELEMENT LOOP START")
    print(80*"=")
    print()
    for ee in range(num_elems_in_scene):
        # Get the pixel coords for the pixels in the bounding box
        px_inds_range_x = np.arange(elem_bound_boxes_px_inds[x_start,ee],
                            elem_bound_boxes_px_inds[x_end,ee])
        px_inds_range_y = np.arange(elem_bound_boxes_px_inds[y_start,ee],
                            elem_bound_boxes_px_inds[y_end,ee])
        (px_inds_grid_x,px_inds_grid_y) = np.meshgrid(px_inds_range_x,
                                                    px_inds_range_y)

        # NOTE: can probably optimise here as a lot of variables are the same
        bound_coords_grid_x = px_coord_grid_x[px_inds_grid_y,px_inds_grid_x]
        bound_coords_grid_y = px_coord_grid_y[px_inds_grid_y,px_inds_grid_x]
        bound_coords_grid_shape = bound_coords_grid_x.shape
        bound_box_coords_flat = np.vstack((bound_coords_grid_x.flatten(),
                                           bound_coords_grid_y.flatten()))

        # We compute the edge function for all pixels in the box to determine if the
        # pixel is inside the element or not
        vert_0 = elem_raster_coords[v0,:,ee]
        vert_1 = elem_raster_coords[v1,:,ee]
        vert_2 = elem_raster_coords[v2,:,ee]
        edge = np.zeros((3,bound_box_coords_flat.shape[1]))
        edge[0,:] = edge_function(vert_1,vert_2,bound_box_coords_flat)
        edge[1,:] = edge_function(vert_2,vert_0,bound_box_coords_flat)
        edge[2,:] = edge_function(vert_0,vert_1,bound_box_coords_flat)

        # Now we check where the edge function is above zero for all edges
        edge_check = np.zeros_like(edge,dtype=np.int8)
        edge_check[edge >= 0.0] = 1
        edge_check = np.sum(edge_check, axis=0)
        # Create a mask with the check, TODO check the 3 here for non triangles
        edge_mask_flat = edge_check == 3
        edge_mask_grid = np.reshape(edge_mask_flat,bound_coords_grid_shape)

        if loop_print:
            print()
            print(80*"-")
            print(f"ELEMENT: {ee}")
            print()
            print("Bounding Box:")
            print(f"Px in box = {bound_box_coords_flat.shape[-1]}")
            print(f"X range = {px_inds_range_x}")
            print(f"Y range = {px_inds_range_y}")
            print(f"Grid shape = {bound_coords_grid_x.shape}")
            print(f"Flat shape = {bound_box_coords_flat.shape}")
            print()
            print("Raster Coords:")
            print(f"Vertex 0 = {vert_0}")
            print(f"Vertex 1 = {vert_1}")
            print(f"Vertex 1 = {vert_2}")
            print()
            print("Edge Function:")
            print(f"Pixels inside = {np.sum(edge_mask_flat)}")
            print(80*"-")

        # Calculate the weights for the masked pixels
        edge_masked = edge[:,edge_mask_flat]
        interp_weights = edge_masked / elem_areas[ee]

        # Compute the depth of all pixels using hyperbolic interp
        px_coord_z = 1/(vert_0[zz] * interp_weights[0,:]
                      + vert_1[zz] * interp_weights[1,:]
                      + vert_2[zz] * interp_weights[2,:])

        field_interp = ((field_frame_divide_z[0,ee] * interp_weights[0,:]
                       + field_frame_divide_z[1,ee] * interp_weights[1,:]
                       + field_frame_divide_z[2,ee] * interp_weights[2,:])
                       * px_coord_z)

        # Get the pixel indices that are inside the element
        px_inds_x_inside = px_inds_grid_x[edge_mask_grid]
        px_inds_y_inside = px_inds_grid_y[edge_mask_grid]

        # Build a mask to replace the depth information if there is already an
        # element in front of the one we are rendering
        px_coord_z_depth_mask = (px_coord_z
                                 < depth_buffer[px_inds_y_inside,px_inds_x_inside])

        # Initialise the z coord to the value in the depth buffer
        px_coord_z_masked = depth_buffer[px_inds_y_inside,px_inds_x_inside]
        # Use the depth mask to overwrite the depth buffer values if points are in
        # front of the values in the depth buffer
        px_coord_z_masked[px_coord_z_depth_mask] = px_coord_z[px_coord_z_depth_mask]

        # Push the masked values into the depth buffer
        depth_buffer[px_inds_y_inside,px_inds_x_inside] = px_coord_z_masked

        # Mask the image buffer using the depth mask
        image_buffer_depth_masked = image_buffer[px_inds_y_inside,px_inds_x_inside]
        image_buffer_depth_masked[px_coord_z_depth_mask] = field_interp[px_coord_z_depth_mask]

        # Push the masked values into the image buffer
        image_buffer[px_inds_y_inside,px_inds_x_inside] = image_buffer_depth_masked
    # END RASTER LOOP

    print()
    print(80*"=")
    print("RASTER LOOP END")
    print(80*"=")

    plot_on = True
    depth_to_plot = np.copy(depth_buffer)
    depth_to_plot[depth_buffer > 10*image_dist] = np.NaN
    image_to_plot = np.copy(image_buffer)
    image_to_plot[depth_buffer > 10*image_dist] = np.NaN
    #===========================================================================
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
        ax.set_title("Origin normal",fontsize=plot_opts.font_head_size)
        ax.set_xlabel(r"x ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
        ax.set_ylabel(r"y ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

        plt.show()

if __name__ == "__main__":
    main()
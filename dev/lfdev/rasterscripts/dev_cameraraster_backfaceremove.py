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
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import convolve2d
import mooseherder as mh
import pyvale


@dataclass(slots=True)
class CameraRasterData:
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

def world_to_raster_coords(cam_data: CameraRasterData,
                           coords_world_with_w: np.ndarray) -> np.ndarray:

    (xx,yy,zz,ww) = (0,1,2,3)
    coords_count = coords_world_with_w.shape[1]

    # Project onto camera coords
    coords_cam = np.matmul(cam_data.world_to_cam_mat,coords_world_with_w)

    # NOTE: w is not 1 when the matrix is a perspective projection! It is only 1
    # here because this is an affine transformation.
    # coords_cam[xx,:] = coords_cam[xx,:] / coords_cam[ww,:]
    # coords_cam[yy,:] = coords_cam[yy,:] / coords_cam[ww,:]
    # coords_cam[zz,:] = coords_cam[zz,:] / coords_cam[ww,:]

    # Coords Image: Perspective divide
    coords_image = np.zeros((2,coords_count))
    coords_image[xx,:] = cam_data.image_dist * coords_cam[xx,:] / -coords_cam[zz,:]
    coords_image[yy,:] = cam_data.image_dist * coords_cam[yy,:] / -coords_cam[zz,:]

    # Coords NDC: Convert to normalised device coords in the range [-1,1]
    coords_ndc = np.zeros((2,coords_count))
    coords_ndc[xx,:] = 2*coords_image[xx,:] / cam_data.image_dims[xx]
    coords_ndc[yy,:] = 2*coords_image[yy,:] / cam_data.image_dims[yy]

    # Coords Raster: Covert to pixel (raster) coords
    # Shape = ([X,Y,Z],num_nodes)
    coords_raster = np.zeros((3,coords_count))
    coords_raster[xx,:] = (coords_ndc[xx,:] + 1)/2 * cam_data.num_pixels[xx]
    coords_raster[yy,:] = (1-coords_ndc[yy,:])/2 * cam_data.num_pixels[yy]
    coords_raster[zz,:] = -coords_cam[zz,:]

    return coords_raster

# TODO: try Numba on this
def edge_function(vert_a: np.ndarray,
                  vert_b: np.ndarray,
                  vert_c: np.ndarray) -> np.ndarray:
    (xx,yy) = (0,1)
    edge_fun = ((vert_c[xx] - vert_a[xx]) * (vert_b[yy] - vert_a[yy])
              - (vert_c[yy] - vert_a[yy]) * (vert_b[xx] - vert_a[xx]))
    return edge_fun

# TODO: try Numba on this too
def average_subpixel_image(subpx_image: np.ndarray,
                           subsample: int) -> np.ndarray:
    conv_mask = np.ones((subsample,subsample))/(subsample**2)
    if subsample > 1:
        subpx_image_conv = convolve2d(subpx_image,conv_mask,mode='same')
        avg_image = subpx_image_conv[round(subsample/2)-1::subsample,
                                     round(subsample/2)-1::subsample]
    else:
        subpx_image_conv = subpx_image
        avg_image = subpx_image

    return avg_image

# Core of the element rasterisation loop
def raster_element(cam_data: CameraRasterData,
                   elem_bound_boxes_px_inds: np.ndarray,
                   elem_raster_coords: np.ndarray,
                   elem_area: float,
                   field_divide_z: np.ndarray
                   ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    (xx,yy,zz) = (0,1,2)
    (x_start,x_end,y_start,y_end) = (0,1,2,3)
    (v0,v1,v2) = (0,1,2)

    # Create the subpixel coords inside the bounding box to test with the
    # edge function. Use the pixel indices of the bounding box.
    bound_subpx_x = np.arange(elem_bound_boxes_px_inds[x_start],
                                elem_bound_boxes_px_inds[x_end],
                                1/cam_data.sub_samp) + 1/(2*cam_data.sub_samp)
    bound_subpx_y = np.arange(elem_bound_boxes_px_inds[y_start],
                                elem_bound_boxes_px_inds[y_end],
                                1/cam_data.sub_samp) + 1/(2*cam_data.sub_samp)
    (bound_subpx_grid_x,bound_subpx_grid_y) = np.meshgrid(bound_subpx_x,
                                                            bound_subpx_y)
    bound_coords_grid_shape = bound_subpx_grid_x.shape
    bound_subpx_coords_flat = np.vstack((bound_subpx_grid_x.flatten(),
                                            bound_subpx_grid_y.flatten()))

    # Create the subpixel indices for buffer slicing later
    subpx_inds_x = np.arange(cam_data.sub_samp*elem_bound_boxes_px_inds[x_start],
                                cam_data.sub_samp*elem_bound_boxes_px_inds[x_end])
    subpx_inds_y = np.arange(cam_data.sub_samp*elem_bound_boxes_px_inds[y_start],
                                cam_data.sub_samp*elem_bound_boxes_px_inds[y_end])
    (subpx_inds_grid_x,subpx_inds_grid_y) = np.meshgrid(subpx_inds_x,
                                                        subpx_inds_y)

    # We compute the edge function for all pixels in the box to determine if the
    # pixel is inside the element or not
    edge = np.zeros((3,bound_subpx_coords_flat.shape[1]))
    edge[0,:] = edge_function(elem_raster_coords[v1,:],
                              elem_raster_coords[v2,:],
                              bound_subpx_coords_flat)
    edge[1,:] = edge_function(elem_raster_coords[v2,:],
                              elem_raster_coords[v0,:],
                              bound_subpx_coords_flat)
    edge[2,:] = edge_function(elem_raster_coords[v0,:],
                              elem_raster_coords[v1,:],
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
    px_coord_z = 1/(elem_raster_coords[v0,zz] * interp_weights[0,:]
                  + elem_raster_coords[v1,zz] * interp_weights[1,:]
                  + elem_raster_coords[v2,zz] * interp_weights[2,:])

    field_interp = ((field_divide_z[v0] * interp_weights[0,:]
                   + field_divide_z[v1] * interp_weights[1,:]
                   + field_divide_z[v2] * interp_weights[2,:])
                   * px_coord_z)


    return (px_coord_z,
            field_interp,
            subpx_inds_grid_x[edge_mask_grid],
            subpx_inds_grid_y[edge_mask_grid])

#===============================================================================
# MAIN
def main() -> None:
    test_case_str = "cyl"

    if test_case_str == "temp":
        # 2D plate, thermal, triangles
        data_path = Path('src/pyvale/simcases/case24_out.e')
    else:
        # 3D cylinder, mechanical, tets
        data_path = Path('src/pyvale/simcases/case21_out.e')

    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_keys = tuple(sim_data.node_vars.keys())
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0

    if test_case_str == "temp":
        field_key = "temperature"
        components = ("temperature",)
        (pv_grid,_) = pyvale.simdata_to_pyvista(sim_data,
                                                    components,
                                                    spat_dim=2)

        roi_pos_world = np.mean(sim_data.coords,axis=0)

    else:
        field_key = "disp_y"
        components = ("disp_x","disp_y","disp_z")
        (pv_grid,_) = pyvale.simdata_to_pyvista(sim_data,
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

    #shape=(3,num_coords)
    coords_world = pv_coords
    coords_count = coords_world.shape[1]
    # shape=(4,num_nodes)
    coords_world_with_w = np.vstack((coords_world,np.ones((1,coords_count))))
    # shape=(nodes_per_elem,num_elems)
    connect = pv_connect
    elem_count = connect.shape[1]
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
    sub_samp: int = 1

    cam_type = "Test"
    if cam_type == "AV507":
        cam_num_px = np.array([2464,2056],dtype=np.int32)
        pixel_size = np.array([3.45e-3,3.45e-3]) # in millimeters!
        focal_leng: float = 25.0

        imaging_rad: float = 500.0 # Not needed for camera data, just for cam pos below
    else:
        cam_num_px = np.array([510,260],dtype=np.int32)
        pixel_size = np.array([10.0e-3,10.0e-3])
        focal_leng: float = 25.0

        imaging_rad: float = 500.0 # Not needed for camera data, just for cam pos below

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
    cam_data = CameraRasterData(num_pixels=cam_num_px,
                                pixel_size=pixel_size,
                                pos_world=cam_pos_world,
                                rot_world=cam_rot,
                                roi_center_world=roi_pos_world,
                                focal_length=focal_leng,
                                sub_samp=sub_samp)

    print()
    print(80*"-")
    print("MESH DATA:")
    print(f"Total Nodes = {coords_world.shape[1]}")
    print(f"Total Elems = {connect.shape[1]}")
    print(f"Nodes per elem = {connect.shape[0]}")
    print(80*"-")
    print()
    print(80*"-")
    print("CAMERA DATA:")
    print(f"Number of pixels [X,Y] = {cam_data.num_pixels}")
    print(f"Pixel size [X,Y] = {cam_data.pixel_size}")
    print(f"Focal length = {cam_data.focal_length}")
    print()
    print(f"Pos. world = {cam_data.pos_world}")
    print(f"ROI center world = {cam_data.roi_center_world}")
    print()
    print(f"Sensor size = {cam_data.sensor_size}")
    print(f"Image dims. = {cam_data.image_dims}")
    print(f"Image dist. = {cam_data.image_dist}")
    print()
    print("Camera to world matrix:")
    print(cam_data.cam_to_world_mat)
    print()
    print("World to camera matrix:")
    print(cam_data.world_to_cam_mat)
    print()
    print(80*"-")

    #---------------------------------------------------------------------------
    # BACK FACE REMOVAL - world method

    # shape=(coord[X,Y,Z,W],node_per_elem,elem_num)
    elem_world_coords = coords_world_with_w[:,connect]
    # shape=(nodes_per_elem,coord[X,Y,Z,W],elem_num)
    elem_world_coords = np.swapaxes(elem_world_coords,0,1)
    elem_world_edge0 = elem_world_coords[1,:-1,:] - elem_world_coords[0,:-1,:]
    elem_world_edge1 = elem_world_coords[2,:-1,:] - elem_world_coords[0,:-1,:]
    elem_world_normals = np.cross(elem_world_edge0,elem_world_edge1,
                                  axisa=0,axisb=0).T
    elem_world_normals_with_w = np.vstack((elem_world_normals,np.ones((1,elem_count))))

    #===========================================================================
    # Method: transform_normals = np.linalg.inv(cam_data.cam_to_world_mat).T
    # I couldn't get this to make sense using the the inv(M).T method from
    # scratchapixel as it seems to ignore rotation of the camera about z.
    # There was a weird results for rotation about z where this method seemed
    # to be correct and the method below was wrong with the sign of the x coord
    # being correct but it still ignores axial rotation about z.
    #
    # Method: transform_normals = cam_data.cam_to_world_mat.T
    # This method is consistent with transforming coords into camera space first
    # and it seems to work for the axial rotation about z but then the sign
    # problem arises for rotation about y.

    #Matrix44f transformNormals = worldToObject.transpose();, world_to_cam
    # NOTE this agrees with scratchapixel which is inv(M).T where
    # M is the cam_to_world matrix (the world_to_cam is inv(cam_to_world))
    #transform_normals = np.linalg.inv(cam_data.cam_to_world_mat).T
    transform_normals = cam_data.cam_to_world_mat.T # This is the same as below starting with elem coords
    elem_cam_normals_check = np.matmul(transform_normals,elem_world_normals_with_w)
    # Chop off the w coord after the matrix multiplication
    elem_cam_normals_check = elem_cam_normals_check[:-1,:]
    elem_cam_normals_check = elem_cam_normals_check / np.linalg.norm(elem_cam_normals_check,axis=0)
    #===========================================================================

    #---------------------------------------------------------------------------
    # BACKFACE REMOVAL - transform method
    coords_cam = np.matmul(cam_data.world_to_cam_mat,coords_world_with_w)

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

    print()
    print(80*"-")
    print("BACK FACE REMOVAL:")
    print(f"Total elements     : {elem_count}")
    print(f"Elements removed   : {elem_count - np.sum(back_face_mask)}")
    print(f"Elements remaining : {np.sum(back_face_mask)}")
    print(80*"-")
    #---------------------------------------------------------------------------

    # Convert world coords of all elements in the scene
    coords_raster = world_to_raster_coords(cam_data,coords_world_with_w)

    # Convert to perspective correct hyperbolic interpolation for z interp
    coords_raster[zz,:] = 1/coords_raster[zz,:]

    # shape=(coord[X,Y,Z],node_per_elem,elem_num)
    elem_raster_coords = coords_raster[:,connect]
    # shape=(nodes_per_elem,coord[X,Y,Z],elem_num)
    elem_raster_coords = np.swapaxes(elem_raster_coords,0,1)

    # NOTE: we have already inverted the raster z coordinate above so to divide
    # by z here we need to multiply
    # shape=(n_nodes,num_time_steps)
    field_divide_z = (field_scalar.T * coords_raster[zz,:]).T
    # shape=(nodes_per_elem,num_elems,num_time_steps)
    field_divide_z = field_divide_z[connect,:]

    # APPLY BACKFACE REMOVAL
    print()
    print(80*"-")
    print("APPLY BACK FACE REMOVAL")
    print("Before:")
    print(f"{elem_raster_coords.shape=}")
    print(f"{field_divide_z.shape=}")
    print()
    elem_raster_coords = elem_raster_coords[:,:,back_face_mask]
    field_divide_z = field_divide_z[:,back_face_mask,:]
    print("After:")
    print(f"{elem_raster_coords.shape=}")
    print(f"{field_divide_z.shape=}")
    print(80*"-")

    #shape=(coord[X,Y,Z],elem_num)
    elem_raster_coord_min = np.min(elem_raster_coords,axis=0)
    elem_raster_coord_max = np.max(elem_raster_coords,axis=0)

    # Check that which nodes are within the 4 edges of the camera image
    #shape=(4_edges_to_check,num_elems)
    mask = np.zeros([4,elem_raster_coords.shape[-1]])
    mask[0,elem_raster_coord_min[xx,:] <= (cam_data.num_pixels[xx]-1)] = 1
    mask[1,elem_raster_coord_min[yy,:] <= (cam_data.num_pixels[yy]-1)] = 1
    mask[2,elem_raster_coord_max[xx,:] >= 0] = 1
    mask[3,elem_raster_coord_max[yy,:] >= 0] = 1
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
    print(f"Total elems =        {elem_count}")
    print(f"Num elems in scene = {num_elems_in_scene}")
    print()
    print(f"{elem_raster_coords.shape=}")
    print(f"{elem_raster_coord_min.shape=}")
    print(f"{elem_raster_coord_max.shape=}")
    print()
    print(f"{field_divide_z.shape=}")
    print(80*"-")

    # Find the indices of the bounding box that each element lies within on the
    # image, bounded by the upper and lower edges of the image
    elem_bound_boxes_px_inds = np.zeros([4,num_elems_in_scene],dtype=np.int32)
    elem_bound_boxes_px_inds[0,:] = bound_box_low(elem_raster_coord_min[xx,:])
    elem_bound_boxes_px_inds[1,:] = bound_box_high(elem_raster_coord_max[xx,:],
                                                   cam_data.num_pixels[xx]-1)
    elem_bound_boxes_px_inds[2,:] = bound_box_low(elem_raster_coord_min[yy,:])
    elem_bound_boxes_px_inds[3,:] = bound_box_high(elem_raster_coord_max[yy,:],
                                                   cam_data.num_pixels[yy]-1)

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

    # Create a depth buffer and an image buffer upsampled for anti-aliasing
    depth_buffer = 1e6*np.ones(cam_data.sub_samp*cam_data.num_pixels).T
    image_buffer = np.full(cam_data.sub_samp*cam_data.num_pixels,0.0).T

    # We only need to loop over elements and slice out and process the bound box
    (x_start,x_end,y_start,y_end) = (0,1,2,3)
    (v0,v1,v2) = (0,1,2)

    frame = -1  # render the last frame
    field_frame_divide_z = field_divide_z[:,:,frame]

    loop_print = False

    time_end_setup = time.perf_counter()

    print()
    print(80*"=")
    print("RASTER ELEMENT LOOP START")
    print(80*"=")
    #---------------------------------------------------------------------------
    time_start_loop = time.perf_counter()
    for ee in range(num_elems_in_scene):

        (px_coord_z,
        field_interp,
        subpx_inds_x_inside,
        subpx_inds_y_inside) = raster_element(cam_data,
                                              elem_bound_boxes_px_inds[:,ee],
                                              elem_raster_coords[:,:,ee],
                                              elem_areas[ee],
                                              field_frame_divide_z[:,ee])

        #-----------------------------------------------------------------------
        # MASK AND PUSH INTO BUFFER

        #  Build a mask to replace the depth information if there is already an
        # element in front of the one we are rendering
        px_coord_z_depth_mask = (px_coord_z
                                 < depth_buffer[subpx_inds_y_inside,subpx_inds_x_inside])

        # Initialise the z coord to the value in the depth buffer
        px_coord_z_masked = depth_buffer[subpx_inds_y_inside,subpx_inds_x_inside]
        # Use the depth mask to overwrite the depth buffer values if points are in
        # front of the values in the depth buffer
        px_coord_z_masked[px_coord_z_depth_mask] = px_coord_z[px_coord_z_depth_mask]

        # Push the masked values into the depth buffer
        depth_buffer[subpx_inds_y_inside,subpx_inds_x_inside] = px_coord_z_masked

        # Mask the image buffer using the depth mask
        image_buffer_depth_masked = image_buffer[subpx_inds_y_inside,subpx_inds_x_inside]
        image_buffer_depth_masked[px_coord_z_depth_mask] = field_interp[px_coord_z_depth_mask]

        # Push the masked values into the image buffer
        image_buffer[subpx_inds_y_inside,subpx_inds_x_inside] = image_buffer_depth_masked

    #---------------------------------------------------------------------------
    # END RASTER LOOP
    depth_avg = average_subpixel_image(depth_buffer,sub_samp)
    image_avg = average_subpixel_image(image_buffer,sub_samp)

    time_end_loop = time.perf_counter()
    print()
    print(80*"=")
    print("RASTER LOOP END")
    print(80*"=")
    print()
    print(80*"=")
    print("PERFORMANCE TIMERS")
    print(f"Setup time = {time_end_setup-time_start_setup} seconds")
    print(f"Loop time  = {time_end_loop-time_start_loop} seconds")
    print(f"Total time = {time_end_loop-time_start_setup} seconds")
    print(80*"=")
    print()

    plot_on = True
    depth_to_plot = np.copy(depth_avg)
    depth_to_plot[depth_avg > 10*cam_data.image_dist] = np.NaN
    image_to_plot = np.copy(image_avg)
    image_to_plot[depth_avg > 10*cam_data.image_dist] = np.NaN
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
        ax.set_title("Field Image",fontsize=plot_opts.font_head_size)
        ax.set_xlabel(r"x ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
        ax.set_ylabel(r"y ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

        plt.show()

if __name__ == "__main__":
    main()
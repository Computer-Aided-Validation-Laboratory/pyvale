#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

"""
NOTE: this module is a feature under developement
"""

from pathlib import Path
import time
from multiprocessing.pool import Pool
import numpy as np
import numba
#import matplotlib.pyplot as plt
import pyvale.dataset as dataset
from pyvale.sensorsim.cameradata import CameraData
from pyvale.sensorsim.cameratools import CameraTools
from pyvale.sensorsim.rendermesh import RenderMesh
from pyvale.sensorsim.renderer import IRenderer, RenderScene
from pyvale.sensorsim.rasteropts import RasterOpts, save_raster
from pyvale.sensorsim.imagetools import ImageTools
#from pyvale.sensorsim.visualimages import plot_field_image
import pyvale.sensorsim.cython.rastercyth as rastercyth


# NOTE: This module is a feature under developement.

#===============================================================================
class RasterNumpy(IRenderer):
    __slots__ = ("scene","opts",)

    def __init__(self, opts: RasterOpts) -> None:
        self.opts = opts


    def render(self,
               scene: RenderScene,
               cam_ind: int = 0,
               frame_ind: int = 0,
               field_ind: int = 0) -> np.ndarray:

        image = RasterNP.raster_frame(
            cam_ind=cam_ind,
            frame_ind=frame_ind,
            field_ind=field_ind,
            cam_data=scene.cameras[cam_ind],
            meshes=scene.meshes,
            opts=self.opts,
            save_path=None,
        )

        return image


    def render_to_disk(self,
                       scene: RenderScene,
                       cam_ind: int = 0,
                       frame_ind: int = 0,
                       field_ind: int = 0,
                       save_path: Path | None = None) -> None:

        if save_path is None:
            save_path = DataSet.create_output_path()

        if not save_path.is_dir():
            raise FileExistsError(f"Save path for render images does not exist:\n{save_path}")

        RasterNP.raster_frame(
            cam_ind=cam_ind,
            frame_ind=frame_ind,
            field_ind=field_ind,
            cam_data=scene.cameras[cam_ind],
            meshes=scene.meshes,
            opts=self.opts,
            save_path=save_path,
        )


    def render_all(self, scene: RenderScene) -> list[np.ndarray]:

        images = RasterNP.raster_scene(
            scene=scene,
            opts=self.opts,
            save_path=None,
        )

        return images


    def render_all_to_disk(self,
                           scene: RenderScene,
                           save_path: Path | None = None) -> None:

        if save_path is None:
            save_path = DataSet.create_output_path()

        if not save_path.is_dir():
            raise FileExistsError(f"Save path for render images does not exist:\n{save_path}")

        RasterNP.raster_scene(scene=scene,opts=self.opts,save_path=save_path)



#===============================================================================
class RasterNP:
    @staticmethod
    def world_to_raster_coords(world_to_cam_mat: np.ndarray,
                               pixels_num: np.ndarray,
                               image_dims: np.ndarray,
                               image_dist: float,
                               coords_world: np.ndarray) -> np.ndarray:
        # coords_world.shape=(num_nodes,coord[X,Y,Z,W])

        # Project onto camera coords
        # coords_raster.shape=(num_nodes,coord[X,Y,Z,W])
        coords_raster = np.matmul(coords_world,world_to_cam_mat.T)

        # NOTE: w is not 1 when the matrix is a perspective projection! It is only 1
        # here when we have an affine transformation
        coords_raster[:,0] = coords_raster[:,0] / coords_raster[:,3]
        coords_raster[:,1] = coords_raster[:,1] / coords_raster[:,3]
        coords_raster[:,2] = coords_raster[:,2] / coords_raster[:,3]

        # Coords Image: Perspective divide
        coords_raster[:,0] = (image_dist * coords_raster[:,0]
                            / -coords_raster[:,2])
        coords_raster[:,1] = (image_dist * coords_raster[:,1]
                            / -coords_raster[:,2])

        # Coords NDC: Convert to normalised device coords in the range [-1,1]
        coords_raster[:,0] = 2*coords_raster[:,0] / image_dims[0]
        coords_raster[:,1] = 2*coords_raster[:,1] / image_dims[1]

        # Coords Raster: Covert to pixel (raster) coords
        # Shape = ([X,Y,Z],num_nodes)
        coords_raster[:,0] = (coords_raster[:,0] + 1)/2 * pixels_num[0]
        coords_raster[:,1] = (1-coords_raster[:,1])/2 * pixels_num[1]
        coords_raster[:,2] = -coords_raster[:,2]

        return coords_raster


    @staticmethod
    def back_face_removal_mask(world_to_cam_mat: np.ndarray,
                               coords_world: np.ndarray,
                               connect: np.ndarray
                               ) -> np.ndarray:
        coords_cam = np.matmul(coords_world,world_to_cam_mat.T)

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
        back_face_mask = proj_elem_to_cam > 1e-12

        return back_face_mask

    @staticmethod
    def crop_and_bound_by_connect(pixels_num: np.ndarray,
                                  coords_raster: np.ndarray,
                                  connectivity: np.ndarray,
                                  ) -> tuple[np.ndarray,np.ndarray]:

        #shape=(num_elems,coord[x,y,z,w])
        if coords_raster.ndim == 2:
            coords_by_elem = coords_raster[connectivity,:]
        else:
            coords_by_elem = coords_raster[connectivity,:,:]

        elem_raster_coord_min = np.min(coords_by_elem,axis=1)
        elem_raster_coord_max = np.max(coords_by_elem,axis=1)

        # Check that min/max nodes are within the 4 edges of the camera image
        #shape=(4_edges_to_check,num_elems)
        crop_mask = np.zeros([connectivity.shape[0],4],dtype=np.int8)
        crop_mask[elem_raster_coord_min[:,0] <= (pixels_num[0]-1), 0] = 1
        crop_mask[elem_raster_coord_min[:,1] <= (pixels_num[1]-1), 1] = 1
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
        elem_bound_boxes_inds[:,0] = RasterNP.elem_bound_box_low(
                                            elem_raster_coord_min[:,0])
        elem_bound_boxes_inds[:,1] = RasterNP.elem_bound_box_high(
                                            elem_raster_coord_max[:,0],
                                            pixels_num[0]-1)
        elem_bound_boxes_inds[:,2] = RasterNP.elem_bound_box_low(
                                            elem_raster_coord_min[:,1])
        elem_bound_boxes_inds[:,3] = RasterNP.elem_bound_box_high(
                                            elem_raster_coord_max[:,1],
                                            pixels_num[1]-1)

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
    def average_buffers(cam_data: CameraData,
                        image_buff_subpx: np.ndarray,
                        depth_buff_subpx: np.ndarray) -> tuple[np.ndarray,np.ndarray]:

        depth_buff_avg = np.empty((cam_data.pixels_num[1],cam_data.pixels_num[0]),
                                    dtype=np.float64)
        image_buff_avg = np.empty((cam_data.pixels_num[1],cam_data.pixels_num[0]),
                                      dtype=np.float64)

        if Path(rastercyth.__file__).suffix in (".so",".dll",".dylib"):
            depth_buff_avg = np.array(
                rastercyth.average_image(depth_buff_subpx,cam_data.sub_samp))


            image_buff_avg = np.array(
                rastercyth.average_image(image_buff_subpx,cam_data.sub_samp))

        else:
            depth_buff_avg = CameraTools.average_subpixel_image(
                depth_buff_subpx,cam_data.sub_samp)
            image_buff_avg = CameraTools.average_subpixel_image(
                image_buff_subpx,cam_data.sub_samp)

        return (image_buff_avg,depth_buff_avg)


    @staticmethod
    def setup_frame(camera: CameraData,
                    mesh: RenderMesh,
                    frame_ind: int = 0,
                    ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:

        connect_in_frame = np.copy(mesh.connectivity)
        coords_deform = np.copy(mesh.coords)

        #-----------------------------------------------------------------------
        # DEFORM MESH WITH DISPLACEMENT
        if mesh.fields_disp is not None:
            # Exclude w coord from mesh deformation
            coords_deform[:,:-1] = (coords_deform[:,:-1]
                                    + mesh.fields_disp[:,frame_ind,:])

        #-----------------------------------------------------------------------
        # Convert all meshes from local to world coords
        coords_deform = np.matmul(coords_deform,mesh.mesh_to_world_mat.T)

        #-----------------------------------------------------------------------
        # Convert world coords of all elements in the scene
        # shape=(num_nodes,coord[x,y,z,w])
        coords_raster = RasterNP.world_to_raster_coords(camera.world_to_cam_mat,
                                                        camera.pixels_num,
                                                        camera.image_dims,
                                                        camera.image_dist,
                                                        coords_deform)

        # Convert to perspective correct hyperbolic interpolation for z interp
        # shape=(num_nodes,coord[x,y,z,w])
        coords_raster[:,2] = 1/coords_raster[:,2]
        # Remove w coord
        coords_raster = coords_raster[:,:-1]

        #-----------------------------------------------------------------------
        # BACKFACE REMOVAL
        # shape=(num_elems,)
        back_face_mask = RasterNP.back_face_removal_mask(camera.world_to_cam_mat,
                                                         coords_deform,
                                                         connect_in_frame)
        connect_in_frame = connect_in_frame[back_face_mask,:]

        #-----------------------------------------------------------------------
        # CROPPING & BOUNDING BOX OPERATIONS
        (crop_mask,
         elem_bound_box_inds) = RasterNP.crop_and_bound_by_connect(
            camera.pixels_num,
            coords_raster,
            connect_in_frame,
        )
        connect_in_frame = connect_in_frame[crop_mask,:]

        #-----------------------------------------------------------------------
        # ELEMENT AREAS FOR INTERPOLATION
        elem_raster_coords = coords_raster[connect_in_frame,:]
        elem_areas = edge_function_slice(elem_raster_coords[:,0,:],
                                         elem_raster_coords[:,1,:],
                                         elem_raster_coords[:,2,:])

        return (coords_raster,connect_in_frame,elem_bound_box_inds,elem_areas)


    @staticmethod
    def raster_elem(elem_raster_coords: np.ndarray,
                    elem_bound_box_inds: np.ndarray,
                    elem_area: float,
                    field_divide_z: np.ndarray,
                    sub_samp: int,
                    ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        # elem_raster_coords.shape=()
        # elem_bound_box_inds.shape=()
        # field_divide_z.shape=()

        # Create the subpixel coords inside the bounding box to test with the
        # edge function. Use the pixel indices of the bounding box.
        bound_subpx_x = np.arange(elem_bound_box_inds[0],
                                  elem_bound_box_inds[1],
                                  1/sub_samp) + 1/(2*sub_samp)
        bound_subpx_y = np.arange(elem_bound_box_inds[2],
                                  elem_bound_box_inds[3],
                                  1/sub_samp) + 1/(2*sub_samp)
        (bound_subpx_grid_x,bound_subpx_grid_y) = np.meshgrid(bound_subpx_x,
                                                              bound_subpx_y)
        bound_coords_grid_shape = bound_subpx_grid_x.shape
        # shape=(coord[x,y],num_subpx_in_box)
        bound_subpx_coords_flat = np.vstack((bound_subpx_grid_x.flatten(),
                                             bound_subpx_grid_y.flatten()))

        # Create the subpixel indices for buffer slicing later
        subpx_inds_x = np.arange(sub_samp*elem_bound_box_inds[0],
                                 sub_samp*elem_bound_box_inds[1])
        subpx_inds_y = np.arange(sub_samp*elem_bound_box_inds[2],
                                 sub_samp*elem_bound_box_inds[3])
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
    def raster_one_mesh(connect_in_frame: np.ndarray,
                          coords_raster: np.ndarray,
                          elem_bound_box_inds: np.ndarray,
                          elem_areas: np.ndarray,
                          field_frame_div_z: np.ndarray,
                          sub_samp: int,
                          image_buff_subpx: np.ndarray,
                          depth_buff_subpx: np.ndarray,
                          ) -> tuple[np.ndarray,np.ndarray]:
        #connect_in_frame.shape=(num_elems,nodes_per_elem)
        #coords_raster.shape=(num_coords,coord[x,y,z,w])
        #elem_bound_box_inds.shape=(num_elems,[min_x,max_x,min_y,max_y])
        #elem_areas.shape=(num_elems,)
        #field_frame_divide_z=(num_coords,)

        for ee in range(connect_in_frame.shape[0]):
            cc = connect_in_frame[ee,:]

            (px_coord_z,
            field_interp,
            subpx_inds_x_in,
            subpx_inds_y_in) = RasterNP.raster_elem(coords_raster[cc,:],
                                                    elem_bound_box_inds[ee,:],
                                                    elem_areas[ee],
                                                    field_frame_div_z[cc],
                                                    sub_samp)


            #  Build a mask to replace the depth information if there is already an
            # element in front of the one we are rendering
            px_coord_z_depth_mask = (px_coord_z
                < depth_buff_subpx[subpx_inds_y_in,subpx_inds_x_in])

            # Initialise the z coord to the value in the depth buffer
            px_coord_z_masked = depth_buff_subpx[subpx_inds_y_in,subpx_inds_x_in]
            # Use the depth mask to overwrite the depth buffer values if points are in
            # front of the values in the depth buffer
            px_coord_z_masked[px_coord_z_depth_mask] = px_coord_z[px_coord_z_depth_mask]

            # Push the masked values into the depth buffer
            depth_buff_subpx[subpx_inds_y_in,subpx_inds_x_in] = px_coord_z_masked

            # Mask the image buffer using the depth mask
            image_buffer_depth_masked = image_buff_subpx[subpx_inds_y_in,subpx_inds_x_in]
            image_buffer_depth_masked[px_coord_z_depth_mask] = field_interp[px_coord_z_depth_mask]

            # Push the masked values into the image buffer
            image_buff_subpx[subpx_inds_y_in,subpx_inds_x_in] = image_buffer_depth_masked

        #---------------------------------------------------------------------------
        # END RASTER LOOP
        return (image_buff_subpx,depth_buff_subpx)

    @staticmethod
    def raster_frame(cam_ind: int,
                     frame_ind: int,
                     field_ind: int,
                     cam_data: CameraData,
                     meshes: list[RenderMesh],
                     opts: RasterOpts,
                     save_path: Path | None
                     ) -> np.ndarray | None:

        depth_buff_subpx = 1e5*cam_data.image_dist*np.ones(cam_data.sub_samp*cam_data.pixels_num).T
        image_buff_subpx = np.full(cam_data.sub_samp*cam_data.pixels_num,0.0).T

        for mm in meshes:
            # coords_raster.shape=(num_coords,coord[x,y,z,w])
            # connect_in_frame.shape=(num_elems_in_scene,nodes_per_elem)
            # elem_bound_box_inds.shape=(num_elems_in_scene,4[x_min,x_max,y_min,y_max])
            # elem_areas.shape=(num_elems,)
            (coords_raster,
            connect_in_frame,
            elem_bound_box_inds,
            elem_areas) = RasterNP.setup_frame(
                cam_data,
                mm,
                frame_ind,
            )

            # NOTE: the z coord has already been inverted in setup so we multiply here
            render_field_div_z = (mm.fields_render[:,frame_ind,field_ind]
                                 *coords_raster[:,2])

            # image_buffer.shape=(num_px_y,num_px_x)
            # depth_buffer.shape=(num_px_y,num_px_x)
            (image_buff_subpx,
            depth_buff_subpx) = RasterNP.raster_one_mesh(
                connect_in_frame,
                coords_raster,
                elem_bound_box_inds,
                elem_areas,
                render_field_div_z,
                cam_data.sub_samp,
                image_buff_subpx,
                depth_buff_subpx
            )


        # TODO: make this configurable
        image_buff_subpx[depth_buff_subpx > 1000*cam_data.image_dist] = np.nan
        depth_buff_subpx[depth_buff_subpx > 1000*cam_data.image_dist] = np.nan

        # Average buffers
        (image_buff_avg,
         depth_buff_avg) = RasterNP.average_buffers(cam_data,
                                                    image_buff_subpx,
                                                    depth_buff_subpx)

        if save_path is None:
            return image_buff_avg

        save_name = ImageTools.get_save_name(cam_ind,frame_ind,field_ind)
        image_save_file = save_path/save_name
        save_raster(image_save_file,image_buff_avg,depth_buff_avg,opts)
        return None



    @staticmethod
    def raster_scene(scene: RenderScene,
                    opts: RasterOpts,
                    save_path: Path | None = None,
                    ) -> list[np.ndarray] | None:

        # TODO: we assume the number of frames and fields is the same per camera
        # Fix this
        frames_num = scene.meshes[0].fields_render.shape[1]
        field_num = scene.meshes[0].fields_render.shape[2]

        (cam_inds,
         frame_inds,
         field_inds) = np.meshgrid(np.arange(0,len(scene.cameras)),
                                    np.arange(0,frames_num),
                                    np.arange(0,field_num))

        cam_inds = cam_inds.flatten()
        frame_inds = frame_inds.flatten()
        field_inds = field_inds.flatten()
        frames_total = cam_inds.shape[0]


        if save_path is None:
            images = []
            for cc in scene.cameras:
                images.append(np.empty((cc.pixels_num[1],
                                        cc.pixels_num[0],
                                        frames_num,
                                        field_num)))
        else:
            images = None
            if not save_path.is_dir():
                save_path.mkdir()


        if opts.parallel is None:
            for ff in range(0,frames_total):
                image = RasterNP.raster_frame(
                    cam_inds[ff],
                    frame_inds[ff],
                    field_inds[ff],
                    scene.cameras[cam_inds[ff]],
                    scene.meshes,
                    opts,
                    save_path,
                )

                if images is not None:
                    images[cam_inds[ff]][:,:,frame_inds[ff],field_inds[ff]] = image

        else:
            with Pool(opts.parallel) as pool:
                processes_with_id = []

                for ff in range(0,frames_total):
                    args = (cam_inds[ff],
                            frame_inds[ff],
                            field_inds[ff],
                            scene.cameras[cam_inds[ff]],
                            scene.meshes,
                            opts,
                            save_path)

                    process = pool.apply_async(
                            RasterNP.raster_frame, args=args
                    )
                    processes_with_id.append({"process": process,
                                              "camera": cam_inds[ff],
                                              "frame": frame_inds[ff],
                                              "field": field_inds[ff]})

                for pp in processes_with_id:
                    image = pp["process"].get()
                    if images is not None:
                        images[cam_inds[ff]][:,:,pp["frame"],pp["field"]] = image

        if images is not None:
            return images

        return None



#-------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------



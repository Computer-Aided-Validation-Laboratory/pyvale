"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import time
import warnings
from pathlib import Path

import numpy as np
from shapely.geometry import Point
from scipy.signal import convolve2d
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
import matplotlib.image as mplim
from PIL import Image

from pyvale.imagesim.imagedefopts import ImageDefOpts
from pyvale.imagesim.cameradataimagedef import CameraImageDef
from pyvale.imagesim.alphashape import alphashape



def load_image(im_path: Path) -> np.ndarray:

    input_im = mplim.imread(im_path)
    input_im = input_im.astype(np.float64)

    # If we have RGB then get rid of it
    if input_im.ndim > 2:
        input_im = input_im[:,:,0]

    return input_im


def save_image(save_file: Path,
               image: np.ndarray,
               n_bits: int = 16) -> None:

    # Need to flip image so coords are top left with Y down
    image = image[::-1,:]

    if n_bits > 8:
        im = Image.fromarray(image.astype(np.uint16))
    else:
        im = Image.fromarray(image.astype(np.uint8))

    im.save(save_file)


def get_pixel_vec_in_m(camera: CameraImageDef) -> tuple[np.ndarray,np.ndarray]:
    mppx = camera.m_per_px
    px_vec_xm = np.arange(mppx/2,camera.fov[0],mppx)
    px_vec_ym = np.arange(mppx/2,camera.fov[1],mppx)
    #px_vec_ym = px_vec_ym[::-1] # flip
    return (px_vec_xm,px_vec_ym)


def get_pixel_grid_in_m(camera: CameraImageDef) -> tuple[np.ndarray,np.ndarray]:
    (px_vec_xm,px_vec_ym) = get_pixel_vec_in_m(camera)
    (px_grid_xm,px_grid_ym) = np.meshgrid(px_vec_xm,px_vec_ym)
    return (px_grid_xm,px_grid_ym)


def get_pixel_vec_in_px(camera: CameraImageDef) -> tuple[np.ndarray,np.ndarray]:
    px_vec_x = np.arange(0,camera.num_px[0],1)
    px_vec_y = np.arange(0,camera.num_px[1],1)
    px_vec_y = px_vec_y[::-1] # flip
    return (px_vec_x,px_vec_y)


def get_pixel_grid_in_px(camera: CameraImageDef) -> tuple[np.ndarray,np.ndarray]:
    (px_vec_x,px_vec_y) = get_pixel_vec_in_px(camera)
    (px_grid_x,px_grid_y) = np.meshgrid(px_vec_x,px_vec_y)
    return (px_grid_x,px_grid_y)


def get_subpixel_vec(camera: CameraImageDef, subsample: int = 2
                     ) -> tuple[np.ndarray,np.ndarray]:

    mppx = camera.m_per_px
    subpx_vec_xm = np.arange(mppx/(2*subsample),camera.fov[0],mppx/subsample)
    subpx_vec_ym = np.arange(mppx/(2*subsample),camera.fov[1],mppx/subsample)
    #subpx_vec_ym = subpx_vec_ym[::-1] #flip
    return (subpx_vec_xm,subpx_vec_ym)


def get_subpixel_grid(camera: CameraImageDef, subsample: int = 2
                     ) -> tuple[np.ndarray,np.ndarray]:

    (subpx_vec_xm,subpx_vec_ym) = get_subpixel_vec(camera,subsample)
    (subpx_grid_xm,subpx_grid_ym) = np.meshgrid(subpx_vec_xm,subpx_vec_ym)
    return (subpx_grid_xm,subpx_grid_ym)


def get_roi_node_vec(camera: CameraImageDef) -> tuple[np.ndarray,np.ndarray]:

    node_vec_x = np.arange(0+camera.roi_loc[0],
                           camera.roi_len[0]+camera.roi_loc[0]+camera.m_per_px/2,
                           camera.m_per_px)
    node_vec_y = np.arange(0+camera.roi_loc[1],
                           camera.roi_len[1]+camera.roi_loc[1]+camera.m_per_px/2,
                           camera.m_per_px)
    node_vec_y = node_vec_y[::-1] # flipud
    return (node_vec_x,node_vec_y)


def get_roi_node_grid(camera: CameraImageDef) -> tuple[np.ndarray,np.ndarray]:

    (node_vec_x,node_vec_y) = get_roi_node_vec(camera)
    (node_grid_x,node_grid_y) = np.meshgrid(node_vec_x,node_vec_y)
    return (node_grid_x,node_grid_y)


def calc_roi_from_nodes(camera: CameraImageDef, nodes: np.ndarray
                        ) -> tuple[np.ndarray,np.ndarray]:


    roi_len_x = np.max(nodes[:,0]) - np.min(nodes[:,0])
    roi_len_y = np.max(nodes[:,1]) - np.min(nodes[:,1])
    roi_len = np.array([roi_len_x,roi_len_y])

    if roi_len[0] > camera.fov[0] or roi_len[1] > camera.fov[1]:
        warnings.warn('ROI is larger than the cameras FOV\n')

    coord_offset = np.array([np.min(nodes[:,0]),np.min(nodes[:,1])])
    return (roi_len,coord_offset)


def calc_res_from_nodes(camera: CameraImageDef, nodes: np.ndarray, border_px: int
                        ) -> float:

    roi_len_x_m = np.max(nodes[:,0]) - np.min(nodes[:,0])
    roi_len_y_m = np.max(nodes[:,1]) - np.min(nodes[:,1])

    roi_len_x_px = camera.num_px[0] - 2*border_px
    roi_len_y_px = camera.num_px[1] - 2*border_px

    m_per_px_in_x = roi_len_x_m/roi_len_x_px
    m_per_px_in_y = roi_len_y_m/roi_len_y_px

    if m_per_px_in_x > m_per_px_in_y:
        m_per_px = m_per_px_in_x
    else:
        m_per_px = m_per_px_in_y

    return m_per_px


def norm_dynamic_range(image: np.ndarray, bits: int) -> np.ndarray:

    ret_image = image
    if bits > 8 and bits < 16:
        ret_image  = ((2**16)/(2**bits))*image
    elif bits < 8:
        raise ValueError('Camera cannot have less than an 8 bit dynamic range\n.')

    return ret_image


def get_image_num_str(im_num: int, width: int , cam_num: int = -1) -> str:

    num_str = str(im_num)
    num_str = num_str.zfill(width)

    if cam_num >= 0:
        num_str = num_str+'_'+str(cam_num)

    return num_str


def rectangle_crop_image(camera: CameraImageDef,
                         image: np.ndarray,
                         corner: tuple[int,int] = (0,0),
                         ) -> np.ndarray:

    if (camera.num_px[0] == image.shape[1] and
        camera.num_px[1] == image.shape[0]):
        return image

    if (corner[0]+camera.num_px[0]) > image.shape[1]:
        raise ValueError('Cannot crop image: '+
                         f'crop edge X of {corner[0]+camera.num_px[0]} is '+
                         'larger than '+
                         f'image size {image.shape[1]}\n.')
    if (corner[1]+camera.num_px[1]) > image.shape[0]:
        raise ValueError('Cannot crop image: '+
                         f'crop edge Y of {corner[1]+camera.num_px[1]} is '+
                         'larger than '+
                         f'image size {image.shape[0]}\n.')

    image = image[corner[1]:camera.num_px[1],corner[0]:camera.num_px[0]]
    return image


def get_im_mask_from_sim(camera: CameraImageDef,
                         image: np.ndarray,
                         nodes: np.ndarray
                         ) -> tuple[np.ndarray,np.ndarray]:

    # Create a mesh of pixel centroid locations
    (px_x_m,px_y_m) = get_pixel_grid_in_m(camera)

    # Convert to np array for compatibility with new alpha shape function
    points = np.array((nodes[:,0]+camera.roi_loc[0],
                       nodes[:,1]+camera.roi_loc[1])).T

    # Calculate the element edge length to use as the alpha radius
    elem_edge = 2*np.max(np.diff(np.sort(nodes[:,0])))
    alpha = elem_edge

    # Find the alpha shape based on the list of nodal points
    # Returns a shapely polygon - use 'within' to find points inside
    a_shape = alphashape(points, alpha, only_outer=True)

    # Create an array of nans to fill as the specimen image
    masked_im = np.empty(image.shape)
    masked_im[:] = np.nan
    im_mask = np.zeros([camera.num_px[1],camera.num_px[0]])

    # Fill the image based on the pixel being within the polygon (alpha-shape)
    # If pixel is not within the specimen set to background default colour
    for yy in range(masked_im.shape[0]):
        for xx in range(masked_im.shape[1]):
            if a_shape.contains(Point(px_x_m[yy,xx],px_y_m[yy,xx])):
                masked_im[yy,xx] = image[yy,xx]
                im_mask[yy,xx] = 1
            else:
                masked_im[yy,xx] = camera.background
                im_mask[yy,xx] = 0

    # Because arrays fill from top down the loop above effectively flips the
    # image, so need to flip it back
    masked_im = masked_im[::-1,:]
    im_mask = im_mask[::-1,:]

    # Return the image of the specimen
    return (masked_im,im_mask)


def upsample_image(camera: CameraImageDef,
                   id_opts: ImageDefOpts,
                   input_im: np.ndarray):
    # Get grid of pixel centroid locations
    (px_vec_xm,px_vec_ym) = get_pixel_vec_in_m(camera)

    # Get grid of sub-pixel centroid locations
    (subpx_vec_xm,subpx_vec_ym) = get_subpixel_vec(camera, id_opts.subsample)

    print(80*"=")
    print(f"{px_vec_xm.shape=}")
    print(f"{px_vec_ym.shape=}")
    print(px_vec_ym)
    print(80*"=")

    # NOTE: See Scipy transition from interp2d docs here:
    # https://scipy.github.io/devdocs/tutorial/interpolate/interp_transition_guide.html
    spline_interp = RectBivariateSpline(px_vec_xm,
                                        px_vec_ym,
                                        input_im.T)
    upsampled_image_interp = lambda x_new, y_new: spline_interp(x_new, y_new).T

    # This function will flip the image regardless of the y vector input so flip it
    # back to FE coords
    upsampled_image =  upsampled_image_interp(subpx_vec_xm,subpx_vec_ym)

    return upsampled_image


def average_subpixel_image(subpx_image,subsample):
    conv_mask = np.ones((subsample,subsample))/(subsample**2)
    if subsample > 1:
        subpx_image_conv = convolve2d(subpx_image,conv_mask,mode='same')
        avg_image = subpx_image_conv[round(subsample/2)-1::subsample,
                                     round(subsample/2)-1::subsample]
    else:
        subpx_image_conv = subpx_image
        avg_image = subpx_image

    return avg_image


def preprocess(input_im: np.ndarray,
               coords: np.ndarray,
                disp_x: np.ndarray,
                disp_y: np.ndarray,
                camera: CameraImageDef,
                id_opts: ImageDefOpts,
                print_on: bool = False
                ) -> tuple[np.ndarray,
                           np.ndarray|None,
                           np.ndarray,
                           np.ndarray,
                           np.ndarray]:

    if print_on:
        print('\n'+'='*80)
        print('IMAGE DEF PRE-PROCESSING\n')

    if not id_opts.save_path.is_dir():
        id_opts.save_path.mkdir()

    if disp_x.ndim == 1:
        disp_x = np.atleast_2d(disp_x).T
    if disp_y.ndim == 1:
        disp_y = np.atleast_2d(disp_y).T

    # This isn't needed for exodus because the first time step in the sim is 0
    if id_opts.add_static_ref:
        num_nodes = coords.shape[0] # type: ignore
        disp_x = np.hstack((np.zeros((num_nodes,1)),disp_x))
        disp_y = np.hstack((np.zeros((num_nodes,1)),disp_y))

    # Image cropping
    input_im = rectangle_crop_image(camera,input_im)

    # Image masking
    if id_opts.mask_input_image or id_opts.def_complex_geom:
        if print_on:
            print('Image masking or complex geometry on, getting image mask.')
            tic = time.perf_counter()

        (masked_im,image_mask) = get_im_mask_from_sim(camera,
                                                input_im,
                                                coords) # type: ignore
        if id_opts.mask_input_image:
            input_im = masked_im
        del masked_im

        if print_on:
            toc = time.perf_counter()
            print(f'Calculating image mask took {toc-tic:.4f} seconds')
    else:
        image_mask = None

    # Image upsampling
    if print_on:
        print('\n'+'-'*80)
        print('GENERATE UPSAMPLED IMAGE\n')
        print(f'Upsampling input image with a {id_opts.subsample}x{id_opts.subsample} subpixel')
        tic = time.perf_counter()

    upsampled_image = upsample_image(camera,id_opts,input_im)

    if print_on:
        toc = time.perf_counter()
        print(f'Upsampling image withtook {toc-tic:.4f} seconds')

    return (upsampled_image,image_mask,input_im,disp_x,disp_y)


def deform_one_image(upsampled_image: np.ndarray,
                    camera: CameraImageDef,
                    id_opts: ImageDefOpts,
                    coords: np.ndarray,
                    disp: np.ndarray,
                    image_mask: np.ndarray | None = None,
                    print_on: bool = True
                    ) -> tuple[np.ndarray,
                                np.ndarray,
                                np.ndarray,
                                np.ndarray,
                                np.ndarray | None]:

    if image_mask is not None:
        if (image_mask.shape[0] != camera.num_px[1]) or (image_mask.shape[1] != camera.num_px[0]):
            if image_mask.size == 0:
                warnings.warn('Image mask not specified, using default mask of ones.')
            else:
                warnings.warn('Image mask size does not match camera, using default mask of ones.')
            image_mask = np.ones([camera.num_px[1],camera.num_px[0]])

    # Get grid of pixel centroid locations
    (px_grid_xm,px_grid_ym) = get_pixel_grid_in_m(camera)
    # Get grid of sub-pixel centroid locations
    (subpx_grid_xm,subpx_grid_ym) = get_subpixel_grid(camera, id_opts.subsample)

    #--------------------------------------------------------------------------
    # Interpolate FE displacements onto the sub-pixel grid
    if print_on:
        print('Interpolating displacement onto sub-pixel grid.')
        tic = time.perf_counter()

    # Interpolate displacements onto sub-pixel locations - nan extrapolation
    subpx_disp_x = griddata((coords[:,0] + disp[:,0] + camera.roi_loc[0],
                             coords[:,1] + disp[:,1] + camera.roi_loc[1]),
                            disp[:,0],
                            (subpx_grid_xm,subpx_grid_ym),
                            method=id_opts.fe_interp,
                            fill_value=np.nan,
                            rescale=id_opts.fe_rescale)

    subpx_disp_y = griddata((coords[:,0] + disp[:,0] + camera.roi_loc[0],
                             coords[:,1] + disp[:,1] + camera.roi_loc[1]),
                            disp[:,1],
                            (subpx_grid_xm,subpx_grid_ym),
                            method=id_opts.fe_interp,
                            fill_value=np.nan,
                            rescale=id_opts.fe_rescale)

    # Ndimage interp can't handle nans so force everything outside the specimen
    # to extrapolate outside the FOV - then use ndimage opts to control
    if id_opts.fe_extrap_outside_fov:
        subpx_disp_ext_vals = 2*camera.fov
    else:
        subpx_disp_ext_vals = (0.0,0.0)

    # Set the nans to the extrapoltion value
    subpx_disp_x[np.isnan(subpx_disp_x)] = subpx_disp_ext_vals[0]
    subpx_disp_y[np.isnan(subpx_disp_y)] = subpx_disp_ext_vals[1]

    if print_on:
        toc = time.perf_counter()
        print('Interpolating displacement with NaN extrap took {:.4f} seconds'.format(toc-tic))

    #--------------------------------------------------------------------------
    # Interpolate sub-pixel gray levels with ndimage toolbox
    if print_on:
        print('Deforming sub-pixel image.')
        tic = time.perf_counter()

    # Use the sub-pixel displacements to deform the image
    def_subpx_x = subpx_grid_xm-subpx_disp_x
    def_subpx_y = subpx_grid_ym-subpx_disp_y
    # Flip needed to be consistent with pixel coords of ndimage
    def_subpx_x = def_subpx_x[::-1,:]
    def_subpx_y = def_subpx_y[::-1,:]

    # NDIMAGE: IMAGE DEF
    # NOTE: need to shift to pixel centroid co-ords from nodal so -0.5 makes the
    # top left 0,0 in pixel co-ords
    def_subpx_x_in_px = def_subpx_x*(id_opts.subsample/camera.m_per_px)-0.5
    def_subpx_y_in_px = def_subpx_y*(id_opts.subsample/camera.m_per_px)-0.5
    # NOTE: prefilter needs to be on to match griddata and interp2D!
    # with prefilter on this exactly matches I2D but 10x faster!
    def_image_subpx = ndimage.map_coordinates(upsampled_image,
                                            [[def_subpx_y_in_px],
                                             [def_subpx_x_in_px]],
                                            prefilter=True,
                                            order= id_opts.image_def_order,
                                            mode= id_opts.image_def_extrap,
                                            cval= id_opts.image_def_extval)

    def_image_subpx = def_image_subpx[0,:,:].squeeze()
    if print_on:
        toc = time.perf_counter()
        print('Deforming sub-pixel image with ndimage took {:.4f} seconds'.format(toc-tic))

    #--------------------------------------------------------------------------
    # Average subpixel image
    if print_on:
        tic = time.perf_counter()

    def_image = average_subpixel_image(def_image_subpx,id_opts.subsample)

    if print_on:
        toc = time.perf_counter()
        print('Averaging sub-pixel imagetook {:.4f} seconds'.format(toc-tic))

    #--------------------------------------------------------------------------
    # DEFORMING IMAGE MASK
    # Only need to do this if there are holes and notches
    if id_opts.def_complex_geom:
        if print_on:
            print('Deforming image mask.')
            tic = time.perf_counter()

        px_disp_x = subpx_disp_x[round(id_opts.subsample/2)-1::id_opts.subsample,
                                 round(id_opts.subsample/2)-1::id_opts.subsample]
        px_disp_y = subpx_disp_y[round(id_opts.subsample/2)-1::id_opts.subsample,
                                 round(id_opts.subsample/2)-1::id_opts.subsample]
        def_px_x = px_grid_xm-px_disp_x
        def_px_y = px_grid_ym-px_disp_y
        # Flip needed to be consistent with pixel coords of ndimage
        def_px_x = def_px_x[::-1,:]
        def_px_y = def_px_y[::-1,:]

        # NDIMAGE: DEFORM IMAGE MASK
        # NOTE: need to shift to pixel centroid co-ords from nodal so -0.5 makes the
        # top left 0,0 in pixel co-ords
        def_px_x_in_px = def_px_x*(1/camera.m_per_px)-0.5
        def_px_y_in_px = def_px_y*(1/camera.m_per_px)-0.5
        # NOTE: prefilter needs to be on to match griddata and interp2D!
        # with prefilter on this exactly matches I2D but 10x faster!
        def_mask = ndimage.map_coordinates(image_mask,
                                            [[def_px_y_in_px],
                                             [def_px_x_in_px]],
                                            prefilter=True,
                                            order=2,
                                            mode='constant',
                                            cval=0)

        def_mask = def_mask[0,:,:].squeeze()
        # Use the deformed image mask to mask the deformed image
        # Mask is 0-1 with 1 being definitely inside the sample 0 outside
        def_image[def_mask<0.51] = camera.background # type: ignore

        if print_on:
            toc = time.perf_counter()
            print('Deforming image mask with ndimage took {:.4f} seconds'.format(toc-tic))

    else:
        def_mask = None

    # Need to flip the image as all processing above is done with y a0s down
    # from the top left hand corner
    def_image = def_image[::-1,:]

    return (def_image,def_image_subpx,subpx_disp_x,subpx_disp_y,def_mask)


def deform_images(input_im: np.ndarray,
                 camera: CameraImageDef,
                 id_opts: ImageDefOpts,
                 coords: np.ndarray,
                 disp_x: np.ndarray,
                 disp_y: np.ndarray,
                 print_on: bool = False) -> None:
    #---------------------------------------------------------------------------
    # Image Pre-Processing
    (upsampled_image,
     image_mask,
     input_im,
     disp_x,
     disp_y) = preprocess(input_im,
                            coords,
                            disp_x,
                            disp_y,
                            camera,
                            id_opts,
                            print_on = print_on)

    #---------------------------------------------------------------------------
    # Image Deformation Loop
    if print_on:
        print('\n'+'='*80)
        print('DEFORMING IMAGES')

    num_frames = disp_x.shape[1]
    ticl = time.perf_counter()

    for ff in range(num_frames):
        if print_on:
            ticf = time.perf_counter()
            print(f'\nDEFORMING FRAME: {ff}')

        (def_image,_,_,_,_) = deform_one_image(upsampled_image,
                                            camera,
                                            id_opts,
                                            coords, # type: ignore
                                            np.array((disp_x[:,ff],disp_y[:,ff])).T,
                                            image_mask=image_mask,
                                            print_on=print_on)

        save_file = id_opts.save_path / str(f'{id_opts.save_tag}_'+
                f'{get_image_num_str(im_num=ff,width=4)}'+
                '.tiff')
        save_image(save_file,def_image,camera.bits)

        if print_on:
            tocf = time.perf_counter()
            print(f'DEFORMING FRAME: {ff} took {tocf-ticf:.4f} seconds')

    if print_on:
        tocl = time.perf_counter()
        print('\n'+'-'*50)
        print(f'Deforming all images took {tocl-ticl:.4f} seconds')
        print('-'*50)

        print('\n'+'='*80)
        print('COMPLETE\n')


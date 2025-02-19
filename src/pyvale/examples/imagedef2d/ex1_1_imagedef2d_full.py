"""
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""

import pickle
from pprint import pprint
from pathlib import Path

import numpy as np

import mooseherder as mh

from pyvale.imagesim.imagedefopts import ImageDefOpts
from pyvale.imagesim.cameradataimagedef import CameraImageDef
import pyvale.imagesim.imagedef as sid


def main() -> None:
    print()
    print('='*80)
    print('PYVALE EXAMPLE: IMAGE DEFORMATION 2D DETAILED')
    print('='*80)

    #---------------------------------------------------------------------------
    # Load image - expects a *.tiff or *.bmp that is grayscale
    im_path = Path('src/data/')
    im_file = 'optspeckle_2464x2056px_spec5px_8bit_gblur1px.tiff'
    im_path = im_path / im_file
    print('\nLoading speckle image from path:')
    print(im_path)

    input_im = sid.load_image(im_path)

    #---------------------------------------------------------------------------
    # Load simulation data - expects a mooseherder.SimData object
    # Read a pickled one or get one from an exodus
    read_exodus = True

    sim_path = Path.cwd()
    if read_exodus:
        sim_path = Path('src/data/')
        sim_file = 'case17_out.e'

        print(f'\nLoading SimData from exodus in path:\n{sim_path}')

        exodus_reader = mh.ExodusReader(sim_path / sim_file)
        sim_data = exodus_reader.read_all_sim_data()

    else:
        sim_path = Path('scripts/imdef_cases/imdefcase8_RampRigidBodyMotion_5_0px')
        sim_file = 'sim_data.pkl'

        print(f'\nLoading pickled SimData from path:\n{sim_path}')

        with open(sim_path / sim_file,'rb') as sim_load_file:
            sim_data = pickle.load(sim_load_file)

    coords = np.array(())
    disp_x = np.array(())
    disp_y = np.array(())

    if sim_data.coords is not None:
        coords = sim_data.coords
    if sim_data.node_vars is not None:
        disp_x = sim_data.node_vars['disp_x']
        disp_y = sim_data.node_vars['disp_y']
    del sim_data

    print(f'coords.shape={coords.shape}')
    print(f'disp_x.shape={disp_x.shape}')
    print(f'disp_y.shape={disp_y.shape}')

    #---------------------------------------------------------------------------
    # INIT IMAGE DEF OPTIONS AND CAMERA
    print('')
    print('='*80)
    print('INIT. IMAGE DEF. OPTIONS AND CAMERA')
    print('')

    #---------------------------------------------------------------------------
    # CREATE IMAGE DEF OPTS
    id_opts = ImageDefOpts()
    id_opts.save_path = sim_path / 'deformed_images'

    # If the input image is just a pattern then the image needs to be masked to
    # show just the sample geometry. This setting generates this image.
    id_opts.mask_input_image = True
    # Set this to True for holes and notches and False for a rectangle
    id_opts.def_complex_geom = False

    # If the input image is much larger than needed it can also be cropped to
    # increase computational speed.
    id_opts.crop_on = True
    id_opts.crop_px = np.array([500,500])

    # Calculates the m/px value based on fitting the specimen/ROI within the camera
    # FOV and leaving a set number of pixels as a border on the longest edge
    id_opts.calc_res_from_fe = True
    id_opts.calc_res_border_px = 10

    # Set this to true to create an undeformed masked image
    if read_exodus:
        id_opts.add_static_ref = 'off'
    else:
        id_opts.add_static_ref = 'pad_disp'

    print('-'*80)
    print('ImageDefOpts:')
    pprint(vars(id_opts))
    print('-'*80)
    print('')

    #---------------------------------------------------------------------------
    # CREATE CAMERA OBJECT
    camera = CameraImageDef()
    # Need to set the number of pixels in [X,Y], the bit depth and the m/px

    # Assume the camera has the same number of pixels as the input image unless we
    # are going to crop/mask the input image
    camera.num_px = np.array([input_im.shape[1],input_im.shape[0]])
    if id_opts.crop_on:
        camera.num_px = id_opts.crop_px

    # Based on the max grey level work out what the bit depth of the image is
    camera.bits = 8
    if max(input_im.flatten()) > (2**8):
        camera.bits = 16

    # Assume 1mm/px to start with, can update this to fit FE data within the FOV
    # using the id_opts above. Or set this manually.
    camera.m_per_px = 1.0e-3 # Overwritten by id_opts.calc_res_from_fe = True

    # Can manually set the ROI location by setting the above to false and setting
    # the camera.roi_loc as the distance from the origin to the bottom left
    # corner of the sample [X,Y]: camera.roi_loc = np.array([1e-3,1e-3])

    # Default ROI is the whole FOV but we want to set this to be based on the
    # furthest nodes, this is set in FE units 'meters' and does not change FOV
    (camera.roi_len,camera.coord_offset) = sid.calc_roi_from_nodes(camera,coords)

    # If we are masking an image we might want to set an optimal resolution based
    # on leaving a specified number of pixels free on each image edge, this will
    # change the FOV in 'meters'
    if id_opts.calc_res_from_fe:
        camera.m_per_px = sid.calc_res_from_nodes(camera,coords, #type: ignore
                                                id_opts.calc_res_border_px)

    # Default ROI is the whole FOV but we want to set this to be based on the
    # furthest nodes, this is set in FE units 'meters' and does not change FOV
    (camera.roi_len,camera.coord_offset) = sid.calc_roi_from_nodes(camera,coords)

    print('-'*80)
    print('CameraData:')
    pprint(vars(camera))
    print('-'*80)
    print('')

    #---------------------------------------------------------------------------
    # PRE-PROCESS AND DEFORM IMAGES
    sid.deform_images(input_im,
                    camera,
                    id_opts,
                    coords,
                    disp_x,
                    disp_y,
                    print_on = True)


if __name__ == "__main__":
    main()


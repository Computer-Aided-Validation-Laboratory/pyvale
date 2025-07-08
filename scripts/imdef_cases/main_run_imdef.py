"""
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
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
    print('Basics IMAGE DEFORMATION 2D MINIMAL')
    print('='*80)
    # This example assumes no experimental image is provided and a masked one
    # needs to be generated from a full speckled image using the simulation data.
    # This example also deals with complex geometry including holes and notches

    im_path = Path('data/speckleimages')
    im_file = 'OptimisedSpeckle_2464_2056_width5.0_8bit_GBlur1.tiff'
    im_path = im_path / im_file

    print(f'\nLoading image to deform from path:{im_path}\n')
    input_im = sid.load_image(im_path)

    sim_path = Path('scripts/imdef_cases')
    #sim_path = sim_path / 'imdefcase0_RandDispx10'
    #sim_path = sim_path / 'imdefcase1_RigidBodyMotion_4_1px'
    sim_path = sim_path / 'imdefcase2_RigidBodyMotion_0_1px'
    #sim_path = sim_path / 'imdefcase3_HydroStrain_1meta'
    #sim_path = sim_path / 'imdefcase4_HydroStrain_1pc'
    #sim_path = sim_path / 'imdefcase5_ShearStrain_1meta'
    #sim_path = sim_path / 'imdefcase6_ShearStrain_1pc'
    #sim_path = sim_path / 'imdefcase7_RampRigidBodyMotion_1_0px'
    sim_file = 'sim_data.pkl'

    print(f'\nLoading pickled sim data in path:\n{sim_path}')
    with open(sim_path / sim_file,'rb') as sim_load_file:
        sim_data = pickle.load(sim_load_file)

    print('\nSetting up image deformation inputs.\n')

    # We require the following variables to run the image deformation:
    # - input_im 2D numpy array of floats which is the input image to deform
    # - coords is a Nx2 numpy array where N is the number of nodes in the ROI
    #   with the nodal coords as [X,Y]
    # - disp_x is a NxF numpy array where N is the number of nodes and F is the
    #   is the number of frames/time steps with displacements in 'm'
    # - disp_y as above but the y directions displacements
    # - ImageDefOpts class - see below
    # - CameraData class - see below

    # Could also just extract a sideset here logical indexing
    if sim_data.coords is not None:
        coords = sim_data.coords
    if sim_data.node_vars is not None:
        disp_x = sim_data.node_vars['disp_x']
        disp_y = sim_data.node_vars['disp_y']

    del sim_data # don't need this anymore so can get rid of it

    id_opts = ImageDefOpts()
    id_opts.save_path = sim_path / 'deformed_images'
    id_opts.mask_input_image = True
    id_opts.def_complex_geom = True
    id_opts.crop_on = True
    id_opts.crop_px = np.array([1600,1000])
    id_opts.calc_res_from_fe = True
    id_opts.calc_res_border_px = 10
    id_opts.add_static_ref = 'off'

    print('\n'+'-'*80)
    print('ImageDefOpts:')
    pprint(vars(id_opts))
    print('-'*80+'\n')

    camera = CameraImageDef()
    camera.num_px = id_opts.crop_px
    camera.bits = 8
    if id_opts.calc_res_from_fe:
        camera.m_per_px = sid.calc_res_from_nodes(camera,coords, #type: ignore
                                             id_opts.calc_res_border_px)
    (camera.roi_len,camera.coord_offset) = sid.calc_roi_from_nodes(camera,coords)

    print('-'*80)
    print('CameraData:')
    pprint(vars(camera))
    print('-'*80)
    print('')

    #---------------------------------------------------------------------------
    # Pre-process and the deform images
    sid.deform_images(input_im,
                    camera,
                    id_opts,
                    coords,
                    disp_x,
                    disp_y,
                    print_on = True)


if __name__ == "__main__":
    main()


"""
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================

GENERATE IMAGE DEFORMATION TEST CASES
"""

from pprint import pprint
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from pyvale.mooseherder import SimData
import pyvale.imagesim.imagedefdiags as idd


# DEFINITIONS
DIAG_ON = True
(xi,yi) = (0,1)
# Colour map for images and vector fields
IM_CMAP = 'gray'
V_CMAP = 'plasma'

@dataclass
class TestCase:
    test_num: int = 1
    test_tag: str = ''
    test_val: float = 0.0
    geom: tuple[float,float] | None = None
    elem_size: tuple[float,float] | None = None
    m_per_px: float = 1.0e-3/5


def save_test_case(save_path: Path,test_case: TestCase,sim_data: SimData):
    print('')
    print('-'*80)
    print(f'TEST CASE {test_case.test_num}')
    print(f'Description: {test_case.test_tag}')
    print('')

    # Create new directory to save the test case to
    test_path = save_path / str(f'imdefcase{test_case.test_num}_{test_case.test_tag}')

    print('Saving test case data to:')
    print(f'{test_path}')
    if not test_path.is_dir():
        test_path.mkdir()

    # Pickle the synthetic sim_data class
    with open(test_path / 'sim_data.pkl','wb') as sim_save_file:
        pickle.dump(sim_data,sim_save_file)

    print('Saving data files complete.')


def main() -> None:
    #---------------------------------------------------------------------------
    print('')
    print('-'*80)
    print('GENERATE IMAGE DEFORMATION TEST CASES')
    print('-'*80)
    # Gets the directory of the current script file
    cwd = Path.cwd()
    print("Current working directory:")
    print(cwd)

    save_path = Path.cwd() / 'scripts' / 'imdef_cases'
    print("Save directory:")
    print(save_path)

    test_case = TestCase()
    pprint(vars(test_case))

    print('')
    print('-'*80)
    print('CREATING GENERIC TEST CASE DATA')
    print('')

    print('Creating generic test case parameters.')
    test_case = TestCase()
    test_case.geom = (78e-3,48e-3)
    test_case.elem_size = (0.25e-3,0.25e-3)
    test_case.m_per_px = 1.0e-3/5

    print('Creating generic SimData nodal array and zero frame data.')
    node_vec_x = np.arange(0,test_case.geom[xi]+test_case.elem_size[xi],
                        test_case.elem_size[xi])
    node_vec_y = np.arange(0,test_case.geom[yi]+test_case.elem_size[yi],
                        test_case.elem_size[yi])
    (node_grid_x,node_grid_y) = np.meshgrid(node_vec_x,node_vec_y)
    node_grid_y = node_grid_y[::-1,:] # flipud

    node_flat_x = np.atleast_2d(node_grid_x.flatten()).T
    node_flat_y = np.atleast_2d(node_grid_y.flatten()).T
    num_nodes = node_flat_x.shape[0]

    gen_data = SimData()
    gen_data.num_spat_dims = 2
    gen_data.coords = np.hstack((node_flat_x,node_flat_y))

    print('Creating generic 2D SimData object.')
    pprint(vars(gen_data))

    #---------------------------------------------------------------------------
    # TEST CASE 1: Large Rigid Body Motion
    test_case.test_num = 1
    test_case.test_tag = 'RigidBodyMotion_4_1px'
    test_case.test_val = 4.1

    # Create generic data object and unique params
    sim_data = gen_data
    sim_data.node_vars = dict()
    sim_data.node_vars['disp_x'] = np.ones((num_nodes,1)
                                           )*test_case.test_val*test_case.m_per_px
    sim_data.node_vars['disp_y'] = np.ones((num_nodes,1)
                                           )*test_case.test_val*test_case.m_per_px

    # Call function to perform general opts and save the pickle
    save_test_case(save_path,test_case,sim_data)

    #---------------------------------------------------------------------------
    # TEST CASE 2: Small Rigid Body Motion
    test_case.test_num = test_case.test_num+1
    test_case.test_tag = 'RigidBodyMotion_0_1px'
    test_case.test_val = 0.1

    sim_data = gen_data
    sim_data.node_vars = dict()
    sim_data.node_vars['disp_x'] = np.ones((num_nodes,1)
                                           )*test_case.test_val*test_case.m_per_px
    sim_data.node_vars['disp_y'] = np.ones((num_nodes,1)
                                           )*test_case.test_val*test_case.m_per_px

    # Call function to perform general opts and save the pickle
    save_test_case(save_path,test_case,sim_data)

    #---------------------------------------------------------------------------
    # TEST CASE 3: Small hydrostatic strain
    test_case.test_num = test_case.test_num+1
    test_case.test_tag = 'HydroStrain_1meta'
    test_case.test_val = -1e-3

    sim_data = gen_data
    sim_data.node_vars = dict()

    eta_h = test_case.test_val
    disp_vec_x = eta_h*(node_vec_x-test_case.geom[xi]/2)
    disp_vec_y = -eta_h*(node_vec_y-test_case.geom[yi]/2)
    (u_x,u_y) = np.meshgrid(disp_vec_x,disp_vec_y)

    if DIAG_ON:
        title =  f'{test_case.test_tag}, u_x'
        idd.plot_diag_image(title,u_x,V_CMAP)
        title =  f'{test_case.test_tag}, u_y'
        idd.plot_diag_image(title,u_y,V_CMAP)

    sim_data.node_vars['disp_x'] = np.atleast_2d(u_x.flatten()).T
    sim_data.node_vars['disp_y'] = np.atleast_2d(u_y.flatten()).T

    save_test_case(save_path,test_case,sim_data)

    #---------------------------------------------------------------------------
    # TEST CASE 4: Large hydrostatic strain
    test_case.test_num = test_case.test_num+1
    test_case.test_tag = 'HydroStrain_1pc'
    test_case.test_val = -1/100

    sim_data = gen_data
    sim_data.node_vars = dict()

    eta_h = test_case.test_val
    disp_vec_x = eta_h*(node_vec_x-test_case.geom[xi]/2)
    disp_vec_y = -eta_h*(node_vec_y-test_case.geom[yi]/2)
    (u_x,u_y) = np.meshgrid(disp_vec_x,disp_vec_y)

    if DIAG_ON:
        title =  f'{test_case.test_tag}, u_x'
        idd.plot_diag_image(title,u_x,V_CMAP)
        title =  f'{test_case.test_tag}, u_y'
        idd.plot_diag_image(title,u_y,V_CMAP)

    sim_data.node_vars['disp_x'] = u_x.flatten()
    sim_data.node_vars['disp_y'] = u_y.flatten()

    save_test_case(save_path,test_case,sim_data)

    #---------------------------------------------------------------------------
    # TEST CASE 5: Small Shear Strain
    test_case.test_num = test_case.test_num+1
    test_case.test_tag = 'ShearStrain_1meta'
    test_case.test_val = 1e-3

    sim_data = gen_data
    sim_data.node_vars = dict()

    etaS = test_case.test_val
    u_x = etaS*(node_grid_y-test_case.geom[xi]/2)
    u_y = etaS*(node_grid_x-test_case.geom[yi]/2)

    if DIAG_ON:
        title =  f'{test_case.test_tag}, u_x'
        idd.plot_diag_image(title,u_x,V_CMAP)
        title =  f'{test_case.test_tag}, u_y'
        idd.plot_diag_image(title,u_y,V_CMAP)

    sim_data.node_vars['disp_x'] = u_x.flatten()
    sim_data.node_vars['disp_y'] = u_y.flatten()

    save_test_case(save_path,test_case,sim_data)

    #---------------------------------------------------------------------------
    # TEST CASE 6: Large Shear Strain
    test_case.test_num = test_case.test_num+1
    test_case.test_tag = 'ShearStrain_1pc'
    test_case.test_val = 1/100

    sim_data = gen_data
    sim_data.node_vars = dict()

    etaS = test_case.test_val
    u_x = etaS*(node_grid_y-test_case.geom[xi]/2)
    u_y = etaS*(node_grid_x-test_case.geom[yi]/2)

    if DIAG_ON:
        title =  f'{test_case.test_tag}, u_x'
        idd.plot_diag_image(title,u_x,V_CMAP)
        title =  f'{test_case.test_tag}, u_y'
        idd.plot_diag_image(title,u_y,V_CMAP)

    sim_data.node_vars['disp_x'] = np.atleast_2d(u_x.flatten()).T
    sim_data.node_vars['disp_y'] = np.atleast_2d(u_y.flatten()).T

    save_test_case(save_path,test_case,sim_data)

    #---------------------------------------------------------------------------
    # TEST CASE 7: Rigid Body Motion RAMP
    # Linear ramp up to 1px of rigid body motion in X and Y
    test_case.test_num = test_case.test_num+1
    test_case.test_tag = 'RampRigidBodyMotion_1_0px'
    test_case.test_val = 1.0
    num_frames = 10

    sim_data = gen_data
    sim_data.node_vars = dict()

    # Create a vector of rigid displacements
    disp_max_m = test_case.test_val*test_case.m_per_px
    disp_inc_m = disp_max_m/num_frames
    disp_vec_m = np.arange(disp_inc_m,disp_max_m+disp_inc_m,disp_inc_m)

    disp_x = np.zeros([num_nodes,num_frames])
    disp_y = np.zeros([num_nodes,num_frames])
    for ff in range(num_frames):
        disp_x[:,ff] = disp_vec_m[ff]
        disp_y[:,ff] = disp_vec_m[ff]

    sim_data.node_vars['disp_x'] = disp_x
    sim_data.node_vars['disp_y'] = disp_y

    save_test_case(save_path,test_case,sim_data)

    #---------------------------------------------------------------------------
    # TEST CASE 8: Rigid Body Motion RAMP
    # Linear ramp up to 5px of rigid body motion in X and Y
    test_case.test_num = test_case.test_num+1
    test_case.test_tag = 'RampRigidBodyMotion_5_0px'
    test_case.test_val = 5.0
    num_frames = 10

    sim_data = gen_data
    sim_data.node_vars = dict()

    # Create a vector of rigid displacements
    disp_max_m = test_case.test_val*test_case.m_per_px
    disp_inc_m = disp_max_m/num_frames
    disp_vec_m = np.arange(disp_inc_m,disp_max_m+disp_inc_m,disp_inc_m)

    disp_x = np.zeros([num_nodes,num_frames])
    disp_y = np.zeros([num_nodes,num_frames])
    for ff in range(num_frames):
        disp_x[:,ff] = disp_vec_m[ff]
        disp_y[:,ff] = disp_vec_m[ff]

    sim_data.node_vars['disp_x'] = disp_x
    sim_data.node_vars['disp_y'] = disp_y

    save_test_case(save_path,test_case,sim_data)

    #---------------------------------------------------------------------------
    # TEST CASE 0: Random Disp x10
    num_frames = 10
    rng = default_rng(42)

    test_case.test_num = 0
    test_case.test_tag = 'RandDispx10'
    test_case.test_val = 1

    sim_data = gen_data
    sim_data.node_vars = dict()

    u_x = (rng.random([node_grid_x.shape[0],node_grid_x.shape[1],num_frames])-0.5
           )*2*test_case.test_val*test_case.m_per_px # type: ignore
    u_y = (rng.random([node_grid_x.shape[0],node_grid_x.shape[1],num_frames])-0.5
           )*2*test_case.test_val*test_case.m_per_px # type: ignore

    if DIAG_ON:
        title =  f'{test_case.test_tag}, u_x'
        idd.plot_diag_image(title,u_x[:,:,-1],V_CMAP)
        title =  f'{test_case.test_tag}, u_y'
        idd.plot_diag_image(title,u_y[:,:,-1],V_CMAP)

    disp_x = np.zeros([num_nodes,num_frames])
    disp_y = np.zeros([num_nodes,num_frames])
    for ff in range(num_frames):
        disp_x[:,ff] = u_x[:,:,ff].flatten()
        disp_y[:,ff] = u_y[:,:,ff].flatten()

    sim_data.node_vars['disp_x'] = disp_x
    sim_data.node_vars['disp_y'] = disp_y

    save_test_case(save_path,test_case,sim_data)

    #---------------------------------------------------------------------------
    # COMPLETE
    if DIAG_ON:
        plt.show()

    print('')
    print('-'*80)
    print('COMPLETE\n')

if __name__ == '__main__':
    main()


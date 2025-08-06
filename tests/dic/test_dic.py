"""
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
license: mit
copyright (c) 2024 the computer aided validation team
================================================================================
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import pyvale.dic as dic

test_dir = os.path.dirname(__file__)
ref_pattern = os.path.abspath(os.path.join(test_dir, "../../src/pyvale/data/plate_rigid_ref0000.tiff"))
def_pattern = os.path.abspath(os.path.join(test_dir, "../../src/pyvale/data/plate_rigid_def0000.tiff"))
roi = dic.RegionOfInterest(ref_image=ref_pattern)
roi.rect_region(x=200, y=200, size_x=100, size_y=100)



def test_image_scan_znssd_affine():
    dic.two_dimensional(reference=ref_pattern,
                               deformed=def_pattern,
                               roi_mask=roi.mask,
                               seed=[250,250],
                               subset_size=31,
                               subset_step=15,
                               max_displacement=2,
                               correlation_criteria="ZNSSD",
                               interpolation_routine="BICUBIC",
                               shape_function="AFFINE",
                               scanning_method="IMAGE_SCAN",
                               output_basepath=test_dir,
                               output_prefix="test_image_scan_znssd_affine_")

    ref_file = os.path.abspath(os.path.join(test_dir, "./reference/ref_image_scan_znssd_affine_plate_rigid.csv"))
    test_file = os.path.abspath(os.path.join(test_dir, "./test_image_scan_znssd_affine_plate_rigid_def0000.csv"))


    with open(ref_file) as f1, open(test_file) as f2:
        assert list(f1) == list(f2)

    os.remove(test_file)

def test_image_scan_znssd_rigid():
    dic.two_dimensional(reference=ref_pattern,
                               deformed=def_pattern,
                               roi_mask=roi.mask,
                               seed=[250,250],
                               subset_size=31,
                               subset_step=15,
                               max_displacement=2,
                               correlation_criteria="ZNSSD",
                               interpolation_routine="BICUBIC",
                               shape_function="RIGID",
                               scanning_method="IMAGE_SCAN",
                               output_basepath=test_dir,
                               output_prefix="test_image_scan_znssd_rigid_")

    ref_file = os.path.abspath(os.path.join(test_dir, "./reference/ref_image_scan_znssd_rigid_plate_rigid.csv"))
    test_file = os.path.abspath(os.path.join(test_dir, "./test_image_scan_znssd_rigid_plate_rigid_def0000.csv"))


    with open(ref_file) as f1, open(test_file) as f2:
        assert list(f1) == list(f2)

    os.remove(test_file)

def test_image_scan_nssd_affine():
    dic.two_dimensional(reference=ref_pattern,
                               deformed=def_pattern,
                               roi_mask=roi.mask,
                               seed=[250,250],
                               subset_size=31,
                               subset_step=15,
                               max_displacement=2,
                               correlation_criteria="NSSD",
                               interpolation_routine="BICUBIC",
                               shape_function="AFFINE",
                               scanning_method="IMAGE_SCAN",
                               output_basepath=test_dir,
                               output_prefix="test_image_scan_nssd_affine_")

    ref_file = os.path.abspath(os.path.join(test_dir, "./reference/ref_image_scan_nssd_affine_plate_rigid.csv"))
    test_file = os.path.abspath(os.path.join(test_dir, "./test_image_scan_nssd_affine_plate_rigid_def0000.csv"))


    with open(ref_file) as f1, open(test_file) as f2:
        assert list(f1) == list(f2)

    os.remove(test_file)

def test_rg_znssd_affine():
    dic.two_dimensional(reference=ref_pattern,
                               deformed=def_pattern,
                               roi_mask=roi.mask,
                               seed=[250,250],
                               subset_size=31,
                               subset_step=15,
                               max_displacement=2,
                               correlation_criteria="ZNSSD",
                               interpolation_routine="BICUBIC",
                               shape_function="AFFINE",
                               scanning_method="RG",
                               output_basepath=test_dir,
                               output_prefix="test_rg_znssd_affine_")

    ref_file = os.path.abspath(os.path.join(test_dir, "./reference/ref_rg_znssd_affine_plate_rigid.csv"))
    test_file = os.path.abspath(os.path.join(test_dir, "./test_rg_znssd_affine_plate_rigid_def0000.csv"))


    with open(ref_file) as f1, open(test_file) as f2:
        assert list(f1) == list(f2)


    os.remove(test_file)

def test_fft_znssd_affine():
    dic.two_dimensional(reference=ref_pattern,
                               deformed=def_pattern,
                               roi_mask=roi.mask,
                               seed=[250,250],
                               subset_size=31,
                               subset_step=15,
                               max_displacement=2,
                               correlation_criteria="ZNSSD",
                               interpolation_routine="BICUBIC",
                               shape_function="AFFINE",
                               scanning_method="FFT",
                               output_basepath=test_dir,
                               output_prefix="test_fft_znssd_affine_")
    
    ref_file = os.path.abspath(os.path.join(test_dir, "./reference/ref_fft_znssd_affine_plate_rigid.csv"))
    test_file = os.path.abspath(os.path.join(test_dir, "./test_fft_znssd_affine_plate_rigid_def0000.csv"))


    with open(ref_file) as f1, open(test_file) as f2:
        assert list(f1) == list(f2)


    os.remove(test_file)



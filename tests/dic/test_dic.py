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

from PIL import Image
import numpy as np
import pyvale.dic as dic
import pyvale.dataset as pyv_data
from pathlib import Path

test_dir = os.path.dirname(__file__)

ref_pattern = pyv_data.dic_plate_rigid_ref()
def_pattern = pyv_data.dic_plate_rigid_def()
def_pattern_25px = pyv_data.dic_plate_rigid_def_25px()
def_pattern_50px = pyv_data.dic_plate_rigid_def_50px()
def_large = [def_pattern_25px, def_pattern_50px]

roi = dic.RegionOfInterest(ref_image=ref_pattern)
roi.rect_region(x=200, y=200, size_x=200, size_y=200)



true_file_00_5 = os.path.abspath(os.path.join(test_dir, "./reference/ref_00_50.csv"))
true_file_01_0 = os.path.abspath(os.path.join(test_dir, "./reference/ref_01_00.csv"))
true_file_25_0 = os.path.abspath(os.path.join(test_dir, "./reference/ref_25_00.csv"))
true_file_50_0 = os.path.abspath(os.path.join(test_dir, "./reference/ref_50_00.csv"))

true_00_5 = np.loadtxt(true_file_00_5, skiprows=1, delimiter=',')
true_01_0 = np.loadtxt(true_file_01_0, skiprows=1, delimiter=',')
true_25_0 = np.loadtxt(true_file_25_0, skiprows=1, delimiter=',')
true_50_0 = np.loadtxt(true_file_50_0, skiprows=1, delimiter=',')




# Create a deformed image where intensities are scaled/offset
ref_image = Image.open(ref_pattern)
files = list(def_pattern.parent.glob(def_pattern.name))
files = sorted(files)
def_image = Image.open(files[0])
ref_arr = np.array(ref_image)
def_arr = np.array(def_image)
original_dtype = def_arr.dtype
scale = 0.5
offset = 50
def_arr = def_arr.astype(np.float32)
def_arr_scaled = def_arr * scale
def_arr_scaled_offset = def_arr * scale + offset
def_arr = def_arr.astype(original_dtype)
def_arr_scaled = def_arr_scaled.astype(original_dtype)
def_arr_scaled_offset = def_arr_scaled_offset.astype(original_dtype)


def test_ssd_rigid():
    dic.calculate_2d(reference=ref_arr,
                     deformed=def_arr,
                     roi_mask=roi.mask,
                     seed=[250,250],
                     subset_size=31,
                     subset_step=15,
                     max_displacement=2,
                     correlation_criteria="SSD",
                     shape_function="AFFINE",
                     method="MULTIWINDOW_RG",
                     output_basepath=test_dir,
                     output_prefix="test_ssd_rigid_")

    output_file_00_5 = os.path.abspath(os.path.join(test_dir, "./test_ssd_rigid_def_img_0000.csv"))
    output_data_00_5 = np.loadtxt(output_file_00_5, skiprows=1, delimiter=',')
    np.testing.assert_allclose(true_00_5[:, :6], output_data_00_5[:, :6], rtol=0.005, atol=1e-6,
                               err_msg="Results from test Do not match ground truth displacement of 0.5px")

    os.remove(output_file_00_5)

def test_nssd_scaled_image_rigid():
    dic.calculate_2d(reference=ref_arr,
                     deformed=def_arr_scaled,
                     roi_mask=roi.mask,
                     seed=[250,250],
                     subset_size=31,
                     subset_step=15,
                     max_displacement=2,
                     correlation_criteria="NSSD",
                     shape_function="AFFINE",
                     method="MULTIWINDOW_RG",
                     output_basepath=test_dir,
                     output_prefix="test_nssd_scaled_image_rigid_")

    output_file_00_5 = os.path.abspath(os.path.join(test_dir, "./test_nssd_scaled_image_rigid_def_img_0000.csv"))
    output_data_00_5 = np.loadtxt(output_file_00_5, skiprows=1, delimiter=',')
    np.testing.assert_allclose(true_00_5[:, :7], output_data_00_5[:, :7], rtol=0.005, atol=1e-6,
                               err_msg="Results from test Do not match ground truth displacement of 0.5px")
    os.remove(output_file_00_5)

def test_znssd_scaled_offset_image_rigid():
    dic.calculate_2d(reference=ref_arr,
                     deformed=def_arr_scaled_offset,
                     roi_mask=roi.mask,
                     seed=[250,250],
                     subset_size=31,
                     subset_step=15,
                     max_displacement=2,
                     correlation_criteria="ZNSSD",
                     shape_function="AFFINE",
                     method="MULTIWINDOW_RG",
                     output_basepath=test_dir,
                     output_prefix="test_znssd_scaled_offset_image_rigid_")

    output_file_00_5 = os.path.abspath(os.path.join(test_dir, "./test_znssd_scaled_offset_image_rigid_def_img_0000.csv"))
    output_data_00_5 = np.loadtxt(output_file_00_5, skiprows=1, delimiter=',')
    np.testing.assert_allclose(true_00_5[:, :7], output_data_00_5[:, :7], rtol=0.005, atol=1e-6,
                               err_msg="Results from test Do not match ground truth displacement of 0.5px")
    os.remove(output_file_00_5)

def test_image_scan_znssd_affine():
    dic.calculate_2d(reference=ref_pattern,
                     deformed=def_pattern,
                     roi_mask=roi.mask,
                     seed=[250,250],
                     subset_size=31,
                     subset_step=15,
                     max_displacement=2,
                     correlation_criteria="ZNSSD",
                     interpolation_routine="BICUBIC",
                     shape_function="AFFINE",
                     method="IMAGE_SCAN",
                     output_basepath=test_dir,
                     output_prefix="test_image_scan_znssd_affine_")

    output_file_00_5 = os.path.abspath(os.path.join(test_dir, "./test_image_scan_znssd_affine_plate_rigid_def0000.csv"))
    output_file_01_0 = os.path.abspath(os.path.join(test_dir, "./test_image_scan_znssd_affine_plate_rigid_def0001.csv"))
    output_data_00_5 = np.loadtxt(output_file_00_5, skiprows=1, delimiter=',')
    output_data_01_0 = np.loadtxt(output_file_01_0, skiprows=1, delimiter=',')

    np.testing.assert_allclose(true_00_5[:, :7], output_data_00_5[:, :7], rtol=0.005, atol=1e-6,
                               err_msg="Results from test Do not match ground truth displacement of 0.5px")

    np.testing.assert_allclose(true_01_0[:, :7], output_data_01_0[:, :7], rtol=0.005, atol=1e-6,
                               err_msg="Results from test Do not match ground truth displacement of 1.0px")

    os.remove(output_file_00_5)
    os.remove(output_file_01_0)

def test_image_scan_znssd_rigid():
    dic.calculate_2d(reference=ref_pattern,
                     deformed=def_pattern,
                     roi_mask=roi.mask,
                     seed=[250,250],
                     subset_size=31,
                     subset_step=15,
                     max_displacement=2,
                     correlation_criteria="ZNSSD",
                     interpolation_routine="BICUBIC",
                     shape_function="RIGID",
                     method="IMAGE_SCAN",
                     output_basepath=test_dir,
                     output_prefix="test_image_scan_znssd_rigid_")

    output_file_00_5 = os.path.abspath(os.path.join(test_dir, "./test_image_scan_znssd_rigid_plate_rigid_def0000.csv"))
    output_file_01_0 = os.path.abspath(os.path.join(test_dir, "./test_image_scan_znssd_rigid_plate_rigid_def0001.csv"))
    output_data_00_5 = np.loadtxt(output_file_00_5, skiprows=1, delimiter=',')
    output_data_01_0 = np.loadtxt(output_file_01_0, skiprows=1, delimiter=',')

    np.testing.assert_allclose(true_00_5[:, :7], output_data_00_5[:, :7], rtol=0.005, atol=1e-6,
                               err_msg="Results from test Do not match ground truth displacement of 0.5px")

    np.testing.assert_allclose(true_01_0[:, :7], output_data_01_0[:, :7], rtol=0.005, atol=1e-6,
                               err_msg="Results from test Do not match ground truth displacement of 1.0px")

    os.remove(output_file_00_5)
    os.remove(output_file_01_0)

def test_image_scan_nssd_affine():
    dic.calculate_2d(reference=ref_pattern,
                     deformed=def_pattern,
                     roi_mask=roi.mask,
                     seed=[250,250],
                     subset_size=31,
                     subset_step=15,
                     max_displacement=2,
                     correlation_criteria="NSSD",
                     interpolation_routine="BICUBIC",
                     shape_function="AFFINE",
                     method="IMAGE_SCAN",
                     output_basepath=test_dir,
                     output_prefix="test_image_scan_nssd_affine_")

    output_file_00_5 = os.path.abspath(os.path.join(test_dir, "./test_image_scan_nssd_affine_plate_rigid_def0000.csv"))
    output_file_01_0 = os.path.abspath(os.path.join(test_dir, "./test_image_scan_nssd_affine_plate_rigid_def0001.csv"))
    output_data_00_5 = np.loadtxt(output_file_00_5, skiprows=1, delimiter=',')
    output_data_01_0 = np.loadtxt(output_file_01_0, skiprows=1, delimiter=',')

    np.testing.assert_allclose(true_00_5[:, :7], output_data_00_5[:, :7], rtol=0.005, atol=1e-6,
                               err_msg="Results from test Do not match ground truth displacement of 0.5px")

    np.testing.assert_allclose(true_01_0[:, :7], output_data_01_0[:, :7], rtol=0.005, atol=1e-6,
                               err_msg="Results from test Do not match ground truth displacement of 1.0px")

    os.remove(output_file_00_5)
    os.remove(output_file_01_0)

def test_rg_znssd_affine():
    dic.calculate_2d(reference=ref_pattern,
                     deformed=def_pattern,
                     roi_mask=roi.mask,
                     seed=[250,250],
                     subset_size=31,
                     subset_step=15,
                     max_displacement=2,
                     correlation_criteria="ZNSSD",
                     interpolation_routine="BICUBIC",
                     shape_function="AFFINE",
                     method="MULTIWINDOW_RG",
                     output_basepath=test_dir,
                     output_prefix="test_rg_znssd_affine_")

    output_file_00_5 = os.path.abspath(os.path.join(test_dir, "./test_rg_znssd_affine_plate_rigid_def0000.csv"))
    output_file_01_0 = os.path.abspath(os.path.join(test_dir, "./test_rg_znssd_affine_plate_rigid_def0001.csv"))
    output_data_00_5 = np.loadtxt(output_file_00_5, skiprows=1, delimiter=',')
    output_data_01_0 = np.loadtxt(output_file_01_0, skiprows=1, delimiter=',')

    np.testing.assert_allclose(true_00_5[:, :7], output_data_00_5[:, :7], rtol=0.005, atol=1e-6,
                               err_msg="Results from test Do not match ground truth displacement of 0.5px")

    np.testing.assert_allclose(true_01_0[:, :7], output_data_01_0[:, :7], rtol=0.005, atol=1e-6,
                               err_msg="Results from test Do not match ground truth displacement of 1.0px")

    os.remove(output_file_00_5)
    os.remove(output_file_01_0)

def test_fft_large():
    dic.calculate_2d(reference=ref_pattern,
                     deformed=def_large,
                     roi_mask=roi.mask,
                     seed=[250,250],
                     subset_size=31,
                     subset_step=15,
                     max_displacement=100,
                     correlation_criteria="ZNSSD",
                     interpolation_routine="BICUBIC",
                     shape_function="RIGID",
                     method="MULTIWINDOW",
                     output_basepath=test_dir,
                     output_prefix="test_fft_")

    output_file_25_0 = os.path.abspath(os.path.join(test_dir, "./test_fft_plate_rigid_def_25px.csv"))
    output_file_50_0 = os.path.abspath(os.path.join(test_dir, "./test_fft_plate_rigid_def_50px.csv"))
    output_data_25_0 = np.loadtxt(output_file_25_0, skiprows=1, delimiter=',')
    output_data_50_0 = np.loadtxt(output_file_50_0, skiprows=1, delimiter=',')

    np.testing.assert_allclose(true_25_0[:, :6], output_data_25_0[:, :6], rtol=0.01, atol=1e-6,
                               err_msg="Results from test Do not match ground truth displacement of 25px")

    np.testing.assert_allclose(true_50_0[:, :6], output_data_50_0[:, :6], rtol=0.01, atol=1e-6,
                               err_msg="Results from test Do not match ground truth displacement of 50px")

    os.remove(output_file_25_0)
    os.remove(output_file_50_0)



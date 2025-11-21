"""
================================================================================
Example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'


import pyvale
ref_pattern = "../../src/pyvale/data/plate_rigid_ref0000.tiff"
def_pattern = "../../src/pyvale/data/plate_rigid_def0000.tiff"
roi = pyvale.dic.RegionOfInterest(ref_image=ref_pattern)
roi.rect_region(x=200,y=200,size_x=100,size_y=100)

pyvale.dic.calculate_2d(reference=ref_pattern,
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
            output_prefix="test_image_scan_znssd_affine_")

pyvale.dic.calculate_2d(reference=ref_pattern,
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
            output_prefix="test_image_scan_znssd_rigid_")

pyvale.dic.calculate_2d(reference=ref_pattern,
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
            output_prefix="test_image_scan_nssd_affine_")

pyvale.dic.calculate_2d(reference=ref_pattern,
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
            output_prefix="test_rg_znssd_affine_")

pyvale.dic.calculate_2d(reference=ref_pattern,
            deformed=def_pattern,
            roi_mask=roi.mask,
            seed=[250,250],
            subset_size=31,
            subset_step=15,
            max_displacement=2,
            correlation_criteria="ZNSSD",
            interpolation_routine="BICUBIC",
            shape_function="AFFINE",
            method="MULTIWINDOW",
            output_prefix="test_fft_znssd_affine_")

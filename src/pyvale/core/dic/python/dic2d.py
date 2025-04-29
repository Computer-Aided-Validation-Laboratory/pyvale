"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""


import numpy as np

# import cython module
from pyvale.core.dic.python import diccppinterface
from pyvale.core.dic.python.dicresults import DICResults


class DIC2D:


    def __init__(self, 
                 reference_image: np.ndarray,
                 deformed_images: np.ndarray,
                 roi_mask: np.ndarray,
                 subset_step: int=10, 
                 subset_size: int=21,
                 correlation_criteria: str="ZNSSD",
                 shape_function: str="affine",
                 interpolation_routine: str="bicubic",
                 max_iterations: int=100,
                 precision: float=0.001,
                 threshold_levenberg: float=0.1,
                 threshold_bruteforce: float=0.2,
                 range_bruteforce: int=10,
                 scanning_method: str="image_scan",
                 out_format: str=".dat",
                 out_layout: str="col",
                 out_base_path: str="./",
                 out_prefix: str="image",
                 out_delimiter: str=" "):

        self.image_ref = reference_image
        self.image_def = deformed_images
        self.roi_mask  = roi_mask
        self.subset_step = subset_step
        self.subset_size = subset_size
        self.max_iterations = max_iterations
        self.precision = precision
        self.threshold_levenberg = threshold_levenberg
        self.threshold_bruteforce = threshold_bruteforce
        self.range_bruteforce = range_bruteforce
        self.corr_crit = correlation_criteria
        self.shape_func = shape_function
        self.interp = interpolation_routine
        self.scanning_method = scanning_method
        self.out_format = out_format
        self.out_layout = out_layout
        self.out_base_path = out_base_path
        self.out_prefix = out_prefix
        self.out_delimiter = out_delimiter


        self.subsets = None
        self.u = None
        self.v = None
        self.p = None
        self.ftol = None
        self.xtol = None
        self.niter = None





    def execute_cpu(self) -> DICResults:
        """
        Executes the c++ 2D DIC routine on CPU architecture.
        """
        results = diccppinterface.cpp_2d_dic_routine(self.image_ref,
                                           self.image_def,
                                           self.roi_mask,
                                           self.subset_step,
                                           self.subset_size,
                                           self.max_iterations,
                                           self.precision,
                                           self.threshold_levenberg,
                                           self.threshold_bruteforce,
                                           self.range_bruteforce,
                                           self.corr_crit,
                                           self.shape_func,
                                           self.interp,
                                           self.scanning_method,
                                           self.out_format,
                                           self.out_layout,
                                           self.out_base_path,
                                           self.out_prefix,
                                           self.out_delimiter)

        #self.subsets = results[0]
        self.niter = results[0]
        self.u = results[1]
        self.v = results[2]
        self.p = results[3]
        self.ftol = results[4]
        self.xtol = results[5]



        


    def execute_gpu(self):
        """
        Executes the c++ 2D DIC routine on GPU architecture.
        """

        print("This is a work in progress...")

        return None


    def build_info(self):
        """
        Returns the build information of the diccppinterface module.
        """
        build = diccppinterface.build_info();
        return build;

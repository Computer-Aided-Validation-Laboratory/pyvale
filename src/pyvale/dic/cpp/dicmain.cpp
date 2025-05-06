// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <cstring>
#include <omp.h>


// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Program Header files
#include "./dicinterpolator.hpp"
#include "./dicbruteforce.hpp"
#include "./dicoptimizer.hpp"
#include "./dicscanmethod.hpp"
#include "./defines.hpp"
#include "./dicutil.hpp"

// cuda Header files
#include "../cuda/malloc.hpp"

namespace py = pybind11;


void engine(const py::array_t<double>& image_ref_arr,
            const py::array_t<double>& image_def_stack_arr,
            const py::array_t<bool>&   image_roi_arr, 
            util::Config &conf){

    // -------------------------------------------------------------------------------------------
    // Initialisation
    // -------------------------------------------------------------------------------------------


    // get raw pointers
    bool* image_roi = static_cast<bool*>(image_roi_arr.request().ptr);
    double* image_ref = static_cast<double*>(image_ref_arr.request().ptr);
    double* image_def_stack = static_cast<double*>(image_def_stack_arr.request().ptr);
    
    // get a list of ss coordinates within RIO.
    util::SubsetData ssdata = util::generate_ss_list(image_roi, conf);


    // TITLE("DIC INITIALISATION");
    INFO_OUT("Width of Images: ", conf.px_horizontal << " [px]");
    INFO_OUT("Height of Images: ", conf.px_vertical << " [px]");
    INFO_OUT("Number of Deformed Images: ", conf.num_def_images);
    INFO_OUT("Subset Step: ", conf.ss_step);
    INFO_OUT("Subset Size: ", conf.ss_size);
    INFO_OUT("Max number of solver iterations: ", conf.max_iter);
    INFO_OUT("Correlation Criterion: ", conf.corr_crit);
    INFO_OUT("Shape Function: ", conf.shape_func);
    INFO_OUT("Interpolation Routine: ", conf.interp_routine);
    INFO_OUT("Image Scan Method: ", conf.scan_method);
    INFO_OUT("Total number of subsets: ", ssdata.num);
    INFO_OUT("Number of OMP threads:", omp_get_max_threads());

    // define our interpolator for the reference image
    interpolator::bicubic_init(image_ref, conf.px_horizontal, conf.px_vertical);

    // initialise the LM optimizer to use the desired correlation criterion and shape func.
    optimizer::init(conf.corr_crit, conf.shape_func);

    // initialise the brute force scan
    std::string brute_method = "SPIRAL";
    brute::init(conf.corr_crit, brute_method);

    // function pointer for the method of scanning the subsets through the image
    void (*scan_ptr)(double *, double *, bool *, util::SubsetData &, util::Config &, int);

    // set the scan_function pointer based on the scan method specified by user.
    if (conf.scan_method=="IMAGE_SCAN") scan_ptr=scanmethod::image;
    else if (conf.scan_method=="IMAGE_SCAN_WITH_BF") scan_ptr=scanmethod::image_with_bf;
    else if (conf.scan_method=="RG") scan_ptr=scanmethod::reliability_guided;
    else {
        std::cerr << "Unknown subset scan type: \'" << conf.scan_method << "\'." << std::endl;
        std::cerr << "Allowed values: \'IMAGE_SCAN\', \'IMAGE_SCAN_WITH_BF\', \'RG\'. " << std::endl;
        return;
    } 

    // -----------------------------------------------------------------------
    // loop over deformed images and perform DIC
    // -----------------------------------------------------------------------
    util::Timer timer("DIC Engine");

    for (int img_num = 0; img_num < conf.num_def_images; img_num++){


        // pointer to starting location of image
        int num_px_in_image = conf.px_horizontal * conf.px_vertical;
        double *image_def = image_def_stack + img_num*num_px_in_image;

        scan_ptr(image_ref, image_def, image_roi, ssdata, conf, img_num);

    }

    // save data
    util::SaveConfig saveconf;
    saveconf.base_path = "./output/";
    saveconf.delimiter = " ";
    saveconf.format = ".dat";
    saveconf.prefix = "results";
    saveconf.layout = "col";
    saveconf.save_at_end = true;

    util::save_to_disk(&saveconf,
                       conf.num_def_images, 
                       &ssdata, conf.num_params);

}






PYBIND11_MODULE(dic2dcpp, m) {
    py::class_<util::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("ss_step", &util::Config::ss_step)
        .def_readwrite("ss_size", &util::Config::ss_size)
        .def_readwrite("max_iter", &util::Config::max_iter)
        .def_readwrite("precision", &util::Config::precision)
        .def_readwrite("threshold_lm", &util::Config::threshold_lm)
        .def_readwrite("threshold_bf", &util::Config::threshold_bf)
        .def_readwrite("range_bf", &util::Config::range_bf)
        .def_readwrite("corr_crit", &util::Config::corr_crit)
        .def_readwrite("shape_func", &util::Config::shape_func)
        .def_readwrite("interp_routine", &util::Config::interp_routine)
        .def_readwrite("scan_method", &util::Config::scan_method)
        .def_readwrite("px_horizontal", &util::Config::px_horizontal)
        .def_readwrite("px_vertical", &util::Config::px_vertical)
        .def_readwrite("num_def_images", &util::Config::num_def_images)
        .def_readwrite("num_params", &util::Config::num_params);

    // Bind the engine function
    m.def("engine", &engine, "Run 2D analysis on input images with config");
}





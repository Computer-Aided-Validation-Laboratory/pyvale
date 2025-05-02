// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <cstring>
#include <chrono>
#include <omp.h>


// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Program Header files
#include "./dicinterpolator.hpp"
#include "./dicbruteforce.hpp"
#include "./dicoptimizer.hpp"
#include "./dicmain.hpp"
#include "./defines.hpp"
#include "./dicutil.hpp"
#include "./dicrg.hpp"

// cuda Header files
#include "../cuda/malloc.hpp"

namespace py = pybind11;


void engine(const py::array_t<double>& image_ref_arr,
            const py::array_t<double>& image_def_stack_arr,
            const py::array_t<bool>&   image_roi_arr, 
            Config &conf){

    // -------------------------------------------------------------------------------------------
    // Initialisation
    // -------------------------------------------------------------------------------------------

    auto s0 = std::chrono::high_resolution_clock::now();
    int px_horizontal = static_cast<int>(image_ref_arr.shape(1));
    int px_vertical   = static_cast<int>(image_ref_arr.shape(0));
    int num_def_images  = static_cast<int>(image_def_stack_arr.shape(0));

    // get raw pointers
    bool* image_roi = static_cast<bool*>(image_roi_arr.request().ptr);
    double* image_ref = static_cast<double*>(image_ref_arr.request().ptr);
    double* image_def_stack = static_cast<double*>(image_def_stack_arr.request().ptr);
    
    // number of parameters for the shape function
    int num_params = util::get_num_params(conf.shape_func);

    // get a list of ss coordinates within RIO.
    util::SubsetData ssdata = util::generate_ss_list(image_roi, px_horizontal, px_vertical, conf.ss_size, conf.ss_step, num_def_images, num_params);


    // TITLE("DIC INITIALISATION");
    INFO_OUT("Height of Images: ", px_vertical << " [px]");
    INFO_OUT("Width of Images: ", px_horizontal << " [px]");
    INFO_OUT("Number of Deformed Images: ", num_def_images);
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
    interpolator::bicubic_init(image_ref, px_horizontal, px_vertical);

    // initialise the LM optimizer to use the desired correlation criterion and shape func.
    optimizer::init(conf.corr_crit, conf.shape_func);

    // initialise the brute force scan
    std::string brute_method = "SPIRAL";
    brute::init(conf.corr_crit, brute_method);

    // for extraction of deformed image from stack
    util::Image image_def;
    image_def.px_horizontal = px_horizontal;
    image_def.px_vertical = px_vertical;

    // function pointer for the method of scanning the subsets through the image
    void (*scan_function)(double *, util::Image *, bool *, util::SubsetData *, int, int, int, double, double, double, double, int);

    // set the scan_function pointer based on the scan method specified by user.
    if (conf.scan_method=="IMAGE_SCAN") scan_function=image_scan;
    else if (conf.scan_method=="IMAGE_SCAN_WITH_BF") scan_function=image_scan_with_bf;
    else if (conf.scan_method=="RG") scan_function=reliability_guided;
    else {
        std::cerr << "Unknown subset scan type: \'" << conf.scan_method << "\'." << std::endl;
        std::cerr << "Allowed values: \'IMAGE_SCAN\', \'IMAGE_SCAN_WITH_BF\', \'RG\'. " << std::endl;
        return;
    } 

    // get end time and calculate DIC duration
    auto f0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> e0 = f0 - s0;
    INFO_OUT("Time taken to Initialize C++ DIC Engine: ", e0.count() << " [s]");



    // -------------------------------------------------------------------------------------------
    // loop over deformed images and perform DIC
    // -------------------------------------------------------------------------------------------


    // timer for 2D DIC engine
    auto s1 = std::chrono::high_resolution_clock::now();

    for (int img_num = 0; img_num < num_def_images; img_num++){

        // pointer to starting location of image
        image_def.num = img_num;
        image_def.vals = image_def_stack + img_num * px_horizontal * px_vertical;

        scan_function(image_ref, &image_def, image_roi, &ssdata, num_def_images, img_num, 
                      conf.max_iter, conf.precision, conf.threshold_lm, 
                      conf.threshold_bf, conf.range_bf, num_params);

    }

    // get end time and calculate DIC duration
    auto f1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> e1 = f1 - s1;
    INFO_OUT("Time taken to Run C++ DIC Engine: ", e1.count() << " [s]")

    // save data
    util::SaveConfig saveconf;
    saveconf.base_path = "./output/";
    saveconf.delimiter = " ";
    saveconf.format = ".dat";
    saveconf.prefix = "results";
    saveconf.layout = "col";
    saveconf.save_at_end = true;
    util::save_to_disk(&saveconf, num_def_images, &ssdata, num_params);

}




// -------------------------------------------------------------------------------------------
// Raw image scan
// -------------------------------------------------------------------------------------------

void image_scan(double *image_ref, 
                util::Image *image_def, 
                bool *image_roi,
                util::SubsetData *ssdata, 
                int num_def_images, 
                int img_num, 
                int max_iter, 
                double precision,
                double threshold_lm,
                double threshold_bf,
                double range_bf,
                int num_params){


    // initialise subsets
    util::Subset ss_def(ssdata->size);
    util::Subset ss_ref(ssdata->size);

    // optimization parameters
    optimizer::Parameters opt(num_params, max_iter, precision, threshold_lm,
                              image_def->px_vertical, image_def->px_horizontal);

    // loop over subsets within the ROI
    #pragma omp parallel for firstprivate(ss_def, ss_ref, opt)
    for (int ss = 0; ss < ssdata->num; ss++){

        // subset coordinate list takes central locations. 
        // Converting to top left corner for optimization routine
        int ss_x = ssdata->coords[ss*2];
        int ss_y = ssdata->coords[ss*2+1];

        // get the deformed subset coordinates and pixel values from the deformed image
        util::extract_ss(ss_x, ss_y, image_def, &ss_def); 


        // perform optimization on subset from deformed image
        optimizer::Results results;
        results = optimizer::solve(ss_x, ss_y, &ss_def, &ss_ref, &opt);


        // append the results for the current subset to result vectors
        util::append_results(num_def_images, img_num, ssdata->num, ss, 
                             results.iter, results.ftol, results.xtol, 
                             results.u, results.v, results.cost, results.p);

    }
}




// -------------------------------------------------------------------------------------------
// Raw image scan with a brute force to find rigid parameters. Good for large displacements
// -------------------------------------------------------------------------------------------

void image_scan_with_bf(double *image_ref, 
                        util::Image *image_def, 
                        bool *image_roi,
                        util::SubsetData *ssdata, 
                        int num_def_images, 
                        int img_num, 
                        int max_iter, 
                        double precision,
                        double threshold_lm,
                        double threshold_bf,
                        double range_bf,
                        int num_params){

    // subsets
    util::Subset ss_def(ssdata->size);
    util::Subset ss_ref(ssdata->size);

    // optimization parameters
    optimizer::Parameters opt(num_params, max_iter, precision, threshold_lm, 
                              image_def->px_vertical, image_def->px_horizontal);

    // brute force scan parameters
    brute::Parameters brute(threshold_bf, range_bf);

    // perform optimization on subset from deformed image
    optimizer::Results results;
    results.iter = 0;

    // counter for each thread
    int ss_thread_num = 0;      

    // temp p values for copy from brute force to optimization.
    double ptemp[6] = {0,0,0,0,0,0};


    // loop over subsets within the ROI
    #pragma omp parallel for firstprivate(ss_def, ss_ref, ss_thread_num, opt, brute, results)
    for (int ss = 0; ss < ssdata->num; ss++){

        // subset coordinate list contains central locations.
        // Converting to top left corner for optimization routine
        int ss_x = ssdata->coords[ss*2];
        int ss_y = ssdata->coords[ss*2+1];

        // get the deformed subset coordinates and pixel values
        util::extract_ss(ss_x, ss_y, image_def, &ss_def); 


        // if this is the first subset in the loop, or, if last subset was a poor match
        // Kick off the next search with a brute force scan
        // using the last set of brute force parameters that gave a good match.
        if ((ss_thread_num == 0) || (results.iter == opt.max_iter)){
            brute::expanding_wavefront(ss_x, ss_y, image_ref, image_def->px_vertical, 
                                       image_def->px_horizontal, &ss_def, &ss_ref, &brute);

            ptemp[0] = brute.p_rigid[0];
            ptemp[1] = brute.p_rigid[1];
            ptemp[2] = 0.0;
            ptemp[3] = 0.0;
            ptemp[4] = 0.0;
            ptemp[5] = 0.0;

            for (int i = 0; i < opt.num_params; i++){
                opt.p[i] = ptemp[i];
            }
        }

        results = optimizer::solve(ss_x, ss_y, &ss_def, &ss_ref, &opt);

        // append the results for the current subset to result vectors
        util::append_results(num_def_images, img_num, ssdata->num, ss, 
                             results.iter, results.ftol, results.xtol,
                             results.u, results.v, results.cost, results.p);

        ss_thread_num++;

    }
}






// -------------------------------------------------------------------------------------------
// Reliability Guided scan of image. (NOT YET IMPLEMENTED)
// -------------------------------------------------------------------------------------------

void reliability_guided(double *image_ref, 
                        util::Image *image_def, 
                        bool *image_roi,
                        util::SubsetData *ssdata,
                        int num_def_images, 
                        int img_num, 
                        int max_iter, 
                        double precision,
                        double threshold_lm,
                        double threshold_bf,
                        double range_bf,
                        int num_params){


     int seed_x = 500; // in corner coordinates
     int seed_y = 500; // in corner coodinates


     rg::reliability_guided_dic_single_seed(image_ref, image_def, image_roi, seed_x, seed_y, ssdata, num_def_images, img_num, max_iter, precision, threshold_lm, threshold_bf, range_bf, num_params);

}


PYBIND11_MODULE(dic2dcpp, m) {
    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("ss_step", &Config::ss_step)
        .def_readwrite("ss_size", &Config::ss_size)
        .def_readwrite("max_iter", &Config::max_iter)
        .def_readwrite("precision", &Config::precision)
        .def_readwrite("threshold_lm", &Config::threshold_lm)
        .def_readwrite("threshold_bf", &Config::threshold_bf)
        .def_readwrite("range_bf", &Config::range_bf)
        .def_readwrite("corr_crit", &Config::corr_crit)
        .def_readwrite("shape_func", &Config::shape_func)
        .def_readwrite("interp_routine", &Config::interp_routine)
        .def_readwrite("scan_method", &Config::scan_method);

    // Bind the engine function
    m.def("engine", &engine, "Run 2D analysis on input images with config");
}





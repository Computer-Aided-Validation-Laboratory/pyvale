// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <cstring>
#include <vector>
#include <chrono>
#include <array>
#include <omp.h>

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


namespace dic {

    void engine_2d(int* image_ref, 
                    int* image_def_stack, 
                    bool* image_roi, 
                    int px_vertical, 
                    int px_horizontal, 
                    int num_def_images,
                    Config &conf,
                    util::SaveConfig &saveconf){

        // -------------------------------------------------------------------------------------------
        // Initialisation
        // -------------------------------------------------------------------------------------------
        
        auto s0 = std::chrono::high_resolution_clock::now();

        const int num_px_image = px_horizontal*px_vertical;

        // number of parameters for the shape function
        int num_params;
        if (conf.shape_func == "rigid") num_params = 2;
        else if (conf.shape_func == "affine") num_params = 6;
        else {
            std::cerr << "Unknown shape function: \'" << conf.shape_func << "\'." << std::endl;
            std::cerr << "Allowed values: \'affine\', \'rigid\'. " << std::endl;
            return;
        }


        // get a list of ss coordinates within RIO.
        util::SubsetData ssdata = util::generate_ss_list(image_roi, px_horizontal, px_vertical, conf.ss_size, conf.ss_step, num_def_images, num_params);


        // TITLE("DIC INITIALISATION");
        // INFO_OUT("Height of Images: ", px_vertical << " [px]");
        // INFO_OUT("Width of Images: ", px_horizontal << " [px]");
        // INFO_OUT("Number of Deformed Images: ", num_def_images);
        // INFO_OUT("Subset Step: ", conf.ss_step);
        // INFO_OUT("Subset Size: ", conf.ss_size);
        // INFO_OUT("Max number of solver iterations: ", max_iter);
        // INFO_OUT("Tolerance cutoff: ", tol);
        // INFO_OUT("Correlation Criterion: ", conf.corr_crit);
        // INFO_OUT("Shape Function: ", conf.shape_func);
        // INFO_OUT("Interpolation Routine: ", interp_routine);
        // INFO_OUT("Image Scan Method: ", conf.scan_method);
        // INFO_OUT("Total number of subsets: ", ssdata.num);
        // INFO_OUT("Number of OMP threads:", omp_get_max_threads());


        // cuglobal::device_info(ssdata->num);
        // exit(0);

        // need to make a copy of the reference image that has been converted to double for the interpolator
        util::Image image_ref_dbl(px_horizontal, px_vertical);
        image_ref_dbl.vals.assign(image_ref, image_ref + num_px_image);

        // define our interpolator for the reference image
        interpolator::bicubic_init(&image_ref_dbl);

        // initialise the LM optimizer to use the desired correlation criterion and shape func.
        optimizer::init(conf.corr_crit, conf.shape_func);

        // initialise the brute force scan
        std::string brute_method = "SPIRAL";
        brute::init(conf.corr_crit, brute_method);

        // for extraction of deformed image from stack
        util::Image image_def(px_horizontal, px_vertical);

        // function pointer for the method of scanning the subsets through the image
        void (*scan_function)(int*, util::Image*, bool*, util::SubsetData*, int, int, int, double, double, double, double, int);

        // set the scan_function pointer based on the scan method specified by user.
        if (conf.scan_method=="image_scan") scan_function=image_scan;
        else if (conf.scan_method=="image_scan_with_brute_force") scan_function=image_scan_with_bf;
        else if (conf.scan_method=="RG") scan_function=reliability_guided;
        else {
            std::cerr << "Unknown subset scan type: \'" << conf.scan_method << "\'." << std::endl;
            std::cerr << "Allowed values: \'image_scan\', \'image_scan_with_brute_force\', \'RG\'. " << std::endl;
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

            // extract a single image from the stack
            util::extract_image(&image_def, image_def_stack, img_num);  
          
            scan_function(image_ref, &image_def, image_roi, &ssdata, num_def_images, img_num, conf.max_iter, conf.precision, conf.threshold_lm, conf. threshold_bf, conf.range_bf, num_params);

        }

        // get end time and calculate DIC duration
        auto f1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> e1 = f1 - s1;
        INFO_OUT("Time taken to Run C++ DIC Engine: ", e1.count() << " [s]")

        // save data
        saveconf.save_at_end = true;
        util::save_to_disk(&saveconf, 1, &ssdata, num_params);

    }




    // -------------------------------------------------------------------------------------------
    // Raw image scan
    // -------------------------------------------------------------------------------------------

    void image_scan(int *image_ref, 
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
        optimizer::Parameters opt(num_params, max_iter, precision, threshold_lm, image_def->px_vertical, image_def->px_horizontal);

        // loop over subsets within the ROI
        #pragma omp parallel for firstprivate(ss_def, ss_ref, opt)
        for (int ss = 0; ss < ssdata->num; ss++){

            // subset coordinate list takes central locations. Converting to top left corner for optimization routine
            int ss_x = ssdata->coords[ss*2];
            int ss_y = ssdata->coords[ss*2+1];

            // get the deformed subset coordinates and pixel values from the deformed image
            util::extract_ss(ss_x, ss_y, image_def, &ss_def); 


            // perform optimization on subset from deformed image
            optimizer::Results results;
            results = optimizer::solve(ss_x, ss_y, &ss_def, &ss_ref, &opt);


            // append the results for the current subset to result vectors
            // append_results(num_def_images, img_num, ss, &results);    
            // exit(0);
        }
    }




    // -------------------------------------------------------------------------------------------
    // Raw image scan with a brute force to find rigid parameters. Good for large displacements
    // -------------------------------------------------------------------------------------------

    void image_scan_with_bf(int *image_ref, 
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
        optimizer::Parameters opt(num_params, max_iter, precision, threshold_lm, image_def->px_vertical, image_def->px_horizontal);
        
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

            // subset coordinate list contains central locations. Converting to top left corner for optimization routine
            int ss_x = ssdata->coords[ss*2];
            int ss_y = ssdata->coords[ss*2+1];

            // get the deformed subset coordinates and pixel values
            util::extract_ss(ss_x, ss_y, image_def, &ss_def); 


            // if this is the first subset in the loop, or, if last subset was a poor match
            // Kick off the next search with a brute force from the last set of brute force parameters that gave a good match.
            if ((ss_thread_num == 0) || (results.iter == opt.max_iter)){
                brute::expanding_wavefront(ss_x, ss_y, image_ref, image_def->px_vertical, image_def->px_horizontal, &ss_def, &ss_ref, &brute);
                // std::cout << brute.p_rigid[0] << " " << brute.p_rigid[1] << std::endl;

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
            // append_results(num_def_images, img_num, ss, &results);    

            ss_thread_num++;

        }
    }






    // -------------------------------------------------------------------------------------------
    // Reliability Guided scan of image. (NOT YET IMPLEMENTED)
    // -------------------------------------------------------------------------------------------

    void reliability_guided(int *image_ref, 
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


        //util::Subset ss_def(ssdata->size);
        //util::Subset ss_ref(ssdata->size);
        //brute::Parameters brute(threshold_bf, range_bf);
        //util::extract_ss(seed_x, seed_y, image_def, &ss_def); 
        //brute::cross_correlation(seed_x, seed_y, image_ref, image_def->px_vertical, image_def->px_horizontal,&ss_def, &ss_ref, &brute);
        DEBUGGER;
        rg::reliability_guided_dic_single_seed(image_ref, image_def, image_roi, seed_x, seed_y, ssdata, num_def_images, img_num, max_iter, precision, threshold_lm, threshold_bf, range_bf, num_params);
                
    }
}

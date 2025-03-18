// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <iostream>
#include <chrono>

// GNU Scientific Library Header files
#include <gsl/gsl_multifit_nlinear.h>

// Program Header files
#include "./dicinterpolator.hpp"
#include "./dicutil.hpp"
#include "./dicoptimization.hpp"
#include "./dicengine.hpp"
#include "./diclm.hpp"


// #ifdef DEBUG
//     #define LOG(x) std::cout << "[VERBOSE] " << x << std::endl;
// #else
//     #define LOG(x)  // Do nothing
// #endif


namespace dic2d {


    // image and subset arrays
    std::vector<double> image_def;
    std::vector<double> subset_def;
    std::vector<double> subset_def_coords_x;
    std::vector<double> subset_def_coords_y;


    void dicengine(int* image_ref, 
                    int* image_def_stack, 
                    int* image_roi, 
                    int px_vertical, 
                    int px_horizontal, 
                    int num_def_images,
                    int subset_step,
                    int subset_size,
                    std::string& corr_crit, 
                    std::string& shape_func,
                    std::string& interp_routine,
                    std::string& scan_method){

        // -------------------------------------------------------------------------------------------
        // Initialisation
        // -------------------------------------------------------------------------------------------


        // total number of subsets
        int edge = 100;
        int n_subsets = util::get_num_subsets(edge, px_horizontal, px_vertical, subset_step);

    
        // timer for 2D DIC engine
        auto s0 = std::chrono::high_resolution_clock::now();

        // need to make a copy of the reference image that has been converted to double for the interpolator
        std::vector<double> image_ref_dbl;
        image_ref_dbl.assign(image_ref, image_ref + px_vertical*px_horizontal);

        // define our interpolator for the reference imageu
        gsl_spline2d *spline = interpolation::create_spline(interp_routine, image_ref_dbl, px_horizontal, px_vertical);

        // setup the optimizer and pass the already create spline object and accelerators.
        optimization::init(num_def_images, n_subsets, corr_crit, interp_routine, shape_func, subset_size, px_horizontal, px_vertical, spline);

        // initialise the LM optimizer that I have been writing
        lm::init(corr_crit, shape_func, subset_size);
        gsl_interp_accel *xacc = gsl_interp_accel_alloc();
        gsl_interp_accel *yacc = gsl_interp_accel_alloc();


        // resize image and subset arrays
        image_def.resize(px_vertical*px_horizontal,0.0);
        subset_def.resize(subset_size*subset_size,0.0);
        subset_def_coords_x.resize(subset_size*subset_size,0.0);
        subset_def_coords_y.resize(subset_size*subset_size,0.0);



        // function pointer for the method of scanning the subsets through the image
        void (*scan_function)(int, int, int, int, int, int, int );
        if (scan_method=="image_scan") scan_function=image_scan;
        else if (scan_method=="RG") scan_function=reliability_guided;
        else {
            std::cerr << "Unknown subset scan type: \'" << scan_method << "\'." << std::endl;
            std::cerr << "Allowed values: \'image_scan\', \'RG\'. " << std::endl;
            return;
        } 



        // -------------------------------------------------------------------------------------------
        // loop over deformed images
        // -------------------------------------------------------------------------------------------

        for (int img_num = 0; img_num < num_def_images; img_num++){

            // extract a single image from the stack
            util::extract_image(image_def, image_def_stack, img_num, px_horizontal, px_vertical);

            // scan_function(num_def_images, n_subsets, edge, px_horizontal, px_vertical, subset_size, subset_step);
            
            
            
            // -------------------------------------------------------------------------------------------
            // TESTING HOMEMADE LM
            // -------------------------------------------------------------------------------------------

            int subset_num = 0;
            for (int ss_y = edge; ss_y < px_vertical-edge; ss_y+=subset_step){
                for (int ss_x = edge; ss_x < px_horizontal-edge; ss_x+=subset_step){


                    // get the subset coordinates and pixel values
                    util::extract_subset(image_def, subset_def,  subset_def_coords_x, 
                                                subset_def_coords_y, ss_x, ss_y, subset_size, 
                                                px_horizontal, px_vertical);


                    // homemade LM optimizer
                    lm::solve(subset_def, subset_def_coords_x, subset_def_coords_y, spline, xacc, yacc, subset_size*subset_size);
                    
                    subset_num++;
                }
            }
        }

        
        
        // -------------------------------------------------------------------------------------------
        // cleanup
        // -------------------------------------------------------------------------------------------

        // get end time and calculate DIC duration
        auto f0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> e0 = f0 - s0;
        std::cout << "Time taken to Run C++ DIC Engine:      " << e0.count() <<  " [s]" << std::endl;

    }






    void image_scan(int n_img, int n_subset, int edge, int px_horizontal, int px_vertical, int subset_size, int subset_step){

        // loop over subsets within the ROI
        int subset_num = 0;
        for (int ss_y = edge; ss_y < px_vertical-edge; ss_y+=subset_step){
            for (int ss_x = edge; ss_x < px_horizontal-edge; ss_x+=subset_step){



                // get the subset coordinates and pixel values
                util::extract_subset(image_def, subset_def,  subset_def_coords_x, 
                                            subset_def_coords_y, ss_x, ss_y, subset_size, 
                                            px_horizontal, px_vertical);


                // update the optimization routine with the subset values
                optimization::set_data(subset_def_coords_x, subset_def_coords_y, subset_def);

                // execute optimization routine. args: seed for next subset, xtol, gtol, ftol, max_iter
                optimization::execute(subset_num, false, 0.001, 0.001, 0.001, 20);

                // optimization::collect_results(n_img, n_subset, subset_num, ss_x, ss_y);

                optimization::print_results(ss_x,ss_y);
                 
                subset_num++;
            }
        }
    }

    void reliability_guided(int n_img, int n_subset, int edge, int px_horizontal, int px_vertical, int subset_size, int subset_step){

        // create image masks
        std::vector<bool> mc(px_horizontal, px_vertical);
        std::vector<bool> mv(px_horizontal, px_vertical);

        std::vector<int> neigh(4,0);

        
        // need to pick an intial subset
        int ss_x_start = 100;
        int ss_y_start = 100;

        for (int ss_y = edge; ss_y < px_vertical-edge; ss_y+=subset_step){
            for (int ss_x = edge; ss_x < px_horizontal-edge; ss_x+=subset_step){


                // get the subset coordinates and pixel values
                util::extract_subset(image_def, subset_def,  subset_def_coords_x, 
                                            subset_def_coords_y, ss_x, ss_y, subset_size, 
                                            px_horizontal, px_vertical);

                // get neighbour indexes
                neigh[0] = (ss_y + 1)  * px_horizontal + ss_x;
                neigh[1] = (ss_y - 1)  * px_horizontal + ss_x;
                neigh[2] = (ss_y) * px_horizontal + ss_x + 1;
                neigh[3] = (ss_y) * px_horizontal + ss_x - 1;
                

            }
        }
    }
}
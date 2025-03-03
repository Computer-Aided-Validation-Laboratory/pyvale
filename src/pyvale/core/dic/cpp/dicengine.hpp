// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <iostream>

// GNU Scientific Library Header files
#include <gsl/gsl_multifit_nlinear.h>

// Program Header files
#include "./dicinterpolator.hpp"
#include "./dicimagemanip.hpp"
#include "./dicoptimization.hpp"


// #ifdef DEBUG
//     #define LOG(x) std::cout << "[VERBOSE] " << x << std::endl;
// #else
//     #define LOG(x)  // Do nothing
// #endif


namespace dic2d {

    
    std::vector<double> subset_ref;
    std::vector<double> subset_def;
    std::vector<double> subset_xvals;
    std::vector<double> subset_yvals;
    std::vector<double> p_arr;
    double ssd_val;

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
                    std::string& interp_routine){


        // LOG("Running DIC Engine")
                
        int subset_num_px = subset_size*subset_size;
        subset_ref.resize(subset_num_px,0.0);
        subset_def.resize(subset_num_px,0.0);
        // LOG("Resizing subset arrays")


        // returns an array of pixel values for each axis
        std::vector<double> xvals = interpolation::xvalues(px_horizontal); 
        std::vector<double> yvals = interpolation::xvalues(px_vertical); 

 

        // returns a pointer to an accelerator object, which is a kind of iterator for interpolation lookups. 
        // It tracks the state of lookups, thus allowing for application of various acceleration strategies.
        gsl_interp_accel *xacc = gsl_interp_accel_alloc();
        gsl_interp_accel *yacc = gsl_interp_accel_alloc();


        // define our interpolator for the reference image
        gsl_spline2d *spline = interpolation::create_spline(interp_routine, &image_ref[0], px_horizontal, px_vertical);


        // setup the optimizer and pass the already create spline object and accelerators.
        optimization::init(interp_routine, shape_func, subset_size, spline, xacc, yacc);


        // deformed image array
        std::vector<double> image_def(px_vertical,px_horizontal,0.0);


        std::vector<double> image_def;
        std::vector<double> subset_def;
        std::vector<double> subset_def_coords_x;
        std::vector<double> subset_def_coords_y;


        // loop over deformed images
        for (unsigned int img_num = 0; img_num < num_def_images; img_num++){



            // extract a single image from the stack
            deformed::extract_image(&image_def[0], &image_def_stack[0], num_def_images, px_horizontal, px_vertical);

            // loop over subsets within the ROI
            int edge = 50;
            for (unsigned int ss_y = edge; ss_y < px_vertical-edge; ss_y+=subset_step){
                for (unsigned int ss_x = edge; ss_x < px_horizontal-edge; ss_x+=subset_step){



                    // get the subset coordinates and pixel values
                    deformed_image::extract_subset(&image_def[0], &subset_def[0],  &subset_def_coords_x[0], 
                                                   &subset_def_coords_y[0], ss_x ss_y, subset_num_px, 
                                                   px_horizontal, px_vertical);

                    // update the optimization routine with the subset values
                    optimization::update_data(subset, subset_def_coords_x, subset_def_coords_y);

                    // execute optimization routine
                    optimization::execute();


                
                }
            }
        }


        exit(0);

    }

}
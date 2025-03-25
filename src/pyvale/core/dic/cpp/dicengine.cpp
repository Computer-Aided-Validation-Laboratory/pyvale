// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <vector>
#include <chrono>

// Program Header files
#include "./dicinterpolator.hpp"
#include "./dicoptimizer.hpp"
#include "./dicengine.hpp"
#include "./dicutil.hpp"




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
                    int max_iter,
                    double tol,
                    std::string& corr_crit, 
                    std::string& shape_func,
                    std::string& interp_routine,
                    std::string& scan_method){

        // -------------------------------------------------------------------------------------------
        // Initialisation
        // -------------------------------------------------------------------------------------------


        int num_px_image = px_horizontal*px_vertical;
        int num_px_subset = subset_size*subset_size;

        // total number of subsets
        int edge = 52;
        int n_subsets = util::get_num_subsets(edge, px_horizontal, px_vertical, subset_step);

    
        // timer for 2D DIC engine
        auto s0 = std::chrono::high_resolution_clock::now();

        // need to make a copy of the reference image that has been converted to double for the interpolator
        std::vector<double> image_ref_dbl;
        image_ref_dbl.assign(image_ref, image_ref + num_px_image);

        // define our interpolator for the reference imageu
        interpolator::bicubic_init(image_ref_dbl, px_horizontal, px_vertical);

        // initialise the LM optimizer that I have been writing
        optimizer::init(corr_crit, shape_func, subset_size);




        // resize image and subset arrays
        image_def.resize(num_px_image,0.0);
        subset_def.resize(num_px_subset,0.0);
        subset_def_coords_x.resize(num_px_subset,0.0);
        subset_def_coords_y.resize(num_px_subset,0.0);



        // function pointer for the method of scanning the subsets through the image
        void (*scan_function)(int, int, int, int, int, int, int, int, double);
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

            scan_function(num_def_images, n_subsets, edge, px_horizontal, px_vertical, subset_size, subset_step, max_iter, tol);
            
        }

        // get end time and calculate DIC duration
        auto f0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> e0 = f0 - s0;
        std::cout << "Time taken to Run C++ DIC Engine:      " << e0.count() <<  " [s]" << std::endl;

    }




    // -------------------------------------------------------------------------------------------
    // Raw image scan
    // -------------------------------------------------------------------------------------------

    void image_scan(int n_img, int n_subset, int edge, int px_horizontal, int px_vertical, int subset_size, int subset_step, int max_iter, double tol){

        // loop over subsets within the ROI
        int subset_num = 0;
        for (int ss_y = edge; ss_y < px_vertical-edge; ss_y+=subset_step){
            for (int ss_x = edge; ss_x < px_horizontal-edge; ss_x+=subset_step){


                // get the subset coordinates and pixel values
                util::extract_subset(image_def, subset_def,  subset_def_coords_x, subset_def_coords_y, ss_x, ss_y, subset_size, px_horizontal, px_vertical);                

                std::cout << ss_x << " " << ss_y <<  " ";
                optimizer::solve(subset_def, subset_def_coords_x, subset_def_coords_y, subset_size*subset_size, tol, max_iter);

                subset_num++;
            }
        }
    }






    // -------------------------------------------------------------------------------------------
    // Reliability Guided scan of image. (NOT YET IMPLEMENTED)
    // -------------------------------------------------------------------------------------------

    void reliability_guided(int n_img, int n_subset, int edge, int px_horizontal, int px_vertical, int subset_size, int subset_step, int max_iter, double tol){

        // create image masks
        std::vector<bool> mc(px_horizontal, px_vertical);
        std::vector<bool> mv(px_horizontal, px_vertical);

        std::vector<int> neigh(4,0);

        
        // need to pick an intial subset
        //int ss_x_start = 100;
        //int ss_y_start = 100;

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

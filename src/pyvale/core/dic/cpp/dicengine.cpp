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
    std::vector<double> ss_def;
    std::vector<double> ss_def_coords_x;
    std::vector<double> ss_def_coords_y;


    void dicengine(int* image_ref, 
                    int* image_def_stack, 
                    bool* image_roi, 
                    int px_vertical, 
                    int px_horizontal, 
                    int num_def_images,
                    int ss_step,
                    int ss_size,
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
        int num_px_ss = ss_size*ss_size;

        // get a list of ss coordinates within RIO.
        std::vector<int> ss_list_x;
        std::vector<int> ss_list_y;
        util::fill_ss_coord_vects(ss_list_x, ss_list_y, image_roi, px_horizontal, px_vertical, ss_size, ss_step);
        int n_ss = ss_list_x.size();

    
        // timer for 2D DIC engine
        auto s0 = std::chrono::high_resolution_clock::now();

        // need to make a copy of the reference image that has been converted to double for the interpolator
        std::vector<double> image_ref_dbl;
        image_ref_dbl.assign(image_ref, image_ref + num_px_image);

        // define our interpolator for the reference imageu
        interpolator::bicubic_init(image_ref_dbl, px_horizontal, px_vertical);

        // initialise the LM optimizer and the output struct
        optimizer::init(corr_crit, shape_func, ss_size);

        // for extraction of deformed image from stack
        std::vector<double> image_def(num_px_image,0.0);

        // resize the deformed subset vectors
        util::resize_ss(ss_def, ss_def_coords_x, ss_def_coords_y, ss_size);



        // function pointer for the method of scanning the subsets through the image
        void (*scan_function)(std::vector<double> &, std::vector<int> &, std::vector<int> &, int, int, int, int, int, double);
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

            scan_function(image_def, ss_list_x, ss_list_y, px_horizontal, px_vertical, n_ss, ss_size, max_iter, tol);
            
        }

        // get end time and calculate DIC duration
        auto f0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> e0 = f0 - s0;
        std::cout << "Time taken to Run C++ DIC Engine:      " << e0.count() <<  " [s]" << std::endl;

    }




    // -------------------------------------------------------------------------------------------
    // Raw image scan
    // -------------------------------------------------------------------------------------------

    void image_scan(std::vector<double> &image_def, std::vector<int> &ss_list_x, std::vector<int> &ss_list_y, int px_horizontal, int px_vertical, int n_ss, int ss_size, int max_iter, double tol){


        int ss_x;
        int ss_y;
        util::Displacement displacement;

        // loop over subsets within the ROI
        for (int ss = 0; ss < n_ss; ss++){

            //convert to corner coordinates
            ss_x = ss_list_x[ss] - ss_size / 2;
            ss_y = ss_list_y[ss] - ss_size / 2;

            // get the subset coordinates and pixel values
            util::extract_ss(image_def, ss_def,  ss_def_coords_x, ss_def_coords_y, ss_x, ss_y, ss_size, px_horizontal, px_vertical);    



            std::cout << ss_list_x[ss] << " " << ss_list_y[ss] <<  " ";
            optimizer::solve(ss_def, ss_def_coords_x, ss_def_coords_y, ss_size*ss_size, tol, max_iter);
            displacement = util::parameters_to_displacement(ss_x,ss_y, optimizer::p);
            // std::cout << ss_list_x[ss] << " " << ss_list_y[ss] << " " << displacement.u << " " << displacement.v << " " << displacement.mag << "\n";

        }

    }






    // -------------------------------------------------------------------------------------------
    // Reliability Guided scan of image. (NOT YET IMPLEMENTED)
    // -------------------------------------------------------------------------------------------

    void reliability_guided(std::vector<double> &image_def, std::vector<int> &ss_list_x, std::vector<int> &ss_list_y, int px_horizontal, int px_vertical, int n_ss, int ss_size, int max_iter, double tol){

        // // create image masks
        // std::vector<bool> mc(px_horizontal, px_vertical);
        // std::vector<bool> mv(px_horizontal, px_vertical);

        // std::vector<int> neigh(4,0);

        
        // // need to pick an intial subset
        // //int ss_x_start = 100;
        // //int ss_y_start = 100;



        // // get the subset coordinates and pixel values
        // util::extract_ss(image_def, ss_def,  ss_def_coords_x, 
        //                             ss_def_coords_y, ss_x, ss_y, ss_size, 
        //                             px_horizontal, px_vertical);

        // // get neighbour indexes
        // neigh[0] = (ss_y + 1)  * px_horizontal + ss_x;
        // neigh[1] = (ss_y - 1)  * px_horizontal + ss_x;
        // neigh[2] = (ss_y) * px_horizontal + ss_x + 1;
        // neigh[3] = (ss_y) * px_horizontal + ss_x - 1;
                
    }


}

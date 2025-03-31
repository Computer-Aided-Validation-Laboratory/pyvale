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

// cuda Header files
#include "../cuda/malloc.hpp"


namespace dic2d {


    // image and subset arrays
    std::vector<double> ss_def;
    std::vector<double> ss_def_coords_x;
    std::vector<double> ss_def_coords_y;


    // result arrays. Not using std::vector because harder to handle with cython
    std::vector<int> ss_coord_list;
    std::vector<int> niter_arr;
    std::vector<double> u_arr;
    std::vector<double> v_arr;
    std::vector<double> p_arr;
    std::vector<double> ftol_arr;
    std::vector<double> xtol_arr;
    int n_ss;


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


        const int num_px_image = px_horizontal*px_vertical;
        const int num_px_ss = ss_size*ss_size;

        // get a list of ss coordinates within RIO.
        util::fill_ss_coord_vects(ss_coord_list, image_roi, px_horizontal, px_vertical, ss_size, ss_step);
        n_ss = ss_coord_list.size() / 2;

        // cuglobal::device_info(n_ss);
        // exit(0);

    
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

        // resize results
        niter_arr.resize(num_def_images * n_ss);
        u_arr.resize(num_def_images * n_ss);
        v_arr.resize(num_def_images * n_ss);
        p_arr.resize(num_def_images * n_ss * 6);
        ftol_arr.resize(num_def_images * n_ss);
        xtol_arr.resize(num_def_images * n_ss);



        // function pointer for the method of scanning the subsets through the image
        void (*scan_function)(std::vector<double> &, std::vector<int> &, int, int, int, int, int, int, int, double);
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

            scan_function(image_def, ss_coord_list, num_def_images, img_num, px_horizontal, px_vertical, n_ss, ss_size, max_iter, tol);
            
        }

        // get end time and calculate DIC duration
        auto f0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> e0 = f0 - s0;
        std::cout << "Time taken to Run C++ DIC Engine:      " << e0.count() <<  " [s]" << std::endl;

    }




    // -------------------------------------------------------------------------------------------
    // Raw image scan
    // -------------------------------------------------------------------------------------------

    void image_scan(std::vector<double> &image_def, std::vector<int> &ss_coord_list, int num_def_images, int img_num, int px_horizontal, int px_vertical, int n_ss, int ss_size, int max_iter, double tol){


        int ss_x;
        int ss_y;

        // loop over subsets within the ROI
        for (int ss = 0; ss < n_ss; ss++){

            // subset coordinate list takes central locations. Converting to top left corner for optimization routine
            ss_x = ss_coord_list[ss*2] - ss_size / 2;
            ss_y = ss_coord_list[ss*2+1] - ss_size / 2;

            // get the deformed subset coordinates and pixel values
            util::extract_ss(image_def, ss_def,  ss_def_coords_x, ss_def_coords_y, ss_x, ss_y, ss_size, px_horizontal, px_vertical);    

            // perform optimization on subset from deformed image
            optimizer::solve(ss_def, ss_def_coords_x, ss_def_coords_y, ss_size*ss_size, tol, max_iter);
            
            // convert shape function parameters to horizontal and vertical displacement
            optimizer::affine_parameters_to_displacement(ss_x,ss_y);

            // append the results for the current subset to result vectors
            append_results(num_def_images, img_num, ss);


            std::cout << ss_coord_list[ss*2] << " " << ss_coord_list[ss*2+1] <<  " ";


        
        }

    }






    // -------------------------------------------------------------------------------------------
    // Reliability Guided scan of image. (NOT YET IMPLEMENTED)
    // -------------------------------------------------------------------------------------------

    void reliability_guided(std::vector<double> &image_def, std::vector<int> &ss_coord_list, int num_def_images, int img_num, int px_horizontal, int px_vertical, int n_ss, int ss_size, int max_iter, double tol){

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



    void append_results(int num_def_images, int img_num, int ss){

            int index = img_num * num_def_images + ss;
            int index_p = 6*index;

            niter_arr[index] = optimizer::iter;
            p_arr[index_p+0] = optimizer::p[0];
            p_arr[index_p+1] = optimizer::p[1];
            p_arr[index_p+2] = optimizer::p[2];
            p_arr[index_p+3] = optimizer::p[3];
            p_arr[index_p+4] = optimizer::p[4];
            p_arr[index_p+5] = optimizer::p[5];
            u_arr[index] = optimizer::u;
            v_arr[index] = optimizer::v;
            ftol_arr[index] = optimizer::ftol;
            xtol_arr[index] = optimizer::xtol;
    }
}

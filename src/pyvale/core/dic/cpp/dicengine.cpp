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

// Program Header files
#include "./dicinterpolator.hpp"
#include "./dicoptimizer.hpp"
#include "./dicengine.hpp"
#include "./defines.hpp"
#include "./dicutil.hpp"

// cuda Header files
#include "../cuda/malloc.hpp"
#include <omp.h>


namespace dic2d {


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
        
        auto s0 = std::chrono::high_resolution_clock::now();

        // TITLE("DIC INITIALISATION");
        // INFO_OUT("Height of Images: ", px_vertical << " [px]");
        // INFO_OUT("Width of Images: ", px_horizontal << " [px]");
        // INFO_OUT("Number of Deformed Images: ", num_def_images);
        // INFO_OUT("Subset Step: ", ss_step);
        // INFO_OUT("Subset Size: ", ss_size);
        // INFO_OUT("Max number of solver iterations: ", max_iter);
        // INFO_OUT("Tolerance cutoff: ", tol);
        // INFO_OUT("Correlation Criterion: ", corr_crit);
        // INFO_OUT("Shape Function: ", shape_func);
        // INFO_OUT("Interpolation Routine: ", interp_routine);
        // INFO_OUT("Image Scan Method: ", scan_method);

        const int num_px_image = px_horizontal*px_vertical;
        const int num_px_ss = ss_size*ss_size;

        // get a list of ss coordinates within RIO.
        util::fill_ss_coord_vects(ss_coord_list, image_roi, px_horizontal, px_vertical, ss_size, ss_step);
        n_ss = ss_coord_list.size() / 2;

        // INFO_OUT("Total number of subsets: ", n_ss);
        // INFO_OUT("Number of OMP threads:", omp_get_max_threads());


        // cuglobal::device_info(n_ss);
        // exit(0);

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
        // util::resize_ss(ss_def, ss_def_coords_x, ss_def_coords_y, ss_size);

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
        
        // get end time and calculate DIC duration
        auto f0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> e0 = f0 - s0;
        // INFO_OUT("Time taken to Initialize C++ DIC Engine: ", e0.count() << " [s]");



        // -------------------------------------------------------------------------------------------
        // loop over deformed images and perform DIC
        // -------------------------------------------------------------------------------------------


        // timer for 2D DIC engine
        auto s1 = std::chrono::high_resolution_clock::now();

        for (int img_num = 0; img_num < num_def_images; img_num++){

            // extract a single image from the stack
            util::extract_image(image_def, image_def_stack, img_num, px_horizontal, px_vertical);

            scan_function(image_def, ss_coord_list, num_def_images, img_num, px_horizontal, px_vertical, n_ss, ss_size, max_iter, tol);
            
        }

        // get end time and calculate DIC duration
        auto f1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> e1 = f1 - s1;
        // INFO_OUT("Time taken to Run C++ DIC Engine: ", e1.count() << " [s]")
        std::cout << e1.count() << std::endl;

    }




    // -------------------------------------------------------------------------------------------
    // Raw image scan
    // -------------------------------------------------------------------------------------------

    void image_scan(std::vector<double> &image_def, std::vector<int> &ss_coord_list, int num_def_images, int img_num, int px_horizontal, int px_vertical, int n_ss, int ss_size, int max_iter, double tol){

        // image and subset arrays
        std::vector<double> ss_def(ss_size*ss_size, 0.0);
        std::vector<double> ss_def_coords_x(ss_size*ss_size, 0.0);
        std::vector<double> ss_def_coords_y(ss_size*ss_size, 0.0);

        optimizer::Parameters params;
        optimizer::init_parameters(&params, ss_size);

        // loop over subsets within the ROI
        #pragma omp parallel for firstprivate(ss_def, ss_def_coords_x, ss_def_coords_y, params)
        for (int ss = 0; ss < n_ss; ss++){

            // subset coordinate list takes central locations. Converting to top left corner for optimization routine
            int ss_x = ss_coord_list[ss*2] - ss_size / 2;
            int ss_y = ss_coord_list[ss*2+1] - ss_size / 2;


            // get the deformed subset coordinates and pixel values
            util::extract_ss(image_def, ss_def,  ss_def_coords_x, ss_def_coords_y, ss_x, ss_y, ss_size, px_horizontal, px_vertical); 

            // perform optimization on subset from deformed image
            optimizer::Results results;
            results = optimizer::solve(ss_x, ss_y, ss_def, ss_def_coords_x, ss_def_coords_y, ss_size*ss_size, tol, max_iter, &params);

            // append the results for the current subset to result vectors
            append_results(num_def_images, img_num, ss, &results);    
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



    void append_results(int num_def_images, int img_num, int ss, optimizer::Results *results){

            int index = img_num * num_def_images + ss;
            int index_p = 6*index;
            niter_arr[index] = results->iter;
            p_arr[index_p+0] = results->p[0];
            p_arr[index_p+1] = results->p[1];
            p_arr[index_p+2] = results->p[2];
            p_arr[index_p+3] = results->p[3];
            p_arr[index_p+4] = results->p[4];
            p_arr[index_p+5] = results->p[5];
            u_arr[index] = results->u;
            v_arr[index] = results->v;
            ftol_arr[index] = results->ftol;
            xtol_arr[index] = results->xtol;
    }
}

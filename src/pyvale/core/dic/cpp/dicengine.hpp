// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#include <iostream>
#include <cmath>
#include "dicsplinec1.hpp"
#include "diccorrelation.hpp"


namespace dic2d {

    
    std::vector<double> subset_ref;
    std::vector<double> subset_def;
    std::vector<double> subset_xvals;
    std::vector<double> subset_yvals;
    std::vector<double> p_arr;
    double ssd_val;

    void dicengine(int* image_ref, 
                    int* image_def, 
                    int* image_roi, 
                    int px_vertical, 
                    int px_horizontal, 
                    int num_def_images,
                    int subset_step,
                    int subset_size,
                    std::string& corr_crit, 
                    std::string& shape_func,
                    std::string& interp_routine){


        std::cout << "LIVE FROM THE ENGINE!" << std::endl;
        exit(0);
                
        subset_ref.resize(subset_size*subset_size,0.0);
        subset_def.resize(subset_size*subset_size,0.0);
        p_arr.resize(6,0.0);



        //create interpolator for the reference image


        // or set params for brute force
        double step = 0.1;
        double subpx_i_min = -0.5;
        double subpx_i_max = 0.5;
        double subpx_j_min = -0.5;
        double subpx_j_max = 0.5;
        double subpx_i_num = (subpx_i_max - subpx_i_min) / step;
        double subpx_j_num = (subpx_j_max - subpx_j_min) / step;
        double subpx_i, subpx_j;
        int subset_min, subset_max;

        // loop over deformed images
        for (unsigned int img = 0; img < num_def_images; img++){

            // loop over subsets
            for (unsigned int ss_i = 0; ss_i < px_vertical; ss_i++){
                for (unsigned int ss_j = 0; ss_j < px_horizontal; ss_j++){


                    //deformed subset values
                    int count = 0;
                    int img_index = 0;
                    for (int ss_xval = 0; ss_xval < subset_size; ss_xval++){
                        for (int ss_yval = 0; ss_yval < subset_size; ss_yval++){

                            img_index = ss_xval * px_horizontal + ss_yval;
                            subset_def[count] = image_def[img_index];
                            count++;

                        }
                    }

                    // Here is it either a brute force search or a minimzation routine
                    // for (int k = 0; k < subpx_i_num; k++){
                    //     for (int l = 0; l < subpx_j_num; l++){

                    //         double subpx_i = subpx_i_min + k * step;
                    //         double subpx_j = subpx_j_min + l * step;

                    //     }
                    // }


                    
                    // minimisation()



                }
            }
        }
    }


    // void minimisations(){



    // }




    // void calculate_displacements() {


    //     // get the interpolation of the entire reference image
    //     interpolator = correlation.spline_interpolation_object(reference_image, 3)

    //     min_x = subset_size // 2
    //     min_y = subset_size // 2
    //     max_x = reference_image.shape[1] - subset_size // 2
    //     max_y = reference_image.shape[0] - subset_size // 2

    //     // dont use subsets if rows/cols < 10
    //     edge_cutoff = 50

    //     x_values = np.arange(min_x+edge_cutoff, max_x-edge_cutoff, subset_step)
    //     y_values = np.arange(min_y+edge_cutoff, max_y-edge_cutoff, subset_step)
    //     shape = (len(y_values), len(x_values), 6) 

    //     total_iterations = x_values.shape[0] * y_values.shape[0]

    //     // Initialize 2D arrays
    //     p_arr = np.zeros(shape)
    //     ssd_arr = np.zeros((len(y_values), len(x_values)))

    //     p = np.array([0.0,0.0,0.0,0.0,0.0,0.0])

    //     // looping over the subsets
    //     for i, x in enumerate(x_values):
    //         for j, y in enumerate(y_values):

    //             subset = correlation.subset(deformed_image, x, y, subset_size)

    //             half_size = subset_size // 2

    //             // reference image subset
    //             x1, x2 = x - half_size, x + half_size + 1
    //             y1, y2 = y - half_size, y + half_size + 1

    //             // list of coordinates 
    //             coords_x = np.arange(x1,x2,1)
    //             coords_y = np.arange(y1,y2,1)

    //             //pixel coordinates of reference subset
    //             xx, yy = np.meshgrid(coords_x,coords_y)

    //             sol = minimize(subset_search_affine_minimizer, p, args=(subset,interpolator,xx,yy),bounds=bounds)
    //             p = sol.x
    //             ssd_val = sol.fun


    //             // value is negative because its deformed subset looking searching in reference image
    //             p_arr[j,i,0:6]  = p
    //             ssd_arr[j,i] = ssd_val

    //             progress_bar.update(1)

    // return p_arr, ssd_arr



    // }





}
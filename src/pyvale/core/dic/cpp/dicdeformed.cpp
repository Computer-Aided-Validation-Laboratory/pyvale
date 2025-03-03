// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <iostream>

// GNU Scientific Library Header files
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_interp2d.h>


// Program Header files



namespace deformed {


    void extract_image(std::vector<double> &image_def,
                       int *image_def_stack, 
                       int image_number,
                       int px_horizontal,
                       int px_vertical){

        // extract a single image
        std::vector<double> img(px_horizontal * px_vertical);
        int count = 0;


        for (int px_vert = 0; px_vert < px_vertical; px_vert++){
            for (int px_hori = 0; px_hori < px_horizontal; px_hori++){

                int index = image_number * px_horizontal * px_vertical + px_vert * px_horizontal + px_hori;
                image_def[count] = image_def_stack[index];

            }
        }
    }

    void extract_subset(std::vector<double> &image_def, 
                        std::vector<double> &subset, 
                        std::vector<double> &subset_coords_x,
                        std::vector<double> &subset_coords_y, 
                        int subset_x, 
                        int subset_y, 
                        int subset_size, 
                        int px_horizontal, 
                        int px_vertical){

        int count = 0;
        for (int px_y = subset_y; px_y < subset_y+subset_size; px_y++){
            for (int px_x = subset_x; px_x < subset_x+subset_size; px_x++){

                // get coordinate values
                subset_coords_x[count] = px_x; 
                subset_coords_y[count] = px_y; 

                // get pixel values
                int index = px_y * px_horizontal + px_x;
                subset[count] = image_def[index];
                count++;
                
            }
        }
    }
}
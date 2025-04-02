// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICUTIL_H
#define DICUTIL_H

// STD library Header files
#include <vector>



namespace util {


    extern double u;
    extern double v;
    extern double mag;

    void extract_image(std::vector<double> &image_def,
                       int *image_def_stack, 
                       int image_number,
                       int px_horizontal,
                       int px_vertical);    
                       
    void extract_ss(std::vector<double> &image_def, 
                        std::vector<double> &ss, 
                        std::vector<double> &ss_coords_x,
                        std::vector<double> &ss_coords_y, 
                        int ss_centre_x, 
                        int ss_centre_y, 
                        int ss_size, 
                        int px_horizontal, 
                        int px_vertical);

    std::vector<int> generate_ss_coord_list(bool *image_roi, int px_horizontal, int px_vertical, int ss_size, int ss_step);
    void resize_ss(std::vector<double> &ss, std::vector<double> &ss_x, std::vector<double> &ss_y, int ss_size);
    void parameters_to_displacement(double ss_x, double ss_y, std::vector<double> &p);
}

#endif //DICUTIL
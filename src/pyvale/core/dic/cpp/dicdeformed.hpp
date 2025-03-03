// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICDEFORMED_H
#define DICDEFORMED_H

// STD library Header files
#include <vector>



namespace deformed {
    void extract_image(std::vector<double> image_def,
                       std::vector<double> &image_def_stack, 
                       int image_number,
                       int px_horizontal,
                       int px_vertical);    
                       
    void extract_subset(std::vector<double> &image_def, 
                        std::vector<double> &subset, 
                        std::vector<double> &subset_coords_x,
                        std::vector<double> &subset_coords_y, 
                        int subset_x, 
                        int subset_y, 
                        int subset_size, 
                        int px_horizontal, 
                        int px_vertical);

}

#endif DICDEFORMED
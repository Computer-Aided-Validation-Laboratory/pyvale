// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <vector>

// Program Header files
#include "./dicinterpolator.cpp"
#include "./dicsmooth.hpp"
#include "./dicinterpolator.hpp"
#include "./dicutil.cpp"

namespace strain {

    
    void engine(std::string interp, std::string tensor, std::vector<int> ss_list) {
    
        std::vector<double> smoothed;

        int ss_step = 10;
        int ss_size = 51;
        int sw = 5;
        int swr = sw / 2;
        int vsg = ((sw-1) * ss_step) + ss_size;


        int num_def_images = 1;

        // loop over the displacement images
        for (int img = 0; img < num_def_images; img++) {
    
            // loop over strain windows within the image
            for (vsg = 0; vsg < ss_list.size(); vsg++){

                // strain calculation
                // get the displacement values for the subsets in the strain winodow
                for (int j = -swr; j < swr; j++){
                    for (int i = -swr; i < swr; swr++){

                        
                         

                    }
                }

            }
        }


    }





} // namespace strain

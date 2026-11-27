// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICCOARSEFINE_H
#define DICCOARSEFINE_H

#include <vector>
#include <string>

namespace coarsefine {

    // Downsample an image by factor 2 using simple averaging
    std::vector<double> downsample(const double* img, int width, int height);

    // Extract a subset patch centered at (cx, cy) from an image of given width
    // Returns false if the patch would go out of bounds
    bool extract_subset(const double* img, int img_w, int img_h,
                        int cx, int cy, int ss_size,
                        std::vector<double>& patch);

    struct CoarseToFineResult {
        double disp_x = 0.0;
        double disp_y = 0.0;
        double peak_val = 0.0;
        bool success = false;
    };

    CoarseToFineResult coarse_to_fine_search(
        const double* img_ref,
        const double* img_def,
        int img_w,
        int img_h,
        int center_x,         // subset center in the reference image
        int center_y,
        int ss_size,          // subset size at full resolution (e.g. 51)
        int max_displacement, // max displacement at full resolution (e.g. 1000)
        bool subpx = true,
        const std::string& peak_method = "GAUSSIAN_2D"
    );

}
#endif // DICCOARSEFINE_H

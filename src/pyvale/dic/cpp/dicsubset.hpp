// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICSUBSET_H
#define DICSUBSET_H

// STD library Header files
#include <vector>

// Program Header files
#include "./dicinterpolator.hpp"

namespace subset {

    struct Grid {
        int num;
        int step;
        int size;
        int num_ss_x;
        int num_ss_y;
        int num_in_mask;
        std::vector<int> coords;
        std::vector<int> mask;
        std::vector<std::vector<int>> neigh;
    };

    /**
     * @brief holds a subset with pixel data and dimensions.
     * 
     * This struct holds the pixel values, coordinates, and dimensions of a square subset.
     */
    struct Pixels {
        std::vector<double> vals;
        std::vector<double> x;
        std::vector<double> y;
        int size;
        int num_px;

        // Constructor to initialize the vectors with ss_size
        Pixels(int ss_size) 
            : vals(ss_size * ss_size, 0.0),
            x(ss_size * ss_size, 0.0),
            y(ss_size * ss_size, 0.0),
            size(ss_size),
            num_px(ss_size * ss_size)
        {}
    };

    /**
     * @brief Extracts a square subset of pixels from an image and stores the data in a Subset object.
     * 
     * This function copies a square region of pixel data from the specified starting coordinates 
     * (`ss_x`, `ss_y`) in the input image into the `ss_def` structure. The size of the square 
     * subset is determined by `ss_def->size`. Both the pixel values and their corresponding 
     * coordinates are stored in `ss_def`.
     * 
     * @param ss_x        X-coordinate (column) of the top-left corner of the subset in the image.
     * @param ss_y        Y-coordinate (row) of the top-left corner of the subset in the image.
     * @param img_def   Pointer to the source image (`util::Image`) from which to extract pixel data.
     * @param ss_def      Pointer to the destination subset (`subset::Pixels`) where extracted pixel 
     *                    values and coordinates are stored.
     */            
    void get_px_from_img(subset::Pixels &ss_ref,
                    const int ss_x, const int ss_y,
                    const int px_hori,
                    const int px_vert,
                    const double *img_def);


    /**
     * @brief Extracts a square subset of pixels from an image and stores the data in a Subset object.
     * 
     * This function copies a square region of pixel data from the specified starting coordinates 
     * (`ss_x`, `ss_y`) in the input image into the `ss_def` structure. The size of the square 
     * subset is determined by `ss_def->size`. Both the pixel values and their corresponding 
     * coordinates are stored in `ss_def`.
     * 
     * @param ss_ref      Pointer to the destination subset (`subset::Pixels`) where extracted pixel info will be stored
     * @param ss_x        X-coordinate (column) of the top-left corner of the subset in the image.
     * @param ss_y        Y-coordinate (row) of the top-left corner of the subset in the image.
     * @param interp_ref  interpolator for the reference image from which to extract pixel data.
     */
    void get_subpx_from_img(subset::Pixels &ss_def, 
                          const double subpx_x, const double subpx_y, 
                          const Interpolator &interp_def);

    /**
     *
     */
    void get_subpx_from_shape_params(subset::Pixels &ss_def, 
                                     const double subpx_x, const double subpx_y,
                                     const std::vector<double>& p,
                                     const Interpolator &interp_def);

    /**
     * @brief Generates a list of subsets based on the provided image ROI and parameters.
     * 
     * This function creates a list of subsets (defined by their coordinates) from a binary mask 
     * (`img_roi`) that indicates the region of interest in the image. The subsets are generated 
     * with specified size and step values.
     * 
     * @param img_roi    Pointer to a binary mask indicating the region of interest in the image.
     * @param px_hori Number of horizontal pixels in the image.
     * @param px_vert   Number of vertical pixels in the image.
     * @param ss_size      Size of each subset (in pixels).
     * @param ss_step      Step size for generating subsets.
     * @return            A subset::Grid object containing the generated subsets and their neighbours.
     */
    subset::Grid create_grid(const bool *img_roi, const int ss_step, const int ss_size, 
                           const int px_hori, const int px_vert, const bool partial=false);

    
    inline bool px_in_img_dims(const int px_x, const int px_y, const int px_hori, 
                        const int px_vert);

    inline bool px_in_roi(const int px_x, const int px_y, const int px_hori, 
                        const int px_vert, const bool *img_roi);
}
#endif // DICSMOOTH_H

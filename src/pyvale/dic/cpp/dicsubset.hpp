// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICSUBSET_H
#define DICSUBSET_H

// STD library Header files
#include <vector>
#include <string>
#include <cmath>

// Program Header files
#include "./dicinterp.hpp"
#include "./dicutil.hpp"

// common_cpp header files
#include "../../common_cpp/util.hpp"

namespace subset {

    struct Grid {
        int num;
        int step;
        int size_x;
        int size_y;
        int num_ss_x;
        int num_ss_y;
        int num_in_mask;
        std::vector<double> coords;
        std::vector<int> mask;
        std::vector<std::vector<int>> neigh;
        std::vector<bool> active_ss;
        int active_total;
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
        int size_x;
        int size_y;
        int num_px;
        int sum;

        // Constructor to initialize the vectors with ss_size
        Pixels(int ss_size_x, int ss_size_y) 
            : vals(ss_size_x * ss_size_y, 0.0),
            x(ss_size_x * ss_size_y, 0.0),
            y(ss_size_x * ss_size_y, 0.0),
            size_x(ss_size_x),
            size_y(ss_size_y),
            num_px(ss_size_x * ss_size_y)
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
     * @param ss_x        X-coordinate (column) of the TOP-LEFT CORNER of the subset in the image.
     * @param ss_y        Y-coordinate (row) of the TOP-LEFT CORNER of the subset in the image.
     * @param img_def   Pointer to the source image (`util::Image`) from which to extract pixel data.
     * @param ss_def      Pointer to the destination subset (`subset::Pixels`) where extracted pixel 
     *                    values and coordinates are stored.
     */            
    void fill_from_img(subset::Pixels &ss_ref,
                    const int ss_x, const int ss_y,
                    const int px_hori,
                    const int px_vert,
                    const Image &img);

    template<typename T>
    void fill_impl(subset::Pixels &ss_ref,
                const std::vector<T> &data,
                int ss_x, int ss_y,
                int px_hori);

    /**
    * @brief Fills a subset of pixels from an image using centre coordinates.
    *
    * Samples pixel values from the interpolator over a grid centred at (cx, cy).
    * Equivalent to fill_from_img_subpx but takes centre coordinates rather than
    * the top-left corner, and stores both pixel coordinates and values in ss_def.
    *
    * @param ss_def    Destination subset where pixel coordinates and values are stored.
    * @param cx        X-coordinate of the CENTRE of the subset in the image.
    * @param cy        Y-coordinate of the CENTRE of the subset in the image.
    * @param interp_def Interpolator for the image from which to sample pixel data.
    */    
    void fill_from_centre_coords(subset::Pixels &ss_def,
                                 const double cx, const double cy,
                                 const Interpolator &interp_def);

    /**
     * @brief Extracts a square subset of pixels from an image and stores the data in a Subset object.
     * 
     * This function copies a square region of pixel data from the specified starting coordinates 
     * (`ss_x`, `ss_y`) in the input image into the `ss_def` structure. The size of the square 
     * subset is determined by `ss_def->size`. Both the pixel values and their corresponding 
     * coordinates are stored in `ss_def`.
     * 
     * @param ss_ref      Pointer to the destination subset (`subset::Pixels`) where extracted pixel info will be stored
     * @param ss_x        X-coordinate (column) of the TOP-LEFT CORNER of the subset in the image.
     * @param ss_y        Y-coordinate (row) of the TOP-LEFT CORNER of the subset in the image.
     * @param interp_ref  interpolator for the reference image from which to extract pixel data.
     */
    void fill_from_img_subpx(subset::Pixels &ss_def, 
                          const double subpx_x, const double subpx_y, 
                          const Interpolator &interp_def);

    /**
    * @brief Populates a deformed subset with interpolated image values using shape function parameters.
    *
    * Applies the shape function to map reference subset coordinates (centred at cx, cy)
    * to deformed image coordinates, then interpolates the image intensity at each mapped location.
    *
    * @param ss_def      Output subset to populate with deformed coordinates and interpolated values.
    * @param cx          Global x-coordinate of the SUBSET CENTRE in the reference image.
    * @param cy          Global y-coordinate of the SUBSET CENTRE in the reference image.
    * @param p           Shape function parameters (e.g. displacement, strain components).
    * @param interp_def  Interpolator for the deformed image.
    * @param shape_func  Shape function type enum.
    */
    void fill_from_shape_params(subset::Pixels &ss_def, 
                                     const double cx, const double cy,
                                     const std::vector<double>& p,
                                     const Interpolator &interp_def,
                                     util::ShapeFunc shape_func);
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
    subset::Grid create_grid(const bool *img_roi, const int ss_step,
                             const int ss_size_x, const int ss_size_y,
                             const int px_hori, const int px_vert,
                             const bool partial);

    
    static inline bool px_in_img_dims(const int px_x, const int px_y, const int px_hori, const int px_vert) {

        if (px_x < 0 || px_y < 0 ||
            px_x >= px_hori ||
            px_y >= px_vert) {
            return false;
        }
        return true;
    }

    static inline bool px_in_roi(const int px_x, const int px_y, const int px_hori, 
                        const int px_vert, const bool *img_roi) {

        int idx = px_y * px_hori + px_x;
        if (!img_roi[idx]) {
            return false;
        }
        return true;
    }



    /**
     * @brief Returns central subset coordinates for given subset corner values
     * and dimensions
     *
     * @param cx[out]    x pixel coordinate of subset centre
     * @param cy[out]    y pixel coordinate of subset centre
     * @param ss_x[in]  x pixel coordinate of subset corner
     * @param ss_y[in]  y pixel coordinate of subset corner
     * @param size_x[in]  subset size in x 
     * @param size_y[in]  subset size in y
     */
    static inline void get_centre(double& cx, double& cy,
                                  const double ss_x, const double ss_y,
                                  const int size_x, const int size_y) {

        cx = ss_x + static_cast<double>(size_x) * 0.5 - 0.5;
        cy = ss_y + static_cast<double>(size_y) * 0.5 - 0.5;

    }
    
    /**
     * @brief Returns corner subset coordinates for given subset centre values
     * and dimensions
     *
     * @param cx[out]    x pixel coordinate of subset centre
     * @param cy[out]    y pixel coordinate of subset centre
     * @param ss_x[in]  x pixel coordinate of subset corner
     * @param ss_y[in]  y pixel coordinate of subset corner
     * @param size_x[in]  subset size in x 
     * @param size_y[in]  subset size in y
     */
    static inline void get_corner(double& corner_x, double& corner_y,
                                const double cx, const double cy,
                                const int size_x, const int size_y) {
        corner_x = cx - static_cast<double>(size_x) * 0.5 + 0.5;
        corner_y = cy - static_cast<double>(size_y) * 0.5 + 0.5;
    }


    // Compute ZNCC between two subsets
    double zncc(const subset::Pixels& ss_ref, const subset::Pixels& ss_def);
}
#endif // DICSUBSET_H

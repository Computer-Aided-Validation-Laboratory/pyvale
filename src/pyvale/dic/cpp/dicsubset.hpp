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
#include "./dicshapefunc.hpp"

// commoncpp header files
#include "../../commoncpp/util.hpp"

struct SubsetGrid {
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
template <typename T, bool StoreCoordinates = true>
struct Subset {
    std::vector<T> vals;
    std::vector<double> x;
    std::vector<double> y;
    int size_x;
    int size_y;
    int num_px;
    T sum;

    // Constructor to initialize the vectors with ss_size
    Subset(int ss_size_x, int ss_size_y)
        : vals(ss_size_x * ss_size_y, T(0)),
        x(StoreCoordinates ? ss_size_x * ss_size_y : 0, 0.0),
        y(StoreCoordinates ? ss_size_x * ss_size_y : 0, 0.0),
        size_x(ss_size_x),
        size_y(ss_size_y),
        num_px(ss_size_x * ss_size_y)
    {}

    static constexpr bool has_coordinates = StoreCoordinates;

    /**
     * @brief Reports whether this subset stores per-pixel coordinates.
     *
     * Coordinate storage is controlled at compile time by @p StoreCoordinates.
     * FFT-only subsets can therefore omit the coordinate arrays entirely.
     */
    bool has_coords() const;

    /**
     * @brief Shifts the subset pixel coordinates into a local coordinate system.
     *
     * @param cx x-coordinate of the subset centre
     * @param cy y-coordinate of the subset centre
     */
    void shift_to_local_coordinates(double cx, double cy);

    /**
    * @brief Fills a subset of pixels from an image using centre coordinates.
    *
    * Samples pixel values from the interpolator over a grid centred at (cx, cy).
    * Equivalent to fill_from_img_subpx but takes centre coordinates rather than
    * the top-left corner, and stores both pixel coordinates and values in the subset.
    *
    * @param cx        X-coordinate of the CENTRE of the subset in the image.
    * @param cy        Y-coordinate of the CENTRE of the subset in the image.
    * @param interp    Interpolator for the image from which to sample pixel data.
     */
    void fill_from_centre_coords(double cx,
                                 double cy,
                                 const Interpolator &interp);

    /**
    * @brief Extracts a square subset of pixels from an image and stores the data in a Subset object.
    * 
    * This function copies a square region of pixel data from the specified starting coordinates
    * (`corner_x`, `corner_y`) in the input image into the subset structure. Both the pixel values
    * and their corresponding coordinates are stored in the subset.
    *
    * @param corner_x    X-coordinate (column) of the TOP-LEFT CORNER of the subset in the image.
    * @param corner_y    Y-coordinate (row) of the TOP-LEFT CORNER of the subset in the image.
    * @param interp      Interpolator for the image from which to extract pixel data.
     */
    void fill_from_img_subpx(double corner_x,
                             double corner_y,
                             const Interpolator &interp);

    /**
    * @brief Populates a deformed subset with interpolated image values using shape function parameters.
    *
    * Applies the shape function to map reference subset coordinates (centred at cx, cy)
    * to deformed image coordinates, then interpolates the image intensity at each mapped location.
    *
    * @param cx          Global x-coordinate of the SUBSET CENTRE in the reference image.
    * @param cy          Global y-coordinate of the SUBSET CENTRE in the reference image.
    * @param params      Shape function parameters (e.g. displacement, strain components).
    * @param interp      Interpolator for the deformed image.
    * @param shape_func  Shape function type enum.
     */
    void fill_from_shape_params(double cx,
                                double cy,
                                const std::vector<double>& params,
                                const Interpolator &interp,
                                util::EShapeFunc shape_func);
};

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
    * @param partial_subset Minimum ROI filling fraction in [0, 1]. Subsets must
    *                       remain entirely inside the image.
    * @return            A SubsetGrid object containing the generated subsets and their neighbours.
    */
SubsetGrid create_grid(const bool *img_roi, const int ss_step,
                            const int ss_size_x, const int ss_size_y,
                            const int px_hori, const int px_vert,
                            const double partial_subset);


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
double zncc(const Subset<double>& ss_ref, const Subset<double>& ss_def);

#endif // DICSUBSET_H

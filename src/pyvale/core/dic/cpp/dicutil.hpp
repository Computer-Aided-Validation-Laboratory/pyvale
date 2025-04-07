// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICUTIL_H
#define DICUTIL_H

// STD library Header files
#include <vector>

// program Header files


namespace util {


    /**
     * @brief holds a subset with pixel data and dimensions.
     * 
     * This struct holds the pixel values, coordinates, and dimensions of a square subset.
     */
    struct Subset {
        std::vector<double> vals;
        std::vector<double> x;
        std::vector<double> y;
        int size;
        int num_px;

        // Constructor to initialize the vectors with ss_size
        Subset(int ss_size) 
            : vals(ss_size * ss_size, 0.0),       
            x(ss_size * ss_size, 0.0),
            y(ss_size * ss_size, 0.0),
            size(ss_size),
            num_px(ss_size * ss_size)
        {}
    };


    /**
     * @brief Represents an image with pixel data and dimensions.
     * 
     * This struct holds the pixel values of an image along with its
     * dimensions. The pixel data is stored in row-major order.
     */
    struct Image {
        std::vector<double> vals;
        int px_horizontal;
        int px_vertical;

        // Constructor to initialize the vectors with ss_size
        Image(int px_h, int px_v) 
            : vals(px_h * px_v, 0.0),
            px_horizontal(px_h), 
            px_vertical(px_v)
        {}

    };


    /**
     * @brief Extracts a single image from a stacked image array and stores it in an `Image` object.
     * 
     * Takes a specific 2D image (identified by `image_number`) from a 3D image stack 
     * (`image_def_stack`) and stores its pixel values into the `vals` field of the provided 
     * `util::Image` structure.
     * 
     * @param image_def        Pointer to a `util::Image` object that will be populated with the extracted image data.
     * @param image_def_stack  Pointer to a flat array representing a stack of images stored sequentially 
     *                         (row-major order).
     * @param image_number     Index of the image to extract from the stack (0-based).
     */
    void extract_image(util::Image *image_def, int *image_def_stack,  int image_number);


           
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
     * @param image_def   Pointer to the source image (`util::Image`) from which to extract pixel data.
     * @param ss_def      Pointer to the destination subset (`util::Subset`) where extracted pixel 
     *                    values and coordinates are stored.
     */            
    void extract_ss(int ss_x, int ss_y, util::Image *image_def, util::Subset *ss_def);


    /**
     * @brief Generates a list of valid subset center coordinates within a region of interest (ROI).
     * 
     * Scans over an image at intervals defined by `ss_step`, testing whether a square 
     * subset of size `ss_size` centered at each position falls entirely within the bounds of the 
     * image and the specified region of interest (`image_roi`). If the entire subset lies within the 
     * ROI, the coordinates of the subset center are added to the output list.
     * 
     * @param image_roi     Boolean mask representing the region of interest (ROI) in the image. 
     *                      Should have `px_horizontal * px_vertical` elements, with `true` indicating 
     *                      valid ROI pixels.
     * @param px_horizontal Width of the image in pixels.
     * @param px_vertical   Height of the image in pixels.
     * @param ss_size       Size (width and height) of the square subset (assumed to be odd).
     * @param ss_step       Step size (in pixels) used to move the subset window across the image.
     * 
     * @return std::vector<int> A flat list of valid (x, y) coordinates where subsets can be extracted. 
     *                          Each coordinate pair is stored consecutively (e.g., [x0, y0, x1, y1, ...]).
     */
    std::vector<int> generate_ss_coord_list(bool *image_roi, int px_horizontal, int px_vertical, int ss_size, int ss_step);


}

#endif //DICUTIL
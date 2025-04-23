// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICUTIL_H
#define DICUTIL_H

// STD library Header files
#include <vector>
#include <unordered_map>

// program Header files


namespace util {


    // Custom hash from above
    // struct PairHash {
    //     std::size_t operator()(const std::pair<int, int>& p) const {
    //         return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    //     }
    // };

    struct SubsetData {
        int num;
        int step;
        int size;
        int num_ss_x;
        int num_ss_y;
        std::vector<int> coords;
        std::vector<bool> mask;

    };


    // result arrays.
    extern std::vector<int> niter_arr;
    extern std::vector<double> u_arr; 
    extern std::vector<double> v_arr;
    extern std::vector<double> p_arr;
    extern std::vector<double> ftol_arr;
    extern std::vector<double> xtol_arr;
    extern std::vector<double> cost_arr;



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
     * @brief Generates a list of subsets based on the provided image ROI and parameters.
     * 
     * This function creates a list of subsets (defined by their coordinates) from a binary mask 
     * (`image_roi`) that indicates the region of interest in the image. The subsets are generated 
     * with specified size and step values.
     * 
     * @param image_roi    Pointer to a binary mask indicating the region of interest in the image.
     * @param px_horizontal Number of horizontal pixels in the image.
     * @param px_vertical   Number of vertical pixels in the image.
     * @param ss_size      Size of each subset (in pixels).
     * @param ss_step      Step size for generating subsets.
     * @return            A SubsetData object containing the generated subsets and their neighbours.
     */
     SubsetData generate_ss_list(bool *image_roi, int px_horizontal, int px_vertical, int ss_size, int ss_step, int num_def_images, int num_params);

    
    /**
     * @brief 
     * 
     * @param num_def_images 
     * @param img_num 
     * @param ss 
     * @param iter 
     * @param ftol 
     * @param xtol 
     * @param u 
     * @param v 
     * @param p 
     */
    void append_results(const int num_def_images, 
                            const int img_num, 
                            const int ss, 
                            const int iter, 
                            const double ftol, 
                            const double xtol, 
                            const double u, 
                            const double v, 
                            const double cost,
                            const std::vector<double> &p);

}

#endif //DICUTIL

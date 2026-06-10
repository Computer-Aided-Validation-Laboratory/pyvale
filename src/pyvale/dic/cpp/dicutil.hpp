// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICUTIL_H
#define DICUTIL_H

// STD library Header files
#include <vector>
#include <string>

// common_cpp header files

// program Header files


namespace util {


    // Custom hash from above
    struct PairHash {
        std::size_t operator()(const std::pair<int, int>& p) const {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };



    struct Config {
        int ss_step;
        int ss_size;
        int max_iter;
        int px_hori;
        int px_vert;
        int num_def_img;
        int num_params;
        double precision;
        double threshold;
        double bf_threshold;
        int max_disp;
        std::pair<int, int> rg_seed;
        std::string corr_crit;
        std::string shape_func;
        std::string interp_routine;
        std::string scan_method;
        std::vector<std::string> filenames;
        bool fft_mad;
        double fft_mad_scale;
        unsigned int debug_level;
    };




    /**
     * @brief Represents an image with pixel data and dimensions.
     * 
     * This struct holds the pixel values of an image along with its
     * dimensions. The pixel data is stored in row-major order.
     */
    struct Image {
        double *vals;
        int px_hori;
        int px_vert;
        int num;
    };


    /**
     * @brief Extracts a single image from a stacked image array and stores it in an `Image` object.
     * 
     * Takes a specific 2D image (identified by `image_number`) from a 3D image stack 
     * (`img_def_stack`) and stores its pixel values into the `vals` field of the provided 
     * `util::Image` structure.
     * 
     * @param img_def        Pointer to a `util::Image` object that will be populated with the extracted image data.
     * @param img_def_stack  Pointer to a flat array representing a stack of images stored sequentially 
     *                         (row-major order).
     * @param image_number     Index of the image to extract from the stack (0-based).
     */
    void extract_image(double *img_def_stack, 
                       int image_number,
                       int px_hori,
                       int px_vert);

    int next_pow2(int n);

    void gen_size_and_step_vector(std::vector<int> &ss_sizes, std::vector<int> &ss_steps, 
                                  const int ss_size, const int ss_step, const int max_disp);

}

#endif //DICUTIL

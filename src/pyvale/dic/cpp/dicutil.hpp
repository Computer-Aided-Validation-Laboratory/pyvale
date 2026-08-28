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
#include <ostream>

// common_cpp header files

// program Header files


namespace util {

    enum class CorrCrit {
        SSD,
        NSSD,
        ZNSSD
    };

    enum class ShapeFunc {
        RIGID,
        AFFINE,
        QUAD
    };

    enum class InterpRoutine {
        BSPLINE,
        HERMITE
    };

    enum class ScanMethod {
        MULTIWINDOW_RG,
        SINGLEWINDOW_RG,
        MULTIWINDOW,
        RASTER
    };


    enum class IncrementalCond {
        IMAGE,
        ITER,
        COST
    };

    enum class FFTPrecision {
        FLOAT32,
        FLOAT64
    };

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
        int max_disp;
        int epi_distance;
        std::vector<int> rg_seeds;
        CorrCrit corr_crit;
        ShapeFunc shape_func;
        InterpRoutine interp_routine;
        ScanMethod scan_method;
        std::vector<std::string> basenames;
        std::vector<std::string> fullpaths;
        bool fft_filter;
        bool fft_save;
        FFTPrecision fft_precision;
        double fft_filter_threshold;
        int fft_filter_radius;
        double fft_filter_corr_power;
        unsigned int debug_level;
        bool stereo;
        bool incremental;
        IncrementalCond incremental_update_cond;
        double incremental_update_val;
        int multiwindow_overlap;
        std::vector<int> multiwindow_subset_size;
        std::vector<int> multiwindow_search_area;
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

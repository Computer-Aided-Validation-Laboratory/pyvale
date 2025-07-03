// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICUTIL_H
#define DICUTIL_H

// STD library Header files
#include <chrono>
#include <vector>
#include <string>
#include <iostream>

// program Header files
#include "./dicinterpolator.hpp"
#include "./defines.hpp"

namespace util {


    // Custom hash from above
    struct PairHash {
        std::size_t operator()(const std::pair<int, int>& p) const {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };

    class Timer {
    public:
        Timer(const std::string& label)
            : label_(label), start_(std::chrono::high_resolution_clock::now()) {}

        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start_;
            INFO_OUT("Time taken for " + label_, elapsed.count() << " [s]");
        }

    private:
        std::string label_;
        std::chrono::high_resolution_clock::time_point start_;
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
        double threshold_lm;
        double threshold_bf;
        int range_bf;
        std::pair<int, int> rg_seed;
        std::string corr_crit;
        std::string shape_func;
        std::string interp_routine;
        std::string scan_method;
        std::vector<std::string> filenames;
    };


    struct SubsetData {
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

    struct SaveConfig {

        std::string basepath;
        std::string prefix;
        std::string delimiter;
        bool binary;
        bool at_end;


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

    struct Results {
        std::vector<double> p;
        double u = 0.0;
        double v = 0.0;
        double mag = 0.0;
        double ftol = 0.0;
        double xtol = 0.0;
        int iter = 0;
        double cost = 0.0;

        Results(size_t num_params) : p(num_params, 0.0) {}
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
     * @param ss_def      Pointer to the destination subset (`util::Subset`) where extracted pixel 
     *                    values and coordinates are stored.
     */            
    void extract_ss(util::Subset &ss_def, 
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
     * @param ss_ref      Pointer to the destination subset (`util::Subset`) where extracted pixel info will be stored
     * @param ss_x        X-coordinate (column) of the top-left corner of the subset in the image.
     * @param ss_y        Y-coordinate (row) of the top-left corner of the subset in the image.
     * @param interp_ref  interpolator for the reference image from which to extract pixel data.
     */
    void extract_ss_subpx(util::Subset &ss_def, 
                          const double subpx_x, const double subpx_y, 
                          const Interpolator &interp_def);



    /**
     */
    int get_num_params(std::string &shape_func);

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
     * @return            A SubsetData object containing the generated subsets and their neighbours.
     */
    SubsetData gen_ss_list(const bool *img_roi, const int ss_step, const int ss_size, 
                           const int px_hori, const int px_vert, const bool partial=false);


    void append_results(int img_num, int ss, 
                        util::Results &res, int num_ss);

    void resize_results(int num_def_img, int num_ss,
                        int num_params, bool at_end);

    void save_to_disk(int img, util::SaveConfig &saveconf,
                      util::SubsetData &ssdata, int num_def_img,
                      int num_params);


    inline bool is_valid_in_dims(const int px_x, const int px_y, const int px_hori, 
                        const int px_vert);

    inline bool is_valid_in_roi(const int px_x, const int px_y, const int px_hori, 
                        const int px_vert, const bool *img_roi);

    inline void write_int(std::ofstream& out, int val);

    inline void write_dbl(std::ofstream& out, double val);

    int next_pow2(int n);
    void gen_size_and_step_vector(std::vector<int> &ss_sizes, std::vector<int> &ss_steps, 
                                  const int ss_size, const int ss_step, const int max_disp);
}

#endif //DICUTIL

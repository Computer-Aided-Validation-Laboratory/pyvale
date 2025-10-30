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
#include <fstream>
#include <atomic>

// program Header files
#include "./dicinterpolator.hpp"
#include "./dicsubset.hpp"
#include "./indicators.hpp"
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
        double opt_threshold;
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




    struct SaveConfig {

        std::string basepath;
        std::string prefix;
        std::string delimiter;
        bool binary;
        bool at_end;
        bool output_unconverged;
        bool shape_params;


    };


    // result arrays.
    extern std::vector<int> niter_arr;
    extern std::vector<double> u_arr; 
    extern std::vector<double> v_arr;
    extern std::vector<double> p_arr;
    extern std::vector<double> ftol_arr;
    extern std::vector<double> xtol_arr;
    extern std::vector<double> cost_arr;





    struct Results {
        std::vector<double> p;
        double u = 0.0;
        double v = 0.0;
        double mag = 0.0;
        double ftol = 0.0;
        double xtol = 0.0;
        int iter = 0;
        double cost = 0.0;
        uint8_t converged = false;

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


           
    

    void append_results(int img_num, int ss, 
                        util::Results &res, int num_ss);

    void resize_results(int num_def_img, int num_ss,
                        int num_params, bool at_end);

    void save_to_disk(int img, const util::SaveConfig &saveconf,
                      const subset::Grid &ss_grid, const int num_def_img,
                      const int num_params, const std::vector<std::string> &filenames);




    
    inline void write_int(std::ofstream& out, int val) {
        out.write(reinterpret_cast<const char*>(&val), sizeof(int));
    }

    inline void write_uint8t(std::ofstream& out, int val) {
        out.write(reinterpret_cast<const char*>(&val), sizeof(uint8_t));
    }

    inline void write_dbl(std::ofstream& out, double val) {
        out.write(reinterpret_cast<const char*>(&val), sizeof(double));
    }

    int next_pow2(int n);

    void gen_size_and_step_vector(std::vector<int> &ss_sizes, std::vector<int> &ss_steps, 
                                  const int ss_size, const int ss_step, const int max_disp);

    void create_progress_bar(indicators::ProgressBar &bar,
                             const std::string &bar_title,
                             const int num_ss);

    void update_progress_bar(indicators::ProgressBar &bar, int i, int num_ss, int &prev_pct);
}

#endif //DICUTIL

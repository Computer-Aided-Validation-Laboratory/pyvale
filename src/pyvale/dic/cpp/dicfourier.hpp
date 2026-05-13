// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICFOURIER_H
#define DICFOURIER_H

// STD library Header files
#include <csignal>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>

// common header files 
#include "../../common_cpp/pocketfft_hdronly.h"
#include <Eigen/Dense>

// DIC Header files
#include "./dicinterp.hpp"
#include "./dicsubset.hpp"
#include "./dicutil.hpp"


struct FFT {
    int ss_size_x;
    int ss_size_y;
    int n_complex;

    // input data
    subset::Pixels ss_def;
    subset::Pixels ss_ref;

    // output data
    std::vector<std::complex<double>> fft_def;
    std::vector<std::complex<double>> fft_ref;
    std::vector<double> cross_corr;

    pocketfft::shape_t shape_in;
    pocketfft::shape_t axes = {0,1};

    pocketfft::stride_t stride_in;
    pocketfft::stride_t stride_out;

    // for subpixel peak position in correlation map
    Eigen::MatrixXd A;
    Eigen::VectorXd b;

    FFT(int ss_size_x_, int ss_size_y_)
        : ss_size_x(ss_size_x_),
        ss_size_y(ss_size_y_),
        n_complex(ss_size_x_/2+1),
        ss_def(ss_size_x_, ss_size_y_),
        ss_ref(ss_size_x_, ss_size_y_),
        fft_def(ss_size_y_*n_complex),
        fft_ref(ss_size_y_*n_complex),
        cross_corr(ss_size_x_ * ss_size_y_),
        A(9,6),
        b(9)
    {
    
        shape_in   = {static_cast<unsigned long>(ss_size_y), static_cast<unsigned long>(ss_size_x)};
        stride_in  = {static_cast<long>(ss_size_x * sizeof(double)), sizeof(double)};
        stride_out = {static_cast<long>(n_complex * sizeof(std::complex<double>)), sizeof(std::complex<double>)};

    }

    void correlate() {

        // forward fft of reference and deformed subsets
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD, ss_ref.vals.data(), fft_ref.data(), 1.0, 1);
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD, ss_def.vals.data(), fft_def.data(), 1.0, 1);
        
        // multiplication of complex fft data
        for (int px = 0; px < ss_size_y * n_complex; px++) {
            fft_def[px] = std::conj(fft_ref[px]) * fft_def[px];
        }

        // inverse FFT to get cross correlation
        pocketfft::c2r(shape_in, stride_out, stride_in, axes, pocketfft::BACKWARD, fft_def.data(), cross_corr.data(), 1.0, 1);
    }

    void correlate_phase() {

        // forward fft of reference and deformed subsets
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD, ss_ref.vals.data(), fft_ref.data(), 1.0, 1);
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD, ss_def.vals.data(), fft_def.data(), 1.0, 1);
        
        // multiplication of complex fft data
        for (int px = 0; px < ss_size_y * n_complex; ++px) {
            std::complex<double> val = std::conj(fft_ref[px]) * fft_def[px];
            double mag = std::abs(val);
            fft_def[px] = (mag > 1e-12) ? val / mag : 0.0;
        }

        // inverse FFT to get cross correlation
        pocketfft::c2r(shape_in, stride_out, stride_in, axes, pocketfft::BACKWARD, fft_def.data(), cross_corr.data(), 1.0, 1);
    }

    void fftshift(std::vector<double>& data, int size_x, int size_y) {
        std::vector<double> temp(size_x * size_y);
        
        int half_x = size_x / 2;
        int half_y = size_y / 2;

        for (int y = 0; y < size_y; ++y) {
            for (int x = 0; x < size_x; ++x) {
                int new_x = (x + half_x) % size_x;
                int new_y = (y + half_y) % size_y;
                temp[new_y * size_x + new_x] = data[y * size_x + x];
            }
        }
        data = temp;
    }

    inline double safe_log(double val, double eps = 1e-7) {
        return std::log(std::max(val, eps));
    }

    inline int wrap(int coord, int size) {
        return (coord + size) % size;
    }


    void get_peak(double &peak_x, double &peak_y, double &max_val, const bool subpx, const std::string &method) {
        max_val = -std::numeric_limits<double>::infinity();
        int x0 = 0, y0 = 0;

        // Step 1: Find the integer peak
        for (int y = 0; y < ss_size_y; ++y) {
            for (int x = 0; x < ss_size_x; ++x) {
                double val = cross_corr[y * ss_size_x + x];
                if (val > max_val) {
                    max_val = val;
                    x0 = x;
                    y0 = y;
                }
            }
        }

        // Step 2: No subpixel refinement requested
        if (!subpx) {
            peak_x = (x0 <= ss_size_x / 2.0) ? x0 : x0 - ss_size_x;
            peak_y = (y0 <= ss_size_y / 2.0) ? y0 : y0 - ss_size_y;
            return;
        }

        int i = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int xw = wrap(x0 + dx, ss_size_x);
                int yw = wrap(y0 + dy, ss_size_y);
                double val = cross_corr[yw * ss_size_x + xw];

                // Safe log for Gaussian
                if (method == "GAUSSIAN_2D" && val <= 0) val = 1e-6;
                double z = (method == "GAUSSIAN_2D") ? std::log(val) : val;

                A(i, 0) = dx * dx;
                A(i, 1) = dy * dy;
                A(i, 2) = dx * dy;
                A(i, 3) = dx;
                A(i, 4) = dy;
                A(i, 5) = 1.0;
                b(i) = z;
                i++;
            }
        }

        // Step 4: Solve least squares
        Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(b);
        double a = coeffs(0), b = coeffs(1), c = coeffs(2);
        double d = coeffs(3), e = coeffs(4);

        // Step 5: Find stationary point (gradient = 0)
        Eigen::Matrix2d H;
        H << 2 * a, c,
            c,     2 * b;
        Eigen::Vector2d g(-d, -e);

        Eigen::Vector2d offset = H.ldlt().solve(g);

        peak_x = x0 + offset(0);
        peak_y = y0 + offset(1);
        peak_x = (peak_x <= ss_size_x / 2.0) ? peak_x : peak_x - ss_size_x;
        peak_y = (peak_y <= ss_size_y / 2.0) ? peak_y : peak_y - ss_size_y;
        //std::cout << peak_x << " " << peak_y << std::endl;
    }


    void get_peak_nowrap(double &peak_x, double &peak_y, double &max_val, const bool subpx, const std::string &method) {
        max_val = -std::numeric_limits<double>::infinity();
        int x0 = 0, y0 = 0;

        // Step 1: Find the integer peak
        for (int y = 0; y < ss_size_y; ++y) {
            for (int x = 0; x < ss_size_x; ++x) {
                double val = cross_corr[y * ss_size_x + x];
                if (val > max_val) {
                    max_val = val;
                    x0 = x;
                    y0 = y;
                }
            }
        }

        // Step 2: No subpixel refinement requested
        if (!subpx) {
            peak_x = x0;
            peak_y = y0;
            return;
        }

        int i = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int xw = wrap(x0 + dx, ss_size_x);
                int yw = wrap(y0 + dy, ss_size_y);
                double val = cross_corr[yw * ss_size_x + xw];
                if (method == "GAUSSIAN_2D" && val <= 0) val = 1e-6;
                double z = (method == "GAUSSIAN_2D") ? std::log(val) : val;
                A(i, 0) = dx * dx;
                A(i, 1) = dy * dy;
                A(i, 2) = dx * dy;
                A(i, 3) = dx;
                A(i, 4) = dy;
                A(i, 5) = 1.0;
                b(i) = z;
                i++;
            }
        }

        // Step 4: Solve least squares
        Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(b);
        double a = coeffs(0), b = coeffs(1), c = coeffs(2);
        double d = coeffs(3), e = coeffs(4);

        // Step 5: Find stationary point
        Eigen::Matrix2d H;
        H << 2 * a, c,
            c,     2 * b;
        Eigen::Vector2d g(-d, -e);
        Eigen::Vector2d offset = H.ldlt().solve(g);

        peak_x = x0 + offset(0);
        peak_y = y0 + offset(1);
    }

    void get_peak_offset(double &peak_x, double &peak_y, double &max_val,
            const bool subpx, const std::string &method) {

        max_val = -std::numeric_limits<double>::infinity();
        int x0 = 0, y0 = 0;

        // Step 1: integer peak
        for (int y = 0; y < ss_size_y; ++y) {
            for (int x = 0; x < ss_size_x; ++x) {
                double val = cross_corr[y * ss_size_x + x];
                if (val > max_val) {
                    max_val = val;
                    x0 = x;
                    y0 = y;
                }
            }
        }

        const double center_x = static_cast<double>(ss_size_x) / 2.0;
        const double center_y = static_cast<double>(ss_size_y) / 2.0;

        // No subpixel refinement requested: map index -> displacement by subtracting center
        if (!subpx) {
            peak_x = static_cast<double>(x0) - center_x;
            peak_y = static_cast<double>(y0) - center_y;

            if (peak_x <= -center_x) peak_x += ss_size_x;
            if (peak_x >  center_x - 1e-12) peak_x -= ss_size_x;
            if (peak_y <= -center_y) peak_y += ss_size_y;
            if (peak_y >  center_y - 1e-12) peak_y -= ss_size_y;
            return;
        }

        // Subpixel refinement: build 3x3 neighborhood and solve quadratic (your existing code)
        int i = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int xw = wrap(x0 + dx, ss_size_x);
                int yw = wrap(y0 + dy, ss_size_y);
                double val = cross_corr[yw * ss_size_x + xw];

                if (method == "GAUSSIAN_2D" && val <= 0) val = 1e-6;
                double z = (method == "GAUSSIAN_2D") ? std::log(val) : val;

                A(i, 0) = dx * dx;
                A(i, 1) = dy * dy;
                A(i, 2) = dx * dy;
                A(i, 3) = dx;
                A(i, 4) = dy;
                A(i, 5) = 1.0;
                b(i) = z;
                i++;
            }
        }

        Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(b);
        double a = coeffs(0), b = coeffs(1), c = coeffs(2);
        double d = coeffs(3), e = coeffs(4);

        Eigen::Matrix2d H;
        H << 2 * a, c,
            c,     2 * b;
        Eigen::Vector2d g(-d, -e);

        Eigen::Vector2d offset = H.ldlt().solve(g);

        // Map integer + fractional index -> displacement relative to center
        double raw_x = static_cast<double>(x0) + offset(0);
        double raw_y = static_cast<double>(y0) + offset(1);

        peak_x = raw_x - center_x;
        peak_y = raw_y - center_y;

        if (peak_x <= -center_x) peak_x += ss_size_x;
        if (peak_x >  center_x - 1e-12) peak_x -= ss_size_x;
        if (peak_y <= -center_y) peak_y += ss_size_y;
        if (peak_y >  center_y - 1e-12) peak_y -= ss_size_y;
    }

    bool zero_norm_subsets(std::vector<double>& ref_vals, 
                        std::vector<double>& def_vals, 
                        const int ss_size_x,
                        const int ss_size_y) {
        const int total_px = ss_size_x * ss_size_y;

        // Compute means
        double mean_def = 0.0;
        double mean_ref = 0.0;
        for (int i = 0; i < total_px; ++i) {
            mean_def += def_vals[i];
            mean_ref += ref_vals[i];
        }
        mean_def /= total_px;
        mean_ref /= total_px;

        // Compute standard deviations
        double std_def = 0.0;
        double std_ref = 0.0;
        for (int i = 0; i < total_px; ++i) {
            std_def += std::pow(def_vals[i] - mean_def, 2);
            std_ref += std::pow(ref_vals[i] - mean_ref, 2);
        }
        std_def = std::sqrt(std_def / total_px);
        std_ref = std::sqrt(std_ref / total_px);


        // small tolerance to avoid division by zero - if std is too small, return without normalizing
        if (std_def < 1e-10 || std_ref < 1e-10) return false;

        // Normalize
        for (int i = 0; i < total_px; ++i) {
            def_vals[i] = (def_vals[i] - mean_def) / std_def;
            ref_vals[i] = (ref_vals[i] - mean_ref) / std_ref;
        }

        return true;
    }

    bool zero_norm_subset(subset::Pixels &ss, 
                        const int ss_size_x,
                        const int ss_size_y) {

        const int total_px = ss_size_x * ss_size_y;

        // Compute means
        double mean_ref = 0.0;
        int idx;
        for (int y = 0; y < ss_size_y; ++y) {
            for (int x = 0; x < ss_size_x; ++x) {

                idx = y * ss.size_x + x;
                mean_ref += ss.vals[idx];
            }
        }
        mean_ref /= total_px;

        // Compute standard deviations
        double std_ref = 0.0;
        for (int y = 0; y < ss_size_y; ++y) {
            for (int x = 0; x < ss_size_x; ++x) {
                idx = y * ss.size_x + x;
                std_ref += std::pow(ss.vals[idx] - mean_ref, 2);
            }
        }
        std_ref = std::sqrt(std_ref / total_px);


        if (std_ref < 1e-10) return false;

        // Normalize
        for (int y = 0; y < ss_size_y; ++y) {
            for (int x = 0; x < ss_size_x; ++x) {
                idx = y * ss.size_x + x;
                ss.vals[idx] = (ss.vals[idx] - mean_ref) / std_ref;
            }
        }

        return true;
    }

    bool zero_norm_subsets_centered(subset::Pixels &ss,
                                    const int ss_size_x,
                                    const int ss_size_y,
                                    const int window_size_x,
                                    const int window_size_y) {
        const int window_half_x = window_size_x / 2;
        const int window_half_y = window_size_y / 2;

        const int ss_half_x = ss_size_x / 2;
        const int ss_half_y = ss_size_y / 2;

        // ---- Compute means ----
        double mean_ref = 0.0;
        int count = 0;

        for (int row = -ss_half_y; row < ss_size_y - ss_half_y; ++row) {
            for (int col = -ss_half_x; col < ss_size_x - ss_half_x; ++col) {

                int x = window_half_x + col;
                int y = window_half_y + row;
                int idx = y * window_size_x + x;

                mean_ref += ss.vals[idx];
                ++count;
            }
        }

        mean_ref /= count;

        // ---- Compute std ----
        double std_ref = 0.0;

        for (int row = -ss_half_y; row < ss_size_y - ss_half_y; ++row) {
            for (int col = -ss_half_x; col < ss_size_x - ss_half_x; ++col) {

                int x = window_half_x + col;
                int y = window_half_y + row;
                int idx = y * window_size_x + x;

                std_ref += std::pow(ss.vals[idx] - mean_ref, 2);
            }
        }

        std_ref = std::sqrt(std_ref / count);

        if (std_ref < 1e-10) return false;

        // ---- Normalize ----
        for (int row = -ss_half_y; row < ss_size_y - ss_half_y; ++row) {
            for (int col = -ss_half_x; col < ss_size_x - ss_half_x; ++col) {

                int x = window_half_x + col;
                int y = window_half_y + row;
                int idx = y * window_size_x + x;

                ss.vals[idx] = (ss.vals[idx] - mean_ref) / std_ref;
            }
        }
        return true;
    }
};

    void get_single_window_fftcc_peak(std::vector<double> &p,
                                      double &max_val,
                                      const double cx, const double cy,
                                      const int ss_size_x, 
                                      const int ss_size_y,
                                      const int window_size_x, 
                                      const int window_size_y,
                                      const Image &img_ref, const Image &img_def,
                                      const Interpolator &interp_def,
                                      const bool debug=false);



    void get_single_window_fftcc_peak_centre(std::vector<double> &p,
                                         double &max_val,
                                         const double cx, const double cy,
                                         const double offset_x, const double offset_y,
                                         const int ss_size_x, 
                                         const int ss_size_y, 
                                         const int window_size_x,
                                         const int window_size_y,
                                         const Image &img_ref, const Image &img_def,
                                         const Interpolator &interp_def,
                                         const bool debug=false);

    void fill_fft_window_with_subset_at_centre(subset::Pixels &ss_ref,
                                     const Image &img_ref,
                                     const int ss_x,
                                     const int ss_y,
                                     const int px_hori,
                                     const int px_vert,
                                     const int ss_size_x,
                                     const int ss_size_y,
                                     const int window_size_x,
                                     const int window_size_y);


    void fill_fft_window_with_subset_at_corner(subset::Pixels &ss_ref,
                                               const Image &img_ref,
                                               const int ss_x,
                                               const int ss_y,
                                               const int px_hori,
                                               const int px_vert,
                                               const int ss_size_x,
                                               const int ss_size_y,
                                               const int window_size_x,
                                               const int window_size_y);

    /**
    * Clamp a subset top-left coordinate so that the window fits inside the image.
    *
    * @param ss_coord      Subset top-left coordinate (x or y)
    * @param window_half   Half-size of the FFT window in this direction
    * @param ss_half       Half-size of the subset in this direction
    * @param img_size      Image size (width or height)
    * @param window_size   FFT window size in this direction
    * @return              Clamped coordinate
    */
    inline int clamp_subset(int ss_coord, int window_half, int ss_half, int img_size, int window_size) {
        return std::clamp(ss_coord - window_half + ss_half, 0, img_size - window_size);
    }

    /**
    * Compute offsets to center the subset within the FFT window.
    *
    * @param window_half   Half-size of the FFT window in this direction
    * @param ss_half       Half-size of the subset in this direction
    * @param ss_coord      Subset top-left coordinate (x or y)
    * @return              Offset for placing subset in window
    */
    inline int compute_subset_offset(int window_half, int ss_half, int ss_coord) {
        return std::min(window_half - ss_half, ss_coord);
    }

    /**
    * Reset a rectangular region inside the FFT reference window to zero.
    *
    * @param vals            window intensity values
    * @param offset_x        x offset of subset in FFT window
    * @param offset_y        y offset of subset in FFT window
    * @param ss_size_x       width of subset
    * @param ss_size_y       height of subset
    * @param window_size_x   total FFT window width
    * @param window_size_y   total FFT window height
    */
    inline void reset_fft_ref_subset(std::vector<double> &vals,
                                     int offset_x, int offset_y,
                                     int ss_size_x, int ss_size_y,
                                     int window_size_x, int window_size_y){

        for (int row = 0; row < ss_size_y; ++row) {
            for (int col = 0; col < ss_size_x; ++col) {
                int target_y = offset_y + row;
                int target_x = offset_x + col;
                int idx_window = target_y * window_size_x + target_x;

                if (idx_window >= window_size_x * window_size_y) {
                    std::cerr << "reset_fft_ref_subset: idx_window out of bounds: "
                            << idx_window << " target_x: " << target_x
                            << " target_y: " << target_y << std::endl;
                    exit(1);
                }

                vals[idx_window] = 0.0;
            }
        }
    }

    double debugcost(subset::Pixels &ss_ref, subset::Pixels &ss_def);
    void smooth_field(std::vector<double>& shift, const subset::Grid& ss_grid, double sigma, int radius);

    
    /**
    * @brief Computes the Hanning window value at a given location.
    *
    *   w(n) = 0.5 * (1 - cos(2πn / (N - 1)))
    *
    * @param row     Row index (0 <= row < size_y)
    * @param col     Column index (0 <= col < size_x)
    * @param size_x  Total number of columns (must be > 1)
    * @param size_y  Total number of rows (must be > 1)
    *
    * @return The 2D Hann window coefficient at (row, col).
    */
    double hanning(const int row, const int col, const int size_x, const int size_y);


    /**
    * @brief Computes the 2D Hamming window value at a given position.
    *
    *   w(n) = 0.54 - 0.46 * cos(2πn / (N - 1))
    *
    * @param row     Row index (0 <= row < size_y)
    * @param col     Column index (0 <= col < size_x)
    * @param size_x  Total number of columns (must be > 1)
    * @param size_y  Total number of rows (must be > 1)
    *
    * @return The 2D Hamming window coefficient at (row, col).
    */
    double hamming(const int row, const int col, const int size_x, const int size_y);



#endif // DICFOURIER_H

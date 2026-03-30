// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include "dicutil.hpp"
#include <iostream>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <omp.h>
#include <csignal>

// Common Header files

// DIC Header files
#include "dicfourier.hpp"
#include "dicsubset.hpp"
#include "dicinterp.hpp"


void smooth_field(std::vector<double>& shift,
                const subset::Grid& ss_grid,
                double sigma = 1.0,
                int radius = 2) {

    std::vector<double> smoothed = shift;

    const int width = ss_grid.num_ss_x;
    const int height = ss_grid.num_ss_y;

    // Precompute Gaussian weights
    std::vector<std::vector<double>> weights(2 * radius + 1, std::vector<double>(2 * radius + 1));
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            double dist2 = dx * dx + dy * dy;
            weights[dy + radius][dx + radius] = std::exp(-dist2 / (2.0 * sigma * sigma));
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            int center_idx = ss_grid.mask[y * width + x];
            if (center_idx == -1) continue;

            double sum = 0.0;
            double weight_sum = 0.0;

            for (int dy = -radius; dy <= radius; ++dy) {
                int ny = y + dy;
                if (ny < 0 || ny >= height) continue;

                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= width) continue;

                    int n_idx = ss_grid.mask[ny * width + nx];
                    if (n_idx == -1) continue;

                    double val = shift[n_idx];
                    double weight = weights[dy + radius][dx + radius];

                    sum += val * weight;
                    weight_sum += weight;
                }
            }

            if (weight_sum > 0.0) {
                smoothed[center_idx] = sum / weight_sum;
            }
        }
    }

    shift = std::move(smoothed);
}


double debugcost(const subset::Pixels &ss_ref, const subset::Pixels &ss_def){
    const int num_px = ss_def.num_px;
    double cost = 0.0;
    double mean_ref = 0.0;
    double mean_def = 0.0;

    // loop over pixel values in reference image
    for (int i = 0; i < num_px; i++){
        mean_ref += ss_ref.vals[i];
        mean_def += ss_def.vals[i];
    }

    mean_ref /= num_px;
    mean_def /= num_px;

    // get cost function denominators
    double sum_squared_ref = 0.0;
    double sum_squared_def = 0.0;
    for (int i = 0; i < num_px; ++i) {
        sum_squared_ref += (ss_ref.vals[i] - mean_ref)*
                            (ss_ref.vals[i] - mean_ref);
        sum_squared_def += (ss_def.vals[i] - mean_def)*
                            (ss_def.vals[i] - mean_def);
    }
    double inv_sum_squared_ref = 1.0 / std::sqrt(sum_squared_ref);
    double inv_sum_squared_def = 1.0 / std::sqrt(sum_squared_def);


    // calcualte cost 
    for (int i = 0; i < num_px; i++){
        double def_norm = ss_def.vals[i] * inv_sum_squared_def;
        double ref_norm = ss_ref.vals[i] * inv_sum_squared_ref;
        cost += (ref_norm - def_norm) * (ref_norm - def_norm);
    }
    return cost;
}






void get_single_window_fftcc_peak(std::vector<double> &p,
                                  const double cx, const double cy,
                                  const int ss_size_x, 
                                  const int ss_size_y, 
                                  const int window_size_x,
                                  const int window_size_y,
                                  const double *img_ref, const double *img_def,
                                  const Interpolator &interp_def){

    // some consts
    const int px_hori = interp_def.px_hori;
    const int px_vert = interp_def.px_vert;
    const int window_half_x = window_size_x/2;
    const int window_half_y = window_size_y/2;
    const int ss_half_x = ss_size_x/2;
    const int ss_half_y = ss_size_y/2;


    // reset p values
    std::fill(p.begin(), p.end(),0.0);

    // class for FFT
    FFT fft(window_size_x, window_size_y);

    // TODO: Add a proper flag for this 
    bool subpx = true;

    // put the subset at the centre of the window
    int corner_x = cx - ss_size_x/2;
    int corner_y = cy - ss_size_y/2;

    fill_fft_window_with_subset_at_corner(fft.ss_ref, img_ref,
                                          corner_x, corner_y, px_hori, px_vert,
                                          ss_size_x, ss_size_y,
                                          window_size_x, window_size_y);

    // populate deformed subset
    subset::fill_from_img_subpx(fft.ss_def,
                                corner_x-window_half_x,
                                corner_y-window_half_y,
                                interp_def);

    // zero norm the subsets
    fft.zero_norm_subset(fft.ss_ref, ss_size_x,ss_size_y);
    fft.zero_norm_subset(fft.ss_def, window_size_x,window_size_y);

    // get peaks from the cross correlation
    double peak_x = 0.0, peak_y = 0.0, max_val = 0.0;
    fft.correlate();
    fft.get_peak_nowrap(peak_x, peak_y, max_val, subpx, "gaussian_2d");

    // coordinate transform
    p[0] = peak_x - window_half_x;
    p[1] = peak_y - window_half_y;

    // debugging
    // std::cout << std::endl;
    // for (int row = 0; row < window_size_y; ++row) {
    //     for (int col = 0; col < window_size_x; ++col) {
    //         int idx  = row*window_size_x+col;
    //         std::cout << col << " " << row << " ";
    //         std::cout << fft.ss_ref.x[idx] << " " << fft.ss_ref.y[idx] << " " << fft.ss_ref.vals[idx] << " ";
    //         std::cout << fft.ss_def.x[idx] << " " << fft.ss_def.y[idx] << " " << fft.ss_def.vals[idx] << " ";
    //         std::cout << fft.cross_corr[idx] << std::endl;
    //     }
    // }
    //
    // std::cout << std::endl;
    // std::cout << peak_x << " " << peak_y << std::endl;
    // exit(0);
}

void get_offcentered_fftcc_peak(double &peak_x, double &peak_y,
                                const int ss_x, const int ss_y,
                                const int ss_size_x, const int ss_size_y,
                                const int window_x, const int window_y,
                                const int window_size_x, const int window_size_y,
                                const double *img_ref, const double *img_def,
                                const Interpolator &interp_def){

    const int px_hori = interp_def.px_hori;
    const int px_vert = interp_def.px_vert;
    const int window_half_x = window_size_x/2;
    const int window_half_y = window_size_y/2;
    const int ss_half_x = ss_size_x/2;
    const int ss_half_y = ss_size_y/2;

    // class for FFT
    FFT fft(window_size_x, window_size_y);

    // TODO: Add a proper flag for this 
    bool subpx = true;

    // put the subset at the centre of the window
    fill_fft_window_with_subset_at_centre(fft.ss_ref, img_ref,
                                ss_x, ss_y, px_hori, px_vert,
                                ss_size_x, ss_size_y,
                                window_size_x, window_size_y);

    // populate fft.ss_def with interpolator values
    subset::fill_from_img_subpx(fft.ss_def, window_x, window_y, interp_def);

    // apply hanning window to ss_def
    for (int row = 0; row < window_size_y; ++row) {
        for (int col = 0; col < window_size_x; ++col) {
            double coeff = hanning(row,col,window_size_x, window_size_y);
            fft.ss_def.vals[row*window_size_x+col] *= coeff;
        }
    }


    // get peaks from the cross correlation
    double max_val = 0.0;
    fft.correlate();
    fft.get_peak(peak_x, peak_y, max_val, subpx, "gaussian_2d");
}

void fill_fft_window_with_subset_at_centre(subset::Pixels &ss_ref,
                                            const double *img_ref,
                                            const int ss_x,
                                            const int ss_y,
                                            const int px_hori,
                                            const int px_vert,
                                            const int ss_size_x,
                                            const int ss_size_y,
                                            const int window_size_x,
                                            const int window_size_y){

const int window_half_x = window_size_x / 2;
const int window_half_y = window_size_y / 2;
const int ss_half_x = ss_size_x / 2;
const int ss_half_y = ss_size_y / 2;

// Iterate over subset pixels using offsets relative to the subset center
for (int row = -ss_half_y; row < ss_size_y - ss_half_y; ++row) {
    for (int col = -ss_half_x; col < ss_size_x - ss_half_x; ++col) {

        int px_x = ss_x + col;
        int px_y = ss_y + row;

        int target_x = window_half_x + col;
        int target_y = window_half_y + row;

        if (px_x < 0 || px_x >= px_hori || px_y < 0 || px_y >= px_vert) {
            std::cout << "Image access out of bounds! px: ("
                        << px_x << ", " << px_y << ")\n";
            continue;
        }
        if (target_x < 0 || target_x >= window_size_x ||
            target_y < 0 || target_y >= window_size_y) {
            std::cout << "Window access out of bounds! target: ("
                        << target_x << ", " << target_y << ")\n";
            continue;
        }

        int idx_img    = px_y * px_hori + px_x;
        int idx_window = target_y * window_size_x + target_x;

        double coeff = hamming(row + ss_half_y, col + ss_half_x, ss_size_x, ss_size_y);

        ss_ref.x[idx_window]    = px_x;
        ss_ref.y[idx_window]    = px_y;
        ss_ref.vals[idx_window] = coeff * img_ref[idx_img];
    }
}
}

void fill_fft_window_with_subset_at_corner(subset::Pixels &ss_ref,
                                            const double *img_ref,
                                            const int ss_x,
                                            const int ss_y,
                                            const int px_hori,
                                            const int px_vert,
                                            const int ss_size_x,
                                            const int ss_size_y,
                                            const int window_size_x,
                                            const int window_size_y){

    const int window_half_x = window_size_x / 2;
    const int window_half_y = window_size_y / 2;
    const int ss_half_x = ss_size_x / 2;
    const int ss_half_y = ss_size_y / 2;

    // Iterate over subset pixels using offsets relative to the subset center
    for (int row = 0; row < ss_size_y; ++row) {
        for (int col = 0; col < ss_size_x; ++col) {

            int px_x = ss_x + col;
            int px_y = ss_y + row;

            if (px_x < 0 || px_x >= px_hori || px_y < 0 || px_y >= px_vert) {
                std::cout << "Image access out of bounds! px: ("
                        << px_x << ", " << px_y << ")\n";
                continue;
            }

            int idx_img    = px_y * px_hori + px_x;
            int idx_window = row * window_size_x + col;
            double coeff = 1.0; //fourier::hamming(row, col, ss_size_x, ss_size_y);
            ss_ref.x[idx_window]    = px_x;
            ss_ref.y[idx_window]    = px_y;
            ss_ref.vals[idx_window] = coeff * img_ref[idx_img];

        }
    }
}

double hanning(const int row, const int col, const int size_x, const int size_y){
    const double hann_row = 0.5 * (1.0 - cos(2.0 * M_PI * row / (size_y - 1)));
    const double hann_col = 0.5 * (1.0 - cos(2.0 * M_PI * col / (size_x - 1)));
    return hann_row * hann_col;
}


double hamming(const int row, const int col, const int size_x, const int size_y){
    const double ham_row = 0.54 - 0.46 * cos(2.0 * M_PI * row / (size_y - 1));
    const double ham_col = 0.54 - 0.46 * cos(2.0 * M_PI * col / (size_x - 1));
    return ham_row * ham_col;
}




// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <csignal>

// Common Header files

// DIC Header files
#include "dicfourier.hpp"
#include "dicsubset.hpp"
#include "dicinterp.hpp"


// Helper: returns the center offset (0-based) for a given size
// odd  size N -> (N-1)/2   e.g. 31 -> 15
// even size N -> (N-1)/2.0 e.g. 32 -> 15.5
inline double half_offset(int size) {
    return (size - 1) * 0.5;
}


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





template<typename Real>
void fill_fft_window_from_img_subpx(FFTPixels<Real> &ss_def,
                                    const double subpx_x,
                                    const double subpx_y,
                                    const Interpolator &interp_def) {
    int count = 0;
    for (int y = 0; y < ss_def.size_y; y++) {
        for (int x = 0; x < ss_def.size_x; x++) {
            const double px_x = subpx_x + x;
            const double px_y = subpx_y + y;
            if (ss_def.has_coords()) {
                ss_def.x[count] = px_x;
                ss_def.y[count] = px_y;
            }
            ss_def.vals[count] = static_cast<Real>(interp_def.eval(0, 0, px_x, px_y));
            count++;
        }
    }
}

template<typename Real>
void get_single_window_fftcc_peak(FFTImpl<Real> &fft,
                                  std::vector<double> &p,
                                  double &max_val,
                                  const double cx, const double cy,
                                  const int ss_size_x, 
                                  const int ss_size_y, 
                                  const int window_size_x,
                                  const int window_size_y,
                                  const Image &img_ref, const Image &img_def,
                                  const Interpolator &interp_def,
                                  const bool debug){

    // some consts
    const int px_hori = interp_def.px_hori;
    const int px_vert = interp_def.px_vert;

    // reset p values
    std::fill(p.begin(), p.end(),0.0);

    // TODO: Add a proper flag for this 
    bool subpx = true;

    // top left corner of the subset
    int corner_x = (int)std::floor(cx - (ss_size_x - 1) * 0.5);
    int corner_y = (int)std::floor(cy - (ss_size_y - 1) * 0.5);

    fill_fft_window_with_subset_at_corner(fft.ss_ref, img_ref,
                                          corner_x, corner_y, px_hori, px_vert,
                                          ss_size_x, ss_size_y,
                                          window_size_x, window_size_y);

    // populate deformed subset
    fill_fft_window_from_img_subpx(fft.ss_def,
                                    cx - half_offset(window_size_x),
                                    cy - half_offset(window_size_y),
                                    interp_def);

    // zero norm the subsets
    bool normed_ref = fft.zero_norm_subset(fft.ss_ref, ss_size_x,ss_size_y);
    bool normed_def = fft.zero_norm_subset(fft.ss_def, window_size_x,window_size_y);

    // get peaks from the cross correlation
    double peak_x = 0.0, peak_y = 0.0;

    if (normed_ref && normed_def){
        fft.correlate();
        fft.get_peak_nowrap(peak_x, peak_y, max_val, subpx, "GAUSSIAN_2D");
    } 

    // coordinate transform
    p[0] = peak_x - half_offset(window_size_x);
    p[1] = peak_y - half_offset(window_size_y);

    // debugging
    if (debug) {
        for (int row = 0; row < window_size_y; ++row) {
            for (int col = 0; col < window_size_x; ++col) {
                int idx  = row*window_size_x+col;
                std::cout << col << " " << row << " ";
                std::cout << fft.ss_ref.x[idx] << " " << fft.ss_ref.y[idx] << " " << fft.ss_ref.vals[idx] << " ";
                std::cout << fft.ss_def.x[idx] << " " << fft.ss_def.y[idx] << " " << fft.ss_def.vals[idx] << " ";
                std::cout << fft.cross_corr[idx] << std::endl;
            }
        }
        std::cout << std::endl;
    }
}


template<typename Real>
void get_single_window_fftcc_peak_centre(FFTImpl<Real> &fft,
                                         std::vector<double> &p,
                                         double &max_val,
                                         const double cx, const double cy,
                                         const double offset_x, const double offset_y,
                                         const int ss_size_x, 
                                         const int ss_size_y, 
                                         const int window_size_x,
                                         const int window_size_y,
                                         const Interpolator &interp_ref,
                                         const Interpolator &interp_def,
                                         const bool debug){

    // some consts
    const int px_hori = interp_def.px_hori;
    const int px_vert = interp_def.px_vert;
    const double window_half_x = half_offset(window_size_x);
    const double window_half_y = half_offset(window_size_y);
    const double ss_half_x = half_offset(ss_size_x);
    const double ss_half_y = half_offset(ss_size_y);


    // reset p values
    std::fill(p.begin(), p.end(),0.0);

    // TODO: Add a proper flag for this 
    bool subpx = true;

    fill_fft_window_with_subset_at_centre(fft.ss_ref, interp_ref,
                                          cx, cy, px_hori, px_vert,
                                          ss_size_x, ss_size_y,
                                          window_size_x, window_size_y);

    // populate deformed subset
    fill_fft_window_from_img_subpx(fft.ss_def,
                                    cx-window_half_x+offset_x,
                                    cy-window_half_y+offset_y,
                                    interp_def);



    // zero norm the subsets
    bool normed_ref = fft.zero_norm_subsets_centered(fft.ss_ref,ss_size_x,ss_size_y, window_size_x, window_size_y);
    bool normed_def = fft.zero_norm_subset(fft.ss_def, window_size_x,window_size_y);

    // get peaks from the cross correlation
    double peak_x = 0.0, peak_y = 0.0;

    if (normed_ref && normed_def){
        fft.correlate();
        fft.get_peak(peak_x, peak_y, max_val, subpx, "GAUSSIAN_2D");
    }

    // coordinate transform
    p[0] = peak_x;
    p[1] = peak_y;
    
    //debugging
    if (debug) {
        for (int row = 0; row < window_size_y; ++row) {
            for (int col = 0; col < window_size_x; ++col) {
                int idx  = row*window_size_x+col;
                std::cout << col << " " << row << " ";
                std::cout << fft.ss_ref.x[idx] << " " << fft.ss_ref.y[idx] << " " << fft.ss_ref.vals[idx] << " ";
                std::cout << fft.ss_def.x[idx] << " " << fft.ss_def.y[idx] << " " << fft.ss_def.vals[idx] << " ";
                std::cout << fft.cross_corr[idx] << std::endl;
            }
        }
        std::cout << std::endl;
    }
}


template<typename Real>
void fill_fft_window_with_subset_at_centre(FFTPixels<Real> &ss_ref,
                                           const Interpolator &interp_ref,
                                           const double cx,
                                           const double cy,
                                           const int px_hori,
                                           const int px_vert,
                                           const int ss_size_x,
                                           const int ss_size_y,
                                           const int window_size_x,
                                           const int window_size_y) {

    const double w_half_x = half_offset(window_size_x);  // e.g. 15.5 for 32
    const double w_half_y = half_offset(window_size_y);
    const double ss_half_x = half_offset(ss_size_x);
    const double ss_half_y = half_offset(ss_size_y);

    // col offset runs over ss pixels: 0, 1, ..., ss_size_x-1
    // pixel coord = cx - ss_half_x + col  (exact, works for even & odd)
    // window coord = col + (w_half_x - ss_half_x)  (centred in window)

    const double win_origin_x = w_half_x - ss_half_x; // offset of ss[0] in window
    const double win_origin_y = w_half_y - ss_half_y;

    for (int row = 0; row < ss_size_y; ++row) {
        for (int col = 0; col < ss_size_x; ++col) {

            // true (possibly fractional) image coordinate
            double px_x = (cx - ss_half_x) + col;
            double px_y = (cy - ss_half_y) + row;

            // nearest integer pixel for bounds check & x/y storage
            int ipx_x = (int)std::round(px_x);
            int ipx_y = (int)std::round(px_y);

            // window index — round to nearest integer window cell
            int target_x = (int)std::round(win_origin_x + col);
            int target_y = (int)std::round(win_origin_y + row);

            int idx_window = target_y * window_size_x + target_x;

            if (ss_ref.has_coords()) {
                ss_ref.x[idx_window] = px_x;
                ss_ref.y[idx_window] = px_y;
            }

            if (ipx_x < 0 || ipx_x >= px_hori || ipx_y < 0 || ipx_y >= px_vert) {
                ss_ref.vals[idx_window] = Real(0);
            } else {
                ss_ref.vals[idx_window] = static_cast<Real>(interp_ref.eval(0, 0, px_x, px_y));
            }
        }
    }
}



template<typename Real, typename T>
void fill_fft_window_with_subset_at_corner_impl(FFTPixels<Real> &ss_ref,
                                            const std::vector<T> &img,
                                            const int corner_x,
                                            const int corner_y,
                                            const int px_hori,
                                            const int px_vert,
                                            const int ss_size_x,
                                            const int ss_size_y,
                                            const int window_size_x,
                                            const int window_size_y){


    // Iterate over subset pixels using offsets relative to the subset center
    for (int row = 0; row < ss_size_y; ++row) {
        for (int col = 0; col < ss_size_x; ++col) {

            int px_x = corner_x + col;
            int px_y = corner_y + row;

            int idx_img    = px_y * px_hori + px_x;
            int idx_window = row * window_size_x + col;
            double coeff = 1.0; //fourier::hamming(row, col, ss_size_x, ss_size_y);
            if (ss_ref.has_coords()) {
                ss_ref.x[idx_window] = px_x;
                ss_ref.y[idx_window] = px_y;
            }

            if (px_x < 0 || px_x >= px_hori || px_y < 0 || px_y >= px_vert) {
                ss_ref.vals[idx_window] = Real(0);
            }
            else { 
                ss_ref.vals[idx_window] = static_cast<Real>(coeff * img[idx_img]);
            }

        }
    }
}

template<typename Real>
void fill_fft_window_with_subset_at_corner(FFTPixels<Real> &ss_ref,
                                           const Image &img_ref,
                                           const int corner_x,
                                           const int corner_y,
                                           const int px_hori,
                                           const int px_vert,
                                           const int ss_size_x,
                                           const int ss_size_y,
                                           const int window_size_x,
                                           const int window_size_y) {
    switch (img_ref.type) {
        case PixelType::UINT8:
            fill_fft_window_with_subset_at_corner_impl(
                ss_ref, img_ref.data8,
                corner_x, corner_y, px_hori, px_vert,
                ss_size_x, ss_size_y,
                window_size_x, window_size_y);
            break;

        case PixelType::UINT16:
            fill_fft_window_with_subset_at_corner_impl(
                ss_ref, img_ref.data16,
                corner_x, corner_y, px_hori, px_vert,
                ss_size_x, ss_size_y,
                window_size_x, window_size_y);
            break;

        case PixelType::UINT32:
            fill_fft_window_with_subset_at_corner_impl(
                ss_ref, img_ref.data32,
                corner_x, corner_y, px_hori, px_vert,
                ss_size_x, ss_size_y,
                window_size_x, window_size_y);
            break;

        case PixelType::UINT32F:
            fill_fft_window_with_subset_at_corner_impl(
                ss_ref, img_ref.data32f,
                corner_x, corner_y, px_hori, px_vert,
                ss_size_x, ss_size_y,
                window_size_x, window_size_y);
            break;

        default:
            throw std::runtime_error("Unsupported pixel type");
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

template void get_single_window_fftcc_peak<float>(FFTf&, std::vector<double>&, double&, double, double, int, int, int, int, const Image&, const Image&, const Interpolator&, bool);
template void get_single_window_fftcc_peak<double>(FFT&, std::vector<double>&, double&, double, double, int, int, int, int, const Image&, const Image&, const Interpolator&, bool);
template void get_single_window_fftcc_peak_centre<float>(FFTf&, std::vector<double>&, double&, double, double, double, double, int, int, int, int, const Interpolator&, const Interpolator&, bool);
template void get_single_window_fftcc_peak_centre<double>(FFT&, std::vector<double>&, double&, double, double, double, double, int, int, int, int, const Interpolator&, const Interpolator&, bool);
template void fill_fft_window_with_subset_at_centre<float>(FFTPixels<float>&, const Interpolator&, double, double, int, int, int, int, int, int);
template void fill_fft_window_with_subset_at_centre<double>(FFTPixels<double>&, const Interpolator&, double, double, int, int, int, int, int, int);
template void fill_fft_window_with_subset_at_corner<float>(FFTPixels<float>&, const Image&, int, int, int, int, int, int, int, int);
template void fill_fft_window_with_subset_at_corner<double>(FFTPixels<double>&, const Image&, int, int, int, int, int, int, int, int);


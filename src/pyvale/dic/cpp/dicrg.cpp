// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <csignal>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <omp.h>
#include <memory>



// Program Header files
#include "./dicutil.hpp"
#include "./defines.hpp"
#include "./dicrg.hpp"
#include "./dicfourier.hpp"


namespace rg {

    int next_pow2(int n) {
        if (n <= 0){
            std::cerr << __FILE__ << " " << __LINE__ << std::endl;
            std::cerr << "Expected a positive integer to calculate next power of 2 " << std::endl;
            std::cerr << "n = " << n << std::endl;
            exit(EXIT_FAILURE);
        }

        // If already a power of 2, return as-is
        if ((n & (n - 1)) == 0) return n;

        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n++;

        // Handle possible overflow
        if (n < 0) return std::numeric_limits<int>::max();

        return n;
    }


    std::vector<int> pow2_between(int n, int x) {
        std::vector<int> result;
        int power = next_pow2(n);
        while (power >= x) {
            result.push_back(power);
            power /= 2;
        }
        return result;
    }

    bool is_valid_point(const int ss_x, const int ss_y, const util::SubsetData &ssdata) {

        int x = ss_x / ssdata.step;
        int y = ss_y / ssdata.step;

        int idx = y * ssdata.num_ss_x + x;

        if ((ss_x % ssdata.step) || (ss_y % ssdata.step)){
            std::cerr << "Subset coordinates (" << ss_x << ", " << ss_y << ") are not a valid subset location." << std::endl;
            std::cerr << "Subset ss_step size: " << ssdata.step << std::endl;
            return false;
            exit(EXIT_FAILURE);
        }
        else if (ssdata.mask[idx] == -1){
            std::cerr << "Subset coordinates (" << ss_x << ", " << ss_y << ") are not a valid subset location." << std::endl;
            std::cerr << "subset mask index: " << idx << std::endl;
            return false;
            exit(EXIT_FAILURE);
        }
        else return true;

        //auto it = ssdata.coords_to_idx.find({ss_x, ss_y});

        //// check if coordinates are in the coordinate list
        //if (it == ssdata.coords_to_idx.end()) {
        //   std::cerr << "Error: coordinates not found in the coordinate list." << std::endl;
        //   std::cerr << "Coordinates: " << ss_x << ", " << ss_y << std::endl;
        //   exit(EXIT_FAILURE);
        //}
        //else return true;
    }


    void get_rigid_shift(double &shift_x, double &shift_y,
                         const int ss_x, const int ss_y,
                         std::vector<std::unique_ptr<fourier::FFT>>& fft_windows,
                         const Interpolator &interp_ref,
                         const double *img_def){


        const int px_hori = interp_ref.px_hori;
        const int px_vert = interp_ref.px_vert;

        double prev_x = 0, prev_y = 0;

        // loop over window sizes
        for (size_t w = 0; w < fft_windows.size(); w++){

            // get the deformed subset values
            util::extract_ss(fft_windows[w]->ss_def, ss_x, ss_y, px_hori, px_vert, img_def);

            // add shift from previous window size;
            double ss_x_shft = ss_x-prev_x;
            double ss_y_shft = ss_y-prev_y;

            // get the reference subset values from interpolator
            util::extract_ss_subpx(fft_windows[w]->ss_ref, ss_x_shft, ss_y_shft, interp_ref);

            // zero normalise the subsets
            fourier::zero_norm_subsets(fft_windows[w]->ss_def.vals, fft_windows[w]->ss_ref.vals, fft_windows[w]->ss_def.size);

            // get peaks from the cross correlation
            double peak_x = 0, peak_y = 0, max_val = 0.0;
            fft_windows[w]->correlate();
            fft_windows[w]->find_peak(peak_x, peak_y, max_val, true, "GAUSSIAN_2D");
            prev_x += peak_x;
            prev_y += peak_y;
        }

        // return the shift once we've reached smallest window size
        shift_x = prev_x;
        shift_y = prev_y;
    }
}



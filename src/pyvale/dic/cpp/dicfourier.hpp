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
#include <Eigen/Dense>

// Program Header files
#include "./dicinterpolator.hpp"
#include "./defines.hpp"
#include "./dicutil.hpp"
#include "./pocketfft_hdronly.h"

namespace fourier {

    struct Shift {

        // number of neighbours to use for removing outliers
        size_t num_neigh;

        //integer shifts
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> cost;
        std::vector<double> max_val;

        // list of neighbours from prev window
        std::vector<int> neighlist;

        void gen_neighlist(const util::SubsetData ssdata,
                           const util::SubsetData ssdata_prev) {

            //util::Timer timer("nearest neighbour collection for :");

            const int prev_step = ssdata_prev.step;

            // For each subset, find 4 nearest neighbours in ssdata_prev
            #pragma omp parallel for
            for (int ss = 0; ss < ssdata.num; ++ss) {

                // corner of subset
                const int ss_x = ssdata.coords[2*ss];
                const int ss_y = ssdata.coords[2*ss+1];

                // Vector to store pairs of (distance, index)
                std::vector<std::pair<double, int>> dist_index_list;

                // loop over a 10x10 section from the previous window
                int idx_x = (ss_x / prev_step);
                int idx_y = (ss_y / prev_step);

                // range of neighbour search
                int min_x = std::max(0,idx_x-5);
                int min_y = std::max(0,idx_y-5);
                int max_x = std::min(ssdata_prev.num_ss_x,idx_x+6);
                int max_y = std::min(ssdata_prev.num_ss_y,idx_y+6);

                for (int y = min_y; y < max_y; y++){
                    for (int x = min_x; x < max_x; x++){

                    // check if point is a valid subset
                    int nss_idx = ssdata_prev.mask[y*ssdata_prev.num_ss_x+x];
                    if (nss_idx == -1) continue;

                    int nss_x = ssdata_prev.coords[2*nss_idx];
                    int nss_y = ssdata_prev.coords[2*nss_idx+1];

                    double dx = (nss_x) - ss_x;
                    double dy = (nss_y) - ss_y;
                    double dist_sq = dx*dx + dy*dy;

                    dist_index_list.emplace_back(dist_sq, nss_idx);
                    }
                }

                // Partial sort to get 4 nearest neighbours
                if (dist_index_list.size() > num_neigh) {
                    std::nth_element(dist_index_list.begin(), dist_index_list.begin() + num_neigh, dist_index_list.end());
                    dist_index_list.resize(num_neigh);
                }
                else {
                    std::cerr << "Could not not find " << num_neigh << " neihbours for point (" << ss_x << ", " << ss_y << ")." << std::endl;
                    std::cerr << "Number of neighbours: " << dist_index_list.size() << std::endl;
                    std::cerr << "Neighbours from previous window: " << std::endl;
                    for (size_t n = 0; n < dist_index_list.size(); n++){
                        int nss_idx = dist_index_list[n].second;
                        int nss_x = ssdata_prev.coords[2*nss_idx];
                        int nss_y = ssdata_prev.coords[2*nss_idx+1];
                        std::cerr << "(" << nss_x << ", " << nss_y << "), ";
                    }
                    std::cerr << std::endl;
                    exit(EXIT_FAILURE);
                }


                // Store neighbours indices into neighlist
                for (size_t i = 0; i < num_neigh; ++i) {
                    //std::cout << ss_x << " " << ss_y << std::endl;
                    neighlist[ss*num_neigh+i] = dist_index_list[i].second;
                    //int nidx = neighlist[ss*num_neigh+i];
                    //std::cout << ssdata_prev.coords[nidx*2] << " " << ssdata_prev.coords[nidx*2+1] << std::endl; 
                }
                //std::cout << std::endl;
            }
            //exit(0);
        }
    };

    extern std::vector<Shift> shifts;

    struct FFT {
        int ss_size;
        int n_complex;
        
        // input data
        util::Subset ss_def;
        util::Subset ss_ref;

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

        FFT(int ss_size_)
            : ss_size(ss_size_), n_complex(ss_size_/2+1), ss_def(ss_size_), ss_ref(ss_size_),
                fft_def(ss_size_*n_complex), fft_ref(ss_size_*n_complex), 
                cross_corr(ss_size_ * ss_size_),  A(9,6), b(9)
        {
       
            shape_in   = {static_cast<unsigned long>(ss_size), static_cast<unsigned long>(ss_size)};
            stride_in  = {static_cast<long>(ss_size * sizeof(double)), sizeof(double)};
            stride_out = {static_cast<long>(n_complex * sizeof(std::complex<double>)), sizeof(std::complex<double>)};

        }



        void correlate() {

            // forward fft of reference and deformed subsets
            pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD, ss_ref.vals.data(), fft_ref.data(), 1.0, 1);
            pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD, ss_def.vals.data(), fft_def.data(), 1.0, 1);
            
            // multiplication of complex fft data
            for (int px = 0; px < ss_size * n_complex; px++) {
                fft_def[px] = std::conj(fft_ref[px]) * fft_def[px];
            }

            // inverse FFT to get cross correlation
            pocketfft::c2r(shape_in, stride_out, stride_in, axes, pocketfft::BACKWARD, fft_def.data(), cross_corr.data(), 1.0, 1);
        }

    void fftshift(std::vector<double>& data, int size) {
        std::vector<double> temp(size * size);

        int half = size / 2;

        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                int new_x = (x + half) % size;
                int new_y = (y + half) % size;
                temp[new_y * size + new_x] = data[y * size + x];
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

       
        void find_peak(double &peak_x, double &peak_y, double &max_val, const bool subpx, const std::string &method) {
            max_val = -std::numeric_limits<double>::infinity();
            int x0 = 0, y0 = 0;

            // Step 1: Find the integer peak
            for (int y = 0; y < ss_size; ++y) {
                for (int x = 0; x < ss_size; ++x) {
                    double val = cross_corr[y * ss_size + x];
                    if (val > max_val) {
                        max_val = val;
                        x0 = x;
                        y0 = y;
                    }
                }
            }

            // Step 2: No subpixel refinement requested
            if (!subpx) {
                peak_x = (x0 < ss_size / 2.0) ? x0 : x0 - ss_size;
                peak_y = (y0 < ss_size / 2.0) ? y0 : y0 - ss_size;
                return;
            }

            int i = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int xw = wrap(x0 + dx, ss_size);
                    int yw = wrap(y0 + dy, ss_size);
                    double val = cross_corr[yw * ss_size + xw];

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
            double a = coeffs(0), b_ = coeffs(1), c = coeffs(2);
            double d = coeffs(3), e = coeffs(4);

            // Step 5: Find stationary point (gradient = 0)
            Eigen::Matrix2d H;
            H << 2 * a, c,
                c,     2 * b_;
            Eigen::Vector2d g(-d, -e);

            Eigen::Vector2d offset = H.ldlt().solve(g);

            peak_x = x0 + offset(0);
            peak_y = y0 + offset(1);
            peak_x = (peak_x < ss_size / 2.0) ? peak_x : peak_x - ss_size;
            peak_y = (peak_y < ss_size / 2.0) ? peak_y : peak_y - ss_size;
            //std::cout << peak_x << " " << peak_y << std::endl;
        }
    };

    void init(std::vector<util::SubsetData> &ssdata,
              std::vector<int> &ss_sizes,
              std::vector<int> &ss_steps,
              const bool *img_roi, const util::Config &conf);

    void mgwd(const std::vector<util::SubsetData> &ssdata,
              const double *img_ref,
              const double *img_def,
              const Interpolator &interp_def);


    void sgwd(const util::SubsetData &ssdata, 
              const int window_size,
              const double *img_ref,
              const double *img_def,
              const Interpolator &interp_def);

    std::pair<double, double> get_prev_shift(const int i, const int ss,
                                       const double ss_x, const double ss_y,
                                       const std::vector<Shift>& shifts,
                                       const std::vector<util::SubsetData>& ssdata);

    double debugcost(util::Subset &ss_ref, util::Subset &ss_def);

    void zero_norm_subsets(std::vector<double>& def_vals, std::vector<double>& ref_vals, int ss_size);

   void smooth_field(std::vector<double>& shift, const util::SubsetData& ssdata, double sigma, int radius);
   void test(double &peak_x, double &peak_y, int ss_x, int ss_y, const int window_size, const double *img_ref, const double *img_def, const Interpolator &interp_def);
}

#endif // DICFOURIER_H

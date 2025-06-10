// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICFOURIER_H
#define DICFOURIER_H

// STD library Header files
#include <fftw3.h>
#include <algorithm>

// Program Header files
#include "./defines.hpp"
#include "./dicutil.hpp"

namespace fourier {



    struct Shift {

        // number of neighbours to use for removing outliers
        int num_neigh;

        //integer shifts
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> cost;
        std::vector<double> peak_x;
        std::vector<double> peak_y;

        // list of neighbours from prev window
        std::vector<int> neighlist;

        void gen_neighlist(const util::SubsetData ssdata,
                           const util::SubsetData ssdata_prev) {

            util::Timer timer("nearest neighbour collection for :");

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
                    for (int n = 0; n < dist_index_list.size(); n++){
                        int nss_idx = dist_index_list[n].second;
                        int nss_x = ssdata_prev.coords[2*nss_idx];
                        int nss_y = ssdata_prev.coords[2*nss_idx+1];
                    }
                }


                // Store neighbours indices into neighlist
                for (int i = 0; i < num_neigh; ++i) {
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

        fftw_complex* fft_def;
        fftw_complex* fft_ref;
        std::vector<double> cross_corr;

        fftw_plan plan_def;
        fftw_plan plan_ref;
        fftw_plan plan_inv;

        FFT(int ss_size_, double* ss_def_vals, double* ss_ref_vals)
            : ss_size(ss_size_), cross_corr(ss_size_ * ss_size_)
        {
            fft_def = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ss_size * (ss_size / 2 + 1));
            fft_ref = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ss_size * (ss_size / 2 + 1));

            #pragma omp critical
            {
                plan_def = fftw_plan_dft_r2c_2d(ss_size, ss_size, ss_def_vals, fft_def, FFTW_ESTIMATE);
                plan_ref = fftw_plan_dft_r2c_2d(ss_size, ss_size, ss_ref_vals, fft_ref, FFTW_ESTIMATE);
                plan_inv = fftw_plan_dft_c2r_2d(ss_size, ss_size, fft_def, cross_corr.data(), FFTW_ESTIMATE);
            }
        }

        ~FFT() {
            fftw_destroy_plan(plan_def);
            fftw_destroy_plan(plan_ref);
            fftw_destroy_plan(plan_inv);
            fftw_free(fft_def);
            fftw_free(fft_ref);
        }

        void correlate() {
            fftw_execute(plan_def);
            fftw_execute(plan_ref);

            for (int px = 0; px < ss_size * (ss_size / 2 + 1); px++) {
                double def_re = fft_def[px][0];
                double def_im = fft_def[px][1];
                double ref_re = fft_ref[px][0];
                double ref_im = fft_ref[px][1];

                // Complex conjugate multiplication
                fft_def[px][0] = def_re * ref_re + def_im * ref_im;
                fft_def[px][1] = def_im * ref_re - def_re * ref_im;
            }

            fftw_execute(plan_inv);
        }

        void find_peak(int &peak_x, int &peak_y, double &max_val) {

            max_val = -std::numeric_limits<double>::infinity();

            for (int y = 0; y < ss_size; ++y) {
                for (int x = 0; x < ss_size; ++x) {
                    double val = cross_corr[y * ss_size + x];
                    if (val > max_val) {
                        max_val = val;
                        peak_x = (x< ss_size / 2) ? x: x - ss_size;
                        peak_y = (y< ss_size / 2) ? y: y - ss_size;
                    }
                }
            }
        }

    };

    void init(std::vector<util::SubsetData> &ssdata, 
              const bool *img_roi, const util::Config conf);

    void mgwd(const std::vector<util::SubsetData> &ssdata,
              const double *img_def, const double *img_ref,
              const int px_hori, const int px_vert);

    std::pair<int, int> get_prev_shift(const int i, const int ss,
                                       const double ss_x, const double ss_y,
                                       const std::vector<Shift>& shifts,
                                       const std::vector<util::SubsetData>& ssdata);

    inline void destroy_fftw_plans(std::vector<fftw_plan>& plans);

    inline void free_fftw_arrays(std::vector<fftw_complex*>& vec);

    double debugcost(util::Subset &ss_ref, util::Subset &ss_def);

    void zero_norm_subsets(std::vector<double>& def_vals, std::vector<double>& ref_vals, int ss_size);

}

#endif // DICFOURIER_H

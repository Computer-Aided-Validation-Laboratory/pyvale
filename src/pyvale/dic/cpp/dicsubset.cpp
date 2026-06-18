// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <omp.h>
#include <iostream>

// Program Header files
#include "./dicsubset.hpp"
#include "./dicshapefunc.hpp"

// common_cpp header files
#include "../../common_cpp/util.hpp"

namespace subset {

     void fill_from_img(subset::Pixels &ss_ref, 
                    const int ss_x, const int ss_y, 
                    const int px_hori,
                    const int px_vert,
                    const Image &img){

        switch (img.type) {
            case PixelType::UINT8:  fill_impl(ss_ref, img.data8,  ss_x, ss_y, px_hori); break;
            case PixelType::UINT16: fill_impl(ss_ref, img.data16, ss_x, ss_y, px_hori); break;
            case PixelType::UINT32: fill_impl(ss_ref, img.data32, ss_x, ss_y, px_hori); break;
        }
    }

    template<typename T>
    void fill_impl(subset::Pixels &ss_ref,
                const std::vector<T> &data,
                int ss_x, int ss_y,
                int px_hori) {

        int count = 0;
        ss_ref.sum = 0.0;

        for (int y = ss_y; y < ss_y + ss_ref.size_y; ++y) {
            for (int x = ss_x; x < ss_x + ss_ref.size_x; ++x) {
                int idx = y * px_hori + x;
                if (ss_ref.has_coords()) {
                    ss_ref.x[count] = x;
                    ss_ref.y[count] = y;
                }
                ss_ref.vals[count] = data[idx];
                ss_ref.sum += data[idx];
                count++;
            }
        }
    }

    double zncc(const subset::Pixels &ss_ref, const subset::Pixels &ss_def) {
        double mean_ref = 0.0;
        double mean_def = 0.0;

        for (int i = 0; i < ss_ref.num_px; ++i) {
            mean_ref += ss_ref.vals[i];
            mean_def += ss_def.vals[i];
        }

        mean_ref /= ss_ref.num_px;
        mean_def /= ss_def.num_px;

        double sum_squared_ref = 0.0;
        double sum_squared_def = 0.0;

        for (int i = 0; i < ss_ref.num_px; ++i) {
            sum_squared_ref += (ss_ref.vals[i] - mean_ref) * (ss_ref.vals[i] - mean_ref);
            sum_squared_def += (ss_def.vals[i] - mean_def) * (ss_def.vals[i] - mean_def);
        }

        const double inv_sum_squared = 1.0 / std::sqrt(sum_squared_ref * sum_squared_def);

        double zncc = 0.0;
        for (int i = 0; i < ss_ref.num_px; ++i) {
            const double def_norm = (ss_def.vals[i] - mean_def);
            const double ref_norm = (ss_ref.vals[i] - mean_ref);
            zncc += ref_norm * def_norm;
        }

        return zncc * inv_sum_squared;
    }

    void fill_from_img_subpx(subset::Pixels &ss_def, 
                          const double subpx_x, const double subpx_y, 
                          const Interpolator &interp_def){

        int count = 0;

        for (int y = 0; y < ss_def.size_y; y++){
            for (int x = 0; x < ss_def.size_x; x++){
                // get coordinate values
                const double px_x = subpx_x + x;
                const double px_y = subpx_y + y;
                if (ss_def.has_coords()) {
                    ss_def.x[count] = px_x; 
                    ss_def.y[count] = px_y; 
                }

                // get pixel values
                ss_def.vals[count] = interp_def.eval(0, 0, px_x, px_y);

                // debugging
                //std::cout << ss_def.x[count] << " " << ss_def.y[count] << " " << ss_def.vals[count] << std::endl;

                count++;
            }
        }
    }

    void fill_from_shape_params(subset::Pixels &ss, 
                                     const double cx, const double cy,
                                     const std::vector<double>& p,
                                     const Interpolator &interp,
                                     util::ShapeFunc shape_func){

        // Get the right shape function
        void (*get_pixel)(double&, double&, const double, const double, const std::vector<double>&);
        switch (shape_func) {
            case util::ShapeFunc::AFFINE:
                get_pixel = &Affine::get_pixel;
                break;
            case util::ShapeFunc::RIGID:
                get_pixel = &Rigid::get_pixel;
                break;
            case util::ShapeFunc::QUAD:
                get_pixel = &Quad::get_pixel;
                break;
        }


        // NOTE: Assuming an odd number subset size
        const double half_x = (ss.size_x - 1) / 2.0;
        const double half_y = (ss.size_y - 1) / 2.0;

        int count = 0;
        ss.sum = 0.0;
        for (int y = 0; y < ss.size_y; y++){
            const double rel_y = y - half_y;
            for (int x = 0; x < ss.size_x; x++){
                double px_x = 0.0;
                double px_y = 0.0;
                get_pixel(px_x, px_y, x - half_x, rel_y, p);
                px_x += cx;
                px_y += cy;
                if (ss.has_coords()) {
                    ss.x[count] = px_x;
                    ss.y[count] = px_y;
                }
                ss.vals[count] = interp.eval(cx, cy, px_x, px_y);
                ss.sum += ss.vals[count];
                count++;
            }
        }
    }

    void fill_from_centre_coords(subset::Pixels &ss_def,
                             const double cx, const double cy,
                             const Interpolator &interp_def) {

        // NOTE: Assuming an odd number subset size
        const double half_x = (ss_def.size_x - 1) / 2.0;
        const double half_y = (ss_def.size_y - 1) / 2.0;

        int count = 0;
        ss_def.sum = 0.0;
        for (int y = 0; y < ss_def.size_y; y++) {
            for (int x = 0; x < ss_def.size_x; x++) {
                const double px_x = cx + x - half_x;
                const double px_y = cy + y - half_y;
                if (ss_def.has_coords()) {
                    ss_def.x[count] = px_x;
                    ss_def.y[count] = px_y;
                }
                ss_def.vals[count] = interp_def.eval(cx, cy,
                                                    px_x,
                                                    px_y);
                ss_def.sum += ss_def.vals[count];
                count++;
            }
        }
    }

    subset::Grid create_grid(const bool *img_roi, const int ss_step,
                             const int ss_size_x, const int ss_size_y,
                             const int px_hori, const int px_vert,
                             const bool partial) {
        
        //Timer timer("subset grid generation for subset size " + std::to_string(ss_size) + " [px] with step " + std::to_string(ss_step) + " [px]:" );

        subset::Grid ss_grid;

        int dx[4] = {ss_step, 0, -ss_step, 0};
        int dy[4] = {0, ss_step, 0, -ss_step};

        int subset_counter = 0;

        int num_ss_x = px_hori / ss_step;
        int num_ss_y = px_vert / ss_step;
        //ss_grid.mask.resize(num_ss_x*num_ss_y, NAN);
        ss_grid.num_ss_x = num_ss_x;
        ss_grid.num_ss_y = num_ss_y;
        ss_grid.num_in_mask = num_ss_x * num_ss_y;
        ss_grid.num = 0;
        ss_grid.step = ss_step;
        ss_grid.size_x = ss_size_x;
        ss_grid.size_y = ss_size_y;

        ss_grid.mask.resize(ss_grid.num_in_mask, -1);
        ss_grid.coords.resize(2*ss_grid.num_in_mask, -1);


        // temp array for storing subset coords for each thread
        std::vector<int> thread_counts(omp_get_max_threads(), 0);

       // First pass: count valid subsets per thread
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < num_ss_y; j++) {
            for (int i = 0; i < num_ss_x; i++) {

                const int ss_x = i * ss_step;
                const int ss_y = j * ss_step;

                // pixel range of subset
                const int xmin = ss_x;
                const int ymin = ss_y;
                const int xmax = ss_x + ss_size_x-1;
                const int ymax = ss_y + ss_size_y-1;

                bool valid = true;
                int valid_count = 0;

                for (int px_y = ymin; px_y <= ymax && valid; px_y++) {
                    for (int px_x = xmin; px_x <= xmax && valid; px_x++) {

                        if (!partial) {
                            if (!px_in_img_dims(px_x, px_y, px_hori, px_vert) ||
                                !px_in_roi(px_x, px_y, px_hori, px_vert, img_roi)) {
                                valid = false;
                                break;
                            }
                        }
                        else {
                            if (!px_in_img_dims(px_x, px_y, px_hori, px_vert)) {
                                valid = false;
                                break;
                            }
                            if (px_in_roi(px_x, px_y, px_hori, px_vert, img_roi)) valid_count++;
                        }
                    }
                }

                if (partial && valid) {
                    valid = (valid_count >= (ss_size_x * ss_size_y) * 0.70);
                }

                if (valid) {
                    int tid = omp_get_thread_num();
                    thread_counts[tid]++;
                }
            }
        }

        // Compute prefix sum to get offsets
        std::vector<int> thread_offsets(omp_get_max_threads(), 0);
        for (int t = 1; t < thread_offsets.size(); t++)
            thread_offsets[t] = thread_offsets[t-1] + thread_counts[t-1];

        int total_valid = thread_offsets.back() + thread_counts.back();
        ss_grid.coords.resize(2 * total_valid);
        ss_grid.num = total_valid;
        ss_grid.active_ss.resize(total_valid, true);
        ss_grid.active_total = total_valid;

        // Reset thread counts to use as writing indices
        std::fill(thread_counts.begin(), thread_counts.end(), 0);

        #pragma omp parallel for collapse(2)
        for (int j = 0; j < num_ss_y; j++) {
            for (int i = 0; i < num_ss_x; i++) {

                // calculate the coordinates of the subset
                const int ss_x = i * ss_step;
                const int ss_y = j * ss_step;

                // pixel range of subset
                const int xmin = ss_x;
                const int ymin = ss_y;
                const int xmax = ss_x + ss_size_x-1;
                const int ymax = ss_y + ss_size_y-1;

                // check if subset is within image and ROI.
                bool valid = true;
                int  valid_count = 0;

                for (int px_y = ymin; px_y <= ymax && valid; px_y++) {
                    for (int px_x = xmin; px_x <= xmax && valid; px_x++) {

                        // When no partial subset filling all px must be within roi
                        if (!partial) {
                            if (!px_in_img_dims(px_x, px_y, px_hori, px_vert) ||
                                !px_in_roi(px_x, px_y, px_hori, px_vert, img_roi)) {
                                valid = false;
                                break;
                            }
                        } 

                        // When partial count num of px in roi. if its outside
                        // the image its still not valid
                        else {
                            if (!px_in_img_dims(px_x, px_y, px_hori, px_vert)) {
                                valid = false;
                                break;
                            }
                            if (px_in_roi(px_x, px_y, px_hori, px_vert, img_roi)) valid_count++;
                        }
                    }

                    if (!valid && !partial) break;
                }

                if (partial && valid) {
                    valid = (valid_count >= (ss_size_x * ss_size_y) * 0.70);
                }

                // if its a valid subset. add it to a list of coordinates
                if (valid) {
                    const int tid = omp_get_thread_num();
                    const int offset = thread_offsets[tid] + thread_counts[tid];
                    ss_grid.coords[2*offset]     = ss_x + static_cast<double>(ss_size_x)/2-0.5;
                    ss_grid.coords[2*offset + 1] = ss_y + static_cast<double>(ss_size_y)/2-0.5;
                    ss_grid.mask[j * num_ss_x + i] = offset;
                    thread_counts[tid]++;
                }
            }
        }

        // resize neighbour list
        ss_grid.neigh.resize(ss_grid.num);

        // neighbours for each of the above subset
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < num_ss_y; ++j) {
            for (int i = 0; i < num_ss_x; ++i) {

                // calculate the coordinates of the subset
                int idx = ss_grid.mask[j * num_ss_x + i];

                if (idx == -1) continue;

                // Clear inner vector and reserve space for 4 neighbors (up/down/left/right)
                ss_grid.neigh[idx].clear();
                ss_grid.neigh[idx].reserve(4);

                for (int d = 0; d < 4; ++d) {
                    int ni = i + dx[d] / ss_step;
                    int nj = j + dy[d] / ss_step;

                    if (ni >= 0 && ni < num_ss_x && nj >= 0 && nj < num_ss_y) {
                        int neigh_idx = ss_grid.mask[nj * num_ss_x + ni];
                        if (neigh_idx != -1) {
                            ss_grid.neigh[idx].push_back(neigh_idx);
                        }
                    }
                }
            }
        }
        return ss_grid;
    }

}

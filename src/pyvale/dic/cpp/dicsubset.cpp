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



namespace subset {

     void get_px_from_img(subset::Pixels &ss_ref, 
                    const int ss_x, const int ss_y, 
                    const int px_hori,
                    const int px_vert,
                    const double *img_def){

        int count = 0;
        int idx;

        for (int px_y = ss_y; px_y < ss_y+ss_ref.size; px_y++){
            for (int px_x = ss_x; px_x < ss_x+ss_ref.size; px_x++){

                // get coordinate values
                ss_ref.x[count] = px_x; 
                ss_ref.y[count] = px_y; 

                // get pixel values
                idx = px_y * px_hori + px_x;
                ss_ref.vals[count] = img_def[idx];
                count++;

                // debugging
                //std::cout << px_x << " " << px_y << " ";
                //std::cout << img_def[idx] << std::endl;
            }
        }
    }

    void get_subpx_from_img(subset::Pixels &ss_def, 
                          const double subpx_x, const double subpx_y, 
                          const Interpolator &interp_def){

        int count = 0;

        for (int y = 0; y < ss_def.size; y++){
            for (int x = 0; x < ss_def.size; x++){
                if (count >= ss_def.size*ss_def.size){
                    std::cerr << "issue with count for subpixel subset population" << std::endl;
                    std::cerr << "count: " << count << std::endl;
                    std::cerr << "subset size: " << ss_def.size << std::endl;
                    std::cerr << "num px (size*size): " << ss_def.size*ss_def.size << std::endl;
                    std::cerr << "subpixel value: " << subpx_x+x << " " << subpx_y+y << std::endl;
                    std::cerr << "subset coordinates: " << " " <<  subpx_x << " " << subpx_y << " " << std::endl;
                    exit(EXIT_FAILURE);
                }
                // get coordinate values
                ss_def.x[count] = subpx_x+x; 
                ss_def.y[count] = subpx_y+y; 

                // get pixel values
                ss_def.vals[count] = interp_def.eval_bicubic(0, 0, ss_def.x[count], ss_def.y[count]);

                // debugging
                //std::cout << ss_def.x[count] << " " << ss_def.y[count] << " " << ss_def.vals[count] << std::endl;

                count++;
            }
        }
        if (count!=ss_def.size*ss_def.size){
            std::cerr << "count for subpixel population is not the same as the number of subset pixels.";
            std::cout << "count: " << count << std::endl;
            std::cerr << "number of pixels: " << ss_def.size*ss_def.size << std::endl; 
            exit(EXIT_FAILURE);
        }
    }

    void get_subpx_from_shape_params(subset::Pixels &ss_def, 
                                     const double subpx_x, const double subpx_y,
                                     const std::vector<double>& p,
                                     const Interpolator &interp_def){

        int count = 0;

        for (int y = 0; y < ss_def.size; y++){
            for (int x = 0; x < ss_def.size; x++){
                if (count >= ss_def.size*ss_def.size){
                    std::cerr << "issue with count for subpixel subset population" << std::endl;
                    std::cerr << "count: " << count << std::endl;
                    std::cerr << "subset size: " << ss_def.size << std::endl;
                    std::cerr << "num px (size*size): " << ss_def.size*ss_def.size << std::endl;
                    std::cerr << "subpixel value: " << subpx_x+x << " " << subpx_y+y << std::endl;
                    std::cerr << "subset coordinates: " << " " <<  subpx_x << " " << subpx_y << " " << std::endl;
                    exit(EXIT_FAILURE);
                }

                // get coordinate values based on shape function parameters
                shapefunc::get_pixel(ss_def.x[count], ss_def.y[count], subpx_x+x, subpx_y+y, p);
                
                // get pixel values from interpolator
                ss_def.vals[count] = interp_def.eval_bicubic(0, 0, ss_def.x[count], ss_def.y[count]);

                count++;
            }
        }
        if (count!=ss_def.size*ss_def.size){
            std::cerr << "count for subpixel population is not the same as the number of subset pixels.";
            std::cout << "count: " << count << std::endl;
            std::cerr << "number of pixels: " << ss_def.size*ss_def.size << std::endl; 
            exit(EXIT_FAILURE);
        }
    }

    subset::Grid create_grid(const bool *img_roi, const int ss_step, 
                           const int ss_size, const int px_hori, 
                           const int px_vert, const bool partial) {
        
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
        ss_grid.size = ss_size;

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
                const int xmax = ss_x + ss_size-1;
                const int ymax = ss_y + ss_size-1;

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
                    valid = (valid_count >= (ss_size * ss_size) * 0.70);
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
                const int xmax = ss_x + ss_size-1;
                const int ymax = ss_y + ss_size-1;

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
                    valid = (valid_count >= (ss_size * ss_size) * 0.70);
                }

                // if its a valid subset. add it to a list of coordinates
                if (valid) {
                    const int tid = omp_get_thread_num();
                    const int offset = thread_offsets[tid] + thread_counts[tid];
                    ss_grid.coords[2*offset] = ss_x;
                    ss_grid.coords[2*offset + 1] = ss_y;
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


    inline bool px_in_img_dims(const int px_x, const int px_y, const int px_hori, 
                        const int px_vert) {

        if (px_x < 0 || px_y < 0 ||
            px_x >= px_hori || px_y >= px_vert) {
            return false;
        }
        return true;
    }

    inline bool px_in_roi(const int px_x, const int px_y, const int px_hori, 
                        const int px_vert, const bool *img_roi) {

        int idx = px_y * px_hori + px_x;
        if (!img_roi[idx]) {
            return false;
        }
        return true;
    }

}

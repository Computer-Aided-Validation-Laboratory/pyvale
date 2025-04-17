// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>

// Program Header files
#include "./dicutil.hpp"


namespace util {


    std::vector<int> ss_coord_list;
    std::vector<int> niter_arr;
    std::vector<double> u_arr;
    std::vector<double> v_arr;
    std::vector<double> p_arr;
    std::vector<double> ftol_arr;
    std::vector<double> xtol_arr;
    std::vector<double> cost_arr;





    void extract_image(util::Image *image_def,
                       int *image_def_stack, 
                       int image_number){

        // extract a single image
        std::vector<double> img(image_def->px_horizontal * image_def->px_vertical);
        int count = 0;

        for (int px_vert = 0; px_vert < image_def->px_vertical; px_vert++){
            for (int px_hori = 0; px_hori < image_def->px_horizontal; px_hori++){

                int index = image_number * image_def->px_horizontal * image_def->px_vertical + px_vert * image_def->px_horizontal + px_hori;
                image_def->vals[count] = image_def_stack[index];
                // std::cout << image_def[count] << " ";
                count++;
            }
            // std::cout << std::endl;
        }
    }




    void extract_ss(int ss_x, int ss_y, util::Image *image_def, util::Subset *ss_def){

        int count = 0;
        int index;

        for (int px_y = ss_y; px_y < ss_y+ss_def->size; px_y++){
            for (int px_x = ss_x; px_x < ss_x+ss_def->size; px_x++){

                // get coordinate values
                ss_def->x[count] = px_x; 
                ss_def->y[count] = px_y; 

                // get pixel values
                index = px_y * image_def->px_horizontal + px_x;
                ss_def->vals[count] = image_def->vals[index];
                count++;
                
            }
        }
    }


    SubsetList generate_ss_list(bool *image_roi, int px_horizontal, int px_vertical, int ss_size, int ss_step, int num_def_images, int num_params) {

        SubsetList result;
        int index;
        int dx[4] = {ss_step, 0, -ss_step, 0};
        int dy[4] = {0, ss_step, 0, -ss_step};

        int subset_counter = 0;

        // First pass: collect valid subset centers and index them
        for (int ss_y = 0; ss_y < px_vertical; ss_y += ss_step) {
            for (int ss_x = 0; ss_x < px_horizontal; ss_x += ss_step) {
                

                // b
                int ss_x_min = ss_x;
                int ss_y_min = ss_y;
                int ss_x_max = ss_x + ss_size;
                int ss_y_max = ss_y + ss_size;

                bool valid = true;
                for (int px_y = ss_y_min; px_y <= ss_y_max && valid; px_y++) {
                    for (int px_x = ss_x_min; px_x <= ss_x_max && valid; px_x++) {
                        if (px_x < 0 || px_y < 0 || px_x >= px_horizontal || px_y >= px_vertical) {
                            valid = false;
                            break;
                        }
                        index = px_y * px_horizontal + px_x;
                        if (!image_roi[index]) {
                            valid = false;
                            break;
                        }
                    }
                }

                if (valid) {
                    result.coords.push_back(ss_x);
                    result.coords.push_back(ss_y);
                    result.coords_to_index[{ss_x, ss_y}] = subset_counter;
                    subset_counter++;
                }
            }
        }
        
        result.n_ss = subset_counter;

        // neighbours for each of the above subset
        for (const auto& kv : result.coords_to_index) {
            const std::pair<int, int>& coord = kv.first;
            int center_idx = kv.second;

            std::vector<int> temp_neigh;

            for (int i = 0; i < 4; ++i) {
                int neigh_x = coord.first + dx[i];
                int neigh_y = coord.second + dy[i];

                int x_min = neigh_x;
                int y_min = neigh_y;
                int x_max = neigh_x + ss_size;
                int y_max = neigh_y + ss_size;

                bool valid = true;

                // checking if the neigbour is valid (in image bounds and within ROI)
                for (int y = y_min; y <= y_max && valid; ++y) {
                    for (int x = x_min; x <= x_max && valid; ++x) {
                        if (x < 0 || y < 0 || x >= px_horizontal || y >= px_vertical) {
                            valid = false;
                            break;
                        }
                        int idx = y * px_horizontal + x;
                        if (!image_roi[idx]) {
                            valid = false;
                            break;
                        }
                    }
                }

                if (valid) {
                    auto it = result.coords_to_index.find({neigh_x, neigh_y});
                    if (it != result.coords_to_index.end()) {
                        temp_neigh.push_back(it->second);
                    }
                }
            }

            result.neighbours[center_idx] = std::move(temp_neigh);
        }


        // resize results
        niter_arr.resize(num_def_images * result.n_ss);
        u_arr.resize(num_def_images * result.n_ss);
        v_arr.resize(num_def_images * result.n_ss);
        p_arr.resize(num_def_images * result.n_ss * num_params);
        ftol_arr.resize(num_def_images * result.n_ss);
        xtol_arr.resize(num_def_images * result.n_ss);
        cost_arr.resize(num_def_images * result.n_ss);
        return result;
    }    



        void append_results(const int num_def_images, 
                            const int img_num, 
                            const int ss, 
                            const int iter, 
                            const double ftol, 
                            const double xtol, 
                            const double u, 
                            const double v, 
                            const double costp,
                            const std::vector<double> &p) {

            int index = img_num * num_def_images + ss;
            int index_p = 6*index;
            niter_arr[index] = iter;
            p_arr[index_p+0] = p[0];
            p_arr[index_p+1] = p[1];
            p_arr[index_p+2] = p[2];
            p_arr[index_p+3] = p[3];
            p_arr[index_p+4] = p[4];
            p_arr[index_p+5] = p[5];
            u_arr[index] = u;
            v_arr[index] = v;
            ftol_arr[index] = ftol;
            xtol_arr[index] = xtol;
            cost_arr[index] = costp;
    }

}   

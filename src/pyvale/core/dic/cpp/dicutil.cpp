// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

// Program Header files
#include "./dicutil.hpp"


namespace util {


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

        // readability
        const int px_horizontal = image_def->px_horizontal;
        const int px_vertical = image_def->px_vertical;

        int count = 0;
        for (int px_y = 0; px_y < px_vertical; px_y++){
            for (int px_x = 0; px_x < px_horizontal; px_x++){
                int idx = image_number * px_horizontal * px_vertical + px_y * px_horizontal + px_x;
                image_def->vals[count] = image_def_stack[idx];
                count++;
            }
        }
    }




    void extract_ss(int ss_x, int ss_y, util::Image *image_def, util::Subset *ss_def){

        int count = 0;
        int idx;

        for (int px_y = ss_y; px_y < ss_y+ss_def->size; px_y++){
            for (int px_x = ss_x; px_x < ss_x+ss_def->size; px_x++){

                // get coordinate values
                ss_def->x[count] = px_x; 
                ss_def->y[count] = px_y; 

                // get pixel values
                idx = px_y * image_def->px_horizontal + px_x;
                ss_def->vals[count] = image_def->vals[idx];
                count++;
                
            }
        }
    }


    SubsetData generate_ss_list(bool *image_roi, int px_horizontal, int px_vertical, int ss_size, int ss_step, int num_def_images, int num_params) {

        SubsetData ssdata;
        int idx;
        int dx[4] = {ss_step, 0, -ss_step, 0};
        int dy[4] = {0, ss_step, 0, -ss_step};

        int subset_counter = 0;

        
        int num_ss_x = px_horizontal / ss_step;
        int num_ss_y = px_vertical / ss_step;
        ssdata.mask.resize(num_ss_x*num_ss_y, false);
        ssdata.num_ss_x = num_ss_x;
        ssdata.num_ss_y = num_ss_y;
        ssdata.num_in_mask = num_ss_x * num_ss_y;
        ssdata.num = 0;
        ssdata.step = ss_step;
        ssdata.size = ss_size;


        // First pass: collect valid subset centers and idx them
        for (int j = 0; j < num_ss_y; j++) {
            for (int i = 0; i < num_ss_x; i++) {
                

                // calculate the coordinates of the subset
                int ss_x = i * ss_step;
                int ss_y = j * ss_step;

                // pixel range of subset
                int ss_x_min = ss_x;
                int ss_y_min = ss_y;
                int ss_x_max = ss_x + ss_size;
                int ss_y_max = ss_y + ss_size;

                // check if subset is within image and ROI.
                bool valid = true;
                for (int px_y = ss_y_min; px_y <= ss_y_max && valid; px_y++) {
                    for (int px_x = ss_x_min; px_x <= ss_x_max && valid; px_x++) {
                        if (px_x < 0 || px_y < 0 || px_x >= px_horizontal || px_y >= px_vertical) {
                            valid = false;
                            break;
                        }
                        idx = px_y * px_horizontal + px_x;
                        if (!image_roi[idx]) {
                            valid = false;
                            break;
                        }
                    }
                }

                // if its a valid subset. add it to a list of coordinates
                if (valid) {
                    ssdata.coords.push_back(ss_x);
                    ssdata.coords.push_back(ss_y);
                    ssdata.mask[j * num_ss_x + i] = true;
                    ssdata.coords_to_idx[{ss_x, ss_y}] = subset_counter;
                    subset_counter++;
                }
            }
        }
        
        ssdata.num = subset_counter;

        // neighbours for each of the above subset
        for (const auto& kv : ssdata.coords_to_idx) {
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
                    auto it = ssdata.coords_to_idx.find({neigh_x, neigh_y});
                    if (it != ssdata.coords_to_idx.end()) {
                        temp_neigh.push_back(it->second);
                    }
                }
            }

            ssdata.neighbours[center_idx] = std::move(temp_neigh);
        }


        // resize results
        niter_arr.resize(num_def_images * ssdata.num_in_mask);
        u_arr.resize(num_def_images * ssdata.num_in_mask);
        v_arr.resize(num_def_images * ssdata.num_in_mask);
        p_arr.resize(num_def_images * ssdata.num_in_mask * num_params);
        ftol_arr.resize(num_def_images * ssdata.num_in_mask);
        xtol_arr.resize(num_def_images * ssdata.num_in_mask);
        cost_arr.resize(num_def_images * ssdata.num_in_mask);
        std::cout << "NITER SIZE " << niter_arr.size() << std::endl;
        return ssdata;
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

        int idx = img_num * num_def_images + ss;
        int idx_p = p.size()*idx;
        niter_arr[idx] = iter;
        u_arr[idx] = u;
        v_arr[idx] = v;
        ftol_arr[idx] = ftol;
        xtol_arr[idx] = xtol;
        cost_arr[idx] = costp;
        for (size_t i = 0; i < p.size(); i++){
            p_arr[idx_p+i] = p[i];
        }
    }


    void save_to_disk(util::SaveConfig *saveconf, const int num_def_images, util::SubsetData *ssdata, const int num_params){

        // readability
        const std::string delimiter = saveconf->delimiter;        
        std::cout << saveconf->delimiter << std::endl;
        std::cout << saveconf->base_path << std::endl;
        std::cout << saveconf->format << std::endl;
        std::cout << saveconf->layout << std::endl;
        std::cout << saveconf->save_at_end << std::endl;

        // loop over images 
        for (int img = 0; img < num_def_images; img++){

            if (saveconf->layout == "col"){

                std::stringstream outfile_str;
                std::ofstream outfile;
    
                // filename
                outfile_str << saveconf->base_path << "/" << saveconf->prefix << img << saveconf->format;
                

                // save in binary format
                if (saveconf->format == ".bin"){
                    outfile.open(outfile_str.str(), std::ios::binary);

                     for (size_t i = 0; i < ssdata->num; ++i) {
                        int idx = img * num_def_images + i;
                        int idx_p = num_params*idx;
                        double magnitude = std::sqrt(u_arr[idx] * u_arr[idx] + v_arr[idx] * v_arr[idx]);

                        outfile.write(reinterpret_cast<const char*>(&ssdata->coords[2*i]), sizeof(int));
                        outfile.write(reinterpret_cast<const char*>(&ssdata->coords[2*i+1]), sizeof(int));
                        outfile.write(reinterpret_cast<const char*>(&u_arr[idx]), sizeof(double));
                        outfile.write(reinterpret_cast<const char*>(&v_arr[idx]), sizeof(double));
                        outfile.write(reinterpret_cast<const char*>(&magnitude), sizeof(double));
                        for (int p = 0; p < num_params; p++){
                            outfile.write(reinterpret_cast<const char*>(&p_arr[idx_p+p]), sizeof(double));
                        }
                        outfile.write(reinterpret_cast<const char*>(&cost_arr[idx]), sizeof(double));
                        outfile.write(reinterpret_cast<const char*>(&ftol_arr[idx]), sizeof(double));
                        outfile.write(reinterpret_cast<const char*>(&xtol_arr[idx]), sizeof(double));
                        outfile.write(reinterpret_cast<const char*>(&niter_arr[idx]), sizeof(int));
                    }
                }


                // save in human readable format
                else if (saveconf->format == ".dat"){

                    outfile.open(outfile_str.str());
                    for (size_t i = 0; i < ssdata->num; ++i) {
                        
                        int idx = img * num_def_images + i;
                        int idx_p = num_params*idx;

                        outfile << ssdata->coords[2*i] << delimiter;
                        outfile << ssdata->coords[2*i+1] << delimiter;
                        outfile << u_arr[idx] << delimiter;
                        outfile << v_arr[idx] << delimiter;
                        outfile << sqrt(u_arr[idx]*u_arr[idx] + v_arr[idx]*v_arr[idx]) << delimiter;
                        for (int p = 0; p < num_params; p++){
                            outfile << p_arr[idx_p+p] << delimiter;
                        }
                        outfile << cost_arr[idx] << delimiter;
                        outfile << ftol_arr[idx] << delimiter;
                        outfile << xtol_arr[idx] << delimiter;
                        outfile << niter_arr[idx] << "\n";
                    }
                }
            }
        }
    }

}   

// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <cmath>

// Program Header files
#include "./defines.hpp"
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

    int get_num_params(std::string &shape_func){
        int num_params;
        if (shape_func == "RIGID") num_params = 2;
        else if (shape_func == "AFFINE") num_params = 6;
        else {
            std::cerr << "Unknown shape function: \'" << shape_func << "\'." << std::endl;
            std::cerr << "Allowed values: \'AFFINE\', \'RIGID\'. " << std::endl;
            exit(EXIT_FAILURE);
        }
        return num_params;
    }



    void extract_ss(util::Subset &ss_def, 
                    int ss_x, int ss_y, 
                    int px_horizontal,
                    int px_vertical,
                    double *image_def){

        int count = 0;
        int idx;

        for (int px_y = ss_y; px_y < ss_y+ss_def.size; px_y++){
            for (int px_x = ss_x; px_x < ss_x+ss_def.size; px_x++){

                // get coordinate values
                ss_def.x[count] = px_x; 
                ss_def.y[count] = px_y; 

                // get pixel values
                idx = px_y * px_horizontal + px_x;
                ss_def.vals[count] = image_def[idx];
                count++;
                
            }
        }
    }


    SubsetData generate_ss_list(bool *image_roi, Config &conf) {
        
        Timer timer("generate subset list");

        SubsetData ssdata;
        const int ss_step = conf.ss_step;
        const int ss_size = conf.ss_size;

        int idx;
        int dx[4] = {ss_step, 0, -ss_step, 0};
        int dy[4] = {0, ss_step, 0, -ss_step};

        int subset_counter = 0;

        
        int num_ss_x = conf.px_horizontal / ss_step;
        int num_ss_y = conf.px_vertical / ss_step;
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
                int xmin = ss_x;
                int ymin = ss_y;
                int xmax = ss_x + ss_size;
                int ymax = ss_y + ss_size;

                // check if subset is within image and ROI.
                bool valid = true;
                for (int px_y = ymin; px_y <= ymax && valid; px_y++) {
                    for (int px_x = xmin; px_x <= xmax && valid; px_x++) {

                       if(!is_valid_pixel(px_x,px_y,conf,image_roi)){
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

                int xmin = neigh_x;
                int ymin = neigh_y;
                int xmax = neigh_x + ss_size;
                int ymax = neigh_y + ss_size;

                bool valid = true;

                // checking if the neigbour is valid
                for (int y = ymin; y <= ymax && valid; ++y) {
                    for (int x = xmin; x <= xmax && valid; ++x) {

                        if(!is_valid_pixel(x,y,conf,image_roi)){
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
        niter_arr.resize(conf.num_def_images * ssdata.num);
        u_arr.resize(conf.num_def_images * ssdata.num);
        v_arr.resize(conf.num_def_images * ssdata.num);
        p_arr.resize(conf.num_def_images * ssdata.num * conf.num_params);
        ftol_arr.resize(conf.num_def_images * ssdata.num);
        xtol_arr.resize(conf.num_def_images * ssdata.num);
        cost_arr.resize(conf.num_def_images * ssdata.num);
        return ssdata;
    }



    void append_results(const int num_def_images, 
                        const int img_num, 
                        const int num_ss,
                        const int ss, 
                        const int iter, 
                        const double ftol, 
                        const double xtol, 
                        const double u, 
                        const double v, 
                        const double costp,
                        const std::vector<double> &p) {

        int idx = img_num * num_ss + ss;
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


    void save_to_disk(util::SaveConfig *saveconf, const int num_def_images, 
                      util::SubsetData *ssdata, const int num_params){

        util::Timer timer("save to disk");


        // readability
        const std::string delimiter = saveconf->delimiter;        

        // loop over images 
        for (int img = 0; img < num_def_images; img++){

            if (saveconf->layout == "col"){

                std::stringstream outfile_str;
                std::ofstream outfile;
    
                // filename
                outfile_str << saveconf->base_path << "/" <<
                saveconf->prefix << img << saveconf->format;

                // save in binary format
                if (saveconf->format == ".bin"){
                    outfile.open(outfile_str.str(), std::ios::binary);

                     for (size_t i = 0; i < ssdata->num; ++i) {
                        int idx = img * ssdata->num + i;
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

                    outfile.close();
                }


                // save in human readable format
                else if (saveconf->format == ".dat"){

                    outfile.open(outfile_str.str());
                    for (size_t i = 0; i < ssdata->num; i++) {
                        
                        int idx = img * ssdata->num + i;
                        int idx_p = num_params*idx;

                        outfile << ssdata->coords[2*i] << delimiter;
                        outfile << ssdata->coords[2*i+1] << delimiter;
                        outfile << u_arr[idx] << delimiter;
                        outfile << v_arr[idx] << delimiter;
                        outfile << sqrt(u_arr[idx]*u_arr[idx]+
                                        v_arr[idx]*v_arr[idx]) << delimiter;
                        for (int p = 0; p < num_params; p++){
                            outfile << p_arr[idx_p+p] << delimiter;
                        }
                        outfile << cost_arr[idx] << delimiter;
                        outfile << ftol_arr[idx] << delimiter;
                        outfile << xtol_arr[idx] << delimiter;
                        outfile << niter_arr[idx] << "\n";
                    }
                    outfile.close();
                }
            }
        }
    }

    bool is_valid_pixel(int px_x, int px_y, Config& conf, 
                        bool *image_roi) {
        if (px_x < 0 || px_y < 0 ||
            px_x >= conf.px_horizontal || px_y >= conf.px_vertical) {
            return false;
        }
        int idx = px_y * conf.px_horizontal + px_x;
        if (!image_roi[idx]) {
            return false;
        }
        return true;
    }


}   

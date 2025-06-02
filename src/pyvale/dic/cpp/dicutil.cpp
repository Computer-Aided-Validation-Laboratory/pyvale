// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <iomanip>
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
    bool at_end;



    void extract_image(double *img_def_stack, 
                       int image_number,
                       int px_hori,
                       int px_vert){

        int count = 0;
        for (int px_y = 0; px_y < px_vert; px_y++){
            for (int px_x = 0; px_x < px_hori; px_x++){
                int idx = image_number * px_hori * px_vert + px_y * px_hori + px_x;
                std::cout << img_def_stack[idx] << " ";
                //img_def->vals[count] = img_def_stack[idx];
                count++;
            }
            std::cout << std::endl;
        }
        exit(0);
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
                    int px_hori,
                    int px_vert,
                    double *img_def){

        int count = 0;
        int idx;

        for (int px_y = ss_y; px_y < ss_y+ss_def.size; px_y++){
            for (int px_x = ss_x; px_x < ss_x+ss_def.size; px_x++){

                // get coordinate values
                ss_def.x[count] = px_x; 
                ss_def.y[count] = px_y; 

                // get pixel values
                idx = px_y * px_hori + px_x;
                ss_def.vals[count] = img_def[idx];
                count++;
                
                // debugging
                //std::cout << px_x << " " << px_y << " " << img_def[idx] << std::endl;
            }
        }
    }


    SubsetData gen_ss_list(bool *img_roi, int ss_step, int ss_size, 
                                int px_hori, int px_vert) {
        
        Timer timer("subset list generation for subset size " + std::to_string(ss_size) +
                    " [px] with step " + std::to_string(ss_step) + " [px]:" );

        SubsetData ssdata;

        int idx;
        int dx[4] = {ss_step, 0, -ss_step, 0};
        int dy[4] = {0, ss_step, 0, -ss_step};

        int subset_counter = 0;

        int num_ss_x = px_hori / ss_step;
        int num_ss_y = px_vert / ss_step;
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

                       if(!is_valid_pixel(px_x,px_y,px_hori,
                                          px_vert,img_roi)){
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

                        if(!is_valid_pixel(x,y,px_hori,
                                           px_vert,img_roi)){

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

            ssdata.neigh[center_idx] = std::move(temp_neigh);
        }

        return ssdata;
    }

    void resize_results(int num_def_img, int num_ss, 
                        int num_params, bool at_end){
        
        util::at_end = at_end;

        if (at_end){
            niter_arr.resize(num_def_img * num_ss);
            u_arr.resize(num_def_img * num_ss);
            v_arr.resize(num_def_img * num_ss);
            p_arr.resize(num_def_img * num_ss * num_params);
            ftol_arr.resize(num_def_img * num_ss);
            xtol_arr.resize(num_def_img * num_ss);
            cost_arr.resize(num_def_img * num_ss);
        }
        else {
            niter_arr.resize(num_ss);
            u_arr.resize(num_ss);
            v_arr.resize(num_ss);
            p_arr.resize(num_ss * num_params);
            ftol_arr.resize(num_ss);
            xtol_arr.resize(num_ss);
            cost_arr.resize(num_ss);
        }
    }


    void append_results(int img_num, int ss, util::Results &res, 
                        int num_ss) {
        int idx;
        if (util::at_end) idx = img_num * num_ss + ss;
        else idx = ss;

        int idx_p = res.p.size()*idx;
        niter_arr[idx] = res.iter;
        u_arr[idx] = res.u;
        v_arr[idx] = res.v;
        ftol_arr[idx] = res.ftol;
        xtol_arr[idx] = res.xtol;
        cost_arr[idx] = res.cost;
        for (size_t i = 0; i < res.p.size(); i++){
            p_arr[idx_p+i] = res.p[i];
        }
    }


    void save_to_disk(int img, util::SaveConfig &saveconf,
                      util::SubsetData &ssdata, int num_def_img,
                      int num_params){

        const std::string delimiter = saveconf.delimiter;

        // open the file
        std::stringstream outfile_str;
        std::ofstream outfile;

        std::string file_ext;
        if (saveconf.binary) file_ext=".bin";
        else file_ext=".dat";

        // filename
        outfile_str << saveconf.basepath << "/" <<
        saveconf.prefix << std::setw(4) << std::setfill('0') << img << file_ext;

        // set the img var to 0 after opening file if not saving at end
        if (!saveconf.at_end) img = 0;

        // save in binary format
        if (saveconf.binary){
            outfile.open(outfile_str.str(), std::ios::binary);

            for (size_t i = 0; i < ssdata.num; ++i) {

                int idx = img * ssdata.num + i;
                int idx_p = num_params*idx;

                double mag = std::sqrt(u_arr[idx]*u_arr[idx]+
                                       v_arr[idx]*v_arr[idx]);

                write_int(outfile, ssdata.coords[2*i]);
                write_int(outfile, ssdata.coords[2*i+1]);
                write_dbl(outfile, u_arr[idx]);
                write_dbl(outfile, v_arr[idx]);
                write_dbl(outfile, mag);
                write_dbl(outfile, cost_arr[idx]);
                write_dbl(outfile, ftol_arr[idx]);
                write_dbl(outfile, xtol_arr[idx]);
                write_int(outfile, niter_arr[idx]);
            }

            outfile.close();
        }
        else {

            outfile.open(outfile_str.str());
            for (size_t i = 0; i < ssdata.num; i++) {

                int idx = img * ssdata.num + i;
                int idx_p = num_params*idx;

                outfile << ssdata.coords[2*i] << delimiter;
                outfile << ssdata.coords[2*i+1] << delimiter;
                outfile << u_arr[idx] << delimiter;
                outfile << v_arr[idx] << delimiter;
                outfile << sqrt(u_arr[idx]*u_arr[idx]+
                                v_arr[idx]*v_arr[idx]) << delimiter;
                //for (int p = 0; p < num_params; p++){
                //    outfile << p_arr[idx_p+p] << delimiter;
                //}
                outfile << cost_arr[idx] << delimiter;
                outfile << ftol_arr[idx] << delimiter;
                outfile << xtol_arr[idx] << delimiter;
                outfile << niter_arr[idx] << "\n";
            }
            outfile.close();
        }
    }

    bool is_valid_pixel(int px_x, int px_y, int px_hori, 
                        int px_vert, bool *img_roi) {
        if (px_x < 0 || px_y < 0 ||
            px_x >= px_hori || px_y >= px_vert) {
            return false;
        }
        int idx = px_y * px_hori + px_x;
        if (!img_roi[idx]) {
            return false;
        }
        return true;
    }


    inline void write_int(std::ofstream& out, int val) {
        out.write(reinterpret_cast<const char*>(&val), sizeof(int));
    }

    inline void write_dbl(std::ofstream& out, double val) {
        out.write(reinterpret_cast<const char*>(&val), sizeof(double));
    }


}

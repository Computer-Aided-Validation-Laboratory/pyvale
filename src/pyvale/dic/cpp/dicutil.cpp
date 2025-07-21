// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

// Program Header files
#include "./dicinterpolator.hpp"
#include "./indicators.hpp"
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
    std::vector<bool> conv_arr;
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



    void extract_ss(util::Subset &ss_ref, 
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

    void extract_ss_subpx(util::Subset &ss_def, 
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

    SubsetData gen_ss_list(const bool *img_roi, const int ss_step, 
                           const int ss_size, const int px_hori, 
                           const int px_vert, const bool partial) {
        
        //Timer timer("subset list generation for subset size " + std::to_string(ss_size) + " [px] with step " + std::to_string(ss_step) + " [px]:" );

        SubsetData ssdata;

        int dx[4] = {ss_step, 0, -ss_step, 0};
        int dy[4] = {0, ss_step, 0, -ss_step};

        int subset_counter = 0;

        int num_ss_x = px_hori / ss_step;
        int num_ss_y = px_vert / ss_step;
        //ssdata.mask.resize(num_ss_x*num_ss_y, NAN);
        ssdata.num_ss_x = num_ss_x;
        ssdata.num_ss_y = num_ss_y;
        ssdata.num_in_mask = num_ss_x * num_ss_y;
        ssdata.num = 0;
        ssdata.step = ss_step;
        ssdata.size = ss_size;

        ssdata.mask.resize(ssdata.num_in_mask, -1);
        ssdata.coords.resize(2*ssdata.num_in_mask, -1);

        // First pass: collect valid subset centers and idx them
        // TODO: Parallelise this with openMP
        for (int j = 0; j < num_ss_y; j++) {
            for (int i = 0; i < num_ss_x; i++) {

                // calculate the coordinates of the subset
                int ss_x = i * ss_step;
                int ss_y = j * ss_step;

                // pixel range of subset
                int xmin = ss_x;
                int ymin = ss_y;
                int xmax = ss_x + ss_size-1;
                int ymax = ss_y + ss_size-1;

                // check if subset is within image and ROI.
                bool valid = true;
                int  valid_count = 0;
                for (int px_y = ymin; px_y <= ymax && valid; px_y++) {
                    for (int px_x = xmin; px_x <= xmax && valid; px_x++) {

                        // When no partial subset filling all px must be within roi
                        if (!partial) {
                            if (!is_valid_in_dims(px_x, px_y, px_hori, px_vert) ||
                                !is_valid_in_roi(px_x, px_y, px_hori, px_vert, img_roi)) {
                                valid = false;
                                break;
                            }
                        } 

                        // When partial count num of px in roi
                        else {
                            if (is_valid_in_dims(px_x, px_y, px_hori, px_vert) &&
                                is_valid_in_roi(px_x, px_y, px_hori, px_vert, img_roi)) {
                                valid_count++;
                            }
                        }
                    }

                    if (!valid && !partial) break;
                }

                // TODO: this is hardcoded so that atleast 70% of pixels in subset must be in ROI
                if (partial) {
                    if (valid_count >= (ss_size*ss_size) * (0.70)) {
                        valid = true;
                    } else {
                        valid = false;
                    }
                }

                // if its a valid subset. add it to a list of coordinates
                if (valid) {
                    ssdata.coords[2*subset_counter] = ss_x;
                    ssdata.coords[2*subset_counter+1] = ss_y;
                    ssdata.mask[j * num_ss_x + i] = subset_counter;
                    subset_counter++;
                }
            }
        }

        ssdata.coords.resize(2*subset_counter);
        ssdata.num = subset_counter;
        ssdata.neigh.resize(ssdata.num);

        // neighbours for each of the above subset
        // TODO: Parallelise with openMP
        for (int j = 0; j < num_ss_y; ++j) {
            for (int i = 0; i < num_ss_x; ++i) {

                // calculate the coordinates of the subset
                int idx = ssdata.mask[j * num_ss_x + i];

                if (idx == -1) continue;

                std::vector<int> temp_neigh;

                for (int d = 0; d < 4; ++d) {
                    int ni = i + dx[d] / ss_step;
                    int nj = j + dy[d] / ss_step;

                    if (ni >= 0 && ni < num_ss_x && nj >= 0 && nj < num_ss_y) {
                        int neigh_idx = ssdata.mask[nj * num_ss_x + ni];
                        if (neigh_idx != -1) {
                            temp_neigh.push_back(neigh_idx);
                        }
                    }
                }

                ssdata.neigh[idx] = temp_neigh;

                // debugging
                //int ss_x = ssdata.coords[2*idx];
                //int ss_y = ssdata.coords[2*idx+1];
                //std::cout << idx << " " << ss_x << " " << ss_y << " ";
                //for (int n = 0; n <  ssdata.neigh[idx].size(); n++){
                //    int nidx = ssdata.neigh[idx][n];
                //    std::cout << ssdata.coords[2*nidx] << " " << ssdata.coords[2*nidx+1] << " ";
                //}
                //std::cout << std::endl;
            }
        }

        //for (const auto& kv : ssdata.coords_to_idx) {
        //    const std::pair<int, int>& coord = kv.first;
        //    int center_idx = kv.second;

        //    std::vector<int> temp_neigh;

        //    for (int i = 0; i < 4; ++i) {
        //        int neigh_x = coord.first + dx[i];
        //        int neigh_y = coord.second + dy[i];

        //        int xmin = neigh_x;
        //        int ymin = neigh_y;
        //        int xmax = neigh_x + ss_size;
        //        int ymax = neigh_y + ss_size;

        //        bool valid = true;

        //        // checking if the neigbour is valid
        //        for (int y = ymin; y <= ymax && valid; ++y) {
        //            for (int x = xmin; x <= xmax && valid; ++x) {

        //                if(!is_valid_pixel(x,y,px_hori,
        //                                   px_vert,img_roi)){

        //                    valid = false;
        //                    break;
        //                }

        //            }
        //        }

        //        if (valid) {
        //            auto it = ssdata.coords_to_idx.find({neigh_x, neigh_y});
        //            if (it != ssdata.coords_to_idx.end()) {
        //                temp_neigh.push_back(it->second);
        //            }
        //        }
        //    }

        //    ssdata.neigh[center_idx] = std::move(temp_neigh);
        //}

        return ssdata;
    }

    void resize_results(int num_def_img, int num_ss, 
                        int num_params, bool at_end){


        util::Timer timer("resizing of result arrays:");
        util::at_end = at_end;

        if (at_end){
            niter_arr.resize(num_def_img * num_ss);
            u_arr.resize(num_def_img * num_ss);
            v_arr.resize(num_def_img * num_ss);
            p_arr.resize(num_def_img * num_ss * num_params);
            ftol_arr.resize(num_def_img * num_ss);
            xtol_arr.resize(num_def_img * num_ss);
            cost_arr.resize(num_def_img * num_ss);
            conv_arr.resize(num_def_img * num_ss);
        }
        else {
            niter_arr.resize(num_ss);
            u_arr.resize(num_ss);
            v_arr.resize(num_ss);
            p_arr.resize(num_ss * num_params);
            ftol_arr.resize(num_ss);
            xtol_arr.resize(num_ss);
            cost_arr.resize(num_ss);
            conv_arr.resize(num_ss);
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
        conv_arr[idx] = res.converged;
        for (size_t i = 0; i < res.p.size(); i++){
            p_arr[idx_p+i] = res.p[i];
        }
    }


    void save_to_disk(int img, const util::SaveConfig &saveconf,
                      const util::SubsetData &ssdata, const int num_def_img,
                      const int num_params, const std::vector<std::string> &filenames){

        const std::string delimiter = saveconf.delimiter;

        // open the file
        std::stringstream outfile_str;
        std::ofstream outfile;

        std::string file_ext;
        if (saveconf.binary) file_ext=".dic2d";
        else file_ext=".csv";

        // Extract the base filename without extension
        std::string full_filename = filenames[img];
        size_t dot_pos = full_filename.find(".");
        if (dot_pos != std::string::npos) {
            full_filename = full_filename.substr(0, dot_pos);
        }

        // output filename
        outfile_str << saveconf.basepath << "/" <<
        saveconf.prefix << full_filename << file_ext;

        // set the img var to 0 after opening file if not saving at end
        if (!saveconf.at_end) img = 0;

        // save in binary format
        if (saveconf.binary){
            outfile.open(outfile_str.str(), std::ios::binary);

            for (int i = 0; i < ssdata.num; ++i) {

                int idx = img * ssdata.num + i;
                //int idx_p = num_params*idx;

                double mag = std::sqrt(u_arr[idx]*u_arr[idx]+
                                       v_arr[idx]*v_arr[idx]);

                // convert from corner to centre subset coords
                double ss_x = ssdata.coords[2*i  ] + static_cast<double>(ssdata.size)/2.0 - 0.5;
                double ss_y = ssdata.coords[2*i+1] + static_cast<double>(ssdata.size)/2.0 - 0.5;

                write_int(outfile, ss_x);
                write_int(outfile, ss_y);
                write_dbl(outfile, u_arr[idx]);
                write_dbl(outfile, v_arr[idx]);
                write_dbl(outfile, mag);
                write_bool(outfile, conv_arr[idx]);
                write_dbl(outfile, cost_arr[idx]);
                write_dbl(outfile, ftol_arr[idx]);
                write_dbl(outfile, xtol_arr[idx]);
                write_int(outfile, niter_arr[idx]);
            }

            outfile.close();
        }
        else {

            outfile.open(outfile_str.str());

            // column headers
            outfile << "subset_x" << delimiter;
            outfile << "subset_y" << delimiter;
            outfile << "displacement_u" << delimiter;
            outfile << "displacement_v" << delimiter;
            outfile << "displacement_mag" << delimiter;
            outfile << "converged" << delimiter;
            outfile << "cost" << delimiter;
            outfile << "ftol" << delimiter;
            outfile << "xtol" << delimiter;
            outfile << "num_iterations\n";

            for (int i = 0; i < ssdata.num; i++) {

                int idx = img * ssdata.num + i;
                //int idx_p = num_params*idx;

                // convert from corner to centre subset coords
                double ss_x = ssdata.coords[2*i  ] + static_cast<double>(ssdata.size)/2.0 - 0.5;
                double ss_y = ssdata.coords[2*i+1] + static_cast<double>(ssdata.size)/2.0 - 0.5;


                outfile << ss_x << delimiter;
                outfile << ss_y << delimiter;
                outfile << u_arr[idx] << delimiter;
                outfile << v_arr[idx] << delimiter;
                outfile << sqrt(u_arr[idx]*u_arr[idx]+
                                v_arr[idx]*v_arr[idx]) << delimiter;
                //for (int p = 0; p < num_params; p++){
                //    outfile << p_arr[idx_p+p] << delimiter;
                //}
                outfile << conv_arr[idx] << delimiter;
                outfile << cost_arr[idx] << delimiter;
                outfile << ftol_arr[idx] << delimiter;
                outfile << xtol_arr[idx] << delimiter;
                outfile << niter_arr[idx] << "\n";
            }
            outfile.close();
        }
    }

    inline bool is_valid_in_dims(const int px_x, const int px_y, const int px_hori, 
                        const int px_vert) {

        if (px_x < 0 || px_y < 0 ||
            px_x >= px_hori || px_y >= px_vert) {
            return false;
        }
        return true;
    }

    inline bool is_valid_in_roi(const int px_x, const int px_y, const int px_hori, 
                        const int px_vert, const bool *img_roi) {

        int idx = px_y * px_hori + px_x;
        if (!img_roi[idx]) {
            return false;
        }
        return true;
    }

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


    void gen_size_and_step_vector(std::vector<int> &ss_sizes, std::vector<int> &ss_steps, 
                                  const int ss_size, const int ss_step, const int max_disp) {

        ss_sizes.clear();
        ss_steps.clear();

        int power = next_pow2(max_disp);

        // Generate sizes down to just above ss_size
        while (power > ss_size) {
            ss_sizes.push_back(power);
            ss_steps.push_back(power / 2);
            power /= 2;
        }

        // Finally, add the original ss_size and ss_step
        ss_sizes.push_back(ss_size);
        ss_steps.push_back(ss_step);

        // debugging
        //for (size_t i = 0; i < ss_sizes.size(); ++i) {
        //    std::cout << "ss_size = " << ss_sizes[i] << ", step = " << ss_steps[i] << std::endl;
        //}
    }

    void create_progress_bar(indicators::ProgressBar &bar,
                             const std::vector<std::string> &filenames,
                             const int img_num, const int num_ss){
        //Hide cursor
        indicators::show_console_cursor(false);
        bar.set_option(indicators::option::BarWidth{50});
        bar.set_option(indicators::option::Start{" ["});
        bar.set_option(indicators::option::Fill{"#"});
        bar.set_option(indicators::option::Lead{"#"});
        bar.set_option(indicators::option::Remainder{"-"});
        bar.set_option(indicators::option::End{"]"});
        bar.set_option(indicators::option::PrefixText{filenames[img_num]});
        bar.set_option(indicators::option::ShowPercentage{true});
        bar.set_option(indicators::option::ShowElapsedTime{true});
    }

    void update_progress_bar(indicators::ProgressBar &bar, int i, int num_ss, std::atomic<int> &prev_pct) {
        int curr_pct = static_cast<float>(i) / static_cast<float>(num_ss) * 100;
        int expected = prev_pct.load();

        // Only update bar if we've passed the previous percentage
        if (curr_pct > expected && prev_pct.compare_exchange_strong(expected, curr_pct)) {
            #pragma omp critical
                bar.set_progress(curr_pct);
        }
    }



}

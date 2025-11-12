// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>

// Program Header files
#include "./dicresults.hpp"
#include "./dicutil.hpp"



OptResultArrays::OptResultArrays(int num_def_img, int num_ss, int num_params, bool at_end){

    common_util::Timer timer("resizing of result arrays:");
    this->at_end = at_end;
    this->num_ss = num_ss;
    this->num_params = num_params;

    if (at_end){
        niter.resize(num_def_img * num_ss);
        u.resize(num_def_img * num_ss);
        v.resize(num_def_img * num_ss);
        p.resize(num_def_img * num_ss * num_params);
        ftol.resize(num_def_img * num_ss);
        xtol.resize(num_def_img * num_ss);
        cost.resize(num_def_img * num_ss);
        conv.resize(num_def_img * num_ss);
    }
    else {
        niter.resize(num_ss);
        u.resize(num_ss);
        v.resize(num_ss);
        p.resize(num_ss * num_params);
        ftol.resize(num_ss);
        xtol.resize(num_ss);
        cost.resize(num_ss);
        conv.resize(num_ss);
    }
}

void OptResultArrays::append(OptResult &res, int img_num, int ss) {
    int idx;
    if (at_end) idx = img_num * num_ss + ss;
    else idx = ss;

    int idx_p = res.p.size()*idx;
    niter[idx] = res.iter;
    u[idx] = res.u;
    v[idx] = res.v;
    ftol[idx] = res.ftol;
    xtol[idx] = res.xtol;
    cost[idx] = res.cost;
    conv[idx] = res.converged;
    for (size_t i = 0; i < res.p.size(); i++){
        p[idx_p+i] = res.p[i];
    }
}

int OptResultArrays::index(const int subset_idx, const int img_num){
    int idx = at_end ? (img_num) * num_ss + subset_idx : subset_idx;
    return idx;
}

int OptResultArrays::index_parameters(const int subset_idx, const int img_num){
    int idx = index(subset_idx, img_num) * num_params;
    return idx;
}


void OptResultArrays::write_to_disk(int img, const common_util::SaveConfig &saveconf,
                    const subset::Grid &ss_grid, const int num_def_img,
                    const std::vector<std::string> &filenames){

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

        for (int i = 0; i < ss_grid.num; ++i) {

            int idx = img * ss_grid.num + i;
            //int idx_p = num_params*idx;

            // if the subset has not converged, set values to nan
            if (!saveconf.output_unconverged && !conv[idx]) {
                u[idx] = NAN;
                v[idx] = NAN;
                for (int pidx = 0; pidx < num_params; pidx++){
                    p[num_params*idx+pidx] = NAN;
                }
                cost[idx] = NAN;
                ftol[idx] = NAN;
                xtol[idx] = NAN;
            }


            double mag = std::sqrt(u[idx]*u[idx]+
                                    v[idx]*v[idx]);

            // convert from corner to centre subset coords
            double ss_x = ss_grid.coords[2*i  ] + static_cast<double>(ss_grid.size)/2.0 - 0.5;
            double ss_y = ss_grid.coords[2*i+1] + static_cast<double>(ss_grid.size)/2.0 - 0.5;

            common_util::write_int(outfile, ss_x);
            common_util::write_int(outfile, ss_y);
            common_util::write_dbl(outfile, u[idx]);
            common_util::write_dbl(outfile, v[idx]);
            common_util::write_dbl(outfile, mag);
            common_util::write_uint8t(outfile, conv[idx]);
            common_util::write_dbl(outfile, cost[idx]);
            common_util::write_dbl(outfile, ftol[idx]);
            common_util::write_dbl(outfile, xtol[idx]);
            common_util::write_int(outfile, niter[idx]);

            if (saveconf.shape_params) {
                for (int pidx = 0; pidx < num_params; pidx++){
                    common_util::write_dbl(outfile, p[num_params*idx+pidx]);
                }
            }

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
        outfile << "num_iterations";

        // column headers for shape parameters
        if (saveconf.shape_params) {
            for (int p = 0; p < num_params; p++){
                outfile << delimiter;
                outfile << "shape_p" << p;
            }
        }

        // newline after headers
        outfile << "\n";

        for (int i = 0; i < ss_grid.num; i++) {

            int idx = img * ss_grid.num + i;
            //int idx_p = num_params*idx;

            // convert from corner to centre subset coords
            double ss_x = ss_grid.coords[2*i  ] + static_cast<double>(ss_grid.size)/2.0 - 0.5;
            double ss_y = ss_grid.coords[2*i+1] + static_cast<double>(ss_grid.size)/2.0 - 0.5;

            // if the subset has not converged, set values to nan
            if (!saveconf.output_unconverged && !conv[idx]) {
                u[idx] = NAN;
                v[idx] = NAN;
                for (int pidx = 0; pidx < num_params; pidx++){
                    p[num_params*idx+pidx] = NAN;
                }
                cost[idx] = NAN;
                ftol[idx] = NAN;
                xtol[idx] = NAN;
            }


            outfile << ss_x << delimiter;
            outfile << ss_y << delimiter;
            outfile << u[idx] << delimiter;
            outfile << v[idx] << delimiter;
            outfile << sqrt(u[idx]*u[idx]+
                            v[idx]*v[idx]) << delimiter;
            outfile << static_cast<int>(conv[idx]) << delimiter;
            outfile << cost[idx] << delimiter;
            outfile << ftol[idx] << delimiter;
            outfile << xtol[idx] << delimiter;
            outfile << niter[idx];

            // write shape parameters if requested
            if (saveconf.shape_params) {
                for (int pidx = 0; pidx < num_params; pidx++){
                    outfile << delimiter;
                    outfile << p[num_params*idx+pidx];
                }
            }

            // newline after each subset
            outfile << "\n";


        }
        outfile.close();
    }
}




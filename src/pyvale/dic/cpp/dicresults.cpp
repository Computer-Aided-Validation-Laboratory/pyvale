// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <sstream>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

// commoncpp header files
#include "../../commoncpp/util.hpp"
#include "../../commoncpp/csvbuffer.hpp"

// DIC Header files
#include "./dicresults.hpp"





ResultArrays::ResultArrays(int num_ss,
                           int num_params,
                           bool stereo){

    common_util::Timer timer("to resize result arrays:", 3);

    this->num_ss = num_ss;
    this->num_params = num_params;
    this->stereo = stereo;

    niter.resize(num_ss, 0.0);
    u.resize(num_ss, 0.0);
    v.resize(num_ss, 0.0);
    p.resize(num_ss * num_params, 0.0);
    ftol.resize(num_ss, 0.0);
    xtol.resize(num_ss, 0.0);
    cost.resize(num_ss, 0.0);
    conv.resize(num_ss, 0.0);
    above_thresh.resize(num_ss);


    // incremental tracking
    // u_last_good.resize(num_ss, 0.0);
    // v_last_good.resize(num_ss, 0.0);
    // du_dt.resize(num_ss, 0.0);
    // dv_dt.resize(num_ss, 0.0);
    // last_success_frame.resize(num_ss, -1);
    // has_good_history.resize(num_ss, 0);

    if (stereo){
        const double nan = std::numeric_limits<double>::quiet_NaN();
        x_world.resize(num_ss, nan);
        y_world.resize(num_ss, nan);
        z_world.resize(num_ss, nan);
        u_world.resize(num_ss, nan);
        v_world.resize(num_ss, nan);
        w_world.resize(num_ss, nan);
        epi_dist_px.resize(num_ss, nan);
    }
}

void ResultArrays::append(OptResult &res, const int i) {
    
    int pp = num_params*i;
    niter[i] = res.iter;
    u[i] = res.u;
    v[i] = res.v;
    ftol[i] = res.ftol;
    xtol[i] = res.xtol;
    cost[i] = res.cost;
    conv[i] = res.converged;
    above_thresh[i] = res.above_thresh;
    for (size_t i = 0; i < num_params; i++){
        p[pp+i] = res.p[i];
    }
}

  void ResultArrays::reset() {
    std::fill(niter.begin(), niter.end(), 0);
    std::fill(u.begin(), u.end(), 0.0);
    std::fill(v.begin(), v.end(), 0.0);
    std::fill(p.begin(), p.end(), 0.0);
    std::fill(ftol.begin(), ftol.end(), 0.0);
    std::fill(xtol.begin(), xtol.end(), 0.0);
    std::fill(cost.begin(), cost.end(), 0.0);
    std::fill(conv.begin(), conv.end(), false);
    std::fill(above_thresh.begin(), above_thresh.end(), false);

    if (stereo) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::fill(x_world.begin(), x_world.end(), nan);
        std::fill(y_world.begin(), y_world.end(), nan);
        std::fill(z_world.begin(), z_world.end(), nan);
        std::fill(u_world.begin(), u_world.end(), nan);
        std::fill(v_world.begin(), v_world.end(), nan);
        std::fill(w_world.begin(), w_world.end(), nan);
        std::fill(epi_dist_px.begin(), epi_dist_px.end(), nan);
    }
  }

// void ResultArrays::get_latest_matches(const ResultArrays &results_def, const int img_num_def) {
//
//     // Naive check that that results_ref and results_def are the same size
//     if (u.size() != results_def.u.size()){
//         niter.resize(num_ss, 0.0);
//         u.resize(num_ss, 0.0);
//         v.resize(num_ss, 0.0);
//         p.resize(num_ss * num_params, 0.0);
//         ftol.resize(num_ss, 0.0);
//         xtol.resize(num_ss, 0.0);
//         cost.resize(num_ss, 0.0);
//         conv.resize(num_ss, 0.0);
//         above_thresh.resize(num_ss);
//
//
//         // incremental tracking
//         u_last_good.resize(num_ss, 0.0);
//         v_last_good.resize(num_ss, 0.0);
//         du_dt.resize(num_ss, 0.0);
//         dv_dt.resize(num_ss, 0.0);
//         last_success_frame.resize(num_ss, -1);
//         has_good_history.resize(num_ss, 0);
//
//         if (stereo){
//             x_world.resize(num_ss,0.0);
//             y_world.resize(num_ss,0.0);
//             z_world.resize(num_ss,0.0);
//             u_world.resize(num_ss,0.0);
//             v_world.resize(num_ss,0.0);
//             w_world.resize(num_ss,0.0);
//         }
//     }
//
//     for (int i = 0; i < u.size(); i++){
//
//         if (results_def.above_thresh[i]){
//
//             if (has_good_history[i]){
//
//                 int dt = img_num_def - last_success_frame[i];
//
//                 if (dt > 0){
//                     du_dt[i] = (results_def.u[i] - u_last_good[i]) / dt;
//                     dv_dt[i] = (results_def.v[i] - v_last_good[i]) / dt;
//                 }
//             }
//
//             // update last good state
//             u_last_good[i] = results_def.u[i];
//             v_last_good[i] = results_def.v[i];
//             last_success_frame[i] = img_num_def;
//             has_good_history[i] = 1;
//             u[i] = results_def.u[i];
//             v[i] = results_def.v[i];
//             p[i] = results_def.p[i];
//             above_thresh[i] = 1;
//         }
//     }
// }
//


// int ResultArrays::index(const int subset_idx, const int results_num){
//     int idx = at_end ? (results_num) * num_ss + subset_idx : subset_idx;
//     return idx;
// }
//
// int ResultArrays::index_parameters(const int subset_idx, const int results_num){
//     int idx = index(subset_idx, results_num) * num_params;
//     return idx;
// }


void ResultArrays::write_to_disk_2d(const common_util::SaveConfig &saveconf,
                                    const SubsetGrid &ss_grid,
                                    const std::string &filename){


    common_util::Timer timer("to write DIC results to disk:", 3);

    const std::string delimiter = saveconf.delimiter;

    std::stringstream outfile_str;
    std::ofstream outfile;

    std::string file_ext;
    if (saveconf.binary) file_ext=".dic2d";
    else file_ext=".csv";

    std::string full_filename = filename;
    size_t dot_pos = full_filename.find(".");
    if (dot_pos != std::string::npos) {
        full_filename = full_filename.substr(0, dot_pos);
    }

    outfile_str << saveconf.basepath << "/"
                << saveconf.prefix
                << full_filename
                << file_ext;

    outfile << std::fixed << std::setprecision(8);

    // set the img var to 0 after opening file if not saving at end
    //if (!saveconf.at_end) results_num = 0;

    // save in binary format
    if (saveconf.binary){
        outfile.open(outfile_str.str(), std::ios::binary);

        for (int i = 0; i < ss_grid.num; ++i) {

            // if the subset has not met threshold, set values to nan
            if (!saveconf.output_below_threshold && !above_thresh[i]) {
                u[i] = NAN;
                v[i] = NAN;
                for (int pp = 0; pp < num_params; pp++){
                    p[num_params*i+pp] = NAN;
                }
                cost[i] = NAN;
                ftol[i] = NAN;
                xtol[i] = NAN;
            }

            // displacement magnitude
            double mag = std::sqrt(u[i]*u[i]+v[i]*v[i]);

            // convert from corner to centre subset coords
            double ss_x = ss_grid.coords[2*i  ];
            double ss_y = ss_grid.coords[2*i+1];

            common_util::write_int(outfile, ss_x);
            common_util::write_int(outfile, ss_y);
            common_util::write_dbl(outfile, u[i]);
            common_util::write_dbl(outfile, v[i]);
            common_util::write_dbl(outfile, mag);
            common_util::write_uint8t(outfile, conv[i]);
            common_util::write_dbl(outfile, cost[i]);
            common_util::write_dbl(outfile, ftol[i]);
            common_util::write_dbl(outfile, xtol[i]);
            common_util::write_int(outfile, niter[i]);

            if (saveconf.shape_params) {
                for (int pp = 0; pp < num_params; pp++){
                    common_util::write_dbl(outfile, p[num_params*i+pp]);
                }
            }

        }

        outfile.close();
    }
    else {

        outfile.open(outfile_str.str());
        common_util::CsvBuffer csv(outfile, delimiter, saveconf.precision);

        // column headers
        outfile << "\"subset_x\"" <<  delimiter;
        outfile << "\"subset_y\"" <<  delimiter;
        outfile << "\"disp_u\"" <<  delimiter;
        outfile << "\"disp_v\"" <<  delimiter;
        outfile << "\"disp_mag\"" <<  delimiter;
        outfile << "\"converged\"" <<  delimiter;
        outfile << "\"cost_zncc\"" <<  delimiter;
        outfile << "\"ftol\"" <<  delimiter;
        outfile << "\"xtol\"" <<  delimiter;
        outfile << "\"num_iter\"" << delimiter;

        // if (saveconf.shape_params) {
        //     for (int p = 0; p < num_params; p++){
        //         outfile << "\"shape_p\"" <<  p;
        //         outfile << delimiter;
        //     }
        // }

        // newline after headers
        outfile << "\n";

        for (int i = 0; i < ss_grid.num; i++) {

            // convert from corner to centre subset coords
            double ss_x = ss_grid.coords[2*i  ];
            double ss_y = ss_grid.coords[2*i+1];

            // if the subset has not met threshold, set values to nan
            if (!saveconf.output_below_threshold && !above_thresh[i]) {
                u[i] = NAN;
                v[i] = NAN;
                for (int pi = 0; pi < num_params; pi++){
                    p[num_params*i+pi] = NAN;
                }
                cost[i] = NAN;
                ftol[i] = NAN;
                xtol[i] = NAN;
            }

            // displacement magnitude
            double mag = std::sqrt(u[i]*u[i]+v[i]*v[i]);

            csv.field(ss_x);
            csv.field(ss_y);
            csv.field(u[i]);
            csv.field(v[i]);
            csv.field(mag);
            csv.field(static_cast<int>(conv[i]));
            csv.field(cost[i]);
            csv.field(ftol[i]);
            csv.field(xtol[i]);
            csv.field(niter[i], true);

            // if (saveconf.shape_params) {
            //     for (int pp = 0; pp < num_params; pp++){
            //         outfile << delimiter;
            //         outfile << p[num_params*i+pp];
            //     }
            // }

            // newline after each subset
            csv.newline();


        }
        csv.flush();
        outfile.close();
    }
}



void ResultArrays::write_to_disk_stereo(const ResultArrays &stereo,
                                        const common_util::SaveConfig &saveconf,
                                        const SubsetGrid &ss_grid,
                                        const std::string &filename){

    const std::string delimiter = saveconf.delimiter;

    std::stringstream outfile_str;
    std::ofstream outfile;

    std::string file_ext;
    if (saveconf.binary) file_ext=".dic3d";
    else file_ext=".csv";

    std::string full_filename = filename;
    size_t dot_pos = full_filename.find(".");
    if (dot_pos != std::string::npos) {
        full_filename = full_filename.substr(0, dot_pos);
    }

    outfile_str << saveconf.basepath << "/"
                << saveconf.prefix
                << full_filename
                << file_ext;

    outfile << std::fixed << std::setprecision(8);

    // set the img var to 0 after opening file if not saving at end
    //if (!saveconf.at_end) results_num = 0;

    // save in binary format
    if (saveconf.binary){
        outfile.open(outfile_str.str(), std::ios::binary);

        for (int i = 0; i < ss_grid.num; ++i) {

            const bool temporal_valid = saveconf.output_below_threshold || above_thresh[i];
            const bool stereo_valid = saveconf.output_below_threshold || stereo.above_thresh[i];
            const double out_u = temporal_valid ? u[i] : NAN;
            const double out_v = temporal_valid ? v[i] : NAN;
            const double out_cost = temporal_valid ? cost[i] : NAN;
            const double out_ftol = temporal_valid ? ftol[i] : NAN;
            const double out_xtol = temporal_valid ? xtol[i] : NAN;
            const double out_stereo_u = temporal_valid ? stereo.u[i] : NAN;
            const double out_stereo_v = temporal_valid ? stereo.v[i] : NAN;
            const double out_stereo_cost = stereo_valid ? stereo.cost[i] : NAN;
            const double out_stereo_ftol = stereo_valid ? stereo.ftol[i] : NAN;
            const double out_stereo_xtol = stereo_valid ? stereo.xtol[i] : NAN;
            const double out_x_world = stereo_valid ? stereo.x_world[i] : NAN;
            const double out_y_world = stereo_valid ? stereo.y_world[i] : NAN;
            const double out_z_world = stereo_valid ? stereo.z_world[i] : NAN;
            const double out_u_world = stereo_valid ? stereo.u_world[i] : NAN;
            const double out_v_world = stereo_valid ? stereo.v_world[i] : NAN;
            const double out_w_world = stereo_valid ? stereo.w_world[i] : NAN;


            // convert from corner to centre subset coords
            double ss_x = ss_grid.coords[2*i  ];
            double ss_y = ss_grid.coords[2*i+1];

            // displacement magnitude
            double mag_temporal = std::sqrt(out_u*out_u+out_v*out_v);
            double mag_stereo = std::sqrt(out_stereo_u*out_stereo_u+out_stereo_v*out_stereo_v);

            common_util::write_int(outfile, ss_x);
            common_util::write_int(outfile, ss_y);
            common_util::write_dbl(outfile, out_u);
            common_util::write_dbl(outfile, out_v);
            common_util::write_dbl(outfile, mag_temporal);
            common_util::write_uint8t(outfile, conv[i]);
            common_util::write_dbl(outfile, out_cost);
            common_util::write_dbl(outfile, out_ftol);
            common_util::write_dbl(outfile, out_xtol);
            common_util::write_int(outfile, niter[i]);
            common_util::write_dbl(outfile, out_stereo_u);
            common_util::write_dbl(outfile, out_stereo_v);
            common_util::write_dbl(outfile, mag_stereo);
            common_util::write_dbl(outfile, out_u_world);
            common_util::write_dbl(outfile, out_v_world);
            common_util::write_dbl(outfile, out_w_world);
            common_util::write_dbl(outfile, out_x_world);
            common_util::write_dbl(outfile, out_y_world);
            common_util::write_dbl(outfile, out_z_world);
            common_util::write_uint8t(outfile, stereo.conv[i]);
            common_util::write_dbl(outfile, out_stereo_cost);
            common_util::write_dbl(outfile, out_stereo_ftol);
            common_util::write_dbl(outfile, out_stereo_xtol);
            common_util::write_int(outfile, stereo.niter[i]);
            common_util::write_dbl(outfile, stereo.epi_dist_px[i]);

            // if (saveconf.shape_params) {
            //     for (int pp = 0; pp < num_params; pp++){
            //         common_util::write_dbl(outfile, p[num_params*i+pp]);
            //         common_util::write_dbl(outfile, stereo.p[stereo.num_params*i+pp]);
            //     }
            // }

        }

        outfile.close();
    }
    else {

        outfile.open(outfile_str.str());
        common_util::CsvBuffer csv(outfile, delimiter, saveconf.precision);

        // column headers
        outfile << "\"subset_x\"" <<  delimiter;
        outfile << "\"subset_y\"" <<  delimiter;
        outfile << "\"disp_u\"" <<  delimiter;
        outfile << "\"disp_v\"" <<  delimiter;
        outfile << "\"disp_mag\"" <<  delimiter;
        outfile << "\"converged\"" <<  delimiter;
        outfile << "\"cost_zncc\"" <<  delimiter;
        outfile << "\"ftol\"" <<  delimiter;
        outfile << "\"xtol\"" <<  delimiter;
        outfile << "\"num_iter\"" << delimiter;
        outfile << "\"stereo_disp_u_px\"" <<  delimiter;
        outfile << "\"stereo_disp_v_px\"" <<  delimiter;
        outfile << "\"stereo_disp_mag_px\"" <<  delimiter;
        outfile << "\"stereo_disp_u_mm\"" << delimiter;
        outfile << "\"stereo_disp_v_mm\"" << delimiter;
        outfile << "\"stereo_disp_w_mm\"" << delimiter;
        outfile << "\"stereo_x_mm\"" << delimiter;
        outfile << "\"stereo_y_mm\"" << delimiter;
        outfile << "\"stereo_z_mm\"" << delimiter;
        outfile << "\"stereo_converged\"" <<  delimiter;
        outfile << "\"stereo_cost_zncc\"" <<  delimiter;
        outfile << "\"stereo_ftol\"" <<  delimiter;
        outfile << "\"stereo_xtol\"" <<  delimiter;
        outfile << "\"stereo_num_iter\"" << delimiter;
        outfile << "\"epi_dist_px\"";
        // if (saveconf.shape_params) {
        //     for (int p = 0; p < num_params; p++){
        //         outfile << "\"shape_p\"" <<  p;
        //         outfile << delimiter;
        //     }
        // }

        // column headers for shape parameters
        // if (saveconf.shape_params) {
        //     for (int p = 0; p < num_params; p++){
        //         outfile << "\"stereo_shape_p\"" <<  p;
        //         outfile << delimiter;
        //     }
        // }

        // newline after headers
        outfile << "\n";

        for (int i = 0; i < ss_grid.num; i++) {

            // convert from corner to centre subset coords
            double ss_x = ss_grid.coords[2*i  ];
            double ss_y = ss_grid.coords[2*i+1];

            // if the subset has not met threshold, set values to nan
                        const bool temporal_valid = saveconf.output_below_threshold || above_thresh[i];
            const bool stereo_valid = saveconf.output_below_threshold || stereo.above_thresh[i];
            const double out_u = temporal_valid ? u[i] : NAN;
            const double out_v = temporal_valid ? v[i] : NAN;
            const double out_cost = temporal_valid ? cost[i] : NAN;
            const double out_ftol = temporal_valid ? ftol[i] : NAN;
            const double out_xtol = temporal_valid ? xtol[i] : NAN;
            const double out_stereo_u = temporal_valid ? stereo.u[i] : NAN;
            const double out_stereo_v = temporal_valid ? stereo.v[i] : NAN;
            const double out_stereo_cost = stereo_valid ? stereo.cost[i] : NAN;
            const double out_stereo_ftol = stereo_valid ? stereo.ftol[i] : NAN;
            const double out_stereo_xtol = stereo_valid ? stereo.xtol[i] : NAN;
            const double out_x_world = stereo_valid ? stereo.x_world[i] : NAN;
            const double out_y_world = stereo_valid ? stereo.y_world[i] : NAN;
            const double out_z_world = stereo_valid ? stereo.z_world[i] : NAN;
            const double out_u_world = stereo_valid ? stereo.u_world[i] : NAN;
            const double out_v_world = stereo_valid ? stereo.v_world[i] : NAN;
            const double out_w_world = stereo_valid ? stereo.w_world[i] : NAN;

            // displacement magnitude
            double mag_= std::sqrt(out_u*out_u+out_v*out_v);
            double mag_stereo = std::sqrt(out_stereo_u*out_stereo_u+out_stereo_v*out_stereo_v);


            csv.field(ss_x);
            csv.field(ss_y);
            csv.field(out_u);
            csv.field(out_v);
            csv.field(mag_);
            csv.field(static_cast<int>(conv[i]));
            csv.field(out_cost);
            csv.field(out_ftol);
            csv.field(out_xtol);
            csv.field(niter[i]);
            csv.field(out_stereo_u);
            csv.field(out_stereo_v);
            csv.field(mag_stereo);
            csv.field(out_u_world);
            csv.field(out_v_world);
            csv.field(out_w_world);
            csv.field(out_x_world);
            csv.field(out_y_world);
            csv.field(out_z_world);
            csv.field(static_cast<int>(stereo.conv[i]));
            csv.field(out_stereo_cost);
            csv.field(out_stereo_ftol);
            csv.field(out_stereo_xtol);
            csv.field(stereo.niter[i]);
            csv.field(stereo.epi_dist_px[i], true);

            // if (saveconf.shape_params) {
            //     for (int pp = 0; pp < num_params; pp++){
            //         outfile << delimiter;
            //         outfile << p[num_params*i+pp];
            //     }
            // }

            // newline after each subset
            csv.newline();


        }
        csv.flush();
        outfile.close();
    }
}

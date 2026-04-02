// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

// common_cpp header files
#include "../../common_cpp/util.hpp"

// DIC Header files
#include "./dicresults.hpp"
#include "./dicutil.hpp"



ResultArrays::ResultArrays(int num_ss,
                           int num_params,
                           bool stereo){

    if (g_debug_level>0) 
        common_util::Timer timer("resizing of result arrays:");

    //this->at_end = at_end;
    this->num_ss = num_ss;
    this->num_params = num_params;

    niter.resize(num_ss, 0.0);
    u.resize(num_ss, 0.0);
    v.resize(num_ss, 0.0);
    p.resize(num_ss * num_params, 0.0);
    ftol.resize(num_ss, 0.0);
    xtol.resize(num_ss, 0.0);
    cost.resize(num_ss, 0.0);
    conv.resize(num_ss, 0.0);
    above_thresh.resize(num_ss);

    if (stereo){
        x_world.resize(num_ss,0.0);
        y_world.resize(num_ss,0.0);
        z_world.resize(num_ss,0.0);
        u_world.resize(num_ss,0.0);
        v_world.resize(num_ss,0.0);
        w_world.resize(num_ss,0.0);
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
    above_thresh[i] = res.above_threshold;
    for (size_t i = 0; i < num_params; i++){
        p[pp+i] = res.p[i];
    }
}

// int ResultArrays::index(const int subset_idx, const int results_num){
//     int idx = at_end ? (results_num) * num_ss + subset_idx : subset_idx;
//     return idx;
// }
//
// int ResultArrays::index_parameters(const int subset_idx, const int results_num){
//     int idx = index(subset_idx, results_num) * num_params;
//     return idx;
// }


void write_to_disk_2d(ResultArrays &temporal,
                      const common_util::SaveConfig &saveconf,
                      const subset::Grid &ss_grid,
                      const std::string &filename){

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

    // set the img var to 0 after opening file if not saving at end
    //if (!saveconf.at_end) results_num = 0;

    // save in binary format
    if (saveconf.binary){
        outfile.open(outfile_str.str(), std::ios::binary);

        for (int i = 0; i < ss_grid.num; ++i) {

            // if the subset has not met threshold, set values to nan
            if (!saveconf.output_below_threshold && !temporal.above_thresh[i]) {
                temporal.u[i] = NAN;
                temporal.v[i] = NAN;
                for (int pp = 0; pp < temporal.num_params; pp++){
                    temporal.p[temporal.num_params*i+pp] = NAN;
                }
                temporal.cost[i] = NAN;
                temporal.ftol[i] = NAN;
                temporal.xtol[i] = NAN;
            }

            // displacement magnitude
            double mag = std::sqrt(temporal.u[i]*temporal.u[i]+temporal.v[i]*temporal.v[i]);

            // convert from corner to centre subset coords
            double ss_x = ss_grid.coords[2*i  ];
            double ss_y = ss_grid.coords[2*i+1];

            common_util::write_int(outfile, ss_x);
            common_util::write_int(outfile, ss_y);
            common_util::write_dbl(outfile, temporal.u[i]);
            common_util::write_dbl(outfile, temporal.v[i]);
            common_util::write_dbl(outfile, mag);
            common_util::write_uint8t(outfile, temporal.conv[i]);
            common_util::write_dbl(outfile, temporal.cost[i]);
            common_util::write_dbl(outfile, temporal.ftol[i]);
            common_util::write_dbl(outfile, temporal.xtol[i]);
            common_util::write_int(outfile, temporal.niter[i]);

            if (saveconf.shape_params) {
                for (int pp = 0; pp < temporal.num_params; pp++){
                    common_util::write_dbl(outfile, temporal.p[temporal.num_params*i+pp]);
                }
            }

        }

        outfile.close();
    }
    else {

        outfile.open(outfile_str.str());

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

        // column headers for shape parameters
        // if (saveconf.shape_params) {
        //     for (int p = 0; p < temporal.num_params; p++){
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
            if (!saveconf.output_below_threshold && !temporal.above_thresh[i]) {
                temporal.u[i] = NAN;
                temporal.v[i] = NAN;
                for (int pi = 0; pi < temporal.num_params; pi++){
                    temporal.p[temporal.num_params*i+pi] = NAN;
                }
                temporal.cost[i] = NAN;
                temporal.ftol[i] = NAN;
                temporal.xtol[i] = NAN;
            }

            // displacement magnitude
            double mag = std::sqrt(temporal.u[i]*temporal.u[i]+temporal.v[i]*temporal.v[i]);

            outfile << ss_x << delimiter;
            outfile << ss_y << delimiter;
            outfile << temporal.u[i] << delimiter;
            outfile << temporal.v[i] << delimiter;
            outfile << mag << delimiter;
            outfile << static_cast<int>(temporal.conv[i]) << delimiter;
            outfile << temporal.cost[i] << delimiter;
            outfile << temporal.ftol[i] << delimiter;
            outfile << temporal.xtol[i] << delimiter;
            outfile << temporal.niter[i];

            // write shape parameters if requested
            // if (saveconf.shape_params) {
            //     for (int pp = 0; pp < temporal.num_params; pp++){
            //         outfile << delimiter;
            //         outfile << temporal.p[temporal.num_params*i+pp];
            //     }
            // }

            // newline after each subset
            outfile << "\n";


        }
        outfile.close();
    }
}



void write_to_disk_stereo(ResultArrays &temporal,
                          ResultArrays &stereo,
                          const common_util::SaveConfig &saveconf,
                          const subset::Grid &ss_grid,
                          const std::string &filename){

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

    // set the img var to 0 after opening file if not saving at end
    //if (!saveconf.at_end) results_num = 0;

    // save in binary format
    if (saveconf.binary){
        outfile.open(outfile_str.str(), std::ios::binary);

        for (int i = 0; i < ss_grid.num; ++i) {

            // if the subset has not met threshold, set values to nan
            if ((!saveconf.output_below_threshold && !temporal.above_thresh[i]) ||
               (!saveconf.output_below_threshold && !stereo.above_thresh[i])) {

                temporal.u[i] = NAN;
                temporal.v[i] = NAN;
                stereo.u[i] = NAN;
                stereo.v[i] = NAN;

                for (int pp = 0; pp < temporal.num_params; pp++){
                    temporal.p[temporal.num_params*i+pp] = NAN;
                      stereo.p[  stereo.num_params*i+pp] = NAN;
                }

                temporal.cost[i] = NAN;
                temporal.ftol[i] = NAN;
                temporal.xtol[i] = NAN;
                stereo.cost[i] = NAN;
                stereo.ftol[i] = NAN;
                stereo.xtol[i] = NAN;
                stereo.x_world[i] = NAN;
                stereo.y_world[i] = NAN;
                stereo.z_world[i] = NAN;
                stereo.u_world[i] = NAN;
                stereo.v_world[i] = NAN;
                stereo.w_world[i] = NAN;
            }


            // convert from corner to centre subset coords
            double ss_x = ss_grid.coords[2*i  ];
            double ss_y = ss_grid.coords[2*i+1];

            // displacement magnitude
            double mag_temporal = std::sqrt(temporal.u[i]*temporal.u[i]+temporal.v[i]*temporal.v[i]);
            double mag_stereo = std::sqrt(stereo.u[i]*stereo.u[i]+stereo.v[i]*stereo.v[i]);

            common_util::write_int(outfile, ss_x);
            common_util::write_int(outfile, ss_y);
            common_util::write_dbl(outfile, temporal.u[i]);
            common_util::write_dbl(outfile, temporal.v[i]);
            common_util::write_dbl(outfile, mag_temporal);
            common_util::write_uint8t(outfile, temporal.conv[i]);
            common_util::write_dbl(outfile, temporal.cost[i]);
            common_util::write_dbl(outfile, temporal.ftol[i]);
            common_util::write_dbl(outfile, temporal.xtol[i]);
            common_util::write_int(outfile, temporal.niter[i]);
            common_util::write_dbl(outfile, stereo.u[i]);
            common_util::write_dbl(outfile, stereo.v[i]);
            common_util::write_dbl(outfile, mag_stereo);
            common_util::write_dbl(outfile, stereo.u_world[i]);
            common_util::write_dbl(outfile, stereo.v_world[i]);
            common_util::write_dbl(outfile, stereo.w_world[i]);
            common_util::write_dbl(outfile, stereo.x_world[i]);
            common_util::write_dbl(outfile, stereo.y_world[i]);
            common_util::write_dbl(outfile, stereo.z_world[i]);
            common_util::write_uint8t(outfile, stereo.conv[i]);
            common_util::write_dbl(outfile, stereo.cost[i]);
            common_util::write_dbl(outfile, stereo.ftol[i]);
            common_util::write_dbl(outfile, stereo.xtol[i]);
            common_util::write_int(outfile, stereo.niter[i]);

            // if (saveconf.shape_params) {
            //     for (int pp = 0; pp < temporal.num_params; pp++){
            //         common_util::write_dbl(outfile, temporal.p[temporal.num_params*i+pp]);
            //         common_util::write_dbl(outfile, stereo.p[stereo.num_params*i+pp]);
            //     }
            // }

        }

        outfile.close();
    }
    else {

        outfile.open(outfile_str.str());

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
        // column headers for shape parameters
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
                        // if the subset has not met threshold, set values to nan
            if ((!saveconf.output_below_threshold && !temporal.above_thresh[i]) ||
               (!saveconf.output_below_threshold && !stereo.above_thresh[i])) {

                temporal.u[i] = NAN;
                temporal.v[i] = NAN;
                stereo.u[i] = NAN;
                stereo.v[i] = NAN;

                for (int pp = 0; pp < temporal.num_params; pp++){
                    temporal.p[temporal.num_params*i+pp] = NAN;
                      stereo.p[  stereo.num_params*i+pp] = NAN;
                }

                temporal.cost[i] = NAN;
                temporal.ftol[i] = NAN;
                temporal.xtol[i] = NAN;
                stereo.cost[i] = NAN;
                stereo.ftol[i] = NAN;
                stereo.xtol[i] = NAN;
                stereo.x_world[i] = NAN;
                stereo.y_world[i] = NAN;
                stereo.z_world[i] = NAN;
                stereo.u_world[i] = NAN;
                stereo.v_world[i] = NAN;
                stereo.w_world[i] = NAN;
            }

            // displacement magnitude
            double mag_temporal = std::sqrt(temporal.u[i]*temporal.u[i]+temporal.v[i]*temporal.v[i]);
            double mag_stereo = std::sqrt(stereo.u[i]*stereo.u[i]+stereo.v[i]*stereo.v[i]);


            outfile << ss_x << delimiter;
            outfile << ss_y << delimiter;
            outfile << temporal.u[i] << delimiter;
            outfile << temporal.v[i] << delimiter;
            outfile << mag_temporal << delimiter;
            outfile << static_cast<int>(temporal.conv[i]) << delimiter;
            outfile << temporal.cost[i] << delimiter;
            outfile << temporal.ftol[i] << delimiter;
            outfile << temporal.xtol[i] << delimiter;
            outfile << temporal.niter[i] << delimiter;
            outfile << stereo.u[i] << delimiter;
            outfile << stereo.v[i] << delimiter;
            outfile << mag_stereo << delimiter;
            outfile << stereo.u_world[i] << delimiter;
            outfile << stereo.v_world[i] << delimiter;
            outfile << stereo.w_world[i] << delimiter;
            outfile << stereo.x_world[i] << delimiter;
            outfile << stereo.y_world[i] << delimiter;
            outfile << stereo.z_world[i] << delimiter;
            outfile << static_cast<int>(stereo.conv[i]) << delimiter;
            outfile << stereo.cost[i] << delimiter;
            outfile << stereo.ftol[i] << delimiter;
            outfile << stereo.xtol[i] << delimiter;
            outfile << stereo.niter[i];

            // write shape parameters if requested
            // if (saveconf.shape_params) {
            //     for (int pp = 0; pp < num_params; pp++){
            //         outfile << delimiter;
            //         outfile << p[num_params*i+pp];
            //     }
            // }

            // newline after each subset
            outfile << "\n";


        }
        outfile.close();
    }
}




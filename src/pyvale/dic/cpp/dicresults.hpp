// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICRESULTS_H
#define DICRESULTS_H

// STD library Header files
#include <vector>
#include <cstdint>

// common_cpp Header files
#include "../../common_cpp/util.hpp"

// DIC Header files
#include "./dicsubset.hpp"
#include "./dicoptimizer.hpp"


class ResultArrays {

    private:


    public:

        int num_ss;
        int num_params;
        bool stereo;

        // result arrays.
        std::vector<int> niter;
        std::vector<double> u; 
        std::vector<double> v;
        std::vector<double> p;
        std::vector<double> ftol;
        std::vector<double> xtol;
        std::vector<double> cost;
        std::vector<uint8_t> conv;
        std::vector<uint8_t> above_thresh;

        // incremental tracking;
        std::vector<double> u_last_good;
        std::vector<double> v_last_good;
        std::vector<double> du_dt;
        std::vector<double> dv_dt;
        std::vector<int> last_success_frame;
        std::vector<char> has_good_history;

        // world coordinates
        std::vector<double> x_world; 
        std::vector<double> y_world;
        std::vector<double> z_world;
        std::vector<double> u_world; 
        std::vector<double> v_world;
        std::vector<double> w_world;

        // constructors
        ResultArrays() = default;
        ResultArrays(int num_ss,
                     int num_params,
                     bool stereo);


        void append(OptResult &res, const int ss);


        void get_latest_matches(const ResultArrays &results_def, const int img_num_def);

        void write_to_disk_2d(const common_util::SaveConfig &saveconf,
                              const subset::Grid &ss_grid,
                              const std::string &filename);


        void write_to_disk_stereo(ResultArrays &stereo,
                                  const common_util::SaveConfig &saveconf,
                                  const subset::Grid &ss_grid,
                                  const std::string &filename);


};



#endif // DICRESULTS_H

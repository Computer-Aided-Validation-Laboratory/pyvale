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
        //int index(const int subset_idx, const int img_num);
        //int index_parameters(const int subset_idx, const int img_num);


};

void write_to_disk_2d(ResultArrays &temporal,
                      const common_util::SaveConfig &saveconf,
                      const subset::Grid &ss_grid,
                      const std::string &filename);


void write_to_disk_stereo(ResultArrays &temporal,
                          ResultArrays &stereo,
                          const common_util::SaveConfig &saveconf,
                          const subset::Grid &ss_grid,
                          const std::string &filename);

#endif // DICRESULTS_H

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

struct OptResult {
    std::vector<double> p;
    double u = 0.0;
    double v = 0.0;
    double mag = 0.0;
    double ftol = 0.0;
    double xtol = 0.0;
    int iter = 0;
    double cost = 0.0;
    uint8_t converged = false;
    uint8_t above_threshold = false;
    OptResult(size_t num_params) : p(num_params, 0.0) {}
};


class OptResultArrays {

    private:
        int num_ss;
        int num_params;
        bool at_end;

    public:
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

        OptResultArrays(int num_def_img, int num_ss, int num_params, bool conf_at_end);
        void append(OptResult &res, int img_num, int ss);
        int index(const int subset_idx, const int img_num);
        int index_parameters(const int subset_idx, const int img_num);
        void write_to_disk(int img, const common_util::SaveConfig &saveconf,
                           const subset::Grid &ss_grid, const int num_def_img,
                           const std::vector<std::string> &filenames);
};


#endif // DICRESULTS_H

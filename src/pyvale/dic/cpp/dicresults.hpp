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

// Program Header files
#include "./dicutil.hpp"

struct Result {
    std::vector<double> p;
    double u = 0.0;
    double v = 0.0;
    double mag = 0.0;
    double ftol = 0.0;
    double xtol = 0.0;
    int iter = 0;
    double cost = 0.0;
    uint8_t converged = false;
    Result(size_t num_params) : p(num_params, 0.0) {}
};


class ResultArrays {

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

        ResultArrays(int num_def_img, int num_ss, int num_params, bool conf_at_end);
        void append(Result &res, int img_num, int ss);
        int index(const int subset_idx, const int img_num);
        int index_parameters(const int subset_idx, const int img_num);
        void write_to_disk(int img, const util::SaveConfig &saveconf,
                           const subset::Grid &ss_grid, const int num_def_img,
                           const std::vector<std::string> &filenames);
};


#endif // DICRESULTS_H

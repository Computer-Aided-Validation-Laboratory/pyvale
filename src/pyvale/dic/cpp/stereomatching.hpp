// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef STEREOMATCHING_HPP
#define STEREOMATCHING_HPP

// STD library Header files
#include <vector>
#include <cmath>


// program header files
#include "../../calib/cpp/calibstereo.hpp"
#include "./dicsubset.hpp"
#include "./dicresults.hpp"
#include "./dicinterp.hpp"
#include "./dicutil.hpp"

// Eigen 
#include <Eigen/Dense>

namespace stereo {

    void matching(const double *img_l,
                const double *img_r,
                const Interpolator &interp_l,
                const Interpolator &interp_r,
                const subset::Grid &ss_grid,
                const util::Config &conf,
                const int img_num_l,
                const int img_num_r,
                const Eigen::Matrix3d &F,
                ResultArrays &result_arrays,
                ResultArrays &matches);

    void matching_strategy3(const double *img_l,
                            const double *img_r,
                            const Interpolator &interp_l,
                            const Interpolator &interp_r,
                            const subset::Grid &ss_grid,
                            const util::Config &conf,
                            const int img_num_l,
                            const int img_num_r,
                            const Eigen::Matrix3d &F,
                            ResultArrays &temporal,
                            ResultArrays &matches);

} // stereo namespace

#endif // STEREOMATCHING_HPP

// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICSINGLEWINDOW_RG_H
#define DICSINGLEWINDOW_RG_H

// STD library Header files
#include <optional>
#include <string>

// common_cpp headers

// Eigen Header files
#include <Eigen/Dense>

// Program Header files
#include "./dicinterp.hpp"
#include "./dicutil.hpp"
#include "./dicsubset.hpp"
#include "./dicresults.hpp"

/**
* @brief reliability guided scan method with incremental updating.
*/
void singlewindow_rg(const Interpolator &interp_ref,
                     const Interpolator &interp_def,
                     const subset::Grid &ss_grid,
                     const util::Config &conf,
                     const int img_num_ref,
                     const int img_num_def,
                     const ResultArrays &results_ref,
                     ResultArrays &results_def,
                     const std::string &mode="temporal",
                     const std::optional<Eigen::Matrix3d> &F=std::nullopt,
                     const std::optional<ResultArrays> &results_def_l=std::nullopt);


#endif //DICSINGLEWINDOW_RG_H

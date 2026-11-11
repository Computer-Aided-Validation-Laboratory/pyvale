// ================================================================================
// pyvale: the python validation engine
// License: MIT
// ================================================================================
#ifndef DICRESIDUAL_H
#define DICRESIDUAL_H

#include <string>
#include <vector>
#include <cstddef>

#include "./dicinterp.hpp"
#include "./dicresults.hpp"

/** Create a dense residual from contiguous double image and DIC arrays. */
void create_residual_map_from_arrays(std::vector<double> &residual_map,
                                     const double *reference,
                                     const double *deformed,
                                     int width,
                                     int height,
                                     const double *u,
                                     const double *v,
                                     const double *ss_x,
                                     const double *ss_y,
                                     std::size_t num_points,
                                     const bool *roi_mask,
                                     int subset_size = 0,
                                     util::EShapeFunc shape_func = util::EShapeFunc::RIGID,
                                     util::EResidualMode mode = util::EResidualMode::NONE);

/** Create a dense signed residual map on the reference image pixel grid. */
void create_residual_map(std::vector<double> &residual_map,
                         const Interpolator &reference,
                         const Interpolator &deformed,
                         const SubsetGrid &subset_grid,
                         const ResultArrays &results,
                         const bool *roi_mask,
                         int subset_size = 0,
                         util::EShapeFunc shape_func = util::EShapeFunc::RIGID,
                         util::EResidualMode mode = util::EResidualMode::NONE);

/** Create and save the residual map(s) for one DIC frame. */
void write_residual_maps(const Interpolator &reference_left,
                         const Interpolator &deformed_left,
                         const Interpolator *reference_right,
                         const Interpolator *deformed_right,
                         const SubsetGrid &subset_grid,
                         const ResultArrays &results_left,
                         const ResultArrays *results_right,
                         const ResultArrays *results_reference_right,
                         const bool *roi_mask,
                         const util::Config &conf,
                         const common_util::SaveConfig &saveconf,
                         const std::string &left_image_filename,
                         const std::string &right_image_filename = "");

/** Write a dense residual map as a headerless image-sized matrix. */
void save_residual_map(const std::vector<double> &residual_map,
                       const std::string &filename,
                       int width,
                       int height,
                       const std::string &delimiter,
                       int precision);

/** Construct the residual output path from the normal DIC output settings. */
std::string residual_filename(const std::string &basepath,
                              const std::string &prefix,
                              const std::string &image_filename,
                              int camera = -1);

#endif // DICRESIDUAL_H

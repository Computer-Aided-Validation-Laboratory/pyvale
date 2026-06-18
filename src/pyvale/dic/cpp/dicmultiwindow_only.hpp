#ifndef DICMULTIWINDOW_ONLY_H
#define DICMULTIWINDOW_ONLY_H

// STD library Header files
#include <vector>

// common_cpp headers

// Eigen Header files
#include <Eigen/Dense>

// Program Header files
#include "./dicinterp.hpp"
#include "./dicutil.hpp"
#include "./dicresults.hpp"
#include "./dicmultiwindow_util.hpp"

/**
 * @brief Multi Window Fast Fourier Transform (FFT) DIC method.
 * correlation is calculated for initial seed point and nearest neighbours.image
 * Scan proceeds along path with better matching subsets. 
 * A full indepth outline of the method can be found here:
 * https://opg.optica.org/ao/abstract.cfm?uri=ao-48-8-1535
 * 
 * @param img_ref pointer to reference image
 * @param img_def pointer to deformed image
 * @param ss_grid pointer to subset information
 * @param conf pointer to DIC config struct
 * @param img_num current image number
 */
void multiwindow_only(const Interpolator &interp_ref,
                      const Interpolator &interp_def,
                      std::vector<WindowLevel> &multiwindow,
                      const util::Config &conf,
                      const int img_num_ref,
                      const int img_num_def,
                      const ResultArrays &results_ref,
                      ResultArrays &results_def);


#endif // DICMULTIWINDOW_ONLY_H

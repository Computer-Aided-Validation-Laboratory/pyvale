// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICSCANMETHOD_H
#define DICSCANMETHOD_H


// STD library Header files

// Program Header files
#include "./dicutil.hpp"
#include "./dicresults.hpp"
#include "./dicmultiwindow.hpp"



namespace scanmethod {



/**
 * @brief straightforward image scan method. 
     * Loops over the subsets as a raster across the image.
     * initial subset locations are distrubuted evenly across the image
 * 
 * @param result_arrays where to populate the results
 * @param img_ref pointer to reference image
 * @param img_def pointer to deformed image
 * @param ss_grid pointer to subset information
 * @param conf pointer to DIC config struct
 * @param img_num current image number
 */
void image(const double *img_ref,
           const Interpolator &interp_def,
           const subset::Grid &ss_grid,
           const util::Config &conf,
           const int img_num_ref,
           const int img_num_def,
           ResultArrays &result_arrays);


/**
 * @brief reliability guided scan method. 
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
void multiwindow_reliability_guided(const double *img_ref,
                                   const double *img_def,
                                   const Interpolator &interp_def,
                                   std::vector<WindowLevel> &multiwindow,
                                   const util::Config &conf,
                                   const int img_num_ref,
                                   const int img_num_def,
                                   ResultArrays &result_arrays);

/**
 * @brief reliability guided scan method using a single grid FFTCC to estimate rigid displacements.
 */
void singlewindow_reliability_guided(const double *img_ref,
                                   const double *img_def,
                                   const Interpolator &interp_def,
                                   std::vector<WindowLevel> &multiwindow,
                                   const util::Config &conf,
                                   const int img_num_ref,
                                   const int img_num_def,
                                   ResultArrays &result_arrays);


/**
 * @brief reliability guided scan method with incremental updating.
 */
void singlewindow_incremental_reliability_guided(const double *img_ref,
                                               const double *img_def,
                                               const Interpolator &interp_ref,
                                               const Interpolator &interp_def,
                                               const subset::Grid &ss_grid,
                                               const util::Config &conf,
                                               const int img_num_ref,
                                               const int img_num_def,
                                               ResultArrays &result_arrays);




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
void multiwindow_only(const double *img_ref,
                      const double *img_def,
                      const Interpolator &interp_def,
                      std::vector<WindowLevel> &multiwindow,
                      const util::Config &conf,
                      const int img_num_ref,
                      const int img_num_def,
                      ResultArrays &result_arrays);

 void multiwindow_reliability_guided_r(const double *img_ref,
                                       const double *img_def,
                                       const Interpolator &interp_ref,
                                       const Interpolator &interp_def,
                                       std::vector<WindowLevel> &multiwindow,
                                       const util::Config &conf,
                                       const int img_num_ref,
                                       const int img_num_def,
                                       ResultArrays &stereo_results,
                                       ResultArrays &temporal_results);



}

#endif // DICSCANMETHOD_H

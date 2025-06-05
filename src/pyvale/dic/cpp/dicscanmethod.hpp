// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICSCANMETHOD_H
#define DICSCANMETHOD_H


// STD library Header files
#include <atomic>

// Program Header files
#include "./dicutil.hpp"



namespace scanmethod {



    void signalHandler(int signal);




/**
 * @brief straightforward image scan method. 
     * Loops over the subsets as a raster across the image.
     * initial subset locations are distrubuted evenly across the image
 * 
 * @param img_ref pointer to reference image
 * @param img_def pointer to deformed image
 * @param img_roi pointer to image roi
 * @param ssdata pointer to subset information
 * @param conf pointer to DIC config struct
 * @param img_num current image number
 */
void image(const double *img_ref, 
           const double *img_def, 
           const bool *img_roi,
           const std::vector<util::SubsetData> &ssdata, 
           const util::Config &conf,
           const int img_num);


/**
 * @brief Image scan with a brute force search method 
 * to handle large displacements or poor images. 
 * initial subset locations are distrubuted evenly across the image
 * and then a brute force search is performed to find the best match
 * for the first subset and any other poorly correlated subsets.
 * 
 * @param img_ref pointer to reference image
 * @param img_def pointer to deformed image
 * @param img_roi pointer to image roi
 * @param ssdata pointer to subset information
 * @param conf pointer to DIC config struct
 * @param img_num current image number
 */
void image_with_bf(const double *img_ref, 
                   const double *img_def, 
                   const bool *img_roi,
                   const std::vector<util::SubsetData> &ssdata, 
                   const util::Config &conf,
                   const int img_num);


/**
 * @brief reliability guided scan method. 
 * correlation is calculated for initial seed point and nearest neighbours.image
 * Scan proceeds along path with better matching subsets. 
 * A full indepth outline of the method can be found here:
 * https://opg.optica.org/ao/abstract.cfm?uri=ao-48-8-1535
 * 
 * @param img_ref pointer to reference image
 * @param img_def pointer to deformed image
 * @param img_roi pointer to image roi
 * @param ssdata pointer to subset information
 * @param conf pointer to DIC config struct
 * @param img_num current image number
 */
void reliability_guided(const double *img_ref, 
                        const double *img_def, 
                        const bool *img_roi,
                        const std::vector<util::SubsetData> &ssdata, 
                        const util::Config &conf,
                        const int img_num);


/**
 * @brief Multi Window Fast Fourier Transform (FFT) DIC method.
 * correlation is calculated for initial seed point and nearest neighbours.image
 * Scan proceeds along path with better matching subsets. 
 * A full indepth outline of the method can be found here:
 * https://opg.optica.org/ao/abstract.cfm?uri=ao-48-8-1535
 * 
 * @param img_ref pointer to reference image
 * @param img_def pointer to deformed image
 * @param img_roi pointer to image roi
 * @param ssdata pointer to subset information
 * @param conf pointer to DIC config struct
 * @param img_num current image number
 */
void multi_window_fourier(const double *img_ref, 
                          const double *img_def, 
                          const bool *img_roi,
                          const std::vector<util::SubsetData> &ssdata, 
                          const util::Config &conf,
                          const int img_num);


}

#endif // DICSCANMETHOD_H

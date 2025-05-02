   // ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICBRUTEFORCE_H
#define DICBRUTEFORCE_H

// STD library Header files
#include <vector>
#include <array>

// Program Header files
#include "./dicoptimizer.hpp"
#include "./dicutil.hpp"

namespace brute {

    struct Parameters {
        std::array<int, 2> p_rigid; // translation vector
        std::array<int, 2> p_rigid_prevmatch; // translation vector
        double threshold_bf;
        int range;
        
        Parameters(double threshold_bf_, int range_bf_)
            : p_rigid{0, 0},
              p_rigid_prevmatch{0, 0},
              threshold_bf(threshold_bf_),
              range(range_bf_) {}

    };
    
    /**
     * @brief Initializes the cost function and search method.
     * 
     * Sets cost function and search method based on the provided string values.
     * If an unrecognized value is provided, default 'SSD' and 'SPIRAL' values are used.
     * 
     * @param cost_function String reference representing the cost function ("SSD", "NSSD", "ZNSSD").
     * @param search_method String reference representing the search method ("EXHAUSTIVE", "SPIRAL").
     */
    void init(std::string &cost_function, std::string &search_method);


    /**
     * @brief Performs a spiral search for the optimal translation.
     * 
     * Searches for the best rigid subset translation by following a spiral path from a starting point.
     * The search stops when the cost function value is below a specified tolerance.
     * 
     * @param ss_x Horizontal coordinate of the starting point.
     * @param ss_y Vertical coordinate of the starting point.
     * @param image_ref Pointer to the reference image.
     * @param px_vertical Vertical size of the image.
     * @param px_horizontal Horizontal size of the image.
     * @param ss_def Pointer to the subset definition.
     * @param ss_ref Pointer to the reference subset.
     * @param brute Pointer to the brute force parameters.
     * @return result is populated in brute.p_rigid.
     */
    void expanding_wavefront(const int ss_x, 
                                  const int ss_y, 
                                  const double *image_ref, 
                                  const int px_vertical, 
                                  const int px_horizontal, 
                                  util::Subset *ss_def, 
                                  util::Subset *ss_ref, 
                                  brute::Parameters *brute);

    /**
     * @brief Performs an exhaustive search for the optimal translation.
     * 
     * Searches all possible rigid subset translations within a specified range to cost function minimum.
     * 
     * @param ss_x Horizontal coordinate of the starting point.
     * @param ss_y Vertical coordinate of the starting point.
     * @param image_ref Pointer to the reference image.
     * @param px_vertical Vertical size of the image.
     * @param px_horizontal Horizontal size of the image.
     * @param ss_def Pointer to the subset definition.
     * @param ss_ref Pointer to the reference subset.
     * @param brute Pointer to the brute force parameters.
     * @return result is populated in brute.p_rigid.
     */
    void exhaustive(const int ss_x, 
                                  const int ss_y, 
                                  const double *image_ref, 
                                  const int px_vertical, 
                                  const int px_horizontal, 
                                  util::Subset *ss_def, 
                                  util::Subset *ss_ref, 
                                  brute::Parameters *brute);

    /**
     * @brief 
     * 
     * @param ss_x 
     * @param ss_y 
     * @param image_ref 
     * @param px_vertical 
     * @param px_horizontal 
     * @param ss_def 
     * @param ss_ref 
     * @param brute 
     */
    void cross_correlation(const int ss_x, 
                        const int ss_y, 
                        const double *image_ref, 
                        const int px_vertical, 
                        const int px_horizontal, 
                        util::Subset *ss_def, 
                        util::Subset *ss_ref, 
                        brute::Parameters *brute);


                         
    /**
     * @brief Computes the Sum of Squared Differences (SSD) cost function.
     * 
     * This function calculates the SSD between the subset in the reference image and the target subset.
     * 
     * @param ss_x Horizontal coordinate of the starting point.
     * @param ss_y Vertical coordinate of the starting point.
     * @param image_ref Pointer to the reference image.
     * @param px_vertical Vertical size of the image.
     * @param px_horizontal Horizontal size of the image.
     * @param ss_def Pointer to the subset definition.
     * @param ss_ref Pointer to the reference subset.
     * @param p0 int value for the x-coordinate of the translation.
     * @param p1 int value for the y-coordinate of the translation.
     * @return The computed SSD value.
     */
    double ssd(const double *image_ref, 
               const int px_vertical, 
               const int px_horizontal, 
               util::Subset *ss_def, 
               util::Subset *ss_ref,
               const int p0,
               const int p1);

    /**
     * @brief Computes the Normalized Sum of Squared Differences (NSSD) cost function.
     * 
     * This function calculates the NSSD between the subset in the reference image and the target subset.
     * The cost is normalized by the sum of squared pixel values in the reference and target subsets.
     * 
     * @param ss_x Horizontal coordinate of the starting point.
     * @param ss_y Vertical coordinate of the starting point.
     * @param image_ref Pointer to the reference image.
     * @param px_vertical Vertical size of the image.
     * @param px_horizontal Horizontal size of the image.
     * @param ss_def Pointer to the subset definition.
     * @param ss_ref Pointer to the reference subset.
     * @param p0 int value for the x-coordinate of the translation.
     * @param p1 int value for the y-coordinate of the translation.
     * @return The computed NSSD value.
     */
    double nssd(const double *image_ref, 
                const int px_vertical, 
                const int px_horizontal, 
                util::Subset *ss_def,
                util::Subset *ss_ref,
                const int p0,
                const int p1);

    /**
     * @brief Computes the Zero-Mean Normalized Sum of Squared Differences (ZNSSD) cost function.
     * 
     * This function calculates the ZNSSD between the subset in the reference image and the target subset.
     * The pixel values are normalized by their mean and the cost is computed based on the squared differences.
     * 
     * @param ss_x Horizontal coordinate of the starting point.
     * @param ss_y Vertical coordinate of the starting point.
     * @param image_ref Pointer to the reference image.
     * @param px_vertical Vertical size of the image.
     * @param px_horizontal Horizontal size of the image.
     * @param ss_def Pointer to the subset definition.
     * @param ss_ref Pointer to the reference subset.
     * @param p0 int value for the x-coordinate of the translation.
     * @param p1 int value for the y-coordinate of the translation.
     * @return The computed ZNSSD value.
     */
    double znssd(const double *image_ref, 
                 const int px_vertical, 
                 const int px_horizontal, 
                 util::Subset *ss_def, 
                 util::Subset *ss_ref,
                 const int p0,
                 const int p1);
}

#endif //BRUTEFORCE

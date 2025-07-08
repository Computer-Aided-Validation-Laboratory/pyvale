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

    /**
     * @brief Parameters for the brute force search method.
     * 
     */
    struct Parameters {
        std::array<int, 2> p_rigid; // translation vector
        std::array<int, 2> p_rigid_prevmatch; // translation vector
        double bf_threshold;
        int range;

        Parameters(double bf_threshold_, int max_disp_)
            : p_rigid{0, 0},
              p_rigid_prevmatch{0, 0},
              bf_threshold(bf_threshold_),
              range(max_disp_) {}

    };
    
    /**
     * @brief Initializes the cost function and search method.
     * 
     * Sets cost function and search method based on the provided string values.
     * If an unrecognized value is provided, default 'SSD' and 'SPIRAL' values are used.
     * 
     * @param cost_function (std::string) reference representing the cost function ("SSD", "NSSD", "ZNSSD").
     * @param search_method (std::string) reference representing the search method ("EXHAUSTIVE", "SPIRAL").
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
     * @param img_ref Pointer to the reference image.
     * @param px_vert Vertical size of the image.
     * @param px_hori Horizontal size of the image.
     * @param ss_def Pointer to the subset definition.
     * @param ss_ref Pointer to the reference subset.
     * @param brute Pointer to the brute force parameters.
     * @return result is populated in brute.p_rigid.
     */
    void expanding_wavefront(const int ss_x, 
                                  const int ss_y, 
                                  const double *img_ref, 
                                  const int px_hori, 
                                  const int px_vert, 
                                  util::Subset &ss_def, 
                                  util::Subset &ss_ref, 
                                  brute::Parameters &brute);

    /**
     * @brief Performs an exhaustive search for the optimal translation.
     * 
     * Searches all possible rigid subset translations within a specified range to cost function minimum.
     * 
     * @param ss_x Horizontal coordinate of the starting point.
     * @param ss_y Vertical coordinate of the starting point.
     * @param img_ref Pointer to the reference image.
     * @param px_vert Vertical size of the image.
     * @param px_hori Horizontal size of the image.
     * @param ss_def Pointer to the subset definition.
     * @param ss_ref Pointer to the reference subset.
     * @param brute Pointer to the brute force parameters.
     * @return result is populated in brute.p_rigid.
     */
    void exhaustive(const int ss_x, 
                                  const int ss_y,
                                  const double *img_ref,
                                  const int px_hori,
                                  const int px_vert,
                                  util::Subset &ss_def,
                                  util::Subset &ss_ref,
                                  brute::Parameters &brute);

    // /**
    //  * @brief 
    //  * 
    //  * @param ss_x 
    //  * @param ss_y 
    //  * @param img_ref 
    //  * @param px_vert 
    //  * @param px_hori 
    //  * @param ss_def 
    //  * @param ss_ref 
    //  * @param brute 
    //  */
    // void cross_correlation(const int ss_x, 
    //                     const int ss_y, 
    //                     const double *img_ref, 
    //                     const int px_vert, 
    //                     const int px_hori, 
    //                     util::Subset *ss_def, 
    //                     util::Subset *ss_ref, 
    //                     brute::Parameters *brute);


                         
    /**
     * @brief Computes the Sum of Squared Differences (SSD) during a brute force search.
     * 
     * This function calculates the SSD between the subset in the reference image and the target subset.
     * 
     * @param ss_x Horizontal coordinate of the starting point.
     * @param ss_y Vertical coordinate of the starting point.
     * @param img_ref Pointer to the reference image.
     * @param px_vert Vertical size of the image.
     * @param px_hori Horizontal size of the image.
     * @param ss_def Pointer to the subset definition.
     * @param ss_ref Pointer to the reference subset.
     * @param p0 int value for the x-coordinate of the translation.
     * @param p1 int value for the y-coordinate of the translation.
     * @return The computed SSD value.
     */
    double ssd(const double *img_ref, 
               const int px_hori, 
               const int px_vert, 
               util::Subset &ss_def, 
               util::Subset &ss_ref,
               const int p0,
               const int p1);

    /**
     * @brief Computes the Normalized Sum of Squared Differences (NSSD) during a brute force search.
     * 
     * This function calculates the NSSD between the subset in the reference image and the target subset.
     * The cost is normalized by the sum of squared pixel values in the reference and target subsets.
     * 
     * @param ss_x Horizontal coordinate of the starting point.
     * @param ss_y Vertical coordinate of the starting point.
     * @param img_ref Pointer to the reference image.
     * @param px_vert Vertical size of the image.
     * @param px_hori Horizontal size of the image.
     * @param ss_def Pointer to the subset definition.
     * @param ss_ref Pointer to the reference subset.
     * @param p0 int value for the x-coordinate of the translation.
     * @param p1 int value for the y-coordinate of the translation.
     * @return The computed NSSD value.
     */
    double nssd(const double *img_ref, 
                const int px_hori, 
                const int px_vert, 
                util::Subset &ss_def,
                util::Subset &ss_ref,
                const int p0,
                const int p1);

    /**
     * @brief Computes the Zero-Mean Normalized Sum of Squared Differences (ZNSSD) during a brute force search.
     * 
     * This function calculates the ZNSSD between the subset in the reference image and the target subset.
     * The pixel values are normalized by their mean and the cost is computed based on the squared differences.
     * 
     * @param ss_x Horizontal coordinate of the starting point.
     * @param ss_y Vertical coordinate of the starting point.
     * @param img_ref Pointer to the reference image.
     * @param px_vert Vertical size of the image.
     * @param px_hori Horizontal size of the image.
     * @param ss_def Pointer to the subset definition.
     * @param ss_ref Pointer to the reference subset.
     * @param p0 int value for the x-coordinate of the translation.
     * @param p1 int value for the y-coordinate of the translation.
     * @return The computed ZNSSD value.
     */
    double znssd(const double *img_ref, 
                 const int px_hori, 
                 const int px_vert, 
                 util::Subset &ss_def, 
                 util::Subset &ss_ref,
                 const int p0,
                 const int p1);


    /**
     * @brief Checks if a point (dx, dy) lies on the perimeter of a square of radius r.
     *
     * A point is considered on the perimeter if its horizontal or vertical distance
     * from the center equals the radius.
     *
     * @param dx Horizontal offset from subset centre location.
     * @param dy Vertical offset from subset centre location.
     * @param r Radius of the square perimeter.
     *
     * @return (bool) True if the point lies on the perimeter, false otherwise.
     */
    inline bool is_perimeter_point(int dx, int dy, int r);
    
    
    /**
     * @brief Checks if a rectangular region is fully contained within image bounds.
     *
     * Evaluates whether the rectangle defined by its corner coordinates lies
     * entirely within the image dimensions.
     *
     * @param xmin (`int`) Minimum x-coordinate of the rectangle.
     * @param ymin (`int`) Minimum y-coordinate of the rectangle.
     * @param xmax (`int`) Maximum x-coordinate of the rectangle.
     * @param ymax (`int`) Maximum y-coordinate of the rectangle.
     * @param width (int) Width of the image.
     * @param height (int) Height of the image.
     *
     * @return (bool) True if the rectangle is fully inside the image, false otherwise.
     */
    inline bool is_within_image(int xmin, int ymin, int xmax, int ymax,
                                int width, int height);
    
    
    /**
     * @brief Checks if two values lie within the specified symmetric integer range.
     *
     * The range is interpreted as [-range, range). This function checks whether
     * both values are within that range.
     *
     * @param p0 (int) first rigid shape function parameter (x).
     * @param p1 (int) Second rigid shape function parameter (y).
     * @param range (int) Half-width of the symmetric range (exclusive).
     *
     * @return (bool) True if both values are within the range, false otherwise.
     */
    inline bool is_within_range(int p0, int p1, int range);


}

#endif //BRUTEFORCE

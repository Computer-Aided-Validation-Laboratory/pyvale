

#ifndef DICSCANMETHOD_H
#define DICSCANMETHOD_H


// STD library Header files

// Program Header files
#include "./dicutil.hpp"



namespace scanmethod {

/**
     * @brief 
     * 
     * @param image_ref 
     * @param image_def 
     * @param ss_coord_list 
     * @param num_def_images 
     * @param img_num 
     * @param max_iter 
     * @param precision 
     * @param threshold_lm 
     * @param threshold_bf 
     * @param range_bf 
     */
void image(double *image_ref, 
                double *image_def, 
                bool *image_roi,
                util::SubsetData &ssdata, 
                util::Config &conf,
                int img_num);


/**
     * @brief 
     * 
     * @param image_ref 
     * @param image_def 
     * @param ss_coord_list 
     * @param num_def_images 
     * @param img_num 
     * @param max_iter 
     * @param precision 
     * @param threshold_lm 
     * @param threshold_bf 
     * @param range_bf 
     */
void image_with_bf(double *image_ref, 
                        double *image_def, 
                        bool *image_roi,
                        util::SubsetData &ssdata, 
                        util::Config &conf,
                        int img_num);


/**
     * @brief 
     * 
     * @param image_ref 
     * @param image_def 
     * @param ss_coord_list 
     * @param num_def_images 
     * @param img_num 
     * @param max_iter 
     * @param precision 
     * @param threshold_lm 
     * @param threshold_bf 
     * @param range_bf 
     */
void reliability_guided(double *image_ref, 
                        double *image_def, 
                        bool *image_roi,
                        util::SubsetData &ssdata, 
                        util::Config &conf,
                        int img_num);

}

#endif // DICSCANMETHOD_H

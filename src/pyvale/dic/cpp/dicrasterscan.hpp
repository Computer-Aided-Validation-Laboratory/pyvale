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
void raster(const Image &img_ref,
           const Interpolator &interp_def,
           const subset::Grid &ss_grid,
           const util::Config &conf,
           const int img_num_ref,
           const int img_num_def,
           ResultArrays &result_arrays);
// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICMULTIWINDOW_UTIL_H
#define DICMULTIWINDOW_UTIL_H

// STD library Header files
#include <csignal>
#include <cstdlib>
#include <cmath>
#include <array>
#include <cstdint>

// common header files 
#include <Eigen/Dense>

// DIC Header files
#include "./dicinterp.hpp"
#include "./dicsubset.hpp"
#include "./dicutil.hpp"




/** 
 * @brief Configuration struct for multiwindow correlation parameters.
 *
 * Members:
 *   - @c overlap : percentage overlap between adjacent subsets at each level (e.g., 50 for 50% overlap).
 *   - @c subset_size : vector of subset sizes for each level, ordered from coarsest (largest window) to finest (smallest window).
 *   - @c search_area : vector of search area sizes (in pixels) for FFT
 *   correlation at each level, ordered from coarsest to finest.
 *   - FFT displacement filtering and intermediate FFT output settings.
 */
struct MultiwindowConfig
{
    std::vector<int> overlap;
    std::vector<int> subset_size;
    std::vector<int> search_area;
    bool fft_filter = true;
    double fft_filter_threshold = 3.0;
    int fft_filter_radius = 3;
    double fft_filter_corr_power = 2.0;
    bool fft_save = false;
};

/**
 * @brief Represents one level in a multiwindow FFT-based correlation hierarchy.
 *
 * A WindowLevel defines:
 *   - the subset layout (grid) at a particular window size,
 *   - the subset grid at a particular window size,
 *   - displacement and correlation data,
 *   - seed relationships to the previous (coarser) level.
 *
 * Members:
 *   - @c u, @c v, @c cost : displacement and correlation data for this level.
 *   - @c peak_amplitude : FFT correlation peak amplitude for each subset.
 *   - @c seed_idx     : up to four indices into the previous level.
 *   - @c seed_count   : number of valid entries in @c seed_idx for each subset.
 * @see WindowLevel::calc_rigid_displacements()
 */
struct WindowLevel {

    SubsetGrid grid;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> cost;
    std::vector<double> peak_amplitude;
    common_util::SaveConfig saveconf;

    // Seed mapping to previous level (up to 4 nearest neighbours)
    std::vector<std::array<int, 4>> seed_idx;
    std::vector<uint8_t> seed_count;

    WindowLevel() = default;
    /**
    * @brief Generate nearest-neighbour mappings from a previous level's grid.
    *
    * For each subset in the current level's layout, finds up to @c max_num_neigh
    * nearest valid subsets from @p layout_prev within a fixed search window (roughly 10x10 cells
    * in the previous level's grid index space). Stores the neighbour count in
    * @c num_neigh_list[ss] and neighbour indices in the flattened @c neigh_list buffer
    * at positions [ss*max_num_neigh + k].
    *
    * This method uses OpenMP for parallelization over current subsets.
    *
    * @param[in] layout_prev Previous level grid layout used as the source of neighbours.
    *
    * @pre  @c layout and @c layout_prev must be initialized.
    * @pre  @c max_num_neigh > 0.
    * @post @c num_neigh_list is resized to @c layout.num and filled.
    * @post @c neigh_list is resized to @c max_num_neigh * layout.num and filled for found neighbours.
    *
    * @note If no neighbours are found for a subset (rare with a sensible ROI and step),
    *       the function prints diagnostics to stderr and terminates the process with EXIT_FAILURE.
    * @note Uses @c std::nth_element to select the closest K neighbours (by squared distance).
    * @note Neighbour validity is determined via @c layout_prev.mask (value -1 indicates invalid).
    *
    * @par Threading
    * Uses `#pragma omp parallel for` over subsets; writes to disjoint regions of
    * `num_neigh_list` and `neigh_list`, making it thread-safe for the given buffers.
    */
    void build_seed_map(const SubsetGrid &grid_prev);



    /**
    * @brief Compute rigid (translation-only) displacements at this level using FFT-CC.
    *
    * For every valid subset in the current level's layout, this method:
    *   1. Seeds an initial guess from the previous level via weighted neighbours (if available).
    *   2. Extracts the reference subset from @p img_ref.
    *   3. Samples the deformed subset from @p interp_def at subpixel-shifted coordinates.
    *   4. Zero-normalizes both subsets (ZNSSD).
    *   5. Performs FFT-based cross-correlation and 2D Gaussian subpixel peak estimation.
    *   6. Stores @c u, @c v (accumulated with the seed) and the peak amplitude (@c max_val).
    *
    * OpenMP is used to parallelize over subsets. Each thread owns a local @c FFT instance.
    * Optionally applies a MAD-based outlier removal at the end if the level is configured to do so.
    *
    * @param[in] prev         Previous level (for seeding); may be ignored if this is the first level.
    * @param[in] img_ref      Pointer to reference image pixel buffer.
    * @param[in] img_def      Pointer to deformed image pixel buffer (not directly sampled if using @p interp_def).
    * @param[in] interp_def   Interpolator providing subpixel access to the deformed image.
    * @param[in] img_num_ref  Index of the reference image in @p filenames (for diagnostics/progress text).
    * @param[in] img_num_def  Index of the deformed image in @p filenames (for diagnostics/progress text).
    * @param[in] filenames    Filenames vector used for progress-bar labeling.
    *
    * @pre  @c layout, @c u, @c v, and @c max_val must be sized to @c layout.num.
    * @pre  @p interp_def must be valid and thread-safe for concurrent sampling (or internally synchronized).
    * @post @c u and @c v contain the estimated translations per subset for this level; @c max_val stores peak amplitudes.
    *
    * @note Currently forces subpixel refinement to true (TODO flag in code).
    * @note Uses Gaussian 2D peak estimation in frequency-domain correlation surface.
    * @note Progress reporting depends on @c g_debug_level and may incur minor overhead.
    *
    * @par Threading
    * Uses `#pragma omp parallel` with a private @c FFT instance per thread and a dynamic schedule.
    * Writes to disjoint indices of @c u, @c v, @c max_val, making it thread-safe.
    */
    void calc_rigid_displacements(const WindowLevel &prev,
                                  const Interpolator &interp_ref,
                                  const Interpolator &interp_def,
                                  const int img_num_ref,
                                  const int img_num_def,
                                  const int window_level,
                                  const int num_levels,
                                  const std::vector<std::string> &filenames,
                                  const util::EFFTPrecision fft_precision,
                                  const MultiwindowConfig &config);



    /**
    * @brief Seed current-level displacement from previous level using inverse-distance weighting.
    *
    * Computes a weighted average of the @p prev level's displacements for the
    * nearest neighbours associated with the current subset @p ss. The weight is
    * defined as w = 1 / (dist^2 + epsilon), where dist is the Euclidean distance
    * between current subset center (@p ss_x, @p ss_y) and the neighbour's center.
    *
    * @param[out] prev_x  Output seed for x-displacement at the current subset.
    * @param[out] prev_y  Output seed for y-displacement at the current subset.
    * @param[in]  prev    Previous window level providing neighbour mappings and displacements.
    * @param[in]  ss      Index of current subset (into this level's layout).
    * @param[in]  ss_x    X-coordinate (pixels) of the current subset center or top-left (consistent with layout).
    * @param[in]  ss_y    Y-coordinate (pixels) of the current subset center or top-left (consistent with layout).
    *
    * @pre  Neighbour lists for the current level were generated against @p prev (via gen_neighlist).
    * @pre  For @p ss, @c prev.num_neigh_list[ss] > 0 and the indices in @c prev.neigh_list are valid.
    * @post @p prev_x and @p prev_y contain the weighted average seed displacement for the current subset.
    *
    * @note Uses a small epsilon (10.0) to avoid singularity and to bound weights.
    * @note Assumes @c prev.u and @c prev.v contain already computed displacements at the previous level.
    */
    void get_displacement_from_prev_window(double &prev_x,
                                           double &prev_y,
                                           const WindowLevel &prev,
                                           const int ss,
                                           const double ss_x,
                                           const double ss_y);



    void remove_outliers_vector(
        std::vector<double>& u,
        std::vector<double>& v,
        const std::vector<double>& max_val,
        double threshold = 3.0,
        int radius = 3,
        double corr_power = 2.0,
        double eps = 1e-6);

};

/**
 * @brief Owns and coordinates the levels of a multiwindow correlation pyramid.
 */
class MultiwindowPyramid {
public:
    std::vector<WindowLevel> levels;
    MultiwindowConfig config;

    WindowLevel &operator[](size_t index) { return levels[index]; }
    const WindowLevel &operator[](size_t index) const { return levels[index]; }
    WindowLevel &back() { return levels.back(); }
    const WindowLevel &back() const { return levels.back(); }
    size_t size() const { return levels.size(); }

    /**
     * @brief Builds a coarse-to-fine multiwindow pyramid.
     */
    void build(const bool *img_roi,
               const util::Config &conf,
               const MultiwindowConfig &mwconf,
               const common_util::SaveConfig &saveconf,
               size_t num_levels = 0);

    /**
     * @brief Rebuilds the pyramid after the reference image and ROI move.
     *
     * The finest level is replaced by @p final_grid, translated by the supplied
     * reference displacement. Coarser levels are rebuilt from @p img_roi.
     */
    void rebuild_for_new_reference(const bool *img_roi,
                                   const SubsetGrid &final_grid,
                                   const std::vector<double> &reference_u,
                                   const std::vector<double> &reference_v,
                                   const util::Config &conf,
                                   const MultiwindowConfig &mwconf,
                                   const common_util::SaveConfig &saveconf,
                                   size_t num_levels);

    /**
     * @brief Calculates rigid FFT displacements from coarse to fine.
     */
    void calc_rigid_displacements(const Interpolator &interp_ref,
                                  const Interpolator &interp_def,
                                  int img_num_ref,
                                  int img_num_def,
                                  const std::vector<std::string> &filenames,
                                  util::EFFTPrecision fft_precision);
};



/**
 * @brief Initialize a multi-resolution FFT-CC pyramid (window levels).
 *
 * Builds the sequence of FFT window levels from the configuration by creating
 * progressively smaller subset sizes (powers of two) down to the final
 * configured subset size, and associates each level with the previous one for
 * seeding/coarse-to-fine propagation.
 *
 * The constructed levels are appended to @p level using `emplace_back(...)` with:
 *   - mask/ROI pointer (@p img_roi),
 *   - step size for this level (half of size for power-of-two levels, or conf.ss_step for the last),
 *   - subset size for this level,
 *   - pixel pitch (conf.px_hori, conf.px_vert),
 *   - a boolean indicating whether there is a finer (next) level,
 *   - the level index,
 *   - a pointer to the previous level's layout (or nullptr for the first level).
 *
 * @param[out] level   Output container to which new levels are appended.
 * @param[in]  img_roi Pointer to the image ROI (mask) used by the layouts. Must remain valid during level construction.
 * @param[in]  conf    Runtime configuration containing max_disp, ss_size, ss_step, px_hori, px_vert.
 * @param[in]  saveconf Configuration for saving intermediate FFT results .
 */
#endif // DICMULTIWINDOW_UTIL_H

// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICMULTIWINDOW_H
#define DICMULTIWINDOW_H

// STD library Header files
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <cmath>

// common header files 
#include <Eigen/Dense>

// DIC Header files
#include "./dicinterp.hpp"
#include "./dicsubset.hpp"
#include "./dicutil.hpp"




/**
 * @brief Represents one level in a multiwindow FFT-based correlation hierarchy.
 *
 * A WindowLevel defines:
 *   - the subset layout (grid) at a particular window size,
 *   - allocated storage for displacement fields (@c u, @c v),
 *   - peak metrics (@c max_val, @c cost),
 *   - neighbour relationships to the previous (coarser) level for seeding.
 *
 * Members:
 *   - @c u, @c v : displacement fields for each subset at this level.
 *   - @c cost    : user-defined cost value per subset (e.g., ZNSSD or debug metric).
 *   - @c max_val : peak height from FFT cross-correlation for this subset.
 *   - @c neigh_list      : flattened list of neighbouring indices (previous level).
 *   - @c num_neigh_list  : number of neighbours used for each subset.
 *   - @c max_num_neigh   : maximum neighbours assigned per subset (default: 4).
 *   - @c level           : multi-window level index (0 = coarsest / largest window).
 *   - @c layout          : subset grid definition (coords, mask, sizes, steps).
 *
 * @note The constructor builds the grid and optionally generates neighbour lists
 *       if a pointer to @p prev_layout is provided.
 *
 * @warning The pointer @p img_roi must remain valid during grid construction.
 *
 * @see WindowLevel::gen_neighlist()
 * @see WindowLevel::calc_rigid_displacements()
 */
struct WindowLevel {

    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> cost;
    std::vector<double> max_val;
    std::vector<int> neigh_list;
    std::vector<int> num_neigh_list;
    size_t max_num_neigh = 4;
    size_t level; // 0 is largest window
    subset::Grid layout;
    bool mad_filter;
    double mad_scale;
    bool fft_save;
    common_util::SaveConfig saveconf;




    WindowLevel(const bool *img_roi,
           const int step,
           const int size,
           const int px_hori,
           const int px_vert,
           const bool allow_outside,
           const size_t level,
           const bool mad_filter,
           const double mad_scale,
           const bool fft_save,
           const common_util::SaveConfig &saveconf,
           const subset::Grid *prev_layout) {

        // create grid for the window
        layout = subset::create_grid(img_roi, step, size, size, px_hori, px_vert, allow_outside);
        u.resize(layout.num);
        v.resize(layout.num);
        cost.resize(layout.num);
        max_val.resize(layout.num);

        this->level = level;
        this->mad_filter = mad_filter;
        this->mad_scale = mad_scale;
        this->fft_save = fft_save;
        this->saveconf = saveconf;

        // create neighbourlist if there's a a previous window size
        if (prev_layout) {
            gen_neighlist(*prev_layout);
        }
    }


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
    void gen_neighlist(const subset::Grid &layout_prev);

    /**
    * @brief Remove local outliers from a displacement component using MAD filtering.
    *
    * For each subset, computes the median and MAD (median absolute deviation) of
    * valid neighbouring values (in a square radius = 2 window in index space).
    * If the current value deviates from the local median by more than
    * @p mad_scale * MAD, it is replaced by the median. Operates on a copy and writes
    * back upon completion to avoid biasing neighbourhood statistics.
    *
    * @param[in,out] u         Displacement component (e.g., u or v) to be filtered in-place.
    * @param[in]     mad_scale Threshold multiplier for MAD-based rejection (larger -> less aggressive).
    *
    * @pre  @c layout must be initialized and consistent with the size of @p u.
    * @post @p u is modified in-place with suspected outliers replaced by the local median.
    *
    * @note Neighbourhood is defined in the grid index space via @c layout.mask with radius = 2.
    * @note If < 4 valid neighbours are found for a subset, that subset is skipped.
    * @note If MAD is extremely small (< 1e-12), the subset is skipped to avoid division blow-up.
    * @complexity Approximately O(N * K log K) where N = number of subsets and K ~ neighbourhood size.
    */
    void remove_outliers(std::vector<double> &u,
                         const double mad_scale);



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
                                  const Image &img_ref,
                                  const Image &img_def,
                                  const Interpolator &interp_def,
                                  const int img_num_ref,
                                  const int img_num_def,
                                  const std::vector<std::string> &filenames);



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
void multiwindow_init(std::vector<WindowLevel> &level, 
                      const bool *img_roi, 
                      const util::Config &conf,
                      const common_util::SaveConfig &saveconf);

#endif // DICMULTIWINDOW_H

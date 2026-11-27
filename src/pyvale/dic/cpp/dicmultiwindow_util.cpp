// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include "dicutil.hpp"
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <csignal>
#include <cstdio>
#include <stdexcept>

// Common Header files
#include "../../commoncpp/progressbar.hpp"
#include "../../commoncpp/defines.hpp"
#include "../../commoncpp/dicsignalhandler.hpp"

// DIC Header files
#include "./dicmultiwindow_util.hpp"
#include "./dicfourier.hpp"
#include "./dicsubset.hpp"
#include "./dicinterp.hpp"


namespace {

WindowLevel make_window_level(const bool *img_roi,
                              const util::Config &conf,
                              const MultiwindowConfig &mwconf,
                              const common_util::SaveConfig &saveconf,
                              size_t level,
                              bool is_final_level) {
    WindowLevel window;
    window.grid = create_grid(img_roi,
                              mwconf.overlap[level],
                              mwconf.subset_size[level],
                              mwconf.subset_size[level],
                              conf.px_hori,
                              conf.px_vert,
                              is_final_level ? conf.partial_subset
                                             : conf.partial_subset_multiwindow);
    window.u.resize(window.grid.num);
    window.v.resize(window.grid.num);
    window.cost.resize(window.grid.num);
    window.peak_amplitude.resize(window.grid.num);
    window.saveconf = saveconf;
    return window;
}

void build_levels(std::vector<WindowLevel> &levels,
                  const bool *img_roi,
                  const util::Config &conf,
                  const MultiwindowConfig &mwconf,
                  const common_util::SaveConfig &saveconf,
                  size_t num_levels) {
    levels.clear();
    levels.reserve(num_levels);

    for (size_t level = 0; level < num_levels; ++level) {
        levels.push_back(make_window_level(img_roi, conf, mwconf, saveconf,
                                           level, level + 1 == mwconf.overlap.size()));
    }

    for (size_t level = 1; level < levels.size(); ++level) {
        levels[level].build_seed_map(levels[level - 1].grid);
    }
}

} // namespace

void MultiwindowPyramid::build(const bool *img_roi,
                               const util::Config &conf,
                               const MultiwindowConfig &mwconf,
                               const common_util::SaveConfig &saveconf,
                               const size_t requested_levels) {
    if (mwconf.overlap.empty()) {
        throw std::invalid_argument("Multiwindow configuration must contain at least one level");
    }
    if (mwconf.subset_size.size() != mwconf.overlap.size() ||
        mwconf.search_area.size() != mwconf.overlap.size()) {
        throw std::invalid_argument(
            "Multiwindow configuration length mismatch: overlap=" +
            std::to_string(mwconf.overlap.size()) +
            ", subset_size=" +
            std::to_string(mwconf.subset_size.size()) +
            ", search_area=" +
            std::to_string(mwconf.search_area.size()));
    }

    const size_t num_levels = requested_levels == 0
        ? mwconf.overlap.size()
        : requested_levels;
    if (num_levels == 0 || num_levels > mwconf.overlap.size()) {
        throw std::invalid_argument("Requested multiwindow level count is out of range");
    }

    config = mwconf;
    build_levels(levels, img_roi, conf, config, saveconf, num_levels);
}

void MultiwindowPyramid::rebuild_for_new_reference(
    const bool *img_roi,
    const SubsetGrid &final_grid,
    const std::vector<double> &reference_u,
    const std::vector<double> &reference_v,
    const util::Config &conf,
    const MultiwindowConfig &mwconf,
    const common_util::SaveConfig &saveconf,
    const size_t num_levels) {

    if (num_levels == 0 || num_levels > mwconf.overlap.size()) {
        throw std::invalid_argument("Requested multiwindow level count is out of range");
    }
    if (final_grid.num != static_cast<int>(reference_u.size()) ||
        final_grid.num != static_cast<int>(reference_v.size())) {
        throw std::invalid_argument("Final grid and reference displacement sizes do not match");
    }

    config = mwconf;
    build_levels(levels, img_roi, conf, config, saveconf, num_levels);

    WindowLevel final_level = make_window_level(img_roi, conf, mwconf, saveconf,
                                                mwconf.overlap.size() - 1, true);
    final_level.grid = final_grid;
    for (int subset = 0; subset < final_level.grid.num; ++subset) {
        if (final_level.grid.active_ss[subset]) {
            final_level.grid.coords[2 * subset] += reference_u[subset];
            final_level.grid.coords[2 * subset + 1] += reference_v[subset];
        }
    }
    final_level.u.assign(final_level.grid.num, 0.0);
    final_level.v.assign(final_level.grid.num, 0.0);
    final_level.cost.assign(final_level.grid.num, 0.0);
    final_level.peak_amplitude.assign(final_level.grid.num, 0.0);

    levels.push_back(std::move(final_level));
    if (levels.size() > 1) {
        levels.back().build_seed_map(levels[levels.size() - 2].grid);
    }
}

void MultiwindowPyramid::calc_rigid_displacements(
    const Interpolator &interp_ref,
    const Interpolator &interp_def,
    const int img_num_ref,
    const int img_num_def,
    const std::vector<std::string> &filenames,
    const util::EFFTPrecision fft_precision) {

    for (size_t level = 0; level < levels.size(); ++level) {
        const WindowLevel &previous = levels[level == 0 ? 0 : level - 1];
        levels[level].calc_rigid_displacements(previous, interp_ref, interp_def,
                                                img_num_ref, img_num_def,
                                                static_cast<int>(level),
                                                static_cast<int>(levels.size()),
                                                filenames, fft_precision, config);
        raise_on_interrupt();
    }
}

void WindowLevel::build_seed_map(const SubsetGrid &layout_prev) {
    constexpr size_t max_seed_count = 4;
    const int prev_step = layout_prev.step;

    seed_count.assign(grid.num, 0);
    seed_idx.assign(grid.num, std::array<int, 4>{-1, -1, -1, -1});

    // Find up to four closest active subsets in the previous level.
    #pragma omp parallel for
    for (int ss = 0; ss < grid.num; ++ss) {
        if (!grid.active_ss[ss]) continue;

        const double cx = grid.coords[2*ss];
        const double cy = grid.coords[2*ss+1];
        std::vector<std::pair<double, int>> candidate_seeds;

        const int idx_x = static_cast<int>(std::floor(cx / prev_step));
        const int idx_y = static_cast<int>(std::floor(cy / prev_step));
        const int min_x = std::max(0, idx_x - 5);
        const int min_y = std::max(0, idx_y - 5);
        const int max_x = std::min(layout_prev.num_ss_x, idx_x + 6);
        const int max_y = std::min(layout_prev.num_ss_y, idx_y + 6);

        for (int y = min_y; y < max_y; ++y) {
            for (int x = min_x; x < max_x; ++x) {
                int nss_idx = layout_prev.mask[y*layout_prev.num_ss_x+x];
                if (nss_idx == -1) continue;

                const double nss_x = layout_prev.coords[2*nss_idx];
                const double nss_y = layout_prev.coords[2*nss_idx+1];

                double dx = (nss_x) - cx;
                double dy = (nss_y) - cy;
                double dist_sq = dx*dx + dy*dy;

                candidate_seeds.emplace_back(dist_sq, nss_idx);
            }
        }

        const size_t num_seeds = std::min(max_seed_count, candidate_seeds.size());
        if (num_seeds == 0) continue;

        if (candidate_seeds.size() > num_seeds) {
            std::nth_element(candidate_seeds.begin(),
                             candidate_seeds.begin() + num_seeds,
                             candidate_seeds.end());
            candidate_seeds.resize(num_seeds);
        }

        seed_count[ss] = static_cast<uint8_t>(num_seeds);
        for (size_t i = 0; i < num_seeds; ++i) {
            seed_idx[ss][i] = candidate_seeds[i].second;
        }
    }
}

void WindowLevel::calc_rigid_displacements(const WindowLevel &prev,
                                           const Interpolator &interp_ref,
                                           const Interpolator &interp_def,
                                           const int img_num_ref,
                                           const int img_num_def,
                                           const int window_level,
                                           const int num_levels,
                                           const std::vector<std::string> &filenames,
                                           const util::EFFTPrecision fft_precision,
                                           const MultiwindowConfig &config){

        const int px_hori = interp_def.px_hori;
        const int px_vert = interp_def.px_vert;
        const int template_size_x = grid.size_x;
        const int template_size_y = grid.size_y;
        const int search_area = config.search_area.at(static_cast<size_t>(window_level));

        // TODO: Add a proper flag for this 
        bool subpx = true;

        // consts
        const int num_ss = grid.num;

        if (grid.num == 0 || grid.active_total == 0) {
            return;
        }

        // set all displacements for multiwindow level to 0
        std::fill(u.begin(), u.end(), 0.0);
        std::fill(v.begin(), v.end(), 0.0);

        // progress bar initialisation
        std::string bar_title = "FFT " + std::to_string(search_area) + "x" + std::to_string(search_area) + " \033[1;4m" + filenames[img_num_ref] + "\033[0m -> \033[1;4m" + filenames[img_num_def] + "\033[0m:";
        ProgressBar pbar(bar_title, grid.active_total);
        std::atomic<int> current_progress = 0;


        auto run_fft_loop = [&](auto &fft) {
            #pragma omp for
            for (int ss = 0; ss < grid.num; ss++){

                // exit when ctrl+C
                if (stop_request) continue;

                if (!grid.active_ss[ss]) continue;

                const double cx = grid.coords[2*ss];
                const double cy = grid.coords[2*ss+1];

                // get the seed for the new window size
                double prev_u = 0.0;
                double prev_v = 0.0;

                if (window_level > 0)
                    get_displacement_from_prev_window(prev_u, prev_v, prev, ss, cx, cy);

                std::vector<double> p(6,0.0);
                double maxv_local = 0.0;
                get_single_window_fftcc_peak_centre(fft, p, maxv_local,
                                                cx, cy,
                                                prev_u, prev_v,
                                                template_size_x, template_size_y,
                                                search_area, search_area,
                                                interp_ref, interp_def, false);

                peak_amplitude[ss] = maxv_local;

                u[ss] = prev_u+p[0];
                v[ss] = prev_v+p[1];

                if (g_debug_level>=2){
                    int progress = current_progress.fetch_add(1);
                    if (omp_get_thread_num()==0) pbar.update(progress+1);
                }
            }
        };

        #ifdef _MSC_VER
            #pragma omp parallel
        #else
            #pragma omp parallel shared(stop_request, prev, interp_ref, interp_def, img_num_ref, search_area, u, v, peak_amplitude)
        #endif
        {
            if (fft_precision == util::EFFTPrecision::FLOAT32) {
                FFTf fft(search_area, search_area);
                run_fft_loop(fft);
            } else {
                FFT fft(search_area, search_area);
                run_fft_loop(fft);
            }
        }

        raise_on_interrupt();

        // remove outliers in fft
        if (config.fft_filter && (window_level != num_levels-1)){
            remove_outliers_vector(u, v,
                                   peak_amplitude, config.fft_filter_threshold,
                                   config.fft_filter_radius, config.fft_filter_corr_power);
        }

        if (config.fft_save){

            // derive base name from deformed filename (remove path and extension)
            std::string base = filenames[img_num_def];
            size_t slash = base.find_last_of("/\\");
            if (slash != std::string::npos) base = base.substr(slash + 1);
            size_t dot = base.find_last_of('.');
            if (dot != std::string::npos) base = base.substr(0, dot);

            std::ostringstream str_size_x;
            std::ostringstream str_size_y;

            str_size_x << std::setw(4) << std::setfill('0') << search_area;
            str_size_y << std::setw(4) << std::setfill('0') << search_area;

            std::string filename = saveconf.basepath + "/fft_displacements_" + base + "_" +
                                   str_size_x.str() + "x" +
                                   str_size_y.str() + ".csv";

            std::ofstream fout(filename);

            fout << "x" << saveconf.delimiter;
            fout << "y" << saveconf.delimiter;
            fout << "u" << saveconf.delimiter;
            fout << "v" << saveconf.delimiter;
            fout << "max_val" << saveconf.delimiter;
            fout << "\n";

            for (int ss = 0; ss < grid.num; ss++){
                if (!grid.active_ss[ss]) continue;
                fout << grid.coords[2*ss] << saveconf.delimiter;
                fout << grid.coords[2*ss+1] << saveconf.delimiter;
                fout << u[ss] << saveconf.delimiter;
                fout << v[ss] << saveconf.delimiter;
                fout << peak_amplitude[ss] << "\n";
            }
            fout.close();
        }

        if (g_debug_level>=2){
            pbar.finish();
        }
    }


void WindowLevel::get_displacement_from_prev_window(double &prev_x, 
                                                    double &prev_y,
                                                    const WindowLevel &prev,
                                                    const int ss,
                                                    const double cx, 
                                                    const double cy) {

    if (seed_count[ss] == 0){
        return;
    }

    const double epsilon = 10.0;
    double weight_sum_x = 0.0;
    double weight_sum_y = 0.0;
    double weight_tot = 0.0;
    double sum_x = 0;
    double sum_y = 0;

    // weighted average of 4 nearest neighbours
    for (size_t j = 0; j < seed_count[ss]; ++j) {

        int nidx = seed_idx[ss][j];
        double cx_neigh = prev.grid.coords[2*nidx];
        double cy_neigh = prev.grid.coords[2*nidx+1];

        double dx = cx - cx_neigh;
        double dy = cy - cy_neigh;
        double dist_sq = dx * dx + dy * dy;
        double weight = 1.0 / (dist_sq + epsilon);

        //sum_x += level[i-1].x[nidx];
        //sum_y += level[i-1].y[nidx];
        weight_sum_x += prev.u[nidx] * weight;
        weight_sum_y += prev.v[nidx] * weight;
        weight_tot += weight;
    }

    //prev_x = sum_x / level[i].num_neigh_list[ss];
    //prev_y = sum_y / level[i].num_neigh_list[ss];
    prev_x = weight_sum_x / weight_tot;
    prev_y = weight_sum_y / weight_tot;

}


static double weighted_median(std::vector<std::pair<double,double>>& vals)
{
    // pair = {value, weight}

    std::sort(vals.begin(), vals.end(),
        [](const auto& a, const auto& b)
        {
            return a.first < b.first;
        });

    double total_w = 0.0;
    for (const auto& v : vals)
        total_w += v.second;

    double accum = 0.0;
    for (const auto& v : vals)
    {
        accum += v.second;

        if (accum >= 0.5 * total_w)
            return v.first;
    }

    return vals.back().first;
}


void WindowLevel::remove_outliers_vector(
    std::vector<double>& u,
    std::vector<double>& v,
    const std::vector<double>& max_val,
    double threshold,
    int radius,
    double corr_power,
    double eps)
{
    if (max_val.empty() || u.empty() || v.empty()) {
        return;
    }

    std::vector<double> u_new = u;
    std::vector<double> v_new = v;

    // ---------------------------------------------------------
    // Global max correlation
    // ---------------------------------------------------------

    double max_corr =
        *std::max_element(max_val.begin(),
                          max_val.end());

    if (max_corr <= eps)
        return;

    double log_max_corr = std::log1p(max_corr);

    // ---------------------------------------------------------
    // Main loop
    // ---------------------------------------------------------

    for (int ss = 0; ss < grid.num; ++ss)
    {
        // subset coords
        int ss_x = grid.coords[2 * ss];
        int ss_y = grid.coords[2 * ss + 1];

        // grid coords
        int idx_x = ss_x / grid.step;
        int idx_y = ss_y / grid.step;

        int min_x = std::max(0, idx_x - radius);
        int min_y = std::max(0, idx_y - radius);

        int max_x =
            std::min(grid.num_ss_x,
                     idx_x + radius + 1);

        int max_y =
            std::min(grid.num_ss_y,
                     idx_y + radius + 1);

        // -----------------------------------------------------
        // Gather weighted neighbours
        // -----------------------------------------------------

        std::vector<std::pair<double,double>> neigh_u;
        std::vector<std::pair<double,double>> neigh_v;

        neigh_u.reserve((2*radius+1)*(2*radius+1));
        neigh_v.reserve((2*radius+1)*(2*radius+1));

        for (int y = min_y; y < max_y; ++y)
        {
            for (int x = min_x; x < max_x; ++x)
            {
                int nss_idx =
                    grid.mask[y * grid.num_ss_x + x];

                if (nss_idx == -1 || nss_idx == ss)
                    continue;

                // ---------------------------------------------
                // Spatial weight
                // ---------------------------------------------

                double dx = double(x - idx_x);
                double dy = double(y - idx_y);

                double dist2 = dx*dx + dy*dy;

                double spatial_w =
                    std::exp(
                        -dist2 /
                        (2.0 * radius * radius));

                // ---------------------------------------------
                // Correlation confidence
                // LOG compression for huge dynamic range
                // ---------------------------------------------

                double confidence =
                    std::log1p(max_val[nss_idx]) /
                    log_max_corr;

                confidence =
                    std::clamp(confidence,
                               0.0,
                               1.0);

                double corr_w =
                    std::pow(confidence,
                             corr_power);

                // ---------------------------------------------
                // Final neighbour weight
                // ---------------------------------------------

                double w = spatial_w * corr_w;

                neigh_u.push_back(
                    {u[nss_idx], w});

                neigh_v.push_back(
                    {v[nss_idx], w});
            }
        }

        // too few neighbours
        if (neigh_u.size() < 5)
            continue;

        // -----------------------------------------------------
        // Weighted vector median
        // -----------------------------------------------------

        double med_u =
            weighted_median(neigh_u);

        double med_v =
            weighted_median(neigh_v);

        // -----------------------------------------------------
        // Compute neighbour residuals
        // -----------------------------------------------------

        std::vector<std::pair<double,double>>
            residuals;

        residuals.reserve(neigh_u.size());

        for (size_t i = 0; i < neigh_u.size(); ++i)
        {
            double du =
                neigh_u[i].first - med_u;

            double dv =
                neigh_v[i].first - med_v;

            double r =
                std::sqrt(du*du + dv*dv);

            double w =
                neigh_u[i].second;

            residuals.push_back({r, w});
        }

        // weighted median residual
        double med_res =
            weighted_median(residuals);

        med_res =
            std::max(med_res, eps);

        // -----------------------------------------------------
        // Current vector residual
        // -----------------------------------------------------

        double du0 = u[ss] - med_u;
        double dv0 = v[ss] - med_v;

        double r0 =
            std::sqrt(du0*du0 + dv0*dv0);

        double normalized_residual =
            r0 / med_res;

        // -----------------------------------------------------
        // Adaptive threshold
        // high confidence vectors trusted more
        // -----------------------------------------------------

        double self_confidence =
            std::log1p(max_val[ss]) /
            log_max_corr;

        self_confidence =
            std::clamp(self_confidence,
                       0.0,
                       1.0);

        // threshold range:
        // low confidence -> 1.5
        // high confidence -> threshold

        double adaptive_threshold =
            1.5 +
            self_confidence *
            (threshold - 1.5);

        // -----------------------------------------------------
        // Reject outlier
        // -----------------------------------------------------

        if (normalized_residual >
            adaptive_threshold)
        {
            u_new[ss] = med_u;
            v_new[ss] = med_v;
        }
    }

    u = std::move(u_new);
    v = std::move(v_new);
}

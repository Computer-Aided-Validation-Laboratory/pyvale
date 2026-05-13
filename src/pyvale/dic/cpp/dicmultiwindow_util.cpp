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

// Common Header files
#include "../../common_cpp/progressbar.hpp"
#include "../../common_cpp/defines.hpp"
#include "../../common_cpp/dicsignalhandler.hpp"

// DIC Header files
#include "./dicmultiwindow_util.hpp"
#include "./dicfourier.hpp"
#include "./dicsubset.hpp"
#include "./dicinterp.hpp"


void multiwindow_init(std::vector<WindowLevel> &level, 
                      const bool *img_roi, 
                      const util::Config &conf,
                      const MultiwindowConfig &mwconf,
                      const common_util::SaveConfig &saveconf) {


    for (size_t lvl = 0; lvl < mwconf.overlap.size(); lvl++) {

        const bool is_last = (lvl == mwconf.overlap.size() - 1);
        const subset::Grid *prev = (lvl > 0) ? &level[lvl-1].layout : nullptr;

        level.emplace_back(img_roi, 
                           mwconf.overlap[lvl], 
                           mwconf.subset_size[lvl],
                           mwconf.search_area[lvl],
                           conf.px_hori, conf.px_vert, 
                           !is_last, lvl, 
                           conf.fft_mad, conf.fft_mad_scale,
                           conf.fft_save, saveconf,
                           prev);

    }
}

void multiwindow_init_partial(std::vector<WindowLevel> &level,
                               const bool *img_roi,
                               const util::Config &conf,
                               const MultiwindowConfig &mwconf,
                               const common_util::SaveConfig &saveconf,
                               const size_t num_levels) {

    for (size_t lvl = 0; lvl < num_levels; lvl++) {
        
        const bool is_last = (lvl == num_levels - 1);
        const subset::Grid *prev = (lvl > 0) ? &level[lvl-1].layout : nullptr;

        level.emplace_back(img_roi,
                           mwconf.overlap[lvl],
                           mwconf.subset_size[lvl],
                           mwconf.search_area[lvl],
                           conf.px_hori, conf.px_vert,
                           !is_last, lvl,
                           conf.fft_mad, conf.fft_mad_scale,
                           conf.fft_save, saveconf,
                           prev);
    }
}

void WindowLevel::gen_neighlist(const subset::Grid &layout_prev) {

    //Timer timer("nearest neighbour collection for :");

    const int prev_step = layout_prev.step;

    // a list containing the number of neighbours from the previous
    // window size for each subset in the current window size
    num_neigh_list.resize(layout.num);

    // we know the neigh_list is going to be a max size of
    // max_neigh*num_ss. we can resize this later once populated
    neigh_list.resize(max_num_neigh*layout.num);

    // shared error state
    std::atomic<bool> failed(false);

    struct ErrorInfo {
        int ss = -1;
        int ss_x = 0;
        int ss_y = 0;
        size_t num_found = 0;
    };

    ErrorInfo error;

    // For each subset, find 4 nearest neighbours in layout_prev
    #pragma omp parallel for
    for (int ss = 0; ss < layout.num; ++ss) {

        // another thread already failed
        if (failed.load()) continue;


        if (!layout.active_ss[ss]) continue;

        // corner of subset
        const int ss_x = layout.coords[2*ss];
        const int ss_y = layout.coords[2*ss+1];

        // Vector to store pairs of (distance, index)
        std::vector<std::pair<double, int>> dist_index_list;

        // loop over a 10x10 section from the previous window
        int idx_x = (ss_x / prev_step);
        int idx_y = (ss_y / prev_step);

        // range of neighbour search
        int min_x = std::max(0,idx_x-5);
        int min_y = std::max(0,idx_y-5);
        int max_x = std::min(layout_prev.num_ss_x,idx_x+6);
        int max_y = std::min(layout_prev.num_ss_y,idx_y+6);

        for (int y = min_y; y < max_y; y++){
            for (int x = min_x; x < max_x; x++){

                // check if point is a valid subset
                int nss_idx = layout_prev.mask[y*layout_prev.num_ss_x+x];
                if (nss_idx == -1) continue;

                int nss_x = layout_prev.coords[2*nss_idx];
                int nss_y = layout_prev.coords[2*nss_idx+1];

                double dx = (nss_x) - ss_x;
                double dy = (nss_y) - ss_y;
                double dist_sq = dx*dx + dy*dy;

                dist_index_list.emplace_back(dist_sq, nss_idx);
            }
        }

        // either use max_num_neigh or size of list if less than max_num_neigh
        int num_neigh = std::min(max_num_neigh, dist_index_list.size());

        // can't find any neighbours.
        if (num_neigh == 0){
            bool expected = false;
            // only first thread records error
            if (failed.compare_exchange_strong(expected, true)) {
                error.ss = ss;
                error.ss_x = ss_x;
                error.ss_y = ss_y;
                error.num_found = dist_index_list.size();
            }
            continue;
        }

        num_neigh_list[ss] = num_neigh;
        std::nth_element(dist_index_list.begin(), dist_index_list.begin() + num_neigh, dist_index_list.end());
        dist_index_list.resize(num_neigh);

        // Store neighbours indices into neighlist
        for (size_t i = 0; i < num_neigh; ++i) {
            neigh_list[ss*max_num_neigh+i] = dist_index_list[i].second;
        }
    }

    // report outside OpenMP
    if (failed.load()) {

        // char msg[512];
        //
        // snprintf(msg,
        //          sizeof(msg),
        //          "Could not find any neighbours from the previous FFT "
        //          "window size for subset (%d, %d). "
        //          "Found %zu neighbours.",
        //          error.ss_x,
        //          error.ss_y,
        //          error.num_found);
        //
        // throw std::runtime_error(msg);
    }
}

void WindowLevel::remove_outliers(std::vector<double> &u,
                                  const double mad_scale) {

    std::vector<double> u_new = u;

    int radius = 2;

    for (int ss = 0; ss < layout.num; ss++) {
        
        // subset coords
        int ss_x = layout.coords[2*ss];
        int ss_y = layout.coords[2*ss+1];

        // subset x and y index in 2d mask
        int idx_x = ss_x / layout.step;
        int idx_y = ss_y / layout.step;

        std::vector<double> neigh_vals;

        int min_x = std::max(0, idx_x-radius);
        int min_y = std::max(0, idx_y-radius);
        int max_y = std::min(layout.num_ss_y, idx_y+radius+1);
        int max_x = std::min(layout.num_ss_x, idx_x+radius+1);

        for (int y = min_y; y < max_y; ++y) {
            for (int x = min_x; x < max_x; ++x) {

                // index of neighbour 
                int nss_idx = layout.mask[y*layout.num_ss_x+x];

                // check if invalid neigh
                if (nss_idx == -1 || nss_idx == ss) continue; 

                neigh_vals.push_back(u[nss_idx]);
            }
        }

        if (neigh_vals.size() < 4) continue;

        // Median
        std::sort(neigh_vals.begin(), neigh_vals.end());
        size_t sz = neigh_vals.size();
        double median = (sz % 2 == 0) ? 0.5 * (neigh_vals[sz/2 - 1] + neigh_vals[sz/2]) : neigh_vals[sz/2];

        // MAD
        std::vector<double> abs_devs;
        abs_devs.reserve(sz);
        for (double v : neigh_vals) abs_devs.push_back(std::abs(v - median));

        std::sort(abs_devs.begin(), abs_devs.end());
        double mad = (sz % 2 == 0) ? 0.5 * (abs_devs[sz/2 - 1] + abs_devs[sz/2]) : abs_devs[sz/2];

        if (mad < 1e-12) continue;

        if (std::abs(u[ss] - median) > mad_scale * mad) {
            u_new[ss] = median;
        }
    }
    u = std::move(u_new);
}

void WindowLevel::calc_rigid_displacements(const WindowLevel &prev,
                                           const Image &img_ref,
                                           const Image &img_def,
                                           const Interpolator &interp_def,
                                           const int img_num_ref,
                                           const int img_num_def,
                                           const std::vector<std::string> &filenames){

        const int px_hori = interp_def.px_hori;
        const int px_vert = interp_def.px_vert;

        // TODO: Add a proper flag for this 
        bool subpx = true;

        // consts
        const int num_ss = layout.num;

        // set all displacements for multiwindow level to 0
        std::fill(u.begin(), u.end(), 0.0);
        std::fill(v.begin(), v.end(), 0.0);

        // progress bar initialisation
        std::string bar_title = "FFT windowing " + std::to_string(search_area) + "x" + std::to_string(search_area) + " for \033[1;4m" + filenames[img_num_ref] + "\033[0m and \033[1;4m" + filenames[img_num_def] + "\033[0m:";
        ProgressBar pbar(bar_title, layout.num);
        std::atomic<int> current_progress = 0;


        #pragma omp parallel shared(stop_request, level, prev, interp_def, search_area, u, v, max_val)
        {


            // class for FFT
            FFT fft(search_area, search_area);

            // loop over subsets for each size/step
            #pragma omp for schedule(dynamic,10)
            for (int ss = 0; ss < layout.num; ss++){

                // exit when ctrl+C
                if (stop_request) continue;

                if (!layout.active_ss[ss]) continue;

                const double cx = layout.coords[2*ss];
                const double cy = layout.coords[2*ss+1];

                // get the seed for the new window size
                double prev_u = 0.0;
                double prev_v = 0.0;

                if (level>0)
                    get_displacement_from_prev_window(prev_u, prev_v, prev, ss, cx, cy);

                std::vector<double> p(6,0.0);
                double maxv_local = 0.0;
                get_single_window_fftcc_peak_centre(fft, p, maxv_local,
                                             cx, cy,
                                             prev_u, prev_v,
                                             template_size, template_size,
                                             search_area, search_area,
                                             img_ref, img_def,
                                             interp_def);
                max_val[ss] = maxv_local;

                u[ss] = prev_u+p[0];
                v[ss] = prev_v+p[1];

                if (g_debug_level>1){
                    int progress = current_progress.fetch_add(1);
                    if (omp_get_thread_num()==0) pbar.update(progress+1);
                }
            }
        }

        // remove outliers in fft
        if (mad_filter){
            //remove_outliers(u, mad_scale);
            //remove_outliers(v, mad_scale);
            remove_outliers_vector(u,v,max_val);
        }

        //smooth_field(level[i].x, current_level, 7.0, 5);
        //smooth_field(level[i].y, current_level, 7.0, 5);

        if (fft_save){

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

            for (int ss = 0; ss < layout.num; ss++){
                fout << layout.coords[2*ss] << saveconf.delimiter;
                fout << layout.coords[2*ss+1] << saveconf.delimiter;
                fout << u[ss] << saveconf.delimiter;
                fout << v[ss] << saveconf.delimiter;
                fout << max_val[ss] << "\n";
            }
            fout.close();
        }

        if (g_debug_level>1){
            pbar.update(current_progress);
            pbar.finish();
        }
    }


void WindowLevel::get_displacement_from_prev_window(double &prev_x, 
                                                    double &prev_y,
                                                    const WindowLevel &prev,
                                                    const int ss,
                                                    const double cx, 
                                                    const double cy) {

    if (num_neigh_list[ss] == 0){
        prev_x = 0.0;
        prev_y = 0.0;
        return;
    }

    const double epsilon = 10.0;
    double weight_sum_x = 0.0;
    double weight_sum_y = 0.0;
    double weight_tot = 0.0;
    double sum_x = 0;
    double sum_y = 0;

    // weighted average of 4 nearest neighbours
    for (size_t j = 0; j < num_neigh_list[ss]; ++j) {

        int nidx = neigh_list[ss*max_num_neigh+j];
        double cx_neigh = prev.layout.coords[2*nidx];
        double cy_neigh = prev.layout.coords[2*nidx+1];

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

    for (int ss = 0; ss < layout.num; ++ss)
    {
        // subset coords
        int ss_x = layout.coords[2 * ss];
        int ss_y = layout.coords[2 * ss + 1];

        // grid coords
        int idx_x = ss_x / layout.step;
        int idx_y = ss_y / layout.step;

        int min_x = std::max(0, idx_x - radius);
        int min_y = std::max(0, idx_y - radius);

        int max_x =
            std::min(layout.num_ss_x,
                     idx_x + radius + 1);

        int max_y =
            std::min(layout.num_ss_y,
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
                    layout.mask[y * layout.num_ss_x + x];

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

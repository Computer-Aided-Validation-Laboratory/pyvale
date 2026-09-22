// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <csignal>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <omp.h>
#include <thread>
#include <chrono>

// Program Header files
#include "./dicsubset.hpp"
#include "./dicfourier.hpp"
#include "./dicoptimizer.hpp"
#include "./dicresults.hpp"
#include "./dicrg.hpp"
#include "../../commoncpp/dicsignalhandler.hpp"


namespace rg {

        bool QueueLocal::try_pop_own_q(const int tid, rg::Point& out) {
        std::lock_guard<std::mutex> lock(locks[tid]);
        if (qs[tid].empty()) return false;
        out = qs[tid].top();
        qs[tid].pop();
        return true;
    }

    bool QueueLocal::try_steal_from_other_q(const int tid, rg::Point& out) {
        std::lock_guard<std::mutex> steal_guard(steal_lock);
        for (size_t i = 0; i < qs.size(); ++i) {
            if (i == tid) continue;
            std::lock_guard<std::mutex> lock(locks[i]);
            if (!qs[i].empty()) {
                out = qs[i].top();
                qs[i].pop();
                return true;
            }
        }
        return false;
    }

    bool QueueLocal::pop(const int tid, rg::Point& out) {
        if (try_pop_own_q(tid, out))
            return true;
        const int max_idle_iters = 100;
        for (int idle = 0; idle < max_idle_iters; ++idle) {
            if (try_steal_from_other_q(tid, out))
                return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    }

    void QueueLocal::push(const int tid, const std::vector<rg::Point>& points) {
        for (const auto& neigh : points) {
            std::lock_guard<std::mutex> lock(locks[tid]);
            qs[tid].push(neigh);
        }
    }

    bool QueueGlobal::pop(const int tid, rg::Point& current) {
        bool found = false;
        {
            std::lock_guard<std::mutex> lock(m);
            if (!q.empty()) {
                current = q.top();
                q.pop();
                found = true;
            }
        }
        if (found) return true;

        active_threads.fetch_sub(1);
        while (active_threads.load() > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            {
                std::lock_guard<std::mutex> lock(m);
                if (!q.empty()) {
                    current = q.top();
                    q.pop();
                    found = true;
                    break;
                }
            }
        }
        if (found) {
            active_threads.fetch_add(1);
            return true;
        }
        return false;
    }

    void QueueGlobal::push(const int tid, const std::vector<rg::Point>& points) {  // fixed: proper method
        if (points.empty()) return;
        std::lock_guard<std::mutex> lock(m);  // fixed: m
        for (const auto& pt : points) {
            q.push(pt);
        }
    }

    bool is_valid_point(const int ss_x, const int ss_y, const subset::Grid &ss_grid) {

        int x = ss_x / ss_grid.step;
        int y = ss_y / ss_grid.step;

        int idx = y * ss_grid.num_ss_x + x;

        if ((ss_x % ss_grid.step) || (ss_y % ss_grid.step)){
            std::cerr << "Subset coordinates (" << ss_x << ", " << ss_y << ") are not a valid subset location." << std::endl;
            std::cerr << "Subset ss_step size: " << ss_grid.step << std::endl;
            return false;
            exit(EXIT_FAILURE);
        }
        else if (ss_grid.mask[idx] == -1){
            std::cerr << "Subset coordinates (" << ss_x << ", " << ss_y << ") are not a valid subset location." << std::endl;
            std::cerr << "subset mask index: " << idx << std::endl;
            return false;
            exit(EXIT_FAILURE);
        }
        else return true;
    }


    bool check_convergence(const int x, const int y, const OptResult &res, std::string &msg, bool direct_neigh) {
        if (!res.above_thresh) {
            std::ostringstream oss;

            oss << (direct_neigh
                    ? "Direct neighbour failed threshold"
                    : "Seed subset failed threshold")
                << "\n"
                << "subset location: " << x << ", " << y << "\n"
                << "displacement: " << res.u << ", " << res.v << "\n"
                << "cost: " << res.cost << "\n"
                << "xtol: " << res.xtol << "\n"
                << "ftol: " << res.ftol << "\n"
                << "above_thresh: " << static_cast<unsigned>(res.above_thresh) << "\n"
                << "converged: " << static_cast<unsigned>(res.converged) << "\n"
                << "iterations: " << res.iter;
            msg = oss.str();
            return false;
        }
        return true;
    }
void retry_bad_points(const Interpolator &interp_ref,
                      const Interpolator &interp_def,
                      const subset::Grid &ss_grid,
                      const util::Config &conf,
                      const ResultArrays &results_ref,
                      ResultArrays &results_def,
                      const std::vector<std::atomic<int>> &computed_mask) {


    for (int required_neigh = 4; required_neigh >= 2; --required_neigh) {

        const std::vector<uint8_t> successful = results_def.above_thresh;
        std::vector<int> retry_indices;

        for (int idx = 0; idx < ss_grid.num; ++idx) {
            if (!computed_mask[idx].load() || !ss_grid.active_ss[idx] || successful[idx]) continue;

            int count = 0;
            for (int nidx : ss_grid.neigh[idx]) count += successful[nidx] != 0;
            if (count >= required_neigh) retry_indices.push_back(idx);
        }

        #pragma omp parallel
        {
            subset::Pixels ss_def(ss_grid.size_x, ss_grid.size_y);
            subset::Pixels ss_ref(ss_grid.size_x, ss_grid.size_y);
            Optimizer opt(conf.shape_func, conf.corr_crit, conf.max_iter,
                          conf.precision, conf.threshold,
                          ss_grid.size_x*ss_grid.size_y);

            #pragma omp for
            for (int i = 0; i < static_cast<int>(retry_indices.size()); ++i) {
                const int idx = retry_indices[i];
                const double cx = ss_grid.coords[2*idx];
                const double cy = ss_grid.coords[2*idx+1];

                subset::fill_from_centre_coords(ss_ref, cx, cy, interp_ref);
                for (int px = 0; px < ss_ref.num_px; ++px) {
                    ss_ref.x[px] -= cx;
                    ss_ref.y[px] -= cy;
                }

                opt.average_params_from_neigh(results_def.p, successful,
                                              ss_grid.neigh[idx]);
                OptResult retry_res(opt.num_params);
                if (ss_ref.sum != 0) {
                    retry_res = opt.solve(cx, cy, ss_ref, ss_def, interp_def);
                }

                if (retry_res.cost > results_def.cost[idx]) {
                    retry_res.u += results_ref.u[idx];
                    retry_res.v += results_ref.v[idx];
                    results_def.append(retry_res, idx);
                }
            }
        }
    }
}

}



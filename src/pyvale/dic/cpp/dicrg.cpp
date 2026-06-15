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
#include "../../common_cpp/dicsignalhandler.hpp"


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
}



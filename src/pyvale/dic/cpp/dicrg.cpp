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

// Program Header files
#include "./dicsubset.hpp"
#include "./dicfourier.hpp"
#include "./dicoptimizer.hpp"
#include "./dicresults.hpp"
#include "./dicrg.hpp"


namespace rg {

    bool try_pop_own_q(std::priority_queue<rg::Point>& q,
                            std::mutex& mtx,
                            rg::Point& out){
        std::lock_guard<std::mutex> lock(mtx);
        if (q.empty()) return false;
        out = q.top();
        q.pop();
        return true;
    }

    bool try_steal_from_other_q(std::vector<std::priority_queue<rg::Point>>& queues,
                        std::vector<std::mutex>& mutexes,
                        std::mutex& steal_mtx,
                        rg::Point& out) {
        std::lock_guard<std::mutex> steal_lock(steal_mtx);
        for (size_t i = 0; i < queues.size(); ++i) {
            std::lock_guard<std::mutex> lock(mutexes[i]);
            if (!queues[i].empty()) {
                out = queues[i].top();
                queues[i].pop();
                return true;
            }
        }
        return false;
    }

    bool pop_next_point(int tid,
                            std::vector<std::priority_queue<rg::Point>>& local_q,
                            std::vector<std::mutex>& queue_mutexes,
                            std::mutex& steal_mutex,
                            rg::Point& current) {

        if (try_pop_own_q(local_q[tid], queue_mutexes[tid], current))
            return true;

        const int max_idle_iters = 100;
        for (int idle = 0; idle < max_idle_iters; ++idle) {
            if (try_steal_from_other_q(local_q, queue_mutexes, steal_mutex, current))
                return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
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


    void check_convergence_or_exit(const int x, const int y, const OptResult &res) {
        if (!res.converged || !res.above_threshold) {
            std::cout << "ERROR: unsuccessful convergence at seed or direct neighbour." << std::endl;
            std::cout << "Please select a different seed location." << std::endl;
            std::cout << std::endl;
            std::cout << "subset location: " << x << ", " << y << std::endl;
            std::cout << "displacement: " << res.u << ", " << res.v << std::endl;
            std::cout << "cost: " << res.cost << std::endl;
            std::cout << "xtol: " << res.xtol << std::endl;
            std::cout << "ftol: " << res.ftol << std::endl;
            std::cout << "above_threshold: " << static_cast<unsigned>(res.above_threshold) << std::endl;
            std::cout << "converged: " << static_cast<unsigned>(res.converged) << std::endl;
            std::cout << "iterations: " << res.iter << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}



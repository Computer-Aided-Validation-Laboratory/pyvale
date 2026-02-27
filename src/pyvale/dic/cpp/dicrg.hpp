// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICRG_H
#define DICRG_H

// STD library Header files
#include <queue>
#include <atomic>
#include <mutex>
// Program Header files
#include "./dicsubset.hpp"
#include "./dicoptimizer.hpp"

namespace rg {

    /**
     * @brief 
     * 
     */
    struct Point {
        int idx;
        double val;

        // Constructor
        Point(int _idx, double _val) : 
            idx(_idx), val(_val) {}

        // Comparison operator for priority queue (higher ZNCC first)
        bool operator<(const Point& other) const {
            return val < other.val;  // Note: priority_queue puts largest elements on top
        }
    };

    /**
    * @brief Retrieves the next point for a thread to process, blocking briefly if queues are empty.
    *
    * First tries the calling thread's own queue via try_pop_own_q(). If empty,
    * repeatedly attempts to steal from other threads via try_steal_from_other_q()
    * up to a fixed number of idle iterations, sleeping 1ms between attempts.
    *
    * @param tid           Index of the calling thread.
    * @param local_q       All thread-local priority queues.
    * @param queue_mutexes Per-queue mutexes protecting each entry in @p local_q.
    * @param steal_mutex   Global mutex serialising steal attempts across threads.
    * @param current       Populated with the next point to process if one is found.
    * @return              True if a point was retrieved, false if all queues remained
    *                      empty after exhausting idle iterations.
    */
    bool pop_next_point(int tid,
                        std::vector<std::priority_queue<rg::Point>>& local_q,
                        std::vector<std::mutex>& queue_mutexes,
                        std::mutex& steal_mutex,
                        rg::Point& current);


    /**
    * @brief Attempts to pop the highest-priority point from a thread's own queue.
    *
    * @param q   The thread-local priority queue to pop from.
    * @param mtx Mutex protecting @p q.
    * @param out Populated with the popped point if successful.
    * @return    True if a point was popped, false if the queue was empty.
    */
    bool try_pop_own_q(std::priority_queue<rg::Point>& q,
                       std::mutex& mtx,
                       rg::Point& out);


    /**
    * @brief Attempts to steal the highest-priority point from any other thread's queue.
    *
    * Iterates over all queues and pops from the first non-empty one found.
    * The global @p steal_mtx is held for the duration to prevent two threads
    * from stealing simultaneously.
    *
    * @param queues    All thread-local priority queues.
    * @param mutexes   Per-queue mutexes, each protecting the corresponding entry in @p queues.
    * @param steal_mtx Global mutex serialising steal attempts across threads.
    * @param out       Populated with the stolen point if successful.
    * @return          True if a point was stolen, false if all queues were empty.
    */
    bool try_steal_from_other_q(std::vector<std::priority_queue<rg::Point>>& queues,
                        std::vector<std::mutex>& mutexes,
                        std::mutex& steal_mtx,
                        rg::Point& out);


    /**
     * @brief 
     * 
     * @param x 
     * @param y 
     * @param px_hori 
     * @param px_vert 
     * @param ss_size 
     * @return true 
     * @return false 
     */
     bool is_valid_point(const int ss_x, const int ss_y, const subset::Grid &ss_grid);



    void check_convergence_or_exit(const int x, const int y, const OptResult &res);

}


#endif // DICRG_H

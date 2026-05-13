// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICRG_H
#define DICRG_H

// STD library Header files
#include <queue>
#include <mutex>
#include <atomic>

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


    struct QueuePolicy {
        virtual bool pop(const int tid, rg::Point& out) = 0;
        virtual void push(const int tid, const std::vector<rg::Point>& points) = 0;
        virtual ~QueuePolicy() = default;
    };

    struct QueueGlobal : QueuePolicy {
        std::priority_queue<rg::Point> q;
        std::mutex m;
        std::atomic<int> active_threads;
        

        explicit QueueGlobal(const int num_threads) {
            active_threads.store(num_threads);
        }

        /**
        * @brief Retrieves the next point from a global priority queue.
        *
        * Coordinates multiple threads to pop the highest-priority point. If the queue is 
        * empty, threads will wait as long as other threads are still active (processing 
        * points that might add new neighbours to the queue).
        *
        * @param tid           Thread ID.
        * @param out       Populated with the next point if found.
        * @return              True if a point was retrieved, false if the queue is 
        *                      empty and all threads are idle.
        */
        bool pop(const int tid, rg::Point& out);

        
        /**
        * @brief Pushes multiple points to the global priority queue in a thread-safe manner.
        *
        * @param tid           Thread ID.
        * @param points       Vector of points to add.
        */
        void push(const int tid, const std::vector<rg::Point>& points);

    };

    struct QueueLocal : QueuePolicy {
        std::vector<std::priority_queue<rg::Point>> qs;
        std::vector<std::mutex> locks;
        std::mutex steal_lock;

        explicit QueueLocal(const int num_threads)
            : qs(num_threads), locks(num_threads) {}    

        /**
        * @brief Retrieves the next point for a thread to process, blocking briefly if queues are empty.
        *
        * First tries the calling thread's own queue via try_pop_own_q(). If empty,
        * repeatedly attempts to steal from other threads via try_steal_from_other_q()
        * up to a fixed number of idle iterations, sleeping 1ms between attempts.
        *
        * @param tid           Thread ID.
        * @param current       Populated with the next point to process if one is found.
        * @return              True if a point was retrieved, false if all queues remained
        *                      empty after exhausting idle iterations.
        */
        bool pop(const int tid, rg::Point& current);


        /**
        * @brief Pushes multiple points to a local threads priority queue in a thread-safe manner.
        *
        * @param tid           Thread ID.
        * @param points    neighbours to add.
        */
        void push(const int tid, const std::vector<rg::Point> &points);

        private:

        /**
        * @brief Attempts to steal the highest-priority point from any other thread's queue.
        *
        * Iterates over all queues and pops from the first non-empty one found.
        * The global @p steal_mtx is held for the duration to prevent two threads
        * from stealing simultaneously.
        *
        * @param tid           Thread ID.
        * @param out       Populated with the stolen point if successful.
        * @return          True if a point was stolen, false if all queues were empty.
        */
        bool try_steal_from_other_q(const int tid, rg::Point& out);


        /**
        * @brief Attempts to pop the highest-priority point from a thread's own queue.
        *
        * @param tid           Thread ID.
        * @param out Populated with the popped point if successful.
        * @return    True if a point was popped, false if the queue was empty.
        */
        bool try_pop_own_q(const int tid, rg::Point& out);
    };








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



    void check_convergence(const int x, const int y, const OptResult &res, bool direct_neigh=false);

}


#endif // DICRG_H

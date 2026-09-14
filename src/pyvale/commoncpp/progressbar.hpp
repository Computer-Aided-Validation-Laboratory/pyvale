#ifndef PROGRESSBAR_H
#define PROGRESSBAR_H

// STD header files
#include <string>
#include <cstddef>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include "./util.hpp"

class ProgressBar {
private:
    std::string message;
    int total_iterations;
    int current_iter;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_update_time;
    int last_iter;
    bool started;
    std::size_t last_progress_chars;
    double next_update_percent;

    static constexpr int BAR_WIDTH = 36;
    static constexpr double UPDATE_INTERVAL_PERCENT = 1.0;

public:
    // Constructor
    ProgressBar(const std::string& msg, int total_iters)
        : message(msg),
          total_iterations(total_iters),
          current_iter(0),
          last_iter(0),
          started(false),
          last_progress_chars(0),
          next_update_percent(0.0) {}

    // Update progress
    void update(int iteration) {
        current_iter = std::clamp(iteration, 0, total_iterations);
        double percent = 100.0;
        if (total_iterations > 0) {
            percent = (static_cast<double>(current_iter) / total_iterations) * 100.0;
            percent = std::clamp(percent, 0.0, 100.0);
        }

        const bool is_first_update = !started;
        const bool is_final_update = current_iter >= total_iterations;
        const bool should_update =
            is_first_update || is_final_update || percent >= next_update_percent;

        if (!should_update) return;

        if (!started) {
            start_time = std::chrono::steady_clock::now();
            last_update_time = start_time;
            started = true;
            hide_cursor();

            std::string time = "[" + common_util::current_datetime_ms() +"] ";

            // Print message line once; keep the progress line open for carriage-return updates.
            std::cout << time << message << "\n";
        }

        while (next_update_percent <= percent) {
            next_update_percent += UPDATE_INTERVAL_PERCENT;
        }

        // Time
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
        std::string time_str = common_util::format_duration_ms(duration);

        last_update_time = now;
        last_iter = current_iter;

        // Progress bar
        std::string bar = make_bar(percent);

        // Build progress line
        std::ostringstream progress;
        progress << "["
                 << bar << "] "
                 << std::fixed << std::setprecision(1)
                 << std::setw(5) << percent << "%  "
                 << "(" << std::setw(6) << current_iter
                 << "/" << std::setw(6) << total_iterations << ")  "
                 << std::setprecision(2)
                 << "Time: " << std::setw(10) << time_str;

        const std::string progress_line = progress.str();
        const std::size_t padding =
            last_progress_chars > progress_line.size()
                ? last_progress_chars - progress_line.size()
                : 0;

        std::cout << "\r" << progress_line << std::string(padding, 32)
                  << std::flush;
        last_progress_chars = progress_line.size();
    }

    // Increment helper
    void tick() {
        update(current_iter + 1);
    }

    // Finish — leaves both message and progress bar visible, cursor on new line
    void finish() {
        if (started && current_iter < total_iterations) {
            update(total_iterations);
        }
        if (started) {
            std::cout << "\n" << std::flush;
        }
        show_cursor();
    }

    // Reset
    void reset() {
        current_iter = 0;
        last_iter = 0;
        started = false;
        last_progress_chars = 0;
        next_update_percent = 0.0;
    }

private:
    // Build visual bar
    std::string make_bar(double percent) const {
        int filled = static_cast<int>(std::round((percent / 100.0) * BAR_WIDTH));
        int empty = BAR_WIDTH - filled;

        std::string bar;
        bar.reserve(BAR_WIDTH);

        bar.append(filled, '#');
        bar.append(empty, '-');

        return bar;
    }

    // Format duration
    std::string format_duration(const std::chrono::milliseconds& duration) const {
        auto total_seconds = duration.count() / 1000;
        auto milliseconds = duration.count() % 1000;

        int hours = total_seconds / 3600;
        int minutes = (total_seconds % 3600) / 60;
        int seconds = total_seconds % 60;

        std::ostringstream oss;
        oss << std::setfill('0');

        if (hours > 0) {
            oss << std::setw(2) << hours << ":"
                << std::setw(2) << minutes << ":"
                << std::setw(2) << seconds << "."
                << std::setw(3) << milliseconds;
        } else {
            oss << std::setw(2) << minutes << ":"
                << std::setw(2) << seconds << "."
                << std::setw(3) << milliseconds;
        }

        return oss.str();
    }

    static void hide_cursor() {
        if (is_terminal()) {
            std::cout << "\033[?25l" << std::flush;
        }
    }

    static void show_cursor() {
        if (is_terminal()) {
            std::cout << "\033[?25h" << std::flush;
        }
    }

    static bool is_terminal() {
#if defined(_WIN32)
        return _isatty(_fileno(stdout)) != 0;
#else
        return isatty(fileno(stdout)) != 0;
#endif
    }
};

#endif // PROGRESSBAR_H

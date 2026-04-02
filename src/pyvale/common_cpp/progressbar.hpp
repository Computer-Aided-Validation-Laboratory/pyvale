#ifndef PROGRESSBAR_H
#define PROGRESSBAR_H

// STD header files
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>

class ProgressBar {
private:
    std::string message;
    int total_iterations;
    int current_iter;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_update_time;
    int last_iter;
    bool started;

    static constexpr int BAR_WIDTH = 30;

public:
    // Constructor
    ProgressBar(const std::string& msg, int total_iters)
        : message(msg),
          total_iterations(total_iters),
          current_iter(0),
          last_iter(0),
          started(false) {}

    // Update progress
    void update(int iteration) {
        if (!started) {
            start_time = std::chrono::steady_clock::now();
            last_update_time = start_time;
            started = true;
            hide_cursor();
            // Print message line once — never overwritten
            std::cout << message << "\n";
            // Print empty progress line so the first \033[1A has a line to go up to
            std::cout << "\n" << std::flush;
        }

        current_iter = iteration;

        // Percentage
        double percent = (static_cast<double>(current_iter) / total_iterations) * 100.0;
        percent = std::clamp(percent, 0.0, 100.0);

        // Time
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
        std::string time_str = format_duration(duration);

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
                 << "Time: [" << std::setw(10) << time_str << "]";

        // Move up 1 to overwrite only the progress line, never the message line
        std::cout << "\033[1A"   // move up 1 line
                  << "\r\033[K"  // clear line
                  << progress.str() << "\n"
                  << std::flush;
    }

    // Increment helper
    void tick() {
        update(current_iter + 1);
    }

    // Finish — leaves both message and progress bar visible, cursor on new line
    void finish() {
        //std::cout << std::endl;
        show_cursor();
    }

    // Reset
    void reset() {
        current_iter = 0;
        last_iter = 0;
        started = false;
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
        std::cout << "\033[?25l" << std::flush;
    }

    static void show_cursor() {
        std::cout << "\033[?25h" << std::flush;
    }
};

#endif // PROGRESSBAR_H

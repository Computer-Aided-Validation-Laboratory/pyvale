#ifndef PROGRESSBAR_H
#define PROGRESSBAR_H

// STD header files
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

class ProgressBar {
private:
    std::string message;
    int total_iterations;
    int current_iter;
    std::chrono::steady_clock::time_point start_time;
    bool started;
    static constexpr int CONSOLE_WIDTH = 100;

public:
    // Constructor
    ProgressBar(const std::string& msg, int total_iters) 
        : message(msg), total_iterations(total_iters), current_iter(0), started(false) {}

    // Update progress (call this each iteration)
    void update(int iteration) {
        if (!started) {
            start_time = std::chrono::steady_clock::now();
            started = true;
            hide_cursor();
        }

        current_iter = iteration;

        // Calculate percentage
        float percent = (static_cast<float>(current_iter) / static_cast<float>(total_iterations)) * 100.0f;

        // Calculate elapsed time
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);

        // Format time as HH:MM:SS.mmm or MM:SS.mmm
        std::string time_str = format_duration(duration);

        // Build the progress string
        std::ostringstream oss;
        oss << "\r" << std::left << std::setw(40) << message << std::right
            << "[" << std::setw(6) << current_iter << "/" << std::setw(6) << total_iterations << "]  "
            << std::fixed << std::setprecision(1) << std::setw(5) << percent << "%  "
            << "Time: [" << std::setw(12) << time_str << "]\t";
            std::cout << "\r" << oss.str() << std::flush;

    }

    // Increment and update (convenient for loops)
    void tick() {
        update(current_iter + 1);
    }

    // Complete the progress bar (prints newline)
    void finish() {
        std::cout << std::endl;
        show_cursor();
    }

    // Reset the progress bar
    void reset() {
        current_iter = 0;
        started = false;
    }

private:
        std::string format_duration(const std::chrono::milliseconds& duration) {
        auto total_seconds = duration.count() / 1000;
        auto milliseconds = duration.count() % 1000;

        int hours = total_seconds / 3600;
        int minutes = (total_seconds % 3600) / 60;
        int seconds = total_seconds % 60;

        std::ostringstream oss;

        if (hours > 0) {
            oss << std::setfill('0')
                << std::setw(2) << hours << ":"
                << std::setw(2) << minutes << ":"
                << std::setw(2) << seconds << "."
                << std::setw(3) << milliseconds;
        } else {
            oss << std::setfill('0')
                << std::setw(2) << minutes << ":"
                << std::setw(2) << seconds << "."
                << std::setw(3) << milliseconds;
        }
        return oss.str();
    }

    static void hide_cursor() {
        std::cout << "\033[?25l" << std::flush;  // ANSI escape code
    }
    
    static void show_cursor() {
        std::cout << "\033[?25h" << std::flush;  // ANSI escape code
    }
};

#endif // PROGRESSBAR_H

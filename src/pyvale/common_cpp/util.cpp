// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <omp.h>
#include <string>
#include <fstream>

// common_cpp header files
#include "./indicators.hpp"

namespace common_util {

    void create_progress_bar(indicators::ProgressBar &bar,
                            const std::string &bar_title,
                            const int num_ss){
        //Hide cursor
        indicators::show_console_cursor(false);
        bar.set_option(indicators::option::BarWidth{50});
        bar.set_option(indicators::option::Start{" ["});
        bar.set_option(indicators::option::Fill{"#"});
        bar.set_option(indicators::option::Lead{"#"});
        bar.set_option(indicators::option::Remainder{"-"});
        bar.set_option(indicators::option::End{"]"});
        bar.set_option(indicators::option::PrefixText{bar_title});
        bar.set_option(indicators::option::ShowPercentage{true});
        bar.set_option(indicators::option::ShowElapsedTime{true});
    }

    void update_progress_bar(indicators::ProgressBar &bar, int i, int num_ss, int &prev_pct) {
        int curr_pct = static_cast<int>((static_cast<float>(i) / num_ss) * 100.0f);

        // Only update if we've passed a new percentage
        if (curr_pct > prev_pct) {
            prev_pct = curr_pct;
            bar.set_progress(curr_pct);
        }
    }

    void set_num_threads(int n){
        omp_set_num_threads(n);
    }
}

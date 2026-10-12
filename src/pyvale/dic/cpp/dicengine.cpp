// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#include "./dicengine.hpp"

#include <numeric>
#include <stdexcept>

SubsetGrid build_initial_grid(const bool *img_roi,
                              const util::Config &conf,
                              const MultiwindowConfig &mwconf,
                              const common_util::SaveConfig &saveconf,
                              MultiwindowPyramid &pyramid) {
    switch (conf.scan_method) {
        case util::EScanMethod::MULTIWINDOW_RG:
        case util::EScanMethod::MULTIWINDOW:
            pyramid.build(img_roi, conf, mwconf, saveconf);
            return pyramid.back().grid;

        case util::EScanMethod::SINGLEWINDOW_RG:
        case util::EScanMethod::RASTER: {
            common_util::Timer timer("to create subset grid:", 2);
            return create_grid(img_roi,
                               conf.ss_step,
                               conf.ss_size,
                               conf.ss_size,
                               conf.px_hori,
                               conf.px_vert,
                               conf.partial_subset);
        }

        default:
            throw std::invalid_argument("Unsupported scan method");
    }
}

bool should_update_ref(const int img_num_def_l,
                       const ResultArrays &results,
                       const util::Config &conf) {
    // Never update on the first deformed image.
    if (img_num_def_l == 1) {
        return false;
    }

    switch (conf.incremental_update_cond) {
        case util::EIncrementalCond::IMAGE: {
            const int interval = static_cast<int>(conf.incremental_update_val);
            return (img_num_def_l - 1) % interval == 0;
        }

        case util::EIncrementalCond::ITER:
            if (results.niter.empty()) return false;
            return std::accumulate(results.niter.begin(), results.niter.end(), 0.0)
                / results.niter.size() > conf.incremental_update_val;

        case util::EIncrementalCond::COST:
            if (results.cost.empty()) return false;
            return std::accumulate(results.cost.begin(), results.cost.end(), 0.0)
                / results.cost.size() > conf.incremental_update_val;

        default:
            throw std::runtime_error(
                "Unknown incremental update condition: " +
                std::to_string(static_cast<int>(conf.incremental_update_cond)));
    }
}

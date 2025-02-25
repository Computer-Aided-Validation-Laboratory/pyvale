// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICCORRELATION_H
#define DICCORRELATION_H

#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>



namespace correlation {

    double ssd(const std::vector<double>& subset_ref, const std::vector<double>& subset_def) {
        
        double ssd_value = 0.0;
        
        for (int i = 0; i < subset_ref.size(); ++i) {
            double diff = subset_ref[i] - subset_def[i];
            ssd_value += diff * diff;
        }

        return ssd_value;

    }


    double nssd(const std::vector<double>& subset_ref, const std::vector<double>& subset_def) {


        auto mean = [](const std::vector<double>& v) {
            return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };

        auto stddev = [](const std::vector<double>& v, double mean_value) {
            double variance = 0.0;
            for (double val : v) {
                variance += (val - mean_value) * (val - mean_value);
            }
            return std::sqrt(variance / v.size());
        };

        double mean_ref = mean(subset_ref);
        double mean_def = mean(subset_def);
        double std_ref = stddev(subset_ref, mean_ref);
        double std_def = stddev(subset_def, mean_def);

        double nssd_value = 0.0;
        for (int i = 0; i < subset_ref.size(); ++i) {
            double normed_ref = (subset_ref[i] - mean_ref) / std_ref;
            double normed_def = (subset_def[i] - mean_def) / std_def;
            double diff = normed_ref - normed_def;
            nssd_value += diff * diff;
        }

        return nssd_value;
    }




    double znssd(const std::vector<double>& subset_ref, const std::vector<double>& subset_def) {


        auto mean = [](const std::vector<double>& v) {
            return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };

        double mean_ref = mean(subset_ref);
        double mean_def = mean(subset_def);

        double znssd_value = 0.0;
        for (size_t i = 0; i < subset_ref.size(); ++i) {
            double zero_mean_ref = subset_ref[i] - mean_ref;
            double zero_mean_def = subset_def[i] - mean_def;
            double diff = zero_mean_ref - zero_mean_def;
            znssd_value += diff * diff;
        }

        return znssd_value;
    }


}

#endif /*DICCORRELATION_H*/



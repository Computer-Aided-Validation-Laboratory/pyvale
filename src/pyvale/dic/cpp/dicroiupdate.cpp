// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// opencv header files
#include <omp.h>
#include <opencv2/opencv.hpp>

// std lib header files
#include <cmath>

// program header files
#include "./dicroiupdate.hpp"


bool* propagate_roi(
    const bool* img_roi,
    const ResultArrays results_def,
    const util::Config conf,
    const subset::Grid ss_grid){


    const int half = conf.ss_size / 2;

    cv::Mat roi_mat(conf.px_vert, conf.px_hori, CV_8U, cv::Scalar(0));

    for (int i = 0; i < results_def.conv.size(); i++)
    {
        if (!results_def.conv[i] || !results_def.above_thresh[i])
            continue;

        int new_cx = static_cast<int>(std::round(ss_grid.coords[2*i + 0] + results_def.u[i]));
        int new_cy = static_cast<int>(std::round(ss_grid.coords[2*i + 1] + results_def.v[i]));

        for (int dy = -half; dy <= half; dy++)
        {
            for (int dx = -half; dx <= half; dx++)
            {
                int px = new_cx + dx;
                int py = new_cy + dy;

                if (px < 0 || px >= conf.px_hori || py < 0 || py >= conf.px_vert)
                    continue;

                roi_mat.at<uchar>(py, px) = 255;
            }
        }
    }
    
    bool do_morph = false;
    if (do_morph)
    {
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(conf.ss_step, conf.ss_step));

        cv::morphologyEx(roi_mat, roi_mat, cv::MORPH_CLOSE, kernel);
    }

    bool* roi_out = new bool[conf.px_hori * conf.px_vert];

    for (int y = 0; y < conf.px_vert; y++)
        for (int x = 0; x < conf.px_hori; x++)
            roi_out[y * conf.px_hori + x] = roi_mat.at<uchar>(y, x) > 0;

    return roi_out;
}

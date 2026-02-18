// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef CALIBSTEREO_H
#define CALIBSTEREO_H


// STD library Header files
#include <iostream>
#include <cstring>
#include <omp.h>
#include <vector>

// master func that gets called from python
void calibrate_stereo(const std::vector<double> &init_params,
                        const std::vector<double> &dots_cam0,
                        const std::vector<double> &dots_cam1,
                        const std::vector<double> &grid,
                        const std::vector<int> &lengths,
                        const int px_hori, const int px_vert, const int num_img_pairs);


/** Camera intrinsic parameters */
struct CamIntrinsics {
    double fx;                      /**< Focal length in x direction [pixels] */
    double fy;                      /**< Focal length in y direction [pixels] */
    double fs;                      /**< Skew coefficient [pixels] */
    double cx;                      /**< Principal point x-coordinate [pixels] */
    double cy;                      /**< Principal point y-coordinate [pixels] */
    std::vector<double> distortion; /**< Distortion coefficients [kappa1, kappa2, p1, p2, kappa3] */
};

/** Stereo camera calibration parameters */
struct Calib {
    CamIntrinsics cam0;              /**< Camera 0 intrinsic parameters */
    CamIntrinsics cam1;              /**< Camera 1 intrinsic parameters */
    std::vector<double> translation; /**< Translation vector [x, y, z] in mm */
    std::vector<double> rotation;    /**< Euler angles [theta, phi, psi] in degrees */
};

#endif

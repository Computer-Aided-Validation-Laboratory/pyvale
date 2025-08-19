// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <Eigen/Core>
#include <iostream>
#include <cstring>
#include <omp.h>
#include <vector>

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

// pyvale header files
#include "./calibopt.hpp"
#include "./calibstereo.hpp"


void stereo_calibration(const std::vector<double> &init_params,
                        const std::vector<double> &dots_cam0,
                        const std::vector<double> &dots_cam1,
                        const std::vector<double> &grid,
                        const std::vector<int> &lengths,
                        const int px_hori, const int px_vert, const int num_img){

    std::cout << init_params.size() << std::endl;
    std::cout <<dots_cam0.size() << std::endl;
    std::cout <<dots_cam1.size() << std::endl;
    std::cout <<grid.size() << std::endl;

    std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    std::cout << num_img << std::endl;
    std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    std::cout << px_hori << std::endl;
    std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    std::cout << px_vert << std::endl;
    std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;


    int num_params = 4*2 + 5*2 + 3 + 3 + 6*num_img;
    std::cout << num_params << std::endl;
    Parameters opt(init_params.size(), 100, 0.001, px_hori, px_vert);

    // assign initial guess for parameter values
    for (int i = 0; i < init_params.size(); i++){
        opt.p[i] = init_params[i];
    }


    // set the initial parameters
    optimization_routine(opt, dots_cam0, dots_cam1, grid, num_img, lengths);
}

PYBIND11_MODULE(calibcpp, m) {
    m.def("stereo_calibration", &stereo_calibration, "stereo_calibration");
}



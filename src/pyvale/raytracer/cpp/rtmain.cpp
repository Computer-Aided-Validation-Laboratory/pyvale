#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Windows.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include "./Eigen/Dense"
#include <cstdint>

#include <fstream>
#include <iostream>
#include <array>
#include <iostream>
#include <vector>
#include "rteigentypes.h"
#include "rtrender.h"


namespace py = pybind11;

void render_scene(const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::vector<py::array_t<int>>& scene_connectivity,
    const std::vector<py::array_t<double>>& scene_coords,
    const std::vector<pybind11::array_t<double>>& scene_face_colors,
    const std::vector<Eigen::Ref<const EiVector3d>> camera_centers,
    const std::vector<Eigen::Ref<const EiVector3d>> pixel_00_centers,
    const std::vector<Eigen::Ref<const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>> matrix_pixel_spacings) {

    size_t num_cameras = camera_centers.size();

    // Iterate over all cameras and render an image for each
    for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
        Eigen::Ref<const EiVector3d> camera_center = camera_centers[camera_idx];
        Eigen::Ref<const EiVector3d> pixel_00_center = pixel_00_centers[camera_idx];
        Eigen::Ref<const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>> matrix_pixel_spacing = matrix_pixel_spacings[camera_idx];

        // Get bytes from the render function and pass back to Python to write it to a file from there
        //return render_ppm_image(test_camera, connectivity, node_coords);
        render_ppm_image(camera_center, pixel_00_center, matrix_pixel_spacing, scene_connectivity, scene_coords, scene_face_colors, image_height, image_width, number_of_samples);
    }
}

PYBIND11_MODULE(rtmaincpp, a) {
    a.def("cpp_render_scene", &render_scene);
}
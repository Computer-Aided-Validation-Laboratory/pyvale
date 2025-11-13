#include "rtrender.h"
#include "rthitrecord.h"
#include "rtrayintersection.h"
#include "rtmathutils.h"
#include <fstream>
#include <iostream>


inline EiVector3d get_face_color(Eigen::Index minRowIndex,
    const double* face_color_ptr) {
    double c1 = face_color_ptr[minRowIndex * 3];
    double c2 = face_color_ptr[minRowIndex * 3 + 1];
    double c3 = face_color_ptr[minRowIndex * 3 + 2];
    EiVector3d face_color;
    face_color << c1, c2, c3;
    return face_color;
}

EiVector3d return_ray_color(const Ray& ray,
    const std::vector < pybind11::array_t<int>>& scene_connectivity,
    const std::vector < pybind11::array_t<double>>& scene_coords,
    const std::vector<pybind11::array_t<double>>& scene_face_colors) {
    EiVectorD3d color_test(3, 3);
    color_test.row(0) << 1.0, 0.0, 0.0;
    color_test.row(1) << 0.0, 1.0, 1.0;
    color_test.row(2) << 1.0, 0.0, 1.0;
    HitRecord intersection_record; // Create HitRecord struct

    size_t num_meshes = scene_coords.size();

    for (size_t mesh_idx = 0; mesh_idx < num_meshes; ++mesh_idx) {
        pybind11::array_t<double> node_coords = scene_coords[mesh_idx];
        pybind11::array_t<int> connectivity = scene_connectivity[mesh_idx];
        pybind11::array_t<double> face_colors = scene_face_colors[mesh_idx];

        long long number_of_elements = connectivity.shape()[0]; // number of triangles/faces, will give us indices for some bits
        double* node_coords_ptr = static_cast<double*>(node_coords.request().ptr);
        int* connectivity_ptr = static_cast<int*>(connectivity.request().ptr);

        // Look for intersection
        IntersectionOutput intersection = intersect_plane(ray, connectivity_ptr, node_coords_ptr, number_of_elements);
        Eigen::Index minRowIndex, minColIndex;

        intersection.t_values.minCoeff(&minRowIndex, &minColIndex); // Find indices of the smallest t_value
        double closest_t = intersection.t_values(minRowIndex, minColIndex);
        if (closest_t < intersection_record.t) {
            intersection_record.t = closest_t;
            intersection_record.barycentric_coordinates = intersection.barycentric_coordinates.row(minRowIndex);
            intersection_record.point_intersection = ray_at_t(closest_t, ray);
            intersection_record.normal_surface = intersection.plane_normals.row(minRowIndex);
            // Get a pointer to the array storing face colors for the mesh if intersected
            double* face_colors_ptr = static_cast<double*>((scene_face_colors[mesh_idx]).request().ptr);
            intersection_record.face_color = get_face_color(minRowIndex, face_colors_ptr);
        }
    }
    if (intersection_record.t != std::numeric_limits<double>::infinity()) { // Instead of keeping a bool hit_anything, check if t value has changed from the default
        set_face_normal(ray, intersection_record.normal_surface);
        return intersection_record.barycentric_coordinates(0) * intersection_record.face_color + intersection_record.barycentric_coordinates(1) * intersection_record.face_color + intersection_record.barycentric_coordinates(2) * intersection_record.face_color;

    }
    // Blue sky gradient
    double a = 0.5 * (ray.direction(1) + 1.0);
    static EiVector3d white, blue;
    white << 1.0, 1.0, 1.0;
    blue << 0.5, 0.7, 1.0;
    return (1.0 - a) * white + a * blue;
}

void render_ppm_image(const Eigen::Ref<const EiVector3d>& camera_center,
    const Eigen::Ref<const EiVector3d>& pixel_00_center,
    const Eigen::Ref<const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>& matrix_pixel_spacing,
    const std::vector < pybind11::array_t<int>>& scene_connectivity,
    const std::vector < pybind11::array_t<double>>& scene_coords,
    const std::vector<pybind11::array_t<double>>& scene_face_colors,
    const int image_height,
    const int image_width,
    const int number_of_samples) {

    std::vector<uint8_t> buffer;
    buffer.reserve(image_width * image_height * 12); // Preallocate memory for the image buffer (conservatively)

    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            EiVector3d pixel_color = EiVector3d::Zero();
            for (int k = 0; k < number_of_samples; k++) {
                double offset[2] = { random_double() - 0.5, random_double() - 0.5 };
                EiVector3d pixel_sample = pixel_00_center +
                    (i + offset[0]) * matrix_pixel_spacing.row(0) +
                    (j + offset[1]) * matrix_pixel_spacing.row(1);
                EiVector3d ray_direction = pixel_sample - camera_center;
                Ray current_ray{ camera_center, ray_direction.normalized() };
                //pixel_color += return_ray_color(current_ray, connectivity, node_coords);
                pixel_color += return_ray_color(current_ray, scene_connectivity, scene_coords, scene_face_colors);
            }
            double gray = 0.2126 * pixel_color[0] + 0.7152 * pixel_color[1] + 0.0722 * pixel_color[2];
            int gray_byte = int(gray / number_of_samples * 255.99);
            buffer.push_back(static_cast<uint8_t>(gray_byte));
            buffer.push_back(static_cast<uint8_t>(gray_byte));
            buffer.push_back(static_cast<uint8_t>(gray_byte));
        }
    }

    std::ofstream image_file;

    // WIP: Will have to make the filename change based on the camera number or some unique identifier, otherwise we will keep on overwriting the same file
    image_file.open("test.ppm");
    if (!image_file.is_open()) {
        std::cerr << "Failed to open the output file.\n";
        return;
    }

    image_file << "P6\n" << image_width << ' ' << image_height << "\n255\n";
    image_file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());

    image_file.close();
    std::cout << "\r Done. \n";
    //return pybind11::bytes(buffer);
}
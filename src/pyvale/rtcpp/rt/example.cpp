// A script to set up a scene and get ray tracing, all within cpp

// #include <iostream>
// #include <Eigen/Dense>

// int main(int, char**) {
//     std::cout << "Hello world" << std::endl;
//     Eigen::Vector3d v = Eigen::Vector3d::Zero();
//     std::cout << v;
// }


#include <iostream>
#include <memory>
// #include "scene.hpp"
#include "materials/material.h"
// #include "shapes.hpp"
#include "geometry/shape.h"
#include "geometry/plane.h"
#include "geometry/sphere.h"
#include "geometry/hittable_list.h"
#include "camera.h"
// #include "render.cpp"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <sstream>

int main(int, char**) {
    Hittable_list scene;
    Hittable_list lights;

    // Materials
    auto red   = std::make_shared<Lambertian>(color(0.65, 0.05, 0.05));
    auto white = std::make_shared<Lambertian>(color(0.73, 0.73, 0.73));
    auto green = std::make_shared<Lambertian>(color(0.12, 0.45, 0.15));
    auto light = std::make_shared<Diffuse_light>(color(13, 13, 13));
    // auto glass = std::make_shared<Refractive>(1.5);

    // First quad
    Eigen::Matrix<double, 4, 3> nodes1;
    nodes1 << 200, 300, 100,
              250, 300, 200,
              250, 350, 200,
              200, 350, 200;
    Eigen::Matrix<double, 4, 3> displacements1 = Eigen::Matrix<double, 4, 3>::Zero();

    // scene.add(std::make_shared<ShapeQuadLin>(nodes1, displacements1, red));

    // Second quad (quadratic)
    Eigen::Matrix<double, 8, 3> nodes2;
    nodes2 << 200, 300, 200,
              250, 300, 200,
              250, 350, 200,
              200, 350, 200,
              225, 300, 200,
              250, 325, 200,
              225, 350, 200,
              200, 325, 200;
    Eigen::Matrix<double, 8, 3> displacements2 = Eigen::Matrix<double, 8, 3>::Zero();

    // scene.add(std::make_shared<ShapeQuadQuad>(nodes2, displacements2, green));

    // Walls and light
    // scene.add(std::make_shared<Plane_yz>(0, 555, 0, 555, 555, green));
    // scene.add(std::make_shared<Plane_yz>(0, 555, 0, 555, 0, red));
    // lights.add(std::make_shared<Plane_xz>(213, 343, 227, 332, 554, light)); // should make this importance sampled
    // scene.add(std::make_shared<Plane_xz>(0, 555, 0, 555, 0, white));
    // scene.add(std::make_shared<Plane_xz>(0, 555, 0, 555, 555, white));
    // scene.add(std::make_shared<Plane_xy>(0, 555, 0, 555, 555, white));
    // scene.add(std::make_shared<Sphere>(point3(278, 100, 250), 100, green));

    // Camera
    Camera camera = Camera(point3(278, 278, -800),
                            point3(278, 278, 0),
                            40.0,
                            200.0 / 200.0,
                            0.01,
                            10.0,
                            vec3(0.,1.,0.)
                    );

    std::ostringstream oss(std::ios::binary);
    camera.render(scene, lights, oss);
    std::string ppm_data = oss.str();

    // 2. Convert to vector<uchar> for imdecode
    std::vector<uchar> buffer(ppm_data.begin(), ppm_data.end());

    // 3. Decode PPM image to cv::Mat
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to decode PPM image.\n";
        return 1;
    }

    // 4. Display the image
    cv::imshow("PPM Image", image);
    cv::waitKey(0);

    return 0;

    // self.screen_width = screen_width
    // self.screen_height = screen_height

    // self.camera = Camera(lookfrom, lookat,field_of_view, screen_width/screen_height, aperture, focus_distance, vup)

    // scene.add_Camera(
    //     point3(278, 278, -800),
    //     point3(278, 278, 0),
    //     200, 200,
    //     40.0,
    //     10.0,
    //     0.01
    // );

    // Render
    // Image img = scene.render(20, 5); // samples per pixel, max depth
    // img.save("output.png");          // Save output to file
    std::cout << "Rendered to output.png" << std::endl;
                
    return 0;
}

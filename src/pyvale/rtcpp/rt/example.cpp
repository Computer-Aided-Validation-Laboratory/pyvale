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
#include "geometry/bvh.h"
#include "materials/material.h"
// #include "shapes.hpp"
#include "geometry/shape.h"
#include "geometry/plane.h"
#include "geometry/sphere.h"
#include "geometry/hittable_list.h"
#include "camera.h"
// #include "render.cpp"

#include <Eigen/Dense>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <chrono>

int main(int, char**) {
    Hittable_list scene;
    Hittable_list lights;

    // Materials
    auto red   = std::make_shared<Lambertian>(color(0.65, 0.05, 0.05));
    auto white = std::make_shared<Lambertian>(color(0.73, 0.73, 0.73));
    auto green = std::make_shared<Lambertian>(color(0.12, 0.45, 0.15));
    auto light = std::make_shared<Diffuse_light>(color(13, 13, 13));
    // auto glass = std::make_shared<Refractive>(1.5);


    Camera camera = Camera(point3(0, 2, 5),
                    point3(0, 2, 0),
                    40.0,
                    200.0 / 200.0,
                    0.01,
                    10.0,
                    vec3(0.,1.,0.),
                    600
    );

    // scene.add(std::make_shared<Sphere>(point3(0, 1, 0), 3, green));
    scene.add(std::make_shared<Sphere>(point3(0, -152, 0), 150, red));

    lights.add(std::make_shared<Quad>(point3(0, 5, 0), vec3(1,0,0), vec3(0,0,1), red));
                

    // First quad
    Eigen::Matrix<double, 4, 3> nodes1;
    nodes1 << -2, -2, 0,
              2, -2, 0,
              2, 2, 0,
              -2, 2, 0;
    Eigen::Matrix<double, 4, 3> displacements1 = Eigen::Matrix<double, 4, 3>::Zero();

    // scene.add(std::make_shared<ShapeQuadLin>(nodes1, displacements1, green));

    // Second quad (quadratic)
    Eigen::Matrix<double, 8, 3> nodes2;
    nodes2 << -2, -2, 0,
              2, -2, 0,
              2, 2, 0,
              -2, 2, 0,
              0, -2, 0,
              2, 0, 0,
              0, 2, 0,
              -2, 0, 0;
    Eigen::Matrix<double, 8, 3> displacements2 = Eigen::Matrix<double, 8, 3>::Zero();

    scene.add(std::make_shared<ShapeQuadQuad>(nodes2, displacements2, green));

    scene = Hittable_list(make_shared<BVH_node>(scene));

    // Walls and light
    // scene.add(std::make_shared<Plane_yz>(0, 555, 0, 555, 555, green));
    // scene.add(std::make_shared<Plane_yz>(0, 555, 0, 555, 0, red));
    // lights.add(std::make_shared<Plane_xz>(213, 343, 227, 332, 554, light)); // should make this importance sampled
    // scene.add(std::make_shared<Plane_xz>(0, 555, 0, 555, 0, white));
    // scene.add(std::make_shared<Plane_xz>(0, 555, 0, 555, 555, white));
    // scene.add(std::make_shared<Plane_xy>(0, 555, 0, 555, 555, white));
    // scene.add(std::make_shared<Sphere>(point3(278, 100, 250), 100, green));

    // Camera
    // Camera camera = Camera(point3(278, 278, -800),
    //                         point3(278, 278, 0),
    //                         40.0,
    //                         200.0 / 200.0,
    //                         0.01,
    //                         10.0,
    //                         vec3(0.,1.,0.)
    //                 );
    
    // Start time
    auto start = std::chrono::high_resolution_clock::now();

    std::ostringstream oss(std::ios::binary);
    camera.render(scene, lights, oss);
    std::string ppm_data = oss.str();

    // End time and show
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Render time: " << elapsed.count() << " seconds\n";


    // Decode PPM image to cv::Mat
    std::vector<uchar> buffer(ppm_data.begin(), ppm_data.end());
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to decode PPM image.\n";
        return 1;
    }

    // Display the image
    cv::namedWindow("image", cv::WINDOW_NORMAL); //namedWindow('image',WINDOW_NORMAL)
    cv::imshow("image", image);
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

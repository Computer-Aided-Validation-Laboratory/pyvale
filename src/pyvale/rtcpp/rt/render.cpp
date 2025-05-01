#include "geometry/hittable.h"
#include "util.h"
#include <math.h>
#include <memory>


#include "ray.h"

#include "geometry/aabb.h"
#include "geometry/hittable_list.h"
#include "geometry/bvh.h"

#include "pdf.h"
#include "camera.h"

#include "geometry/plane.h"
#include "geometry/sphere.h"
#include "geometry/box.h"
#include "geometry/shape.h"

struct Scene_info {
    color background;
    shared_ptr<Hittable_list> hittable_list;
    Camera& cam;
    shared_ptr<Hittable_list> importance_sampled_list;
};

#include "materials/material.h"



#include <iostream>
#include <thread>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#define MAX_THREADS 16

class Material;



color get_raycolor(const Ray& ray, const Scene_info& scene) {
    Hit_record hit;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (ray.depth <= 0)
        return color(0,0,0);

    // scene.hittable_list.hit passes hit as reference and store distance collison, uv data and surface normal orientation.
    // If the ray hits nothing, return the background color.
    if (!scene.hittable_list->hit(ray, 0.001, infinity, hit))
        return scene.background;


    return hit.material_ptr->get_color(ray, scene, hit);
}


// #include <Eigen/Dense>

// Eigen::MatrixXd add_matrices(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
//     return a + b;
// }





void render_pixel_chunck(int start_j, int final_j, double *r_ptr , double*g_ptr , double*b_ptr , 
                         const int image_height , const int image_width , const int samples_per_pixel, const int max_depth , const Scene_info& scene) {

    for (int j = start_j; j > final_j; --j) {
        //std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            color pixel_color;
            for (int s = 0; s < samples_per_pixel; ++s) {

                //antialiasing
                auto u = (i + random_double()) / ((double)image_width-1.);
                auto v = (j + random_double()) / ((double)image_height-1.);

                Ray ray = scene.cam.get_ray(u, v, max_depth);
                pixel_color += get_raycolor(ray, scene);
            }


            auto scale = 1.0 / samples_per_pixel;

            auto r = sqrt(scale * pixel_color.x());
            auto g = sqrt(scale * pixel_color.y());
            auto b = sqrt(scale * pixel_color.z());

            *r_ptr++ =  r;
            *g_ptr++ =  g;
            *b_ptr++ =  b;

        }
    }
}

void render_image(py::array_t<double>& r_arr, py::array_t<double>& g_arr, py::array_t<double>& b_arr, 
                const int samples_per_pixel, const int max_depth,  const Scene_info& scene) {
    double *r_ptr = (double *) r_arr.request().ptr,
           *g_ptr = (double *) g_arr.request().ptr,
           *b_ptr = (double *) b_arr.request().ptr;


    const int image_height = (const int)r_arr.shape(0),  image_width = (const int)r_arr.shape(1);
    const int processor_count = std::thread::hardware_concurrency();
    const int number_of_threads = 1;//(processor_count > MAX_THREADS) ? MAX_THREADS : processor_count;





    std::vector<std::thread> threads(number_of_threads);

    for (int i = 0 ; i < number_of_threads; ++i) {
        int pixel_chunck_height_start = image_height -1 - (int) (round(((float)image_height)*  ( ((float)(i))  /  ((float)(number_of_threads)))));
        //std::cout << pixel_chuncks_height_start[i]  <<"\n";
        int pixel_chunck_height_finish = image_height -1  - (int) (round(((float)image_height)*  ( ((float)(i+1))  /  ((float)(number_of_threads)))));
        //std::cout << pixel_chuncks_height_finish[i]  <<"\n";
        int pixel_chunck_ptr_start =  image_width * (int) (round(((float)image_height)*  ( ((float)(i))  /  ((float)(number_of_threads)))));
        //std::cout << pixel_chuncks_ptr_start[i]  <<"\n";
        threads[i]  = std::thread(render_pixel_chunck ,pixel_chunck_height_start  , pixel_chunck_height_finish ,   r_ptr + pixel_chunck_ptr_start,g_ptr + pixel_chunck_ptr_start, b_ptr + pixel_chunck_ptr_start,  
                                  image_height ,  image_width ,  samples_per_pixel, max_depth , std::ref(scene));
        
    }


    for (int i = 0 ; i < number_of_threads; ++i) {
        threads[i].join();
    }
}





PYBIND11_MODULE(render, m) {
    m.def("render_image", &render_image);

    py::class_<Hittable_list, shared_ptr<Hittable_list>>(m, "Hittable_list")
        .def(py::init<>())
        .def("add", &Hittable_list::add);

    py::class_<Scene_info>(m, "Scene_info")
        .def(py::init<color, shared_ptr<Hittable_list>, Camera&, shared_ptr<Hittable_list>>());




    py::class_<vec3>(m, "vec3")
        .def(py::init<double,double,double>())
        .def("length",&vec3::length)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self *= double())
        .def(py::self /= double())
        .def(double() * py::self)
        .def(py::self * double())
        .def(-py::self);



    py::class_<texture,shared_ptr<texture>>(m, "texture");


    py::class_<solid_color,texture,shared_ptr<solid_color>>(m, "solid_color")
        .def(py::init<vec3>())   
        .def(py::init<double,double,double>());   

    py::class_<checker_texture,texture,shared_ptr<checker_texture>>(m, "checker_texture")
        .def(py::init< shared_ptr<solid_color>, shared_ptr<solid_color> >());   








    py::class_<Material,shared_ptr<Material>>(m, "Material");


    py::class_<Refractive,Material,shared_ptr<Refractive>>(m, "Refractive")
        .def(py::init<double>());       

    py::class_<Diffuse,Material,shared_ptr<Diffuse>>(m, "Diffuse")
        .def(py::init<shared_ptr<texture>>());  
/*
    py::class_<metal,Material,shared_ptr<metal>>(m, "metal")
        .def(py::init<vec3, double>()); 
*/
    py::class_<Diffuse_light, Material ,shared_ptr<Diffuse_light>>(m, "Diffuse_light")
        .def(py::init<shared_ptr<texture>>());  








    py::class_<Hittable,shared_ptr<Hittable>>(m, "Hittable");

    py::class_<Sphere,Hittable,shared_ptr<Sphere>>(m, "Sphere")
        .def(py::init<point3, double, shared_ptr<Material>>());  

    py::class_<Box,Hittable,shared_ptr<Box>>(m, "Box")
        .def(py::init<point3, point3, shared_ptr<Material>>());   

    py::class_<Plane_xy,Hittable,shared_ptr<Plane_xy>>(m, "Plane_xy")
        .def(py::init<double, double, double, double, double, shared_ptr<Material>>(),
         py::arg("x0"), py::arg("x1"), py::arg("y0") ,py::arg("y1"),py::arg("z"), py::arg("Material"));   

    py::class_<Plane_xz,Hittable,shared_ptr<Plane_xz>>(m, "Plane_xz")
        .def(py::init<double, double, double, double, double, shared_ptr<Material>>(),
         py::arg("x0"), py::arg("x1"), py::arg("z0") ,py::arg("z1"),py::arg("y"), py::arg("Material")); 

    py::class_<Plane_yz,Hittable,shared_ptr<Plane_yz>>(m, "Plane_yz")
        .def(py::init<double, double, double, double, double, shared_ptr<Material>>(), 
         py::arg("y0"), py::arg("y1"), py::arg("z0") ,py::arg("z1"),py::arg("x"), py::arg("Material")); 

    py::class_<Quad,Hittable, shared_ptr<Quad>>(m, "Quad")
        .def(py::init<point3, vec3, vec3, shared_ptr<Material>>(),
         py::arg("Q"), py::arg("u"), py::arg("v"), py::arg("Material"));

    py::class_<Tri,Hittable,shared_ptr<Tri>>(m, "Tri")
         .def(py::init<point3, vec3, vec3, shared_ptr<Material>>(),
          py::arg("Q"), py::arg("u"), py::arg("v"), py::arg("Material"));
 

    py::class_<BVH_node,Hittable,shared_ptr<BVH_node>>(m, "BVH_node")
        .def(py::init<Hittable_list&, double, double>());  

    py::class_<Translate, Hittable, shared_ptr<Translate>>(m, "Translate")
        .def(py::init<shared_ptr<Hittable>, vec3>());
    py::class_<Rotate_y, Hittable, shared_ptr<Rotate_y>>(m, "Rotate_y")
        .def(py::init<shared_ptr<Hittable>, double>());


    py::class_<Camera>(m, "Camera")
        .def(py::init<vec3, vec3,
            double, 
            double, double, double, vec3>());

    py::class_<ShapeQuadLin,Hittable,shared_ptr<ShapeQuadLin>>(m, "ShapeQuadLin")
        .def(py::init<Eigen::Matrix<double, 4, 3>, Eigen::Matrix<double, 4, 3>, shared_ptr<Material>>());
    py::class_<ShapeQuadQuad,Hittable,shared_ptr<ShapeQuadQuad>>(m, "ShapeQuadQuad")
        .def(py::init<Eigen::Matrix<double, 8, 3>, Eigen::Matrix<double, 8, 3>, shared_ptr<Material>>());


}

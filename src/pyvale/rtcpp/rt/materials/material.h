#ifndef MATERIAL_H
#define MATERIAL_H

#include "util.h"
#include "textures/texture.h"
#include "ray.h"
#include "pdf.h"

struct Hit_record;




// color get_raycolor(const Ray& ray, const Scene_info& scene);



double schlick(double cosine, double n) {
    auto r0 = (1-n) / (1+n);//falta a�adir n de ambiente (aqui se supone = 1)
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}

class scatter_record {
    public:
      color attenuation;
      shared_ptr<PDF> pdf_ptr;
      bool skip_pdf;
      Ray skip_pdf_ray;
  };


class Material {
    public:
        // virtual color emitted(double u, double v, const point3& position) const {
        //     return color(0,0,0);
        // }

        // virtual color get_color(const Ray& ray, const Scene_info& scene , const Hit_record& hit) const = 0;


        virtual color emitted(
            const Ray& r_in, const Hit_record& rec, double u, double v, const point3& p
        ) const {
            return color(0,0,0);
        }
    
        virtual bool scatter(const Ray& r_in, const Hit_record& rec, scatter_record& srec) const {
            return false;
        }
    
        virtual double scattering_pdf(const Ray& r_in, const Hit_record& rec, const Ray& scattered)
        const {
            return 0;
        }

};

class Diffuse_light : public Material {
    public:
        Diffuse_light(shared_ptr<texture> tex) : tex(tex) {}
        Diffuse_light(const color& emit) : tex(make_shared<solid_color>(emit)) {}
  
        color emitted(const Ray& r_in, const Hit_record& rec, double u, double v, const point3& p)
        const override {
            if (!rec.orientation)
                return color(0,0,0);
            return tex->value(u, v, p);
        }
  
    private:
      shared_ptr<texture> tex;
  };


class Lambertian : public Material {
    public:
        Lambertian(const color& albedo) : tex(make_shared<solid_color>(albedo)) {}
        Lambertian(shared_ptr<texture> tex) : tex(tex) {}

        bool scatter(const Ray& r_in, const Hit_record& rec, scatter_record& srec) const override {
            srec.attenuation = tex->value(rec.u, rec.v, rec.position);
            srec.pdf_ptr = make_shared<Cosine_PDF>(rec.normal);
            srec.skip_pdf = false;
            return true;
        }

        double scattering_pdf(const Ray& r_in, const Hit_record& rec, const Ray& scattered)
        const override {
            auto cos_theta = dot(rec.normal, normalize(scattered.get_direction()));
            return cos_theta < 0 ? 0 : cos_theta/pi;
        }

    private:
        shared_ptr<texture> tex;
};
// class Refractive : public Material {
//     public:
//         Refractive(double n) : n(n) {}

//         virtual color get_color(const Ray& ray, const Scene_info& scene, const Hit_record& hit) const {

            
//             /*
//             if hit_orientation == UPWARDS:
//             #ray enter in the material
//                 if hit_orientation == UPDOWN:
//             #ray get out of the material
//             */

             
//             double n1_div_n2 = (hit.orientation == UPWARDS) ? (1.0 / n) : (n);

//             vec3 ray_direction = normalize(ray.get_direction());


//             double cos_theta = fmin(dot(-ray_direction, hit.normal), 1.0);
//             double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

//             // total internal reflexion
//             if (n1_div_n2 * sin_theta > 1.0 ) {
//                 vec3 reflected_ray = reflect(ray_direction, hit.normal);
//                 return get_raycolor(Ray(hit.position, reflected_ray, ray.get_depth() - 1, ray.get_time()), scene);
//             }

//             double reflect_prob = schlick(cos_theta, n1_div_n2);
//             if (random_double() < reflect_prob)
//             {
//                 vec3 reflected_ray = reflect(ray_direction, hit.normal);
//                 return get_raycolor(Ray(hit.position, reflected_ray, ray.get_depth() - 1, ray.get_time()), scene);
//             }

//             vec3 refracted_ray = refract(ray_direction, hit.normal, n1_div_n2);
//             return get_raycolor(Ray(hit.position, refracted_ray, ray.get_depth() - 1, ray.get_time()), scene);
//         }

//     public:
//         double n;
// };


// class Diffuse_light : public Material {
//     public:
//         Diffuse_light(shared_ptr<texture> a) : emit(a) {}

//         virtual color get_color(const Ray& ray, const Scene_info& scene , const Hit_record& hit) const {
//             return emit->value(hit.u, hit.v, hit.position);
//         }

//     public:
//         shared_ptr<texture> emit;
// };


// class Diffuse : public Material {
//     public:
//         Diffuse(shared_ptr<texture> a) : reflectance(a) {}


//         virtual color get_color(const Ray& ray, const Scene_info& scene , const Hit_record& hit) const {

//             //std::cout << random_int(0, 1) << " \n";
//             /*Cosine_PDF PDF(hit.normal);
            
//             vec3 scatter_direction = PDF.generate();
//             Ray scattered_ray = Ray(hit.position, scatter_direction, ray.depth - 1, ray.get_time());
//             double pdf_val = PDF.value(scatter_direction);
//             double NdotL = clamp(dot(hit.normal, scatter_direction), 0., 1.);
//             color radiance = get_raycolor(scattered_ray, scene);

//             return radiance *  reflectance->value(hit.u, hit.v, hit.position) * NdotL / (pdf_val * pi);
//             */

//             if (!scene.importance_sampled_list->objects.empty()) {

                
//                 Mixture_PDF pdf(make_shared<Hittable_PDF>(scene.importance_sampled_list, hit.position), make_shared<Cosine_PDF>(hit.normal));
//                 vec3 scatter_direction = pdf.generate();
//                 double pdf_val = pdf.value(scatter_direction);
//                 double NdotL = clamp(dot(hit.normal, scatter_direction), 0., 1.);
//                 Ray scattered_ray = Ray(hit.position + hit.normal * 0.001, scatter_direction, ray.depth - 1, ray.get_time());

//                 color radiance = get_raycolor(scattered_ray, scene);
//                 return radiance * reflectance->value(hit.u, hit.v, hit.position) * NdotL / (pdf_val * pi);
//             }
//             else{
//                 Cosine_PDF pdf(hit.normal);
//                 vec3 scatter_direction = pdf.generate();
//                 Ray scattered_ray = Ray(hit.position, scatter_direction, ray.depth - 1, ray.get_time());
//                 double pdf_val = pdf.value(scatter_direction);
//                 double NdotL = clamp(dot(hit.normal, scatter_direction), 0., 1.);
//                 color radiance = get_raycolor(scattered_ray, scene);

//                 return radiance * reflectance->value(hit.u, hit.v, hit.position) * NdotL / (pdf_val * pi);
//             }            
//         }


//     public:
//         shared_ptr<texture> reflectance;
// };

/*
class metal : public Material {
    public:
        metal(const color& a, double f) : reflectance(a), fuzz(f < 1 ? f : 1) {}

        virtual bool scatter(
            const Ray& ray, const Hit_record& hit, color& attenuation, Ray& scattered
        ) const {
            vec3 reflected_ray = reflect(normalize(ray.direction()), hit.normal);
            scattered = Ray(hit.position, reflected_ray + fuzz*random_in_unit_sphere(), ray.time());
            attenuation = reflectance;
            return (dot(scattered.direction(), hit.normal) > 0);
        }

    public:
        color reflectance;
        double fuzz;
};

*/
#endif

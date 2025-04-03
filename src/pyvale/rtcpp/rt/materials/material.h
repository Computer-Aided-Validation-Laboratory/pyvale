#ifndef MATERIAL_H
#define MATERIAL_H
#include "../util.h"
#include "../textures/texture.h"

struct Hit_record;




color get_raycolor(const Ray& ray, const Scene_info& scene);



double schlick(double cosine, double n) {
    auto r0 = (1-n) / (1+n);//falta añadir n de ambiente (aqui se supone = 1)
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}




class Material {
    public:
        virtual color emitted(double u, double v, const point3& position) const {
            return color(0,0,0);
        }

        virtual color get_color(const Ray& ray, const Scene_info& scene , const Hit_record& hit) const = 0;

};



class Refractive : public Material {
    public:
        Refractive(double n) : n(n) {}

        virtual color get_color(const Ray& ray, const Scene_info& scene, const Hit_record& hit) const {

            
            /*
            if hit_orientation == UPWARDS:
            #ray enter in the material
                if hit_orientation == UPDOWN:
            #ray get out of the material
            */

             
            double n1_div_n2 = (hit.orientation == UPWARDS) ? (1.0 / n) : (n);

            vec3 ray_direction = normalize(ray.get_direction());


            double cos_theta = fmin(dot(-ray_direction, hit.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

            // total internal reflexion
            if (n1_div_n2 * sin_theta > 1.0 ) {
                vec3 reflected_ray = reflect(ray_direction, hit.normal);
                return get_raycolor(Ray(hit.position, reflected_ray, ray.get_depth() - 1, ray.get_time()), scene);
            }

            double reflect_prob = schlick(cos_theta, n1_div_n2);
            if (random_double() < reflect_prob)
            {
                vec3 reflected_ray = reflect(ray_direction, hit.normal);
                return get_raycolor(Ray(hit.position, reflected_ray, ray.get_depth() - 1, ray.get_time()), scene);
            }

            vec3 refracted_ray = refract(ray_direction, hit.normal, n1_div_n2);
            return get_raycolor(Ray(hit.position, refracted_ray, ray.get_depth() - 1, ray.get_time()), scene);
        }

    public:
        double n;
};


class Diffuse_light : public Material {
    public:
        Diffuse_light(shared_ptr<texture> a) : emit(a) {}

        virtual color get_color(const Ray& ray, const Scene_info& scene , const Hit_record& hit) const {
            return emit->value(hit.u, hit.v, hit.position);
        }

    public:
        shared_ptr<texture> emit;
};


class Diffuse : public Material {
    public:
        Diffuse(shared_ptr<texture> a) : reflectance(a) {}


        virtual color get_color(const Ray& ray, const Scene_info& scene , const Hit_record& hit) const {

            //std::cout << random_int(0, 1) << " \n";
            /*Cosine_PDF PDF(hit.normal);
            
            vec3 scatter_direction = PDF.generate();
            Ray scattered_ray = Ray(hit.position, scatter_direction, ray.depth - 1, ray.get_time());
            double pdf_val = PDF.value(scatter_direction);
            double NdotL = clamp(dot(hit.normal, scatter_direction), 0., 1.);
            color radiance = get_raycolor(scattered_ray, scene);

            return radiance *  reflectance->value(hit.u, hit.v, hit.position) * NdotL / (pdf_val * pi);
            */

            if (!scene.importance_sampled_list->objects.empty()) {

                
                Mixture_PDF pdf(make_shared<Hittable_PDF>(scene.importance_sampled_list, hit.position), make_shared<Cosine_PDF>(hit.normal));
                vec3 scatter_direction = pdf.generate();
                double pdf_val = pdf.value(scatter_direction);
                double NdotL = clamp(dot(hit.normal, scatter_direction), 0., 1.);
                Ray scattered_ray = Ray(hit.position + hit.normal * 0.001, scatter_direction, ray.depth - 1, ray.get_time());

                color radiance = get_raycolor(scattered_ray, scene);
                return radiance * reflectance->value(hit.u, hit.v, hit.position) * NdotL / (pdf_val * pi);
                

                /*Cosine_PDF cosine_pdf(hit.normal);
                Hittable_PDF pdf(scene.importance_sampled_list, hit.position);
                vec3 scatter_direction;

                int r = random_int(0, 1);
                if (r == 1) {
                    scatter_direction = cosine_pdf.generate();

                }
                else {
                    scatter_direction = pdf.generate();

                }

                //double x0 = 213, x1= 343, z0 = 227, z1 = 332, k = 549.9999;


                //scatter_direction = pdf.generate();



                double pdf_val = 0.5*pdf.value(scatter_direction)  + 0.5 * cosine_pdf.value(scatter_direction);
                double NdotL = clamp(dot(hit.normal, scatter_direction), 0., 1.);
                Ray scattered_ray = Ray(hit.position + hit.normal * 0.001, scatter_direction, ray.depth - 1, ray.get_time());

                color radiance = get_raycolor(scattered_ray, scene);
                return radiance * reflectance->value(hit.u, hit.v, hit.position) * NdotL / (pdf_val * pi);*/

                


                /*Cosine_PDF cosine_PDF(hit.normal);

                vec3 scatter_direction;
                Ray scattered_ray;
                double pdf_val;

                double radius = 82.5;
                vec3 center = vec3(277, 555, 277);


                double target_distance = (center - hit.position).length();
                double cos_theta_max = sqrt(1 - clamp(radius * radius / (target_distance * target_distance), 0., 1.));
                
                int r = random_int(0, 1);
                if (r == 1) {
                    scatter_direction = scene.importance_sampled_list->objects[0]->random(hit.position);

                }
                else {
                    scatter_direction = cosine_PDF.generate();

                }

                if (dot(scatter_direction, normalize(center - hit.position)) > cos_theta_max)
                {
                    pdf_val = 0.5 * scene.importance_sampled_list->objects[0]->pdf_value(hit.position, scatter_direction);
                }
                else
                {
                    pdf_val = 0.5*cosine_PDF.value(scatter_direction);
                }

                
                double NdotL = clamp(dot(hit.normal, scatter_direction) , 0., 1.);
                
                scattered_ray = Ray(hit.position + hit.normal*0.001, scatter_direction, ray.depth - 1, ray.get_time());



                
                color radiance = get_raycolor(scattered_ray, scene);
                
                return radiance * reflectance->value(hit.u, hit.v, hit.position) * NdotL / (pdf_val * pi);*/
                //std::cout << NdotL << " \n";
                //std::cout << radiance.x() << " " << radiance.y() << " " << radiance.z() << " "  << " " << "\n";
            }
            else{
                Cosine_PDF pdf(hit.normal);
                vec3 scatter_direction = pdf.generate();
                Ray scattered_ray = Ray(hit.position, scatter_direction, ray.depth - 1, ray.get_time());
                double pdf_val = pdf.value(scatter_direction);
                double NdotL = clamp(dot(hit.normal, scatter_direction), 0., 1.);
                color radiance = get_raycolor(scattered_ray, scene);

                return radiance * reflectance->value(hit.u, hit.v, hit.position) * NdotL / (pdf_val * pi);
            }
            /*Cosine_PDF pdf(hit.normal);

            vec3 scatter_direction = pdf.generate();
            Ray scattered_ray = Ray(hit.position, scatter_direction, ray.depth - 1, ray.get_time());
            double pdf_val = pdf.value(scatter_direction);
            double NdotL = clamp(dot(hit.normal, scatter_direction), 0., 1.);
            color radiance = get_raycolor(scattered_ray, scene);
            
            return radiance *  reflectance->value(hit.u, hit.v, hit.position) * NdotL / (pdf_val * pi);*/
            
        }


    public:
        shared_ptr<texture> reflectance;
};

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

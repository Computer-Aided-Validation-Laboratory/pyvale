#ifndef PDF_H
#define PDF_H
#include "util.h"
#include "geometry/hittable.h"
#include "onb.h"



vec3 random_in_unit_disk() {
    while (true) {
        auto position = vec3(random_double(-1, 1), random_double(-1, 1), 0);
        if (position.length_squared() >= 1) continue;
        return position;
    }
}

vec3 random_unit_vector() {
    auto a = random_double(0, 2 * pi);
    auto z = random_double(-1, 1);
    auto r = sqrt(1 - z * z);
    return vec3(r * cos(a), r * sin(a), z);
}

vec3 random_in_unit_sphere() {
    while (true) {
        auto position = vec3::random(-1, 1);
        if (position.length_squared() >= 1) continue;
        return position;
    }
}

vec3 random_in_hemisphere(const vec3& normal) {
    vec3 in_unit_sphere = random_in_unit_sphere();
    if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

inline vec3 random_cosine_direction() {
    auto r1 = random_double();
    auto r2 = random_double();
    auto z = sqrt(1-r2);

    auto phi = 2*pi*r1;
    auto x = cos(phi)*sqrt(r2);
    auto y = sin(phi)*sqrt(r2);

    return vec3(x, y, z);
}


inline vec3 random_to_sphere(double radius, double distance_squared) {
    auto r1 = random_double();
    auto r2 = random_double();


    double cos_theta_max = sqrt(1 - clamp(radius * radius / (distance_squared * distance_squared), 0., 1.));


    double z = 1. + r2 * (cos_theta_max - 1.);

    auto phi = 2*pi*r1;
    auto x = cos(phi)*sqrt(1-z*z);
    auto y = sin(phi)*sqrt(1-z*z);

    return vec3(x, y, z);
}


class PDF  {
    public:
        virtual ~PDF() {}

        virtual double value(const vec3& direction) const = 0;
        virtual vec3 generate() const = 0;
};


class Cosine_PDF : public PDF {
    public:
        Cosine_PDF(const vec3& w) { uvw.build_from_w(w); }

        virtual double value(const vec3& direction) const {
            auto cosine = dot(normalize(direction), uvw.w());
            return (cosine <= 0) ? 0.001 : cosine/pi;
        }

        virtual vec3 generate() const  {
            return uvw.local(random_cosine_direction());
        }

    public:
        onb uvw;
};




class Mixture_PDF : public PDF {
    public:
        Mixture_PDF(shared_ptr<PDF> p0, shared_ptr<PDF> p1) {
            p[0] = p0;
            p[1] = p1;
        }

        virtual double value(const vec3& direction) const {
            return 0.5 * p[0]->value(direction) + 0.5 *p[1]->value(direction);
        }

        virtual vec3 generate() const {
            if (random_double() < 0.5)
                return p[0]->generate();
            else
                return p[1]->generate();
        }

    public:
        shared_ptr<PDF> p[2];
};

class Hittable_PDF : public PDF {
    public:
        Hittable_PDF(shared_ptr<Hittable> p, const point3& origin) : ptr(p), o(origin) {}

        virtual double value(const vec3& direction) const {
            return ptr->pdf_value(o, direction);
        }

        virtual vec3 generate() const {
            return ptr->random(o);
        }

    public:
        point3 o;
        shared_ptr<Hittable> ptr;
};



#endif

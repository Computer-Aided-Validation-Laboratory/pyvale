#ifndef TEXTURE_H
#define TEXTURE_H
#include "../util.h"
#include <iostream>


class texture  {
    public:
        virtual color value(double u, double v, const vec3& position) const = 0;
};


class solid_color : public texture {
    public:

        solid_color(color c) : color_value(c) {}

        solid_color(double red, double green, double blue) : solid_color(color(red,green,blue)) {}

        virtual color value(double u, double v, const vec3& position) const {
            return color_value;
        }

    private:
        color color_value;
};


class checker_texture : public texture {
    public:

        checker_texture(shared_ptr<texture> t0, shared_ptr<texture> t1): even(t0), odd(t1) {}

        virtual color value(double u, double v, const vec3& position) const {
            auto sines = sin(10*position.x())*sin(10*position.y())*sin(10*position.z());
            if (sines < 0)
                return odd->value(u, v, position);
            else
                return even->value(u, v, position);
        }

    public:
        shared_ptr<texture> odd;
        shared_ptr<texture> even;
};



#endif

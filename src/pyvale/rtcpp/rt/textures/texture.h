#ifndef TEXTURE_H
#define TEXTURE_H
#include "../util.h"
#include <iostream>
#include "textures/perlin.h"
#include "vec3.h"
#include <random>
#include <cmath>

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

class Noise_texture : public texture {
    public:
        Noise_texture(double scale) : scale(scale) {}

        color value(double u, double v, const point3& p) const override {
            Eigen::Vector3d pe;
            pe << p.x(), p.y(), p.z();
            return color(.5, .5, .5) * (1 + std::sin(scale * p.z() + 10 * noise.turb(pe, 7)));
        }

    private:
        Perlin noise;
        double scale;
};

// Dots of 3 pixels
class Dot_texture : public texture {
    public:
        // (int num_speckles=20, double min_radius=0.05, double max_radius=0.05, unsigned seed = 42) :
        Dot_texture(int grid_x=10, int grid_y=10, double min_radius=0.04, double max_radius=0.04,
                    double jitter_strength = 0.02, unsigned seed = 42)
            : gen(seed),
            jitter_dist(-jitter_strength, jitter_strength),
            radius_dist(min_radius, max_radius)
        {
            for (int i = 0; i < grid_y; ++i) {
                for (int j = 0; j < grid_x; ++j) {
                    // Uniform grid spacing
                    double u = (j + 0.5) / grid_x;
                    double v = (i + 0.5) / grid_y;

                    // Apply small random jitter
                    u += jitter_dist(gen);
                    v += jitter_dist(gen);

                    // Clamp to [0,1] in case of boundary overshoot
                    u = std::clamp(u, 0.0, 1.0);
                    v = std::clamp(v, 0.0, 1.0);

                    double radius = radius_dist(gen);
                    speckles.push_back({u, v, radius});
                }
            }
        }

        color value(double u, double v, const point3& p) const override {
            for (const auto& s : speckles) {
                double dx = u - s.x;
                double dy = v - s.y;
                if (dx * dx + dy * dy <= s.radius * s.radius) {
                    return spckle_color; // inside speckle
                }
            }
            return background_color; 
        }
    private:
        struct Speckle {
            double x, y, radius;
        };

        std::vector<Speckle> speckles;
        std::mt19937 gen;
        std::uniform_real_distribution<> jitter_dist;
        std::uniform_real_distribution<> radius_dist;

        color background_color = color(1.0, 1.0, 1.0);
        color spckle_color = color( 0.0, 0.0, 0.0);

};


#endif

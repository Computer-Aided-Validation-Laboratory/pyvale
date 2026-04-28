// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// ray tracer header files
#include "rtelemconstants.h"
#include "rtmathutils.h"
#include "rtcolorsampling.h"


namespace texsampler{

    // Pointer to the selected function
    EiVector3d (*sample_texture)(const Texture& texture, const EiArray2d& uvs);
    int lower_boundary, upper_boundary; // Boundaries for loops going over texel neighbourhoods. Not used now


    // Nearest neighbour method
    EiVector3d sample_greyscale_nearest_neighbour(const Texture& texture,
        const EiArray2d& uvs) {
        int height = texture.height;
        int width = texture.width;
        // Clip between 0.0 and 1.0
        double u = clip(uvs(0), 0.0, 1.0);
        double v = clip(uvs(1), 0.0, 1.0);

        // Get the nearest integer and cast to index into the texture array
        int ix = static_cast<int>(u * (width - 1)); //
        int iy = static_cast<int>((1.0 - v) * (height - 1)); // Flipped to have it match the image coordinates starting in top left
        double g = texture.data[iy * width + ix]; // Row-major offset to access indices [iy, ix] in a flattened array via a pointer
        EiVector3d output;
        output << g, g, g;

        return output;
    }

    // Functions for finding coefficients

    // Lanczos 2
    inline double kernel_lanczos2(const double x) {
        constexpr double tol_lanczos_centre_snap = 1e-5; // Placeholder value; in Lloyd's zig code this was tol.texture.lacsoz_centre_snap, so probably stored elsewhere
        constexpr int window_width = 2;
        const double abs_x = std::abs(x);
        if (abs_x < tol_lanczos_centre_snap) {
            return 1.0;
        }
        else if (abs_x >= window_width) {
            return 0.0;
        }
        const double pi_x = M_PI * x;
        const double pi_x_window = pi_x / static_cast<double>(window_width);
        return (std::sin(pi_x) / pi_x) * (std::sin(pi_x_window) / pi_x_window);
    }

    // Lanczos 3
     inline double kernel_lanczos3(const double x) {
        constexpr double tol_lanczos_centre_snap = 1e-5; // Placeholder value; in Lloyd's zig code this was tol.texture.lacsoz_centre_snap, so probably stored elsewhere
        constexpr int window_width = 3;
        const double abs_x = std::abs(x);
        if (abs_x < tol_lanczos_centre_snap) {
            return 1.0;
        }
        else if (abs_x >= window_width) {
            return 0.0;
        }
        const double pi_x = M_PI * x;
        const double pi_x_window = pi_x / static_cast<double>(window_width);
        return (std::sin(pi_x) / pi_x) * (std::sin(pi_x_window) / pi_x_window);
    }

    //Catmull-Rom - cubic coefficient
    inline double kernel_catmull_rom(const double x){
        const double abs_x = std::abs(x);
        if (abs_x <= 1.0) {
            return ((1.5 * abs_x - 2.5) * abs_x + 0.0) * abs_x + 1.0;
        } else if (abs_x < 2.0) {
            //return ((-0.5 * abs_x + 2.5) * abs_x - 4.0) * abs_x;
            return ((-0.5 * abs_x + 2.5) * abs_x - 4.0) * abs_x + 2.0; // End of envelope calculation tells me it should have 2.
        }
        return 0.0;
    }

    // Mitchell Netravali - cubic coefficient
    inline double kernel_mitchell_netravali(const double x) {
        const double abs_x = std::abs(x);
        const double B = 1.0 / 3.0;
        const double C = 1.0 / 3.0;
        if (abs_x < 1.0) {
            return ((12.0 - 9.0 * B - 6.0 * C) * abs_x * abs_x * abs_x +
                (-18.0 + 12.0 * B + 6.0 * C) * abs_x * abs_x +
                (6.0 - 2.0 * B)) / 6.0;
        }
        else if (abs_x < 2.0) {
            return ((-B - 6.0 * C) * abs_x * abs_x * abs_x +
                (6.0 * B + 30.0 * C) * abs_x * abs_x +
                (-12.0 * B - 48.0 * C) * abs_x +
                (8.0 * B + 24.0 * C)) / 6.0;
        }
        return 0.0;
    }

     //B-Spline - cubic coefficient
    inline double kernel_bspline(const double x){
        const double abs_x = std::abs(x);
        if (abs_x < 1.0) {
            return (3.0 * abs_x * abs_x * abs_x - 6.0 * abs_x * abs_x + 4.0) / 6.0;
        }
        else if (abs_x < 2.0) {
            const double t = 2.0 - abs_x;
            return t * t * t / 6.0; 
        }
        return 0.0;
    }

    // Quintic Spline
    inline double kernel_quintic_spline(const double x) {
        const double abs_x = std::abs(x);
        if (abs_x >= 3.0) {
            return 0.0;
        }
        if (abs_x <= 1.0) {
            return ((((-(1.0 / 12.0) * abs_x + (1.0 / 4.0)) * abs_x + 0.0) * abs_x - (1.0 / 2.0)) * abs_x + (11.0 / 20.0));
        }
        else if (abs_x <= 2.0){
            const double t = abs_x - 1.0;
            return (((((1.0 / 24.0) * t - (1.0 / 6.0)) * t + (1.0 / 6.0)) * t - (5.0 / 12.0)) * t + (13.0 / 60.0));
        }
        else {
            const double u = abs_x - 2.0;
            return (((((-(1.0 / 120.0) * u + (1.0 / 24.0)) * u - (1.0 / 12.0)) * u + (1.0 / 12.0)) * u - (1.0 / 24.0)) * u + (1.0 / 120.0));
        }
    }
    

    // Setter
    void set(TextureSampler sampler_type){
        switch (sampler_type){
            case TextureSampler::NEAREST_NEIGHBOUR:
                sample_texture = &sample_greyscale_nearest_neighbour;
                break;
        case::TextureSampler::LANCZOS_2:
        // Lanczos is defined as [-lanczos_window_width + 1, lanczos_window_width] so we have bounds [-1, 2]
                sample_texture = &sample_greyscale<kernel_lanczos2, -1, 2>;
                break;
        case::TextureSampler::LANCZOS_3:
                sample_texture = &sample_greyscale<kernel_lanczos3, -2, 3>;
                break;
        case::TextureSampler::CATMULL_ROM:
                sample_texture = &sample_greyscale<kernel_catmull_rom, -1, 2>;
                break;
        case::TextureSampler::MITCHELL_NETRAVALI:
                sample_texture = &sample_greyscale<kernel_mitchell_netravali, -1, 2>;
                break;
        case::TextureSampler::BSPLINE:
                sample_texture = &sample_greyscale<kernel_bspline, -1, 2>;
                break;
        case::TextureSampler::QUINTIC_SPLINE:
                sample_texture = &sample_greyscale<kernel_quintic_spline, -2, 3>;
                break;
        }
    }
}
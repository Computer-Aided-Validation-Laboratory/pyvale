// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// ray tracer header files
#include "rtelemconstants.h"
#include "rtmathutils.h"
#include "rtcolorsampling.h"

// TO DO: Rewrite this similarly to DIC interpolation module with a namespace and function pointer setter as having to replace the interpolators in every intersection function manually is a bit cumbersome and error-prone

// Nearest neighbour method for texture rendering tests
EiVector3d sample_texture_nearest_neighbour(const Texture& texture,
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

// C++ transcription of the cubic interpolants from Lloyd

// Lanczos 3
double lanczos_filter(const double x, const int window_width) {
    constexpr double tol_lanczos_centre_snap = 1e-5; // Placeholder value; in Lloyd's zig code this was tol.texture.lacsoz_centre_snap, so probably stored elsewhere
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

EiVector3d sample_texture_lanczos3(const Texture& texture,
    const EiArray2d& uvs) {
    constexpr int LANCZOS_WINDOW_WIDTH = 3; // Lanczos 3 => Sinc window width = 3, so the window extends between -3 and 3
    // Retrieve values
    int height = texture.height;
    int width = texture.width;
    double u = uvs(0);
    double v = uvs(1);
    // Texture grid coordinates based on (u,v) of where in the texture the ray hit
    double texel_x = u * width - 0.5;
    double texel_y = v * height - 0.5; 
    // Indices of the texel whose center is the closest to the texture hit point. Lanczos window is centered around those
    int center_x = static_cast<int>(std::round(texel_x));
    int center_y = static_cast<int>(std::round(texel_y));
    double color = 0.0; // Single value that will have to be stashed into EiVector3d at the end
    double total_weight = 0.0; // Sum of weights over all neighbours used for normalisation
    constexpr int starting_index = -LANCZOS_WINDOW_WIDTH + 1;
    // Loop over offsets in texel space from the central texel, so e.g., from -2 to 3 for window_width = 3 => window of 2 * window_width in each dimension
    int sample_x, sample_y;
    double sample_weight;
    for (int offset_y = starting_index; offset_y <= LANCZOS_WINDOW_WIDTH; ++offset_y){
        for (int offset_x = starting_index; offset_x  <= LANCZOS_WINDOW_WIDTH; ++offset_x){
            // Texel indices of the currently processed neighbour texel (found via center_texel + offset)
            sample_x = std::clamp(center_x + offset_x, 0, width  - 1);
            sample_y = std::clamp(center_y + offset_y, 0, height - 1);
            // Find the weight contributed by this neighbour texel to the final interpolated result
            sample_weight = lanczos_filter(texel_x - (center_x + offset_x), LANCZOS_WINDOW_WIDTH) * lanczos_filter(texel_y - (center_y + offset_y), LANCZOS_WINDOW_WIDTH);
            // Colour is a weighted colour sum
            color += sample_weight * texture.data[sample_y * width + sample_x];
            total_weight += sample_weight;
        }
    }
    // Normalise - important near edges
    double color_normalised = color / total_weight;
    EiVector3d output;
    output << color_normalised, color_normalised, color_normalised;

    return output;
}


//Catmull-Rom
double cubic_coefficient_CatmullRom(double x){
    const double abs_x = std::abs(x);
    if (abs_x <= 1.0) {
        return ((1.5 * abs_x - 2.5) * abs_x + 0.0) * abs_x + 1.0;
    } else if (abs_x < 2.0) {
        return ((-0.5 * abs_x + 2.5) * abs_x - 4.0) * abs_x;
    }
    return 0.0;
}

// Mitchell Netravali
double cubic_coefficient_MitchellNetravali(double x) {
    const double r = std::abs(x);
    const double B = 1.0 / 3.0;
    const double C = 1.0 / 3.0;
    if (r < 1.0) {
        return ((12.0 - 9.0 * B - 6.0 * C) * r * r * r +
            (-18.0 + 12.0 * B + 6.0 * C) * r * r +
            (6.0 - 2.0 * B)) / 6.0;
    }
    else if (r < 2.0) {
        return ((-B - 6.0 * C) * r * r * r +
            (6.0 * B + 30.0 * C) * r * r +
            (-12.0 * B - 48.0 * C) * r +
            (8.0 * B + 24.0 * C)) / 6.0;
    }
    return 0.0;
}

//B-Spline 
double cubic_coefficient_BSpline(double x){
    const double r = std::abs(x);
    if (r < 1.0) {
        return (3.0 * r * r * r - 6.0 * r * r + 4.0) / 6.0;
    }
    else if (r < 2.0) {
        const double t = 2.0 - r;
        return t * t * t / 6.0; 
    }
    return 0.0;
}


// Quintic Spline
double quintic_spline(double x) {
    const double r = std::abs(x);
    if (r >= 3.0) {
        return 0.0;
    }
    if (r <= 1.0) {
        return ((((-(1.0 / 12.0) * r + (1.0 / 4.0)) * r + 0.0) * r - (1.0 / 2.0)) * r + (11.0 / 20.0));
    }
    else if (r <= 2.0){
        const double t = r - 1.0;
        return (((((1.0 / 24.0) * t - (1.0 / 6.0)) * t + (1.0 / 6.0)) * t - (5.0 / 12.0)) * t + (13.0 / 60.0));
    }
    else {
        const double u = r - 2.0;
        return (((((-(1.0 / 120.0) * u + (1.0 / 24.0)) * u - (1.0 / 12.0)) * u + (1.0 / 12.0)) * u - (1.0 / 24.0)) * u + (1.0 / 120.0));
    }
 }
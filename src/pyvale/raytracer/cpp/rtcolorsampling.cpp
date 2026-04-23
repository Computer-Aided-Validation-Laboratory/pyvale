// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// ray tracer header files
#include "rtelemconstants.h"
#include "rtmathutils.h"
#include "rtcolorsampling.h"

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


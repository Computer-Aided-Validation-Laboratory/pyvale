// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef RTELEMCONST_H
#define RTELEMCONST_H

#include <cstdint>

// The below are specified, so it is always obvious where the indexing, loops, or array sizes come from
// Doesn't hurt performance, and it is very useful for working with flat arrays
constexpr int8_t NODE_COORDINATES = 3; // number of coordinates per each mesh node (x,y,z). Used for some of flat indexing
constexpr int8_t UV_COORDINATES = 2; // number of coordinates used for texturing (u,v). Used for some indexing

/**
 * @brief Enum storing the number of nodes per element.
 * 
 * This is to avoid hard-coding where possible and make it work with the Python data.
 * Unscoped (not enum class) because we want the implicit casts to integers for this one - it is used literally all over the codebase.
 */
enum ElementNodeCount : int8_t {
    TRI3 = 3,
    TRI6 = 6,
    QUAD4 = 4,
    QUAD8 = 8,
    QUAD9 = 9
};

/**
 * @brief Enum storing the shading type configuration.
 * 
 * Shading types:
 * FLAT - Use geometric normals for all elements.
 * BLENDED - Use shading normals: based on angle-averaged nodal coordinates for TRI3 and QUAD4, and re-calculated Jacobians for curved elements.
 * ANGLE_AVG_BLENDED - Use shading normals calculated from the angle-averaged nodal coordinates for ALL element types.
 * 
 * Ensure that these match the enum in Python. Integers used to avoid using strings in C-interface.
 * Scoped enum, so will not implicitly convert to int.
 */
 
enum class ShadingType : int {
    FLAT = 0,
    BLENDED = 1,
    ANGLE_AVG_BLENDED = 2
};

/**
 * @brief Enum with SurfaceType data. Can be either SOLID_COLOR or TEXTURE.
 * 
 * Ensure that these match the enum in Python. Integers used to avoid using strings in C-interface.
 */
enum class SurfaceType : int {
    SOLID_COLOR = 0, 
    TEXTURE = 1
};

/**
 * @brief Enum with MaterialType data.
 * 
 * Material types:
 * DIFFUSE - Matte/Lambertian material, e.g., wood.
 * SPECULAR - Material that reflects, e.g., polished metal, mirrors.
 * REFRACTIVE - Material that reflects and refracts, e.g., glass, certain plastics, water.
 * UNLIT - Material for which there are no shading effects applied. Default if not set.
 * 
 * Ensure that these match the enum in Python. Integers used to avoid using strings in C-interface.
 */
enum MaterialType : int { 
    DIFFUSE = 1, 
    SPECULAR = 2, 
    REFRACTIVE = 3,
    UNLIT = 4 
};

/**
 * @bried Enum specifying ObjectType data for refractive materials. Can be either SHELL or SOLID.
 * 
 * This choice will affect the refractive behaviour.
 * 
 * Ensure that these match the enum in Python. Integers used to avoid using strings in C-interface.
 */
enum class ObjectType : int {
    SOLID = 0,
    SHELL = 1
};

#endif // RTELEMCONST_H
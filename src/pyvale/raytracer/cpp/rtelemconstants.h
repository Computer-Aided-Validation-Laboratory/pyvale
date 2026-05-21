// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================
#pragma once
constexpr int NODE_COORDINATES = 3; // number of coordinates per each mesh node (x,y,z). Used for some of flat indexing
constexpr int UV_COORDINATES = 2; // number of coordinates used for texturing (u,v). Used for some indexing

// Enum storing the number of nodes per element, so we can avoid hard-coding where possible and make it work with the Python data
// Unscoped (not enum class) because we want the implicit casts to integers
enum ElementNodeCount {
    TRI3 = 3,
    TRI6 = 6,
    QUAD4 = 4,
    QUAD8 = 8,
    QUAD9 = 9
};


// Scoped enum, so will not implicitly convert to int
// Ensure that these match the enum in Python. Integers used to avoid using strings in C-interface
enum class ShadingType{
    FLAT = 0,
    BLENDED = 1,
    ANGLE_AVG_BLENDED = 2
};

enum class SurfaceType{
    SOLID_COLOR = 0, 
    TEXTURE = 1
};

enum MaterialType : int { 
    NOT_DEFINED = 0,
    DIFFUSE = 1, 
    SPECULAR = 2, 
    REFRACTIVE = 3,
    UNLIT = 4 
};

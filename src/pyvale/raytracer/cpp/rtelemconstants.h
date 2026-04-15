// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================
#pragma once
constexpr int NODE_COORDINATES = 3; // number of coordinates per each mesh node (x,y,z). Used for some of flat indexing

// Enum storing the number of nodes per element, so we can avoid hard-coding where possible and make it work with the Python data
// Unscoped (not enum class) because we want the implicit casts to integers
enum ElementNodeCount {
    TRI3 = 3,
    TRI6 = 6,
    QUAD4 = 4,
    QUAD8 = 8,
    QUAD9 = 9
};
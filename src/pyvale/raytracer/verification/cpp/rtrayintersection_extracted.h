// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once

// STD header files
#include <array>
#include <vector>

// ray tracer header files
#include "rteigentypes.h"
#include "rtray.h"
#include "rthitrecord.h"
// #include "rtbvh.h"
#include "rtelemconstants.h"
// #include "rtcolorsampling.h"
#include "rtshapefuncs.h"

// Struct size: 
// elem_interp_coords: MAX_ELEMENTS_PER_LEAF (currently = 4) x 3 (u,v,w per element) x 8 (double) = 96 bytes
// geometric_normals: MAX_ELEMENTS_PER_LEAF (currently = 4) x 3 (x,y,z per element) x 8 (double) = 96 bytes
// t_values: MAX_ELEMENTS_PER_LEAF (currently = 4) x 8 (double) = 32 bytes 
// Total: 224 bytes
struct IntersectionOutput {
    Eigen::ArrayXXd elem_interp_coords; // E.g., barycentric coordinates for TRI3, bilinear interpolation coords for QUAD4
    EiVectorD3d geometric_normals;
    Eigen::Array<double, Eigen::Dynamic, 1> t_values;
};

/* ********************************************** 
 * Curved elements: Single-element intersections
********************************************** */
// Return t at the intersection (or +infinity if none); also fill the geometric normal surface_normals_out and the parametric (xi, eta) at the hit.

double intersect_tri6(const Ray &ray,
    const std::array<EiVector3d, ElementNodeCount::TRI6> nodes,
    EiVector3d &surface_normals_out,
    Eigen::Vector2d &uv);

double intersect_quad8(const Ray& ray,
    const std::array<EiVector3d, ElementNodeCount::QUAD8>& nodes,
    EiVector3d& surface_normals_out,
    Eigen::Vector2d& xi_eta_out);

double intersect_quad9(const Ray& ray,
    const std::array<EiVector3d, ElementNodeCount::QUAD9>& nodes,
    EiVector3d& surface_normals_out,
    Eigen::Vector2d& xi_eta_out);
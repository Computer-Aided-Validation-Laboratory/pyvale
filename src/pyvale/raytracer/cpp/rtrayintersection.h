// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once

// STD header files
#include <array>

// ray tracer header files
#include "rteigentypes.h"
#include "rtray.h"
#include "rthitrecord.h"
#include "rtbvh.h"
#include "rtelemconstants.h"

inline EiVector3d get_face_color(Eigen::Index minRowIndex,
    std::vector<double>& face_color);

struct IntersectionOutput {
    Eigen::ArrayXXd barycentric_coordinates; // might be worth changing the name as these are not really barycentric coordinates for quads
    EiVectorD3d plane_normals;
    Eigen::Array<double, Eigen::Dynamic, 1> t_values;
};

EiVectorD3d cross_rowwise(const EiVectorD3d& mat1, const EiVectorD3d& mat2);

EiArrayD3d lerp_vectorised (const EiArrayD3d& points_A, const EiArrayD3d& points_B, const Eigen::Array<double, Eigen::Dynamic, 1> weights);

inline Eigen::Array<double, Eigen::Dynamic, 1> dot_rowwise (const EiArrayD3d& mat1, const EiArrayD3d& mat2);

IntersectionOutput intersect_bvh_triangles(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_triangle_count);

    /*
IntersectionOutput intersect_bvh_quads(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_quad_count);
*/
void intersect_BLAS(const Ray& ray,
    const BLAS& mesh_bvh,
    IntersectionOutput &out_intersection,
    HitRecord &intersection_record);

void intersect_TLAS(const Ray& ray,
    const TLAS& scene_TLAS,
    IntersectionOutput &out_intersection,
    HitRecord& out_intersection_record);
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
#include "rtcolorsampling.h"

struct IntersectionOutput {
    Eigen::ArrayXXd elem_interp_coords; // E.g., barycentric coordinates for TRI3, bilinear interpolation coords for QUAD4
    EiVectorD3d plane_normals;
    Eigen::Array<double, Eigen::Dynamic, 1> t_values;
};

/* ********************************************** 
 * Eigen algebra functions
********************************************** */

EiVectorD3d cross_rowwise(const EiVectorD3d& mat1, const EiVectorD3d& mat2);

//inline Eigen::Array<double, Eigen::Dynamic, 1> dot_rowwise (const EiArrayD3d& mat1, const EiArrayD3d& mat2);
// Had to be moved here to allow the templated intersect_bvh_quad to compile 
inline EiArrayD1d dot_rowwise (const EiArrayD3d& mat1, const EiArrayD3d& mat2){
    // Eigen should automatically convert EiVectorD3d to EiArrayD3d, so no need to do that while calling the function
    // These change just the object behaviour: arrays are for coefficient-wise operations, matrices for linear algebra. No data copying
    // However, if that breaks, just use e.g., mat1.array()
    return (mat1 * mat2).rowwise().sum();
}

EiArrayD3d lerp_vectorised (const EiArrayD3d& points_A, const EiArrayD3d& points_B, const Eigen::Array<double, Eigen::Dynamic, 1> weights);

/* ********************************************** 
 * Overwrite intersection output - textures
********************************************** */

void overwrite_intersection_quad4_tex(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx);

void overwrite_intersection_quad8_tex(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx);

void overwrite_intersection_quad9_tex(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx);

void overwrite_intersection_tri3_tex(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx);

void overwrite_intersection_tri6_tex(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx);

/* ********************************************** 
 * Overwrite intersection output - solid colour
********************************************** */

void overwrite_intersection_tri3_col(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx);

void overwrite_intersection_any_col(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx);

/* ********************************************** 
 * Ray-mesh element intersections
********************************************** */

IntersectionOutput intersect_bvh_tri3(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_triangle_count);
    
IntersectionOutput intersect_bvh_tri6(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_triangle_count);

template<QuadType element_node_count>
IntersectionOutput intersect_bvh_quad(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_quad_count);
    
/* ********************************************** 
 * Ray-acceleration structure intersections
********************************************** */

void intersect_BLAS(const Ray& ray,
    const BLAS& mesh_bvh,
    IntersectionOutput& out_intersection,
    HitRecord& intersection_record);

void intersect_TLAS(const Ray& ray,
    const TLAS& scene_TLAS,
    IntersectionOutput& out_intersection,
    HitRecord& out_intersection_record);

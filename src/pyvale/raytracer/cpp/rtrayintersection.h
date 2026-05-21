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
#include "rtbvh.h"
#include "rtelemconstants.h"
#include "rtcolorsampling.h"
#include "rtshapefuncs.h"

struct IntersectionOutput {
    Eigen::ArrayXXd elem_interp_coords; // E.g., barycentric coordinates for TRI3, bilinear interpolation coords for QUAD4
    EiVectorD3d geometric_normals;
    //EiVectorD3d shading_normals;
    Eigen::Array<double, Eigen::Dynamic, 1> t_values;
};

/* ********************************************** 
 * Overwrite intersection output - QUAD4
********************************************** */
// Heavily templated with constexpr to create compile-time variants of these functions depending on the shading and surface types, without introducing overhead
// or 20 different functions that each would need to be updated manually if something changed

template <ShadingType Mode, SurfaceType Surface>
void overwrite_intersection_quad4(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx) 
{
    // Interpolation coordinates from the ray-quad intersection; u,v in [0,1]
    const double u = intersection_record.elem_interp_coords(0);
    const double v = intersection_record.elem_interp_coords(1); 
    // Weights for bilinear interpolation
    const std::array<double, ElementNodeCount::QUAD4> shape_weights = compute_shape_quad4(u, v);
    
    // Get node normals if the shading type calls for them
    std::array<double, ElementNodeCount::QUAD4 * NODE_COORDINATES> node_normals;
    if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED) {
        get_face_data_array(min_row_idx, Node.node_normals, ElementNodeCount::QUAD4, &node_normals[0]);  // node_normals are shaped (nodes_per_element, 3)
    }
    EiArray3d shading_normal(0.0, 0.0, 0.0);

    // Write texture or color depending on the surface type
    if constexpr (Surface == SurfaceType::TEXTURE){
         // Find (u,v) coordinates for each node of the intersected element
        std::array<double, ElementNodeCount::QUAD4 * UV_COORDINATES> element_uvs; // Flat array so we can pass a pointer to get_face_uvs. Texture (u,v) for each node of the intersected mesh element
        get_face_uvs(min_row_idx, Node.face_color, ElementNodeCount::QUAD4, &element_uvs[0]); // element_uvs are shaped (nodes_per_element, 2) - one (u,v) pair for every element node
        EiArray2d uvs(0.0, 0.0);
        // Interpolate final (u,v) coordinates of this face
        for (int i = 0; i < ElementNodeCount::QUAD4; ++i) {
            EiArray2d node_uv;
            node_uv << element_uvs[UV_COORDINATES * i], element_uvs[UV_COORDINATES * i + 1];
            uvs += shape_weights[i] * node_uv;
            // The compiler evaluates this at compile-time and removes the check
            if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED) {
                EiArray3d node_normal;
                node_normal << node_normals[i * NODE_COORDINATES], node_normals[i * NODE_COORDINATES + 1], node_normals[i * NODE_COORDINATES + 2];
                shading_normal += shape_weights[i] * node_normal;
            }
        }
        // These uvs can be sent to sample the texture and the output returned to return_ray_color, regardless  of the element type down the line
        intersection_record.face_color = texsampler::sample_texture(texture, uvs);
    }
    else if constexpr (Surface == SurfaceType::SOLID_COLOR){
        const EiVector3d color_data = get_face_color(min_row_idx, Node.face_color); 
        EiVector3d interpolated_color(0.0, 0.0, 0.0);
        // Interpolate face colour using shape functions
        for (int i = 0; i < ElementNodeCount::QUAD4; ++i) {
            interpolated_color += shape_weights[i] * color_data;
            // The compiler evaluates this at compile-time and removes the check
            if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED) {
                EiArray3d node_normal;
                node_normal << node_normals[i * NODE_COORDINATES], node_normals[i * NODE_COORDINATES + 1], node_normals[i * NODE_COORDINATES + 2];
                shading_normal += shape_weights[i] * node_normal;
            }
        }
        intersection_record.face_color = interpolated_color;
    }
    // Save shading normal depending on the chosen type
    if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED) {
        intersection_record.normal_shading = shading_normal; // Save the shading normal from angle-averaged node normals found in the for loop above

    } else if constexpr (Mode == ShadingType::BLENDED) {
        // Blended for anything that is not TRI3 = shading normal is from Jacobians
        std::array<EiVector3d, ElementNodeCount::QUAD4> node_coords;
        get_face_data_vector(min_row_idx, Node.node_coords, ElementNodeCount::QUAD4, &node_coords[0]);
        Eigen::Matrix<double, 3, 2> jacobian = get_face_Jacobian_quad4(u, v, node_coords);
        intersection_record.normal_shading = (jacobian.col(0).cross(jacobian.col(1))).transpose();

    } else if constexpr (Mode == ShadingType::FLAT) {
        // Use geometric normal as the shading normal
        intersection_record.normal_shading = intersection_record.normal_surface;
    }
}

/* ********************************************** 
 * Overwrite intersection output - QUAD8
********************************************** */

template <ShadingType Mode, SurfaceType Surface>
void overwrite_intersection_quad8(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx) 
{
    // Interpolation coordinates from the ray-quad intersection; u,v in [0,1]
    const double u = intersection_record.elem_interp_coords(0);
    const double v = intersection_record.elem_interp_coords(1); 
    // Map from [0,1] (typical for (u,v)) to [-1, 1] (typical for FEM shape functions)
    const double xi = 2.0 * u - 1.0;
    const double eta = 2.0 * v - 1.0;
     // Shape functions (weights) for QUAD8
    std::array<double, ElementNodeCount::QUAD8> shape_weights = compute_shape_quad8(xi, eta);
    
    // Get node normals if the shading type calls for them
    std::array<double, ElementNodeCount::QUAD8 * NODE_COORDINATES> node_normals;
    if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED) {
        get_face_data_array(min_row_idx, Node.node_normals, ElementNodeCount::QUAD8, &node_normals[0]);  // node_normals are shaped (nodes_per_element, 3)
    }
    EiArray3d shading_normal(0.0, 0.0, 0.0);

    // Write texture or color depending on the surface type
    if constexpr (Surface == SurfaceType::TEXTURE){
         // Find (u,v) coordinates for each node of the intersected element
        std::array<double, ElementNodeCount::QUAD8 * UV_COORDINATES> element_uvs; // Flat array so we can pass a pointer to get_face_uvs. Texture (u,v) for each node of the intersected mesh element
        get_face_uvs(min_row_idx, Node.face_color, ElementNodeCount::QUAD8, &element_uvs[0]); // element_uvs are shaped (nodes_per_element, 2) - one (u,v) pair for every element node
        EiArray2d uvs(0.0, 0.0);
        // Interpolate final (u,v) coordinates of this face
        for (int i = 0; i < ElementNodeCount::QUAD8; ++i) {
            EiArray2d node_uv;
            node_uv << element_uvs[UV_COORDINATES * i], element_uvs[UV_COORDINATES * i + 1];
            uvs += shape_weights[i] * node_uv;
            // The compiler evaluates this at compile-time and removes the check
            if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED) {
                EiArray3d node_normal;
                node_normal << node_normals[i * NODE_COORDINATES], node_normals[i * NODE_COORDINATES + 1], node_normals[i * NODE_COORDINATES + 2];
                shading_normal += shape_weights[i] * node_normal;
            }
        }
        // These uvs can be sent to sample the texture and the output returned to return_ray_color, regardless  of the element type down the line
        intersection_record.face_color = texsampler::sample_texture(texture, uvs);
    }
    else if constexpr (Surface == SurfaceType::SOLID_COLOR){
        const EiVector3d color_data = get_face_color(min_row_idx, Node.face_color); 
        EiVector3d interpolated_color(0.0, 0.0, 0.0);
        // Interpolate face colour using shape functions
        for (int i = 0; i < ElementNodeCount::QUAD8; ++i) {
            interpolated_color += shape_weights[i] * color_data;
            // The compiler evaluates this at compile-time and removes the check
            if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED) {
                EiArray3d node_normal;
                node_normal << node_normals[i * NODE_COORDINATES], node_normals[i * NODE_COORDINATES + 1], node_normals[i * NODE_COORDINATES + 2];
                shading_normal += shape_weights[i] * node_normal;
            }
        }
        intersection_record.face_color = interpolated_color;
    }
    // Save shading normal depending on the chosen type
    if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED) {
        intersection_record.normal_shading = shading_normal; // Save the shading normal from angle-averaged node normals found in the for loop above

    } else if constexpr (Mode == ShadingType::BLENDED) {
        // Blended for anything that is not TRI3 = shading normal is from Jacobians
        std::array<EiVector3d, ElementNodeCount::QUAD8> node_coords;
        get_face_data_vector(min_row_idx, Node.node_coords, ElementNodeCount::QUAD8, &node_coords[0]);
        Eigen::Matrix<double, 3, 2> jacobian = get_face_Jacobian_quad8(xi, eta, node_coords);
        intersection_record.normal_shading = (jacobian.col(0).cross(jacobian.col(1))).transpose();

    } else if constexpr (Mode == ShadingType::FLAT) {
        // Use geometric normal as the shading normal
        intersection_record.normal_shading = intersection_record.normal_surface;
    }
}

/* ********************************************** 
 * Overwrite intersection output - QUAD9
********************************************** */

template <ShadingType Mode, SurfaceType Surface>
void overwrite_intersection_quad9(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx) 
{
    // Interpolation coordinates from the ray-quad intersection; u,v in [0,1]
    const double u = intersection_record.elem_interp_coords(0);
    const double v = intersection_record.elem_interp_coords(1); 
    // Map from [0,1] (typical for (u,v)) to [-1, 1] (typical for FEM shape functions)
    const double xi = 2.0 * u - 1.0;
    const double eta = 2.0 * v - 1.0;
     // Shape functions (weights) for QUAD8
    std::array<double, ElementNodeCount::QUAD9> shape_weights = compute_shape_quad9(xi, eta);
    
    // Get node normals if the shading type calls for them
    std::array<double, ElementNodeCount::QUAD9 * NODE_COORDINATES> node_normals;
    if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED) {
        get_face_data_array(min_row_idx, Node.node_normals, ElementNodeCount::QUAD9, &node_normals[0]);  // node_normals are shaped (nodes_per_element, 3)
    }
    EiArray3d shading_normal(0.0, 0.0, 0.0);

    // Write texture or color depending on the surface type
    if constexpr (Surface == SurfaceType::TEXTURE){
         // Find (u,v) coordinates for each node of the intersected element
        std::array<double, ElementNodeCount::QUAD9 * UV_COORDINATES> element_uvs; // Flat array so we can pass a pointer to get_face_uvs. Texture (u,v) for each node of the intersected mesh element
        get_face_uvs(min_row_idx, Node.face_color, ElementNodeCount::QUAD9, &element_uvs[0]); // element_uvs are shaped (nodes_per_element, 2) - one (u,v) pair for every element node
        EiArray2d uvs(0.0, 0.0);
        // Interpolate final (u,v) coordinates of this face
        for (int i = 0; i < ElementNodeCount::QUAD9; ++i) {
            EiArray2d node_uv;
            node_uv << element_uvs[UV_COORDINATES * i], element_uvs[UV_COORDINATES * i + 1];
            uvs += shape_weights[i] * node_uv;
            // The compiler evaluates this at compile-time and removes the check
            if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED) {
                EiArray3d node_normal;
                node_normal << node_normals[i * NODE_COORDINATES], node_normals[i * NODE_COORDINATES + 1], node_normals[i * NODE_COORDINATES + 2];
                shading_normal += shape_weights[i] * node_normal;
            }
        }
        // These uvs can be sent to sample the texture and the output returned to return_ray_color, regardless  of the element type down the line
        intersection_record.face_color = texsampler::sample_texture(texture, uvs);
    }
    else if constexpr (Surface == SurfaceType::SOLID_COLOR){
        const EiVector3d color_data = get_face_color(min_row_idx, Node.face_color); 
        EiVector3d interpolated_color(0.0, 0.0, 0.0);
        // Interpolate face colour using shape functions
        for (int i = 0; i < ElementNodeCount::QUAD9; ++i) {
            interpolated_color += shape_weights[i] * color_data;
            // The compiler evaluates this at compile-time and removes the check
            if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED) {
                EiArray3d node_normal;
                node_normal << node_normals[i * NODE_COORDINATES], node_normals[i * NODE_COORDINATES + 1], node_normals[i * NODE_COORDINATES + 2];
                shading_normal += shape_weights[i] * node_normal;
            }
        }
        intersection_record.face_color = interpolated_color;
    }

    // Save shading normal depending on the chosen type
    if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED) {
        intersection_record.normal_shading = shading_normal; // Save the shading normal from angle-averaged node normals found in the for loop above

    } else if constexpr (Mode == ShadingType::BLENDED) {
        // Blended for anything that is not TRI3 = shading normal is from Jacobians
        std::array<EiVector3d, ElementNodeCount::QUAD9> node_coords;
        get_face_data_vector(min_row_idx, Node.node_coords, ElementNodeCount::QUAD9, &node_coords[0]);
        Eigen::Matrix<double, 3, 2> jacobian = get_face_Jacobian_quad9(xi, eta, node_coords);
        intersection_record.normal_shading = (jacobian.col(0).cross(jacobian.col(1))).transpose();

    } else if constexpr (Mode == ShadingType::FLAT) {
        // Use geometric normal as the shading normal
        intersection_record.normal_shading = intersection_record.normal_surface;
    }
}

/* ********************************************** 
 * Overwrite intersection output - TRI3
********************************************** */

template <ShadingType Mode, SurfaceType Surface>
void overwrite_intersection_tri3(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx) 
{
    if constexpr (Surface == SurfaceType::TEXTURE){
        std::array<double, ElementNodeCount::TRI3 * UV_COORDINATES> element_uvs; // Shape (faces, 2) but flat. Texture (u,v) for each node of mesh element
        get_face_uvs(min_row_idx, Node.face_color, ElementNodeCount::TRI3, &element_uvs[0]); // element_uvs are shaped (nodes_per_element, 2) - one (u,v) pair for every element node
        EiArray2d uv0, uv1, uv2;
        uv0 << element_uvs[0], element_uvs[1]; // (u,v) for node 0
        uv1 << element_uvs[2], element_uvs[3]; // (u,v) for node 1
        uv2 << element_uvs[4], element_uvs[5]; // (u,v) for node 2 
        // Original arrangement that works if barycentric coordinates are stored as (w, u, v) - otherwise there is a mismatch and it does not render correctly
        EiArray2d uvs = intersection_record.elem_interp_coords(0) * uv0 + intersection_record.elem_interp_coords(1) * uv1 + intersection_record.elem_interp_coords(2) * uv2;
        // Barycentric interpolation that actually works if we want to store barycentric coordinates as (u, v, w)
        //EiArray2d uvs = intersection_record.elem_interp_coords(2) * uv0 + intersection_record.elem_interp_coords(0) * uv1 + intersection_record.elem_interp_coords(1) * uv2;  // Final (u,v)
        intersection_record.face_color = texsampler::sample_texture(texture, uvs);
    }
    else if constexpr (Surface == SurfaceType::SOLID_COLOR){
        const EiVector3d color_data = get_face_color(min_row_idx, Node.face_color); 
        // Barycentric interpolation
        intersection_record.face_color = intersection_record.elem_interp_coords(0) * color_data + intersection_record.elem_interp_coords(1) * color_data + intersection_record.elem_interp_coords(2) * color_data;
    }

    // Save shading normal depending on the chosen type
    // For TRI3 we do not have Jacobians, so blended = angle averaged blended shading based on node normals
    if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED || Mode == ShadingType::BLENDED){
         std::array<double, ElementNodeCount::TRI3 * NODE_COORDINATES> node_normals; // Shape (faces, 2) but flat. Texture (u,v) for each node of mesh element
        get_face_data_array(min_row_idx, Node.node_normals, ElementNodeCount::TRI3, &node_normals[0]); // element_uvs are shaped (nodes_per_element, 2) - one (u,v) pair for every element node
        EiArray3d normal_0, normal_1, normal_2;
        normal_0 << node_normals[0], node_normals[1], node_normals[2];
        normal_1 << node_normals[3], node_normals[4], node_normals[5];
        normal_2 << node_normals[6], node_normals[7], node_normals[8];
        // Original that should work if we store barycentric coordinates as (w, u, v)
        EiVector3d shading_normal = intersection_record.elem_interp_coords(0) * normal_0 + intersection_record.elem_interp_coords(1) * normal_1 + intersection_record.elem_interp_coords(2) * normal_2;
        intersection_record.normal_shading = shading_normal;
    }
    else if constexpr (Mode == ShadingType::FLAT){ // Geometric/flat shading
        intersection_record.normal_shading = intersection_record.normal_surface;
    }
}

/* ********************************************** 
 * Overwrite intersection output - TRI6
********************************************** */

template <ShadingType Mode, SurfaceType Surface>
void overwrite_intersection_tri6(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx){

    // Get solid surface color
    EiVector3d color_data = get_face_color(min_row_idx, Node.face_color); 

     // Get barycentric coordinates (g, h)
    const double g = intersection_record.elem_interp_coords(0);
    const double h = intersection_record.elem_interp_coords(1);
    const double ray = intersection_record.elem_interp_coords(2);
    // Compute quadratic shape functions
    Eigen::VectorXd N = compute_shape_tri6(g, h); // size = 6

    // Get node normals if the shading type calls for them
    std::array<double, ElementNodeCount::TRI6 * NODE_COORDINATES> node_normals; // Shape (faces, 3) but flat
    if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED){
        get_face_data_array(min_row_idx, Node.node_normals, ElementNodeCount::TRI6, &node_normals[0]); // element_uvs are shaped (nodes_per_element, 3)    
    }
    EiArray3d shading_normal(0.0, 0.0, 0.0);
    
    if constexpr (Surface == SurfaceType::TEXTURE){
        std::array<double, ElementNodeCount::TRI6 * UV_COORDINATES> element_uvs;
        get_face_uvs(min_row_idx, Node.face_color, ElementNodeCount::TRI6, &element_uvs[0]);

        // Extract UVs for each node
        EiArray2d uv0, uv1, uv2, uv3, uv4, uv5;

        uv0 << element_uvs[0],  element_uvs[1];
        uv1 << element_uvs[2],  element_uvs[3];
        uv2 << element_uvs[4],  element_uvs[5];
        uv3 << element_uvs[6],  element_uvs[7];
        uv4 << element_uvs[8],  element_uvs[9];
        uv5 << element_uvs[10], element_uvs[11];

        uv3 = 0.5 * (uv0 + uv1); // edge 0-1
        uv4 = 0.5 * (uv1 + uv2); // edge 1-2
        uv5 = 0.5 * (uv2 + uv0); // edge 2-0
        // Interpolate UVs using shape functions
        EiArray2d uvs =
          N(0) * uv0
        + N(1) * uv1
        + N(2) * uv2
        + N(3) * uv3
        + N(4) * uv4
        + N(5) * uv5;
         // uvs = g*uv0 + h*uv1 + ray*uv2;

        // uvs = intersection_record.elem_interp_coords(2) * uv0 + intersection_record.elem_interp_coords(0) * uv1 + intersection_record.elem_interp_coords(1) * uv2;
        intersection_record.face_color = texsampler::sample_texture(texture, uvs);
        if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED){
            for(int i = 0; i < ElementNodeCount::TRI6; i++){
            double shape_weight = N(i);
            EiArray3d node_normal;
            node_normal << node_normals[i * NODE_COORDINATES], node_normals[i * NODE_COORDINATES + 1], node_normals[i * NODE_COORDINATES + 2];
            shading_normal += shape_weight * node_normal;
        }
        }
    }
    else if (Surface == SurfaceType::SOLID_COLOR){
        EiVector3d interpolated_color(0.0, 0.0, 0.0);
        for(int i = 0; i < ElementNodeCount::TRI6; i++){
            double shape_weight = N(i);
            interpolated_color += shape_weight * color_data;
            if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED){
                EiArray3d node_normal;
                node_normal << node_normals[i * NODE_COORDINATES], node_normals[i * NODE_COORDINATES + 1], node_normals[i * NODE_COORDINATES + 2];
                shading_normal += shape_weight * node_normal;
            }
        }
    intersection_record.face_color = interpolated_color;
    }

    // Save shading normal depending on the chosen type
    if constexpr (Mode == ShadingType::ANGLE_AVG_BLENDED) {
        intersection_record.normal_shading = shading_normal.matrix(); // Save the shading normal from angle-averaged node normals found in the for loop above

    } else if constexpr (Mode == ShadingType::BLENDED) {
        // Blended for anything that is not TRI3 = shading normal is from Jacobians
        std::array<EiVector3d, ElementNodeCount::TRI6> node_coords;
        get_face_data_vector(min_row_idx, Node.node_coords, ElementNodeCount::TRI6, &node_coords[0]);
        Eigen::Matrix<double, 3, 2> jacobian = get_face_Jacobian_tri6(g, h, node_coords);

    } else if constexpr (Mode == ShadingType::FLAT) {
        // Use geometric normal as the shading normal
        intersection_record.normal_shading = intersection_record.normal_surface;
    }
}

/* ********************************************** 
 * Ray-mesh element intersections
********************************************** */

IntersectionOutput intersect_bvh_tri3(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_triangle_count);

IntersectionOutput intersect_bvh_quad4(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_quad_count);

// For curved elements, these are leaf entry points - i.e., not directly ray-element intersection calculators

IntersectionOutput intersect_bvh_quad8(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_quad_count);

IntersectionOutput intersect_bvh_quad9(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_quad_count);

IntersectionOutput intersect_bvh_tri6(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_triangle_count);
    
/* ********************************************** 
 * Curved elements: Single-element intersections
********************************************** */
// Return t at the intersection (or +infinity if none); also fill the geometric normal surface_normals_out and the parametric (xi, eta) at the hit.

double intersect_quad8(const Ray& ray,
    const std::array<EiVector3d, ElementNodeCount::QUAD8>& nodes,
    EiVector3d& surface_normals_out,
    Eigen::Vector2d& xi_eta_out);

double intersect_quad9(const Ray& ray,
    const std::array<EiVector3d, ElementNodeCount::QUAD9>& nodes,
    EiVector3d& surface_normals_out,
    Eigen::Vector2d& xi_eta_out);

double intersect_tri6(const Ray &ray,
    const std::array<EiVector3d, ElementNodeCount::TRI6> nodes,
    EiVector3d &surface_normals_out,
    Eigen::Vector2d &uv);
    
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
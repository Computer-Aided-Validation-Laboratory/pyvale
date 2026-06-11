// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef RTRAYINTERSECT_H
#define RTRAYINTERSECT_H

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

// ================================================================================
// Intersection output structure
// ================================================================================

/**
 * @brief Aggregated output of ray–element intersection tests within a BVH node.
 *
 * Stores interpolation coordinates on the element (e.g. barycentric, bilinear,
 * or parametric coordinates), geometric normals, and ray parameters t for all
 * elements in a BVH node.
 */
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

// ================================================================================
// Overwrite intersection output - QUAD4
// ================================================================================
// Heavily templated with constexpr to create compile-time variants of these functions depending on the shading and surface types,
// without introducing overhead or 20 different functions that each would need to be updated manually if something changed

/**
 * @brief Writes QUAD4 intersection data into a hit record.
 *
 * Uses the stored interpolation coordinates to:
 *  - Compute shape-function weights for QUAD4,
 *  - Interpolate texture UVs or per-face color depending on surface type,
 *  - Compute shading normals based on shading mode (flat, blended, angle-averaged).
 *
 * @tparam Surface (SurfaceType) Surface representation (texture or solid color)
 * @tparam Mode (ShadingType) Shading mode (flat, blended, angle-averaged blended)
 *
 * @param[in,out] intersection_record (HitRecord&) Hit record to update
 * @param[in] Node (const BLAS_Node&) BLAS node containing nodal data for the element
 * @param[in] texture (const Texture&) Texture descriptor, used if Surface=TEXTURE
 * @param[in] min_row_idx (Eigen::Index) Index of the element within the BLAS node
 */
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
    const std::array<double, ElementNodeCount::QUAD4> shape_weights = shapefuncs::compute_shape_quad4(u, v);
    
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
        Eigen::Matrix<double, 3, 2> jacobian = shapefuncs::get_face_Jacobian_quad4(u, v, node_coords);
        intersection_record.normal_shading = (jacobian.col(0).cross(jacobian.col(1))).transpose();

    } else if constexpr (Mode == ShadingType::FLAT) {
        // Use geometric normal as the shading normal
        intersection_record.normal_shading = intersection_record.normal_surface;
    }
}

// ================================================================================
// Overwrite intersection output - QUAD8
// ================================================================================

/**
 * @brief Writes QUAD8 intersection data into a hit record.
 *
 * Maps the stored (u,v) interpolation coordinates from [0,1] to the FEM
 * parametric domain [-1,1], evaluates QUAD8 shape functions, and then:
 *  - Interpolates UVs or solid colors based on surface type,
 *  - Computes shading normals using either nodal normals or Jacobians
 *    depending on the shading mode.
 *
 * @tparam Surface (SurfaceType) Surface representation (texture or solid color)
 * @tparam Mode (ShadingType) Shading mode (flat, blended, angle-averaged blended)
 *
 * @param[in,out] intersection_record (HitRecord&) Hit record to update
 * @param[in] Node (const BLAS_Node&) BLAS node containing nodal data for the element
 * @param[in] texture (const Texture&) Texture descriptor, used if Surface=TEXTURE
 * @param[in] min_row_idx (Eigen::Index) Index of the element within the BLAS node
 */
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
    std::array<double, ElementNodeCount::QUAD8> shape_weights = shapefuncs::compute_shape_quad8(xi, eta);
    
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
        Eigen::Matrix<double, 3, 2> jacobian = shapefuncs::get_face_Jacobian_quad8(xi, eta, node_coords);
        intersection_record.normal_shading = (jacobian.col(0).cross(jacobian.col(1))).transpose();

    } else if constexpr (Mode == ShadingType::FLAT) {
        // Use geometric normal as the shading normal
        intersection_record.normal_shading = intersection_record.normal_surface;
    }
}

// ================================================================================
//Overwrite intersection output - QUAD9
// ================================================================================

/**
 * @brief Writes QUAD9 intersection data into a hit record.
 *
 * Maps the stored (u,v) interpolation coordinates from [0,1] to the FEM
 * parametric domain [-1,1], evaluates QUAD8 shape functions, and then:
 *  - interpolates UVs or solid colors based on surface type,
 *  - computes shading normals using either nodal normals or Jacobians
 *    depending on the shading mode.
 *
 * @tparam Surface (SurfaceType) Surface representation (texture or solid color)
 * @tparam Mode (ShadingType) Shading mode (flat, blended, angle-averaged blended)
 *
 * @param[in,out] intersection_record (HitRecord&) Hit record to update.
 * @param[in] Node (const BLAS_Node&) BLAS node containing nodal data for the element.
 * @param[in] texture (const Texture&) Texture descriptor, used if Surface=TEXTURE.
 * @param[in] min_row_idx (Eigen::Index) Index of the element within the BLAS node.
 */
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
    std::array<double, ElementNodeCount::QUAD9> shape_weights = shapefuncs::compute_shape_quad9(xi, eta);
    
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
        Eigen::Matrix<double, 3, 2> jacobian = shapefuncs::get_face_Jacobian_quad9(xi, eta, node_coords);
        intersection_record.normal_shading = (jacobian.col(0).cross(jacobian.col(1))).transpose();

    } else if constexpr (Mode == ShadingType::FLAT) {
        // Use geometric normal as the shading normal
        intersection_record.normal_shading = intersection_record.normal_surface;
    }
}

// ================================================================================
// Overwrite intersection output - TRI3
// ================================================================================

/**
 * @brief Writes TRI3 intersection data into a hit record.
 *
 * Uses barycentric coordinates stored in the hit record to:
 *  - interpolate UVs for textured surfaces, or
 *  - interpolate solid colors for per-face fields,
 *  - compute shading normals using either nodal normals (blended modes)
 *    or geometric normals (flat shading).
 *
 * @tparam Surface (SurfaceType) Surface representation (texture or solid color)
 * @tparam Mode (ShadingType) Shading mode (flat, blended, angle-averaged blended)
 *
 * @param[in,out] intersection_record (HitRecord&) Hit record to update
 * @param[in] Node (const BLAS_Node&) BLAS node containing nodal data for the element
 * @param[in] texture (const Texture&) Texture descriptor, used if Surface=TEXTURE
 * @param[in] min_row_idx (Eigen::Index) Index of the element within the BLAS node
 */
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

// ================================================================================
// Overwrite intersection output - TRI6
// ================================================================================

/**
 * @brief Writes TRI6 intersection data into a hit record.
 *
 * Uses quadratic shape functions for TRI6 to:
 *  - interpolate per-node UVs or solid colors,
 *  - compute shading normals either from nodal normals (angle-averaged)
 *    or from the Jacobian (blended), or use geometric normals (flat).
 *
 * @tparam Surface (SurfaceType) Surface representation (texture or solid color)
 * @tparam Mode (ShadingType) Shading mode (flat, blended, angle-averaged blended)
 *
 * @param[in,out] intersection_record (HitRecord&) Hit record to update
 * @param[in] Node (const BLAS_Node&) BLAS node containing nodal data for the element
 * @param[in] texture (const Texture&) Texture descriptor, used if Surface=TEXTURE
 * @param[in] min_row_idx (Eigen::Index) Index of the element within the BLAS node
 */
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
    Eigen::VectorXd N = shapefuncs::compute_shape_tri6(g, h); // size = 6

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
        Eigen::Matrix<double, 3, 2> jacobian = shapefuncs::get_face_Jacobian_tri6(g, h, node_coords);
        intersection_record.normal_shading = (jacobian.col(0).cross(jacobian.col(1))).transpose();

    } else if constexpr (Mode == ShadingType::FLAT) {
        // Use geometric normal as the shading normal
        intersection_record.normal_shading = intersection_record.normal_surface;
    }
}

// ================================================================================
// Ray-mesh element intersections
// ================================================================================
// For curved elements, these are leaf entry points - i.e., not directly ray-element
// intersection calculators - these are stored in the next section

/**
 * @brief Intersects a ray with TRI3 elements stored in a BVH node.
 *
 * Vectorised implementation of the Möller–Trumbore algorithm for all TRI3
 * elements in a node. Returns interpolation coordinates (barycentrics),
 * geometric normals, and t values for each element.
 *
 * @param[in] ray (const Ray&) Ray to intersect with the triangles
 * @param[in] node_coords (const std::vector<double>&) Packed nodal coordinates
 *            for all TRI3 elements in this node
 * @param[in] bvh_node_triangle_count (unsigned int) Number of TRI3 elements in this node
 *
 * @return (IntersectionOutput) Intersection data for all triangles in the node.
 */
IntersectionOutput intersect_bvh_tri3(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_triangle_count);

/**
 * @brief Intersects a ray with QUAD4 elements stored in a BVH node.
 *
 * Uses the bilinear patch intersection algorithm for general (possibly
 * non-planar) quads to compute (u,v) coordinates, geometric normals and
 * t values for each QUAD4 element.
 *
 * @param[in] ray (const Ray&) Ray to intersect with the quads
 * @param[in] node_coords (const std::vector<double>&) Packed nodal coordinates
 *            for all QUAD4 elements in this node
 * @param[in] bvh_node_quad_count (unsigned int) Number of QUAD4 elements in this node
 *
 * @return (IntersectionOutput) Intersection data for all quads in the node.
 */
IntersectionOutput intersect_bvh_quad4(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_quad_count);

/**
 * @brief Intersects a ray with QUAD8 elements stored in a BVH node.
 *
 * For curved QUAD8 elements, this function is a leaf entry point:
 *  - Evaluates per-element intersection (via Newton-based solver),
 *  - Returns per-element parametric coordinates and geometric normals.
 *
 * @param[in] ray (const Ray&) Ray to intersect with the elements
 * @param[in] node_coords (const std::vector<double>&) Packed nodal coordinates
 *            for all QUAD8 elements in this node
 * @param[in] bvh_node_quad_count (unsigned int) Number of QUAD8 elements in this node
 *
 * @return (IntersectionOutput) Intersection data for all QUAD8 elements in the node.
 */
IntersectionOutput intersect_bvh_quad8(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_quad_count);
    
/**
 * @brief Intersects a ray with QUAD9 elements stored in a BVH node.
 *
 * For curved QUAD9 elements, this function:
 *  - Runs sub-triangle Möller–Trumbore seeding,
 *  - Refines the solution with a damped Newton solver on the quadratic patch,
 *  - Returns parametric (xi, eta), geometric normals and t values.
 *
 * @param[in] ray (const Ray&) Ray to intersect with the elements
 * @param[in] node_coords (const std::vector<double>&) Packed nodal coordinates
 *            for all QUAD9 elements in this node
 * @param[in] bvh_node_quad_count (unsigned int) Number of QUAD9 elements in this node
 *
 * @return (IntersectionOutput) Intersection data for all QUAD9 elements in the node.
 */
IntersectionOutput intersect_bvh_quad9(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_quad_count);

/**
 * @brief Intersects a ray with TRI6 elements stored in a BVH node.
 *
 * Uses a combination of sub-triangle linear intersections and a Newton
 * solver on the quadratic surface to obtain intersection t and parametric
 * coordinates for TRI6 elements.
 *
 * @param[in] ray (const Ray&) Ray to intersect with the elements
 * @param[in] node_coords (const std::vector<double>&) Packed nodal coordinates
 *            for all TRI6 elements in this node
 * @param[in] bvh_node_triangle_count (unsigned int) Number of TRI6 elements in this node
 *
 * @return (IntersectionOutput) Intersection data for all TRI6 elements in the node.
 */
IntersectionOutput intersect_bvh_tri6(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_triangle_count);
    
// ================================================================================
// Curved elements: Single-element intersections
// ================================================================================
// Return t at the intersection (or +infinity if none); also fill the geometric normal surface_normals_out
// and the parametric (xi, eta) at the hit.

/**
 * @brief Intersects a ray with a single QUAD8 element.
 *
 * Runs a Newton solver on the QUAD8 surface to find the intersection point,
 * returning the distance t, the geometric normal at the hit, and the parametric
 * coordinates (xi, eta) in [-1,1]^2.
 *
 * @param[in] ray (const Ray&) Ray to intersect with the element
 * @param[in] nodes (const std::array<EiVector3d, ElementNodeCount::QUAD8>&)
 *            Nodal coordinates of the QUAD8 element
 * @param[out] surface_normals_out (EiVector3d&) Geometric normal at the intersection
 * @param[out] xi_eta_out (Eigen::Vector2d&) Parametric coordinates (xi, eta) at the hit
 *
 * @return (double) Ray distance t to the intersection, or +infinity if no hit.
 */
double intersect_quad8(const Ray& ray,
    const std::array<EiVector3d, ElementNodeCount::QUAD8>& nodes,
    EiVector3d& surface_normals_out,
    Eigen::Vector2d& xi_eta_out);

/**
 * @brief Intersects a ray with a single QUAD9 element.
 *
 * Combines sub-triangle seeding and a damped Newton method on the QUAD9
 * surface to find a robust intersection point and associated parametric
 * coordinates.
 *
 * @param[in] ray (const Ray&) Ray to intersect with the element
 * @param[in] nodes (const std::array<EiVector3d, ElementNodeCount::QUAD9>&)
 *            Nodal coordinates of the QUAD9 element
 * @param[out] surface_normals_out (EiVector3d&) Geometric normal at the intersection
 * @param[out] xi_eta_out (Eigen::Vector2d&) Parametric coordinates (xi, eta) at the hit
 *
 * @return (double) Ray distance t to the intersection, or +infinity if no hit.
 */
double intersect_quad9(const Ray& ray,
    const std::array<EiVector3d, ElementNodeCount::QUAD9>& nodes,
    EiVector3d& surface_normals_out,
    Eigen::Vector2d& xi_eta_out);

/**
 * @brief Intersects a ray with a single TRI6 element.
 *
 * Uses a linear TRI3 sub-triangulation for initial guesses, followed by a
 * Newton iteration on the TRI6 surface, to find t and quadratic barycentric
 * coordinates (g, h) at the intersection.
 *
 * @param[in] ray (const Ray&) Ray to intersect with the element
 * @param[in] nodes (const std::array<EiVector3d, ElementNodeCount::TRI6>&)
 *            Nodal coordinates of the TRI6 element
 * @param[out] surface_normals_out (EiVector3d&) Geometric normal at the intersection
 * @param[out] uv (Eigen::Vector2d&) Triangle-relative parametric coordinates (g, h) at the hit
 *
 * @return (double) Ray distance t to the intersection, or +infinity if no hit.
 */
double intersect_tri6(const Ray &ray,
    const std::array<EiVector3d, ElementNodeCount::TRI6> nodes,
    EiVector3d &surface_normals_out,
    Eigen::Vector2d &uv);
    
// ================================================================================
// Ray-acceleration structure intersections
// ================================================================================

/**
 * @brief Intersects a ray with a single BLAS (mesh BVH).
 *
 * Traverses the BLAS tree for the given mesh, finds the closest intersected
 * element, and updates the output IntersectionOutput and HitRecord with
 * the corresponding intersection data.
 * 
 * Assumes that one BLAS can store only one mesh. This mesh can contain only:
 *  - One material type;
 *  - One element type;
 *  - One surface type (i.e., we can't mix textures and solid colours).
 *
 * @param[in] ray (const Ray&) Ray to trace through the BLAS
 * @param[in] mesh_bvh (const BLAS&) BLAS representing the mesh BVH
 * @param[out] intersection_record (HitRecord&) Hit record to populate with final intersection info
 */
void intersect_BLAS(const Ray& ray,
    const BLAS& mesh_bvh,
    HitRecord& intersection_record);

/**
 * @brief Intersects a ray with the TLAS (scene-level BVH).
 *
 * Traverses the TLAS to identify potentially intersected BLASes, then
 * descends into each relevant BLAS to find the closest hit across the
 * entire scene.
 *
 * @param[in] ray (const Ray&) Ray to trace through the scene
 * @param[in] scene_TLAS (const TLAS&) Top-level acceleration structure for the scene
 * @param[out] out_intersection_record (HitRecord&) Final hit record containing intersection details
 *
 * @return (bool) True if an intersection was found, otherwise false.
 */
bool intersect_TLAS(const Ray& ray,
    const TLAS& scene_TLAS,
    HitRecord& out_intersection_record);

#endif // RTINTERSECT_H
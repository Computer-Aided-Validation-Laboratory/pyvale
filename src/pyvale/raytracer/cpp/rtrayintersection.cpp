// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD header files
#include <iostream>
#include <limits>

// ray tracer header files
#include "rtrayintersection.h"
#include "rtelemconstants.h"
#include "rtmathutils.h"
#include "rtbvh.h"
#include "rtcolorsampling.h"

EiVectorD3d cross_rowwise(const EiVectorD3d& mat1, const EiVectorD3d& mat2) {
    // Row-wise cross product for 2 matrices (i.e., treating each row as a vector).
    // Also works for multiplying a matrix with a row vector, so the input order determines the multiplication order. Happy days.
    // Written because this otherwise can't be a one-liner like in NumPy - Eigen's cross product works only for vector types.

    // We shouldn't need this check in principle due to how EiVectorD3d is defined - remove it?
    if (mat1.cols() != 3 || mat2.cols() != 3) {
        std::cerr << "Error: matrices need to have exactly 3 columns to find the cross product" << std::endl;
        return {};
    }
    // This also should be possible to comment out in production since it is used strictly for intersection code, where these are always created to be the same.
    if (mat1.rows() != mat2.rows()){
        std::cerr << "Error: matrices need to have the same number of rows to find the cross product" << std::endl;
        return {};
    }
    long long number_of_rows = mat1.rows(); // number of rows. Long long to match the type from Eigen::Index
    EiVectorD3d cross_product_result(number_of_rows, 3);
    cross_product_result.col(0) = mat1.col(1).cwiseProduct(mat2.col(2)) - mat1.col(2).cwiseProduct(mat2.col(1));
    cross_product_result.col(1) = mat1.col(2).cwiseProduct(mat2.col(0)) - mat1.col(0).cwiseProduct(mat2.col(2));
    cross_product_result.col(2) = mat1.col(0).cwiseProduct(mat2.col(1)) - mat1.col(1).cwiseProduct(mat2.col(0));
    return cross_product_result;
}

inline EiArrayD1d dot_rowwise (const EiArrayD3d& mat1, const EiArrayD3d& mat2){
    // Eigen should automatically convert EiVectorD3d to EiArrayD3d, so no need to do that while calling the function
    // These change just the object behaviour: arrays are for coefficient-wise operations, matrices for linear algebra. No data copying
    // However, if that breaks, just use e.g., mat1.array()
    return (mat1 * mat2).rowwise().sum();
}

EiArrayD3d lerp_vectorised (const EiArrayD3d& points_A,
    const EiArrayD3d& points_B,
    const EiArrayD1d weights){
    // Linear interpolation between points stored in arrays points_A and points_B using weights from vector weights.
    // Vectorised version of calculating (1-weight) * point_a + weight * point_b
    if (points_A.rows() != points_B.rows() || points_A.rows() != weights.rows()) {
        std::cerr << "Error: points_A, points_B and weights need to have the same number of rows for lerp" << std::endl;
        return {};
    }
    if (points_A.cols() != 3 || points_B.cols() != 3) {
        std::cerr << "Error: matrices need to have exactly 3 columns to find the interpolation." << std::endl;
        return {};
    }
    // Replicate the (N, 1) weights array to (N, 3) so we can take advantage of Eigen's coefficient-wise array operations
    EiArrayD3d weights_replicated = weights.replicate(1, 3);
    return (1.0 - weights_replicated ) * points_A + weights_replicated * points_B;
}


void overwrite_intersection_quad4_tex(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx){     
    // Texture color save for QUAD4           

    // Find (u,v) coordinates for each node of the intersected element
    std::array<double, ElementNodeCount::QUAD4 * UV_COORDINATES> element_uvs; // Flat array so we can pass a pointer to get_face_uvs. Texture (u,v) for each node of mesh element
    get_face_uvs(min_row_idx, Node.face_color, ElementNodeCount::QUAD4, &element_uvs[0]); // element_uvs are shaped (nodes_per_element, 2) - one (u,v) pair for every element node
    // Interpolation coordinates from the ray-quad intersection
    const double u = intersection_record.elem_interp_coords(0);
    const double v = intersection_record.elem_interp_coords(1);
    // Convert to Eigen format so we can use it for the maths below
    EiArray2d uv0, uv1, uv2, uv3;
    uv0 << element_uvs[0], element_uvs[1]; // (u,v) for node 0
    uv1 << element_uvs[2], element_uvs[3]; // (u,v) for node 1
    uv2 << element_uvs[4], element_uvs[5]; // (u,v) for node 2
    uv3 << element_uvs[6], element_uvs[7]; // (u,v) for node 3
    // Weights for bilinear interpolation
    const double w0 = (1.0 - u) * (1.0 - v);
    const double w1 = u * (1.0 - v);
    const double w2 = u * v;
    const double w3 = (1.0 - u) * v;

    const EiArray2d uvs = w0 * uv0 + w1 * uv1 + w2 * uv2 + w3 * uv3; // Final (u,v)
    // These uvs can be sent to sample the texture and the output returned to return_ray_color, regardless  of the element type down the line
    intersection_record.face_color = texsampler::sample_texture(texture, uvs);
}

void overwrite_intersection_quad8_tex(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx){     
    // Texture color save for QUAD4           

    // Find (u,v) coordinates for each node of the intersected element
    std::array<double, ElementNodeCount::QUAD8 * UV_COORDINATES> element_uvs; // Flat array so we can pass a pointer to get_face_uvs. Texture (u,v) for each node of mesh element
    get_face_uvs(min_row_idx, Node.face_color, ElementNodeCount::QUAD8, &element_uvs[0]); // element_uvs are shaped (nodes_per_element, 2) - one (u,v) pair for every element node
    // Interpolation coordinates from the ray-quad intersection
    const double u = intersection_record.elem_interp_coords(0);
    const double v = intersection_record.elem_interp_coords(1);
    // Map from [0,1] (typical for (u,v)) to [-1, 1] (typical for FEM shape functions)
    const double xi = 2.0 * u - 1.0;
    const double eta = 2.0 * v - 1.0;
    // Pre-compute squares
    const double xi2 = xi * xi;
    const double eta2 = eta * eta;
    
     // Shape functions (weights) for QUAD8
    std::array<double, ElementNodeCount::QUAD8> N;
    // Corners
    N[0] = -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta); // Bottom left
    N[1] = -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta); // Bottom right
    N[2] = -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta); // Top right
    N[3] = -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta); // Top left
    // Mid-edges
    N[4] =  0.5 * (1.0 - xi2) * (1.0 - eta); // Bottom mid-edge
    N[5] =  0.5 * (1.0 + xi)  * (1.0 - eta2); // Right mid-edge
    N[6] =  0.5 * (1.0 - xi2) * (1.0 + eta); // Top mid-edge
    N[7] =  0.5 * (1.0 - xi)  * (1.0 - eta2); // Left mid-edge

    // Interpolate final (u,v) coordinates
    EiArray2d uvs(0.0, 0.0);
    for (int i = 0; i < ElementNodeCount::QUAD8; ++i) {
        EiArray2d node_uv;
        node_uv << element_uvs[UV_COORDINATES * i], element_uvs[UV_COORDINATES * i + 1];
        uvs += N[i] * node_uv;
    }

    // These uvs can be sent to sample the texture and the output returned to return_ray_color, regardless  of the element type down the line
    intersection_record.face_color = texsampler::sample_texture(texture, uvs);
}

void overwrite_intersection_quad9_tex(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx){     
    // Texture color save for QUAD4           

    // Find (u,v) coordinates for each node of the intersected element
    std::array<double, ElementNodeCount::QUAD9 * UV_COORDINATES> element_uvs; // Flat array so we can pass a pointer to get_face_uvs. Texture (u,v) for each node of mesh element
    get_face_uvs(min_row_idx, Node.face_color, ElementNodeCount::QUAD9, &element_uvs[0]); // element_uvs are shaped (nodes_per_element, 2) - one (u,v) pair for every element node
    // Interpolation coordinates from the ray-quad intersection
    const double u = intersection_record.elem_interp_coords(0);
    const double v = intersection_record.elem_interp_coords(1);
    // Map from [0,1] (typical for (u,v)) to [-1, 1] (typical for FEM shape functions)
    const double xi = 2.0 * u - 1.0;
    const double eta = 2.0 * v - 1.0;
    // Pre-compute squares
    const double xi2 = xi * xi;
    const double eta2 = eta * eta;
    
     // Shape functions (weights) for QUAD9
    std::array<double, ElementNodeCount::QUAD9> N;
    // Corners
    N[0] =  0.25 * xi * (xi - 1.0) * eta * (eta - 1.0); // Bottom left
    N[1] =  0.25 * xi * (xi + 1.0) * eta * (eta - 1.0); // Bottom right
    N[2] =  0.25 * xi * (xi + 1.0) * eta * (eta + 1.0); // Top right
    N[3] =  0.25 * xi * (xi - 1.0) * eta * (eta + 1.0); // Top left
    // Mid-edges
    N[4] =  0.5 * (1.0 - xi2) * eta * (eta - 1.0); // Bottom mid-edge
    N[5] =  0.5 * xi * (xi + 1.0) * (1.0 - eta2);  // Right mid-edge
    N[6] =  0.5 * (1.0 - xi2) * eta * (eta + 1.0); // Top mid-edge
    N[7] =  0.5 * xi * (xi - 1.0) * (1.0 - eta2);  // Left mid-edge
    // Center node
    N[8] =  (1.0 - xi2) * (1.0 - eta2); 

    // Interpolate final (u,v) coordinates
    EiArray2d uvs(0.0, 0.0);
    for (int i = 0; i < ElementNodeCount::QUAD9; ++i) {
        EiArray2d node_uv;
        node_uv << element_uvs[UV_COORDINATES * i], element_uvs[UV_COORDINATES * i + 1];
        uvs += N[i] * node_uv;
    }

    // These uvs can be sent to sample the texture and the output returned to return_ray_color, regardless  of the element type down the line
    intersection_record.face_color = texsampler::sample_texture(texture, uvs);
}


void overwrite_intersection_tri3_tex(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx){
    // Texture color save for TRI3

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
    // These uvs can be sent to sample the texture and the output returned to return_ray_color, regardless  of the element type down the line
    intersection_record.face_color = texsampler::sample_texture(texture, uvs);
}

void overwrite_intersection_tri3_col(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx){
    // Solid surface color save for TRI3 with barycentric interpolation

    EiVector3d color_data = get_face_color(min_row_idx, Node.face_color); 
    // Barycentric interpolation
    intersection_record.face_color = intersection_record.elem_interp_coords(0) * color_data + intersection_record.elem_interp_coords(1) * color_data + intersection_record.elem_interp_coords(2) * color_data;
};


void overwrite_intersection_any_col(HitRecord& intersection_record,
    const BLAS_Node& Node,
    const Texture& texture,
    Eigen::Index min_row_idx){
    // Solid surface color save for any element other than TRI3 (no interpolation)
    intersection_record.face_color = get_face_color(min_row_idx, Node.face_color); // Write solid color without any interpolation
}

IntersectionOutput intersect_bvh_tri3(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_triangle_count) {

    // Go through all the triangles and find an intersection of each triangle with a ray
    int NODES_PER_ELEMENT = 3; // Number of nodes per triangle/quad. Used for some of flat indexing.
    double EPSILON = 1e-6;
    // Ray data broadcasted to use in vectorised operations on matrices
    // This is faster than doing it in a loop
    EiVectorD3d ray_directions = ray.direction.replicate(bvh_node_triangle_count, 1);
    EiArrayD3d ray_origins = ray.origin.replicate(bvh_node_triangle_count, 1).array();

    // Define default negative output if there is no intersection
    IntersectionOutput negative_output{
        Eigen::ArrayXXd(bvh_node_triangle_count, NODE_COORDINATES),
        EiVectorD3d::Zero(bvh_node_triangle_count, NODE_COORDINATES),
        Eigen::Vector<double, Eigen::Dynamic>::Constant(bvh_node_triangle_count, 1, std::numeric_limits<double>::infinity())
    };

    // Calculations - edges and normals
    EiMatrixDd edge0(bvh_node_triangle_count, NODE_COORDINATES), nEdge2(bvh_node_triangle_count, NODE_COORDINATES); // shape (faces, 3) each
    EiArrayDd  nodes0(bvh_node_triangle_count, NODE_COORDINATES);
    // Go over all TRI3s in the node to find corner and edge coordinates
    // Node coords are stored as [xyz, xyz, xyz...]
    for (int triangle_idx = 0; triangle_idx < bvh_node_triangle_count; triangle_idx++) {
        int node_0 = triangle_idx * NODES_PER_ELEMENT;
        int node_1 = triangle_idx * NODES_PER_ELEMENT + 1;
        int node_2 = triangle_idx * NODES_PER_ELEMENT + 2;

        for (int j = 0; j < NODE_COORDINATES; j++) {
            edge0(triangle_idx, j) = node_coords[node_1 * NODE_COORDINATES + j] - node_coords[node_0 * NODE_COORDINATES + j];
            nodes0(triangle_idx, j) = node_coords[node_0 * NODE_COORDINATES + j];
            // Skip edge1 because it never gets used in the calculations anyway and calculate negative Edge2 as this is what we will need
            nEdge2(triangle_idx, j) = node_coords[node_2 * NODE_COORDINATES + j] - node_coords[node_0 * NODE_COORDINATES + j];
            //std::cout << "nEdge2 : " << nEdge2(triangle_idx,j) << std::endl;
        }
    }
    EiVectorD3d plane_normals = cross_rowwise(edge0, nEdge2); // not normalised! Shape (faces, 3)

    // Step 1: Quantities for the Moller Trumbore method
    EiArrayD3d p_vec = cross_rowwise(ray_directions, nEdge2); // Assigns a vector to an array variable, but Eigen automatically converts so long as the underlying sizes are correct at initialization. Shape (faces, 3)
    EiArrayD1d discriminants = dot_rowwise(edge0, p_vec);  // Row-wise dot product; shape (faces, 1)

    // Step 2: Culling.
    //Discriminant negative -> triangle is back-facing. If discriminant is close to 0, ray and triangle are parallel and ray misses the triangle.
    EiBoolMask valid_mask = (discriminants > EPSILON) && (discriminants > 0);
    if (!valid_mask.any()) {
        //std::cout << "Condition 1 triggered" << std::endl;
        return negative_output; // No intersection - return infinity
    }

    // Step 3: Test if ray is in front of the triangle
    EiArrayD1d inverse_discriminants = discriminants.inverse(); // Element-wise inverse. shape (faces, 1)
    EiArrayD3d t_vec = ray_origins - nodes0; // shape (faces, 3)
    EiArrayD1d barycentric_u = dot_rowwise(t_vec, p_vec) * inverse_discriminants; // shape (faces, 1)
    valid_mask = valid_mask && (barycentric_u >= 0) && (barycentric_u <= 1);
    if (!valid_mask.any()) {
        //std::cout << "Condition 2 triggered" << std::endl;
        return negative_output; // No intersection - return infinity
    }

    EiArrayD3d q_vec = cross_rowwise(t_vec.matrix(), edge0); // shape (faces, 3)
    EiArrayD1d barycentric_v = dot_rowwise(ray_directions, q_vec) * inverse_discriminants; // shape (faces, 1)
    // Check barycentric_v and sum
    valid_mask = valid_mask && (barycentric_v >= 0) && ((barycentric_u + barycentric_v) <= 1);
    // t values
    EiArrayD1d t_values = dot_rowwise(nEdge2, q_vec) * inverse_discriminants; // shape (faces, 1)
    valid_mask = valid_mask && (t_values >= ray.t_min) && (t_values <= ray.t_max);

    // Iterate through all t_values and set them to infinity if they don't satisfy the conditions imposed by the mask
    for (int i = 0; i < t_values.rows(); ++i) {
        for (int j = 0; j < t_values.cols(); ++j) {
            if (!valid_mask(i, j)) {
                t_values(i, j) = std::numeric_limits<double>::infinity();
            }
        }
    }
    // Create an array for barycentric coordinates so we can do things element-wise with those
    Eigen::ArrayXXd barycentric_coordinates(bvh_node_triangle_count, NODE_COORDINATES);
    barycentric_coordinates.col(0) = 1.0 - barycentric_u - barycentric_v; // barycentric_w
    barycentric_coordinates.col(1) = barycentric_u;
    barycentric_coordinates.col(2) = barycentric_v;
    /* Original order (u, v, w), but not representative of our and the texture does not render correctly 
    barycentric_coordinates.col(0) = barycentric_u;
    barycentric_coordinates.col(1) = barycentric_v;
    barycentric_coordinates.col(2) = 1.0 - barycentric_u - barycentric_v; // barycentric_w
    */
    return IntersectionOutput{ barycentric_coordinates, plane_normals, t_values };
}

IntersectionOutput intersect_bvh_quad(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_quad_count){
    // Go through all the quads and find an intersection of each quad with a ray
    // Method based on NVIDIA 2019, E. Haines, T. Akenine-Möller (eds.), Ray Tracing Gems, https://doi.org/10.1007/978-1-4842-4427-2_8
    // More specifically: Chapter 8, "Cool Patches: A Geometric Approach to Ray/Bilinear Patch Intersections" by A. Reshetov

    // IMPORTANT NOTE: This should work for QUAD8 and QUAD9 in both the VTK and Exodus order without any changes as they also store corner nodes at indices 0-3 in the connectivity array,
    // and this is developed for non-planar quadrilaterals. Still, as of now, it was only tested with QUAD4

    static const int NODES_PER_ELEMENT = 4;
    static const int COORDS_PER_ELEMENT = NODES_PER_ELEMENT * NODE_COORDINATES;
    double EPSILON = 1e-6;
    
    // 1. COORDINATES AND EDGES
    // Ray data broadcasted to use in vectorised operations on matrices
    // This is faster than doing it in a loop
    EiVectorD3d ray_directions = ray.direction.replicate(bvh_node_quad_count, 1);
    EiArrayD3d ray_origins = ray.origin.replicate(bvh_node_quad_count, 1).array();

    // Define default negative output if there is no intersection
    IntersectionOutput negative_output{
        Eigen::ArrayXXd(bvh_node_quad_count, NODE_COORDINATES), // elem_interp_coords
        EiVectorD3d::Zero(bvh_node_quad_count, NODE_COORDINATES), //  plane_normals
        Eigen::Vector<double, Eigen::Dynamic>::Constant(bvh_node_quad_count, 1, std::numeric_limits<double>::infinity()) // t_values
    };

    // 4 corner coordinates, where bl - bottom left, br - bottom right, tr - top right, tl - top left. 
    EiArrayD3d corners_bl(bvh_node_quad_count, NODE_COORDINATES), corners_br(bvh_node_quad_count, NODE_COORDINATES), corners_tr(bvh_node_quad_count, NODE_COORDINATES), corners_tl(bvh_node_quad_count, NODE_COORDINATES); // shape (faces, 3) each
    EiArrayD3d edges_bottom(bvh_node_quad_count, NODE_COORDINATES), edges_top(bvh_node_quad_count, NODE_COORDINATES), edges_right(bvh_node_quad_count, NODE_COORDINATES), edges_left(bvh_node_quad_count, NODE_COORDINATES); // shape (faces, 3) each

    //std::cout << "Getting corners and edges" << std::endl;
    // Go over all quads in the node to find corner and edge coordinates
    // Node coords are stored as [xyz, xyz, xyz...]
    for (int quad_idx = 0; quad_idx < bvh_node_quad_count; quad_idx++) {
        int base_stride = quad_idx * COORDS_PER_ELEMENT;
        // Go over x, y, z coordinates
        for (int j = 0; j < NODE_COORDINATES; j++) {
            // Access n-th node via base_stride + n, where n e [0, 3] for QUAD4
            // Then use access x, y, z coordinates in flattened array
            // E.g., [base_stride + 1 * NODE_COORDINATES + 2] would be node_1, coordinate z
            corners_bl(quad_idx, j) = node_coords[base_stride + 0 * NODE_COORDINATES + j]; // Bottom left corner coordinates; q00 in paper
            corners_br(quad_idx, j) = node_coords[base_stride + 1 * NODE_COORDINATES + j]; // Bottom right corner coordinates; q10 in paper
            corners_tr(quad_idx, j) = node_coords[base_stride + 2 * NODE_COORDINATES + j]; // Top right corner coordinates; q11 in paper
            corners_tl(quad_idx, j) = node_coords[base_stride + 3 * NODE_COORDINATES + j]; // Top left corner coordinates; q01 in paper
            edges_left(quad_idx, j) = corners_tl(quad_idx, j) - corners_bl(quad_idx, j); // Left edge; e00 in paper
            edges_right(quad_idx, j) = corners_tr(quad_idx, j) -corners_br(quad_idx, j); // Right edge; e11 in paper
            edges_bottom(quad_idx, j) = corners_br(quad_idx,j) - corners_bl(quad_idx, j); // Bottom edge; e10 in paper
            edges_top(quad_idx, j) = corners_tr(quad_idx, j) - corners_tl(quad_idx, j); // Top edge
        }
    }

    EiArrayD3d normals = cross_rowwise(corners_br-corners_bl, corners_tl-corners_tr).array(); // not normalised! Shape (faces,3)
    // Translate only the coordinates that need ray-origin subtraction
    corners_bl -= ray_origins;
    corners_br -= ray_origins;

    // 2. SOLVING THE INTERSECTION

    // 1. Solve quadratic a + b*u + c*u^2
    //std::cout << "Getting a,b,c" << std::endl;
    EiArrayD1d a = dot_rowwise(cross_rowwise(corners_bl, ray_directions), edges_left); // Row-wise dot product; shape (faces, 1)
    //((cross_rowwise(corners_bl, ray_directions)).array() * edges_left).rowwise().sum(); // Row-wise dot product; shape (faces, 1))
    EiArrayD1d c = dot_rowwise(normals, ray_directions); //
    //(normals * ray_directions.array()).rowwise().sum(); // Row-wise dot product; shape (faces, 1)
    EiArrayD1d b = dot_rowwise(cross_rowwise(corners_br, ray_directions), edges_right);
    //((cross_rowwise(corners_br, ray_directions)).array() * edges_right).rowwise().sum(); // Row-wise dot product; shape (faces, 1))
    b = b - (a + c); 
    EiArrayD1d discriminants = b.square() - 4*a*c; // Shape (faces, 1)
    // Discriminant negative -> triangle is back-facing. If discriminant is close to 0, ray and triangle are parallel and ray misses the triangle.
    EiBoolMask valid_mask = (discriminants > EPSILON) && (discriminants > 0); // Shape (faces, 1)
    if (!valid_mask.any()) {
        //std::cout << "Condition 1 triggered" << std::endl;
        return negative_output; // No intersection - return infinity
    }
    discriminants = discriminants.sqrt(); // Pre-compute square root of discriminant
    // Two roots (u_1, u_2)
    EiArrayD1d u1(bvh_node_quad_count, 1), u2(bvh_node_quad_count, 1);

    // We can have c == 0 or c!=0 so create two masks for these cases to handle the vectorised version properly
    // c == 0 => Trapezoid, only one root
    EiBoolMask c_zero_mask = c.abs() < EPSILON; // Shape (faces, 1)
    // c != 0 => Stanford model
    EiBoolMask c_nonzero_mask = !c_zero_mask; // Shape (faces, 1)

    // Additionally check b values to avoid division by 0
    EiBoolMask b_zero_mask = b.abs() < EPSILON;
    EiArrayD1d safe_b = (!b_zero_mask).select(b, 1.0);

    // 2. Compute roots
    // 2.1. Trapezoid branch
    //u1 = c_zero_mask.select(-a/b, u1); // Equivalent of if (c == 0) { u_1 = -a/b; } else { u_1 = u1; } (where u1 is undefined here, but will be found for the other branch)
    // Safer version with avoiding division by b ~ 0. Set to -1 otherwise as solution not found
    EiArrayD1d trapezoid_u1 = (!b_zero_mask).select(-a / safe_b, -1.0);
    u1 = c_zero_mask.select(trapezoid_u1, u1); // 
    u2 = c_zero_mask.select(-1.0, u2);

    // 2.2. Stanford branch
    //std::cout << "Stanford branch" << std::endl;
    // We cannot use copysign for Eigen arrays, so use Eigen's boolean selections instead
    // If b>=0, we want to add the discriminant, else we subtract it
    
    // Copysign will not work on an Eigen array, so create a mask based on the sign of b, which updates the sign of the discriminant
    EiArrayD1d copysign_disc_b = (b >= 0.0).select(discriminants, -discriminants); // Shape (faces, 1)
    EiArrayD1d tmp_root = (-b - copysign_disc_b) / 2.0; // Shape (faces, 1). Part of the quadratic root (-b +/- sqrt(discriminant))/ 2 with missing "a" in the denominator
    EiBoolMask tmp_zero_mask = tmp_root.abs() < EPSILON; // Shape (faces, 1). Check that tmp != 0
    u1 = (c_nonzero_mask).select(tmp_root/c, u1); // Numerically stable root
    // tmp_root ~ 0 branch to avoid division by 0
    u2 = (c_nonzero_mask && tmp_zero_mask).select(-1.0, u2);
    // tmp_root != 0 branch - use Viete's formula for u1*u2
    u2 = (c_nonzero_mask && !tmp_zero_mask).select(a/tmp_root, u2);

    // 3. Evaluate and check whether the solution lies inside the patch (quad)
    // Initialize output values with default negatives. (u,v) are the solutions; we are looking for (u,v) for the smallest t > 0 
    EiArrayD1d t_values = EiArrayD1d::Constant(bvh_node_quad_count, 1, std::numeric_limits<double>::infinity()); // Array, not a vector for now as this enables element-wise access in Eigen
    EiArrayD1d u_values = EiArrayD1d::Constant(bvh_node_quad_count, 1, -1.0);
    EiArrayD1d v_values = EiArrayD1d::Constant(bvh_node_quad_count, 1, -1.0);

    // 3.1. Evaluate case 0.0 <= u1 <= 1.0
    //std::cout << "u1" << std::endl;
    EiBoolMask u1_valid = valid_mask && (0.0 <= u1) && (u1 <= 1.0);
    EiArrayD3d pa1 = lerp_vectorised(corners_bl, corners_br, u1);
    EiArrayD3d pb1 = lerp_vectorised(edges_left, edges_right, u1);
    EiVectorD3d n1 = cross_rowwise(ray_directions, pb1.matrix());
    EiArrayD1d det1 = n1.array().square().rowwise().sum();
    EiBoolMask det1_valid = det1 > EPSILON; // Non-zero discriminant
    EiVectorD3d n1_cross = cross_rowwise(n1, pa1);
    EiArrayD1d t1 = dot_rowwise(n1_cross, pb1); // TEST OF DOT_ROWWISE AS WELL SINCE N1_CROSS ISN'T AN ARRAY
    EiArrayD1d v1 = dot_rowwise(n1_cross, ray_directions);
    // Create a hit mask if we are in the u1 branch, discriminant is valid, and t1 > 0 and 0.0 <= v1 <= det1
    EiBoolMask hit_mask1 = u1_valid && det1_valid && (t1 > 0.0) && (v1 >= 0.0) && (det1 >= v1);
    // Update values where we have a hit
    t_values = hit_mask1.select(t1/det1, t_values);
    u_values = hit_mask1.select(u1, u_values);
    v_values = hit_mask1.select(v1/det1, v_values);

    // 3.2. Evaluate case 0.0 <= u2 <= 1.0 - Slightly different since u1 might be good and we need 0 < t2 < t1
    //std::cout << "u2" << std::endl;
    EiBoolMask u2_valid = valid_mask && (0.0 <= u2) && (u2 <= 1.0);
    EiArrayD3d pa2 = lerp_vectorised(corners_bl, corners_br, u2);
    EiArrayD3d pb2 = lerp_vectorised(edges_left, edges_right, u2);
    EiVectorD3d n2 = cross_rowwise(ray_directions, pb2.matrix());
    EiArrayD1d det2 = n2.array().square().rowwise().sum();
    EiBoolMask det2_valid = det2 > EPSILON; // Non-zero discriminant
    EiVectorD3d n2_cross = cross_rowwise(n2, pa2);
    EiArrayD1d t2 = dot_rowwise(n2_cross, pb2) / det2; //
    EiArrayD1d v2 = dot_rowwise(n2_cross, ray_directions);
    // Hit mask for u2 branch, discriminant is valid, 0 < t2 < t and 0 <= v2 <= det2
    EiBoolMask hit_mask2 = u2_valid && det2_valid && (t2 > 0.0) && (t_values > t2) && (det2 >= v2) && (v2 >= 0);
    // Update values where we have a hit
    t_values = hit_mask2.select(t2, t_values);
    u_values = hit_mask2.select(u2, u_values);
    v_values = hit_mask2.select(v2/det2, v_values);

    // 4. Interpolate final results, find geometric normals and texture coordinates
    //std::cout << "Final results" << std::endl;
    EiArrayD3d du = lerp_vectorised(edges_left, edges_top, v_values);
    EiArrayD3d dv = lerp_vectorised(edges_left, edges_right, u_values);
    EiVectorD3d geometric_normals = cross_rowwise(du, dv);
    // Assign the final outputs and return
    // Create an array for barycentric coordinates so we can do things element-wise with those
    Eigen::ArrayXXd quad_coordinates = Eigen::ArrayXXd::Zero(bvh_node_quad_count, NODE_COORDINATES);
    quad_coordinates.col(0) = u_values;
    quad_coordinates.col(1) = v_values;
    // Leave last column at 0 - matches out output format and this is also what they do explicitly in the paper
    return IntersectionOutput{ quad_coordinates, geometric_normals, t_values };
    }

struct Ray_old { Eigen::Vector3d o, d; Ray_old(Eigen::Vector3d o_, Eigen::Vector3d d_) : o(o_), d(d_) {} };

void ray_convert(Ray_old &ray_old, const Ray &ray) {

    for (int i = 0; i < 3; ++i) {
        ray_old.o(i) = ray.origin(i);
        ray_old.d(i) = ray.direction(i);
    }
}

struct Quadratic_tet {
    std::vector<Eigen::Vector3d> nodes;

    // Nodes for the 4 faces - each face has 6 nodes: 3 corners, 3 mid-edges
    // Numbering for 1 tetrahedron example
    // const int face_indices[4][6] = {
    //     {0, 2, 1, 6, 5, 4}, // Face 0 (bottom)
    //     {0, 1, 3, 4, 8, 7}, // Face 1
    //     {1, 2, 3, 5, 9, 8}, // Face 2
    //     {2, 0, 3, 6, 7, 9}  // Face 3
    // };

      // Correct numbering for NGSolve/Netgen
      const int face_indices[4][6] = {
      {0, 1, 2, 4, 7, 5}, // Face 0 (bottom) (v0, v1, v2)
      {0, 1, 3, 4, 8, 6}, // Face 1 (v0, v1, v3)
      {1, 2, 3, 7, 9, 8}, // Face 2 (v1, v2, v3)
      {0, 2, 3, 5, 9, 6}  // Face 3 (v0, v2, v3)
      };

    Quadratic_tet(std::vector<Eigen::Vector3d> nodes_) :
        nodes(nodes_) {}

    // Quadratic triangle shape functions (g, h)
    // r = 1 - g - h
    static Eigen::VectorXd get_face_N(double g, double h) {
        double r = 1.0 - g - h;
        Eigen::VectorXd N(6);
        N << r*(2*r-1), g*(2*g-1), h*(2*h-1), 4*g*r, 4*g*h, 4*h*r;
        return N;
    }

    static Eigen::Matrix<double, 3, 2> get_face_Jacobian(double g, double h, const std::vector<Eigen::Vector3d>& f_nodes) {
        double r = 1.0 - g - h;
        // Derivatives of N wrt g and h
        double dNdu[6] = { -(4*r-1), 4*g-1, 0, 4*(r-g), 4*h, -4*h };
        double dNdv[6] = { -(4*r-1), 0, 4*h-1, -4*g, 4*g, 4*(r-h) };

        Eigen::Matrix<double, 3, 2> J = Eigen::Matrix<double, 3, 2>::Zero();
        for (int i = 0; i < 6; ++i) {
            J.col(0) += f_nodes[i] * dNdu[i];
            J.col(1) += f_nodes[i] * dNdv[i];
        }
        return J;
    }

    double intersect(const Ray_old &r, Eigen::Vector3d &n_out, Eigen::Vector2d &uv) const {
      
    // Define imprecision parameters for the initial guess, they are especially needed for the cases
    // where the ray is close to being tangential to the linear triangle.
    // If the discriminant is close to zero, the ray misses the triangle.
    // Imprecision levels should be relatively large for the initial guess.
    const double eps_init_guess1 = 1e-10; // Imprecision for linear tringle's discriminant
    const double eps_init_guess2 = 0.1; // Imprecision for linear tringle's isoparametric coordinates (u, v)
    const double eps_t = 1e-5; // Imprecision for t

    const int iter_max = 50; // Maximum number of iterations for Newton-Raphson method.

      // Define imprecision parameters for the solution.
      // Imprecision levels should be relatively small for solution.
      const double eps_sol1 = 1e-7; // Imprecision for the residual
      const double eps_sol2 = 1e-8; // Imprecision for the quadratic triangle's isoparametric coordinates (g, h)
      const double eps_sol3 = 1e-10; // Imprecision for the discriminant

    double min_t = std::numeric_limits<double>::infinity();
    bool intersect = false;

    // 1. Intersect ray with the 4 triangles
    // Use the code already used for Traingle structure.
    // Impose the defined imprecision levels

    // Mapping for sub-triangulation (indices within the 6-node f_nodes vector)
    // Quadratic layout: 0,1,2 are corners; 3,4,5 are midpoints of (0-1), (1-2), (2-0)
    int sub_tris[4][3] = {
        {0, 3, 5}, // Bottom-left
        {3, 1, 4}, // Bottom-right
        {5, 4, 2}, // Top
        {3, 4, 5}  // Center
    };

    // Barycentric coordinate offsets for the sub-triangles to map back to (g, h)
    // These represent the (g, h) coordinates of the nodes in f_nodes
    Eigen::Vector2d nodes_gh[6] = {
        {0,0}, {1,0}, {0,1}, {0.5,0}, {0.5,0.5}, {0,0.5}
    };

    for (int f = 0; f < 4; ++f) {
        std::vector<Eigen::Vector3d> f_nodes(6);
        for(int i=0; i<6; ++i) f_nodes[i] = nodes[face_indices[f][i]];

        // Sub-triangulation
        double best_sub_t = std::numeric_limits<double>::infinity();
        Eigen::Vector2d best_gh_guess(0.33, 0.33);
        bool found_guess = false;

        for (int s = 0; s < 4; ++s) {
            Eigen::Vector3d v0 = f_nodes[sub_tris[s][0]];
            Eigen::Vector3d v1 = f_nodes[sub_tris[s][1]];
            Eigen::Vector3d v2 = f_nodes[sub_tris[s][2]];

            Eigen::Vector3d edge1 = v1 - v0, edge2 = v2 - v0;
            Eigen::Vector3d pvec = r.d.cross(edge2);
            double det = edge1.dot(pvec);
            if (fabs(det) < eps_init_guess1) continue;

            double invDet = 1.0 / det;
            Eigen::Vector3d tvec = r.o - v0;
            double u_sub = tvec.dot(pvec) * invDet;
            if (u_sub < 0 - eps_init_guess2 || u_sub > 1 + eps_init_guess2) continue;

            Eigen::Vector3d qvec = tvec.cross(edge1);
            double v_sub = r.d.dot(qvec) * invDet;
            if (v_sub < 0 - eps_init_guess2 || (u_sub + v_sub) > 1 + eps_init_guess2) continue;

            double t_sub = edge2.dot(qvec) * invDet;
            if (t_sub > eps_t && t_sub < best_sub_t) {
                best_sub_t = t_sub;
                // Interpolate the global (g, h) from the sub-triangle's local barycentrics
                double w_sub = 1.0 - u_sub - v_sub;
                best_gh_guess = w_sub * nodes_gh[sub_tris[s][0]] + 
                                u_sub * nodes_gh[sub_tris[s][1]] + 
                                v_sub * nodes_gh[sub_tris[s][2]];
                found_guess = true;
            }
        }

        if (!found_guess) continue;

        // 2. Start Newton-Raphson at t_guess, and gh from the right triangle's centroid 
        // or approximated (u, v) from the linear triangle. 
        Eigen::Vector2d gh = best_gh_guess;
        double t = best_sub_t;

        for (int iter = 0; iter < iter_max; ++iter) {
            Eigen::VectorXd N = get_face_N(gh.x(), gh.y());
            Eigen::Vector3d P = Eigen::Vector3d::Zero();
            for(int i=0; i<6; ++i) P += N[i] * f_nodes[i];

            Eigen::Vector3d res = r.o + t * r.d - P;
            if (res.norm() < eps_sol1) {
                // Check if hit is within quadratic triangle bounds
                    if (gh.x() >= 0 - eps_sol2 && gh.y() >= 0 - eps_sol2 && (gh.x() + gh.y()) <= 1 + eps_sol2) {
                    if (t < min_t && t > eps_t) {
                        min_t = t;
                        Eigen::Matrix<double, 3, 2> J = get_face_Jacobian(gh.x(), gh.y(), f_nodes);
                        Eigen::Vector3d normal = J.col(0).cross(J.col(1)).normalized();

                            // Ensure normal points against the ray (outward for convex)
                            if (normal.dot(r.d) > 0) {
                                normal = -normal;
                            }
                            n_out = normal;
                            uv = gh;
                            intersect = true;
                    }
                }
                break;
            }

            // Solve [rd, -dP/du, -dP/dv] * [dt, du, dv]^T = -res
            Eigen::Matrix<double, 3, 2> J = get_face_Jacobian(gh.x(), gh.y(), f_nodes);
            Eigen::Matrix3d M;
            M.col(0) = r.d;
            M.col(1) = -J.col(0);
            M.col(2) = -J.col(1);

            if (std::abs(M.determinant()) < eps_sol3) break;
            
            // Eigen::Vector3d delta = M.colPivHouseholderQr().solve(-res);
            Eigen::Vector3d delta = M.inverse() * (-res);
            t += delta.x();
            gh.x() += delta.y();
            gh.y() += delta.z();

            if (gh.x() < -0.5 || gh.y() < -0.5 || (gh.x() + gh.y()) > 1.5) break;
        }
    }

    return intersect ? min_t : std::numeric_limits<double>::infinity();
}
};



void load_quad_tets(const std::vector<double>& node_coords, 
                    std::vector<Quadratic_tet> &quadratic_tets,
                    const unsigned int bvh_node_quad_tet_count,
                    enum ElementNodeCount nodes_per_element) {
                    

   const int coords_per_element = nodes_per_element * NODE_COORDINATES; // number of elements times 3 coordinates each

   for (int quad_tet_idx = 0; quad_tet_idx < bvh_node_quad_tet_count; quad_tet_idx++) {

    std::vector<Eigen::Vector3d> nodes;
    
    int index_min = quad_tet_idx * coords_per_element;

        for (int i = 0; i < 10; i++) {

            Eigen::Vector3d node(0, 0, 0);

            node(0) = node_coords[index_min + i * 3 + 0]; // X-component
            node(1) = node_coords[index_min + i * 3 + 1]; // Y-component
            node(2) = node_coords[index_min + i * 3 + 2]; // Z-component

            nodes.emplace_back(node);
            // std::cerr << "Loaded " << i << "\n" << node << "\n";
        }

        quadratic_tets.emplace_back(nodes);

        // std::cerr << "Loaded " << quadratic_tets.size() << " quadratic tetrahedrons" << "\n";
    }
}


IntersectionOutput intersect_bvh_quad_tet(const Ray& ray,
    std::vector<Quadratic_tet> quadratic_tets,
    const unsigned int bvh_node_quad_tet_count) {

    // Go through all the tetrahedron and find an intersection of each triangle with a ray

    // Define default negative output if there is no intersection
    IntersectionOutput negative_output{
        Eigen::ArrayXXd(bvh_node_quad_tet_count, 3),
        EiVectorD3d::Zero(bvh_node_quad_tet_count, 3),
        Eigen::Vector<double, Eigen::Dynamic>::Constant(bvh_node_quad_tet_count, 1, std::numeric_limits<double>::infinity())
    };

    // Calculations - go through all the tetrahedrons
    
    // Convert between 2 types of rays, need to get rid of this later
    Eigen::Vector3d a;
    Eigen::Vector3d b;
    Ray_old ray_old(a, b);
    ray_convert(ray_old, ray);


    EiVectorD3d plane_normals(bvh_node_quad_tet_count, 3);
    Eigen::ArrayXXd t_values(bvh_node_quad_tet_count, 1);
    Eigen::ArrayXXd barycentric_u(bvh_node_quad_tet_count, 1);
    Eigen::ArrayXXd barycentric_v(bvh_node_quad_tet_count, 1);

    for (int quad_tet_idx = 0; quad_tet_idx < bvh_node_quad_tet_count; quad_tet_idx++) {
        Eigen::Vector3d n_tmp;
        Eigen::Vector2d uv_tmp;
        double t = quadratic_tets[quad_tet_idx].intersect(ray_old, n_tmp, uv_tmp);

        // Convert the intersection results to acceptable format
        for (int i = 0; i < 3; ++i) {
            plane_normals(quad_tet_idx, i) = n_tmp(i);
        }
        
        t_values(quad_tet_idx) = t;
        barycentric_u(quad_tet_idx) = uv_tmp.x();
        barycentric_v(quad_tet_idx) = uv_tmp.y();
        
    }

    // Mask inappropriate values based on t_values
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask;
    valid_mask = (t_values > 0.0); // t=0.0 means no intersection with the tet
    if (!valid_mask.any()) {
        //std::cout << "Condition 1 triggered" << std::endl;
        return negative_output; // No intersection - return infinity
    }

    valid_mask = valid_mask && (t_values >= ray.t_min) && (t_values <= ray.t_max);

    // Iterate through all t_values and set them to infinity if they don't satisfy the conditions imposed by the mask
    for (int i = 0; i < t_values.rows(); ++i) {
        for (int j = 0; j < t_values.cols(); ++j) {
            if (!valid_mask(i, j)) {
                t_values(i, j) = std::numeric_limits<double>::infinity();
            }
        }
    }

    // Create an array for barycentric coordinates so we can do things element-wise with those
    Eigen::ArrayXXd barycentric_coordinates(bvh_node_quad_tet_count, 3);
    barycentric_coordinates.col(0) = barycentric_u;
    barycentric_coordinates.col(1) = barycentric_v;
    barycentric_coordinates.col(2) = 1.0 - barycentric_u - barycentric_v; // barycentric_w

    return IntersectionOutput{ barycentric_coordinates, plane_normals, t_values };
}

bool intersect_AABB (const Ray& ray, const AABB& AABB) {
    // Slab method for ray-AABB intersection
    double t_axis[6]; // t values for each axis, so [0,1] are for x, [2,3] for y, and [4,5] for z
    EiVector3d inverse_direction = 1/(ray.direction.array()); // Divide first to use cheaper multiplication later

    // Find ray intersections with planes defining the AABB in X, Y, Z
    for (int i = 0; i < 3; ++i) {
        t_axis[2*i] = (AABB.corner_min[i] - ray.origin(i)) * inverse_direction(i);
        t_axis[2*i + 1] = (AABB.corner_max[i] - ray.origin(i)) * inverse_direction(i);
    }
    //Overlap test
    // Find the minimum t for each axis (x, y, z), then find maximum of these for (x,y,z)
    double t_min = std::max(std::max(std::min(t_axis[0], t_axis[1]), std::min(t_axis[2], t_axis[3])), std::min(t_axis[4], t_axis[5]));
    // Find the maximum t for each axis (x, y, z), then find minimum of these for (x,y,z)
    double t_max = std::min(std::min(std::max(t_axis[0], t_axis[1]), std::max(t_axis[2], t_axis[3])), std::max(t_axis[4], t_axis[5]));

    // t_min < t_max - Ray which just touches a corner, edge, or face of the AABB will be considered non-intersecting
    // t_min <= t_max - Rays which touch the box boundary are considered intersecting. A bit of a degenerate case, but decided to include it here, hence more relaxed inequality.
    // t_min < ray.t_max - Clip to ray segment
    return t_min <= t_max && t_max > 0.0 && t_min < ray.t_max; // False => No overlap => Ray does not intersect the AABB.
}

void intersect_BLAS(const Ray& ray,
    const BLAS& mesh_bvh,
    IntersectionOutput& out_intersection,
    HitRecord& intersection_record) {

    //std::cout << "  BLAS: Starting BVH intersection test" << std::endl;

    // Find the number of nodes per mesh element NOW to limit branching
    // This is valid only if we assume that one mesh can contain only one type of element
    Texture texture = mesh_bvh.texture;

    // this could be stored in BLAS and assigned when we build it to remove these checks
    void (*overwrite_intersection_function_ptr)(HitRecord&, const BLAS_Node&, const Texture& texture, Eigen::Index min_row_idx); // Saving data to HitRecord depending on the surface type (color/texture) and element type
    overwrite_intersection_function_ptr = mesh_bvh.overwrite_intersection_function_ptr;

    // Function pointer to the appropriate intersection function. Nb4 this syntax means that they should require the same arguments
    IntersectionOutput (*intersection_function_ptr)(const Ray&, const std::vector<double>& node_coords, const unsigned int bvh_node_element_count); // Ray-mesh element intersection (TRI3, QUAD4, etc.)
    intersection_function_ptr = mesh_bvh.intersection_function_ptr;
        
    // Create stack to intersect BLAS nodes. Stack (LIFO) so DFS
    std::vector<int> stack; // Store node indices on the stack
    stack.push_back(mesh_bvh.root_idx);

     while(!stack.empty()){
        const BLAS_Node& Node = mesh_bvh.tree_nodes[stack.back()];
        stack.pop_back();

        if (!intersect_AABB(ray, Node.bounding_box)) continue; // Early exit if ray does not intersect the AABB of the node

        // No children => Leaf node => Intersect triangles
        if (Node.left_child_idx == -1) {
            
            //std::cout << "We are trying to intersect elements in node now" << std::endl;
            
            out_intersection = intersection_function_ptr(ray, Node.node_coords, Node.element_count);

            Eigen::Index min_row_idx, min_col_idx;
            //std::cout << "Number of t_values: " << out_intersection.t_values.size() << std::endl;

            out_intersection.t_values.minCoeff(&min_row_idx, &min_col_idx); // Find indices of the smallest t_value
            double closest_t = out_intersection.t_values(min_row_idx, min_col_idx);
            //std::cout << "Closest t found: " << closest_t << std::endl;

            // Store the closest intersection if it is closer than the previously stored one
            if (closest_t < intersection_record.t) {
                intersection_record.t = closest_t;
                intersection_record.elem_interp_coords = out_intersection.elem_interp_coords.row(min_row_idx);
                intersection_record.point_intersection = ray_at_t(closest_t, ray);
                intersection_record.normal_surface = out_intersection.plane_normals.row(min_row_idx);
                //intersection_record.face_color = get_face_color(min_row_idx, Node.face_color); // the OG part
                overwrite_intersection_function_ptr(intersection_record, Node, texture, min_row_idx);
            } 
        }
        else { // Not a leaf node => Test children nodes for intersections
            // DFS order
            int left = Node.left_child_idx;
            int right = left + 1;
            if (right != 0) stack.push_back(right);
            if(left != -1) stack.push_back(left);
            // Potential improvement: testing node distance vs. ray to push the farther one first, so we trasverse closer child first.
            // How to: Compare t_near from intersect_AABB for both children and intersect the closer one first
        }   
     }
}

void intersect_TLAS(const Ray& ray,
    const TLAS& scene_TLAS,
    IntersectionOutput& out_intersection,
    HitRecord& out_intersection_record){

    //std::cout << "TLAS: Starting BVH intersection test" << std::endl;
     std::vector<int> stack; // Store node indices on the stack
     stack.push_back(0); // Push root index

     while(!stack.empty()){
        const TLAS_Node& Node = scene_TLAS.tlas_nodes[stack.back()];
        stack.pop_back();

        if (!intersect_AABB(ray, Node.bounding_box)) continue; // Early exit if ray does not intersect the AABB of the node
        if (Node.left_child_idx == -1) {
            // No children => Leaf node => Intersect triangles
            //std::cout << "TLAS: Leaf node reached with " << Node.blas_count << " BLASes." << std::endl;
            int node_max_index = Node.min_blas_idx + Node.blas_count;
            for (int i = Node.min_blas_idx; i < node_max_index; ++i){
                // Note: Comment out the below check if MAX_ELEMENTS_PER_LEAF = 1; in build_TLAS because then TLAS node AABB = BLAS AABB, so this check is unnecessary
                if (!intersect_AABB(ray, scene_TLAS.blases[i].bounding_box)) continue; // Early exit if the ray does not intersect the AABB of the BLAS (mesh).
                //std::cout << " TLAS: Intersected BLAS index: " << i << std::endl;
                intersect_BLAS(ray, scene_TLAS.blases[i], out_intersection, out_intersection_record);
            }
        }
        else { // Not a leaf node => Test children nodes for intersections
            // DFS order
            int left = Node.left_child_idx;
            int right = left + 1;
            if (right != 0) stack.push_back(right);
            if(left != -1) stack.push_back(left);
        }
     }
}
        
           
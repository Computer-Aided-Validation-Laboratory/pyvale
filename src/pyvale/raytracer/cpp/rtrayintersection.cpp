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
#include "rtshapefuncs.h"
#include "rtmaterials.h"
#include "rteigentypes.h"



/* ********************************************** 
 * TRI3 intersection
********************************************** */

IntersectionOutput intersect_bvh_tri3(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_triangle_count) {

    // Go through all the triangles and find an intersection of each triangle with a ray
    static constexpr int NODES_PER_ELEMENT = static_cast<int>(ElementNodeCount::TRI3); // Number of nodes per triangle/quad. Used for some of flat indexing.
    static constexpr double EPSILON = 1e-12;
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
        int base_stride = triangle_idx * NODES_PER_ELEMENT * NODE_COORDINATES;
        // Go over x, y, z coordinates
        for (int j = 0; j < NODE_COORDINATES; j++) {
            // Access n-th node via base_stride + n, where n e [0, 2] for TRI3
            // Then use access x, y, z coordinates in flattened array
            // E.g., [base_stride + 1 * NODE_COORDINATES + 2] would be node_1, coordinate z
            nodes0(triangle_idx, j) = node_coords[base_stride + 0 * NODE_COORDINATES + j];
            edge0(triangle_idx, j) = node_coords[base_stride + 1 * NODE_COORDINATES + j] - node_coords[base_stride + 0 * NODE_COORDINATES + j];
            nEdge2(triangle_idx, j) = node_coords[base_stride + 2 * NODE_COORDINATES + j] - node_coords[base_stride + 0 * NODE_COORDINATES + j];
        }
    }

    EiVectorD3d geometric_normals = cross_rowwise(edge0, nEdge2); // not normalised! Shape (faces, 3)

    // Step 1: Quantities for the Moller Trumbore method
    EiArrayD3d p_vec = cross_rowwise(ray_directions, nEdge2); // Assigns a vector to an array variable, but Eigen automatically converts so long as the underlying sizes are correct at initialization. Shape (faces, 3)
    EiArrayD1d discriminants = dot_rowwise(edge0, p_vec);  // Row-wise dot product; shape (faces, 1)

    // Step 2: Culling.
    //Discriminant negative -> triangle is back-facing. If discriminant is close to 0, ray and triangle are parallel and ray misses the triangle.
    EiBoolMask valid_mask = discriminants.abs() > EPSILON;
    // PREVIOUS VERSION 
    /*
    EiBoolMask valid_mask = (discriminants > EPSILON) && (discriminants > 0); // old
    if (!valid_mask.any()) {
        //std::cout << "Condition 1 triggered" << std::endl;
        return negative_output; // No intersection - return infinity
    }
*/
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
    EiArrayD1d barycentric_w = 1.0 - barycentric_u - barycentric_v; // barycentric_w
    barycentric_coordinates.col(0) = barycentric_w;
    barycentric_coordinates.col(1) = barycentric_u;
    barycentric_coordinates.col(2) = barycentric_v;
    //EiVectorD3d shading_normals = ;
    /* Original order (u, v, w), but not representative of our and the texture does not render correctly 
    barycentric_coordinates.col(0) = barycentric_u;
    barycentric_coordinates.col(1) = barycentric_v;
    barycentric_coordinates.col(2) = barycentric_w;
    */
    return IntersectionOutput{ barycentric_coordinates, geometric_normals, t_values };
}

/* ********************************************** 
 *  QUAD4 intersection
********************************************** */

IntersectionOutput intersect_bvh_quad4(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_quad_count){
    // Go through all the quads and find an intersection of each quad with a ray
    // Method based on NVIDIA 2019, E. Haines, T. Akenine-Möller (eds.), Ray Tracing Gems, https://doi.org/10.1007/978-1-4842-4427-2_8 that should work for non-planar quads
    // More specifically: Chapter 8, "Cool Patches: A Geometric Approach to Ray/Bilinear Patch Intersections" by A. Reshetov

    static constexpr int COORDS_PER_ELEMENT = static_cast<int>(ElementNodeCount::QUAD4) * NODE_COORDINATES;
    static constexpr double EPSILON = 1e-12; // This works sensibly so long as we don't have a mesh of size like 0.001 (in whatever world units are chosen), but adaptive epsilon setting could probably be useful
    
    // 1. COORDINATES AND EDGES
    // Ray data broadcasted to use in vectorised operations on matrices
    // This is faster than doing it in a loop
    EiVectorD3d ray_directions = ray.direction.replicate(bvh_node_quad_count, 1);
    EiArrayD3d ray_origins = ray.origin.replicate(bvh_node_quad_count, 1).array();

    // Define default negative output if there is no intersection
    IntersectionOutput negative_output{
        Eigen::ArrayXXd(bvh_node_quad_count, NODE_COORDINATES), // elem_interp_coords
        EiVectorD3d::Zero(bvh_node_quad_count, NODE_COORDINATES), //  geometric_normals
        Eigen::Vector<double, Eigen::Dynamic>::Constant(bvh_node_quad_count, 1, std::numeric_limits<double>::infinity()) // t_values
    };

    // 4 corner coordinates, where bl - bottom left, br - bottom right, tr - top right, tl - top left. 
    EiArrayD3d corners_bl(bvh_node_quad_count, NODE_COORDINATES), corners_br(bvh_node_quad_count, NODE_COORDINATES), corners_tr(bvh_node_quad_count, NODE_COORDINATES), corners_tl(bvh_node_quad_count, NODE_COORDINATES); // shape (faces, 3) each
    EiArrayD3d edges_bottom(bvh_node_quad_count, NODE_COORDINATES), edges_top(bvh_node_quad_count, NODE_COORDINATES), edges_right(bvh_node_quad_count, NODE_COORDINATES), edges_left(bvh_node_quad_count, NODE_COORDINATES); // shape (faces, 3) each

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

    EiArrayD3d normals = cross_rowwise(corners_br-corners_bl, corners_tl-corners_tr).array(); // not normalised! Shape (faces,3); original code
    // Translate only the coordinates that need ray-origin subtraction
    corners_bl -= ray_origins;
    corners_br -= ray_origins;

    // 2. SOLVING THE INTERSECTION

    // 1. Solve quadratic a + b*u + c*u^2
    EiArrayD1d a = dot_rowwise(cross_rowwise(corners_bl, ray_directions), edges_left); // Row-wise dot product; shape (faces, 1)
    EiArrayD1d c = dot_rowwise(normals, ray_directions); //
    EiArrayD1d b = dot_rowwise(cross_rowwise(corners_br, ray_directions), edges_right);
    b = b - (a + c); 
    EiArrayD1d discriminants = b.square() - 4*a*c; // Shape (faces, 1)
    // Discriminant negative -> triangle is back-facing. If discriminant is close to 0, ray and triangle are parallel and ray misses the triangle.
    //EiBoolMask valid_mask = (discriminants > EPSILON) && (discriminants > 0); // Shape (faces, 1)
    EiBoolMask valid_mask = (discriminants > EPSILON); // Shape (faces, 1)
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
    //EiBoolMask u1_valid = valid_mask && (0.0 <= u1) && (u1 <= 1.0);
    //EiBoolMask u1_valid = valid_mask && (EPSILON <= u1) && (u1 <= 1.0);
    EiBoolMask u1_valid = valid_mask && (0.0 <= u1) && (u1 <= 1.0);
    EiArrayD3d pa1 = lerp_vectorised(corners_bl, corners_br, u1);
    EiArrayD3d pb1 = lerp_vectorised(edges_left, edges_right, u1);
    EiVectorD3d n1 = cross_rowwise(ray_directions, pb1.matrix());
    EiArrayD1d det1 = n1.array().square().rowwise().sum();
    EiBoolMask det1_valid = det1 > EPSILON; // Non-zero discriminant
    EiVectorD3d n1_cross = cross_rowwise(n1, pa1);
    //EiArrayD1d t1 = dot_rowwise(n1_cross, pb1); 
    EiArrayD1d t1 = dot_rowwise(n1_cross, pb1) / det1; 
    EiArrayD1d nodes_1 = dot_rowwise(n1_cross, ray_directions);
    // Create a hit mask if we are in the u1 branch, discriminant is valid, and t1 > 0 and 0.0 <= nodes_1 <= det1
    //EiBoolMask hit_mask1 = u1_valid && det1_valid && (t1 > 0.0) && (nodes_1 >= 0.0) && (det1 >= nodes_1);
    //EiBoolMask hit_mask1 = u1_valid && det1_valid && (t1 > EPSILON) && (nodes_1 >= EPSILON) && (det1 >= nodes_1);
    EiBoolMask hit_mask1 = u1_valid && det1_valid && (t1 > EPSILON) && (nodes_1 >= 0.0) && (det1 >= nodes_1);


    // Update values where we have a hit
    t_values = hit_mask1.select(t1, t_values);
    u_values = hit_mask1.select(u1, u_values);
    v_values = hit_mask1.select(nodes_1/det1, v_values);

    // 3.2. Evaluate case 0.0 <= u2 <= 1.0 - Slightly different since u1 might be good and we need 0 < t2 < t1
    EiBoolMask u2_valid = valid_mask && (0.0 <= u2) && (u2 <= 1.0);
    EiArrayD3d pa2 = lerp_vectorised(corners_bl, corners_br, u2);
    EiArrayD3d pb2 = lerp_vectorised(edges_left, edges_right, u2);
    EiVectorD3d n2 = cross_rowwise(ray_directions, pb2.matrix());
    EiArrayD1d det2 = n2.array().square().rowwise().sum();
    EiBoolMask det2_valid = det2 > EPSILON; // Non-zero discriminant
    EiVectorD3d n2_cross = cross_rowwise(n2, pa2);
    EiArrayD1d t2 = dot_rowwise(n2_cross, pb2) / det2; //
    EiArrayD1d nodes_2 = dot_rowwise(n2_cross, ray_directions);
    // Hit mask for u2 branch, discriminant is valid, 0 < t2 < t and 0 <= nodes_2 <= det2
    //EiBoolMask hit_mask2 = u2_valid && det2_valid && (t2 > 0.0) && (t_values > t2) && (det2 >= nodes_2) && (nodes_2 >= 0);
    EiBoolMask hit_mask2 = u2_valid && det2_valid && (t2 > EPSILON) && (t_values > t2) && (det2 >= nodes_2) && (nodes_2 >= 0);

    // Update values where we have a hit
    t_values = hit_mask2.select(t2, t_values);
    u_values = hit_mask2.select(u2, u_values);
    v_values = hit_mask2.select(nodes_2/det2, v_values);

    // Apply ray segment bounds
    EiBoolMask in_range = (t_values >= ray.t_min) && (t_values <= ray.t_max);
    for (int i = 0; i < t_values.rows(); ++i){
        if (!in_range(i, 0)){
            t_values(i, 0) = std::numeric_limits<double>::infinity();
        } 
    }

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


/* ********************************************** 
 *  Precision parameters for QUAD8 and QUAD9
********************************************** */

// Sub-triangle Moller-Trumbore acceptance slack (on barycentrics)
static const double eps_sub_bary = 1e-6;
// Minimum |det| accepted in the sub-triangle MT solve
static const double eps_sub_det = 1e-12;
// Minimum t accepted (initial guess and final acceptance floor)
static const double eps_t_min = 1e-7;

// Newton residual tolerance: max(eps_res_abs, eps_res_rel * find_element_diagonal)
static const double eps_res_rel = 1e-10;
static const double eps_res_abs = 1e-12;

// Parametric slack on [-1, 1]^2 for final acceptance
static const double eps_param_accept = 1e-6;
// Trust region: abort the Newton seed if (xi, eta) leaves this box
static const double xi_eta_trust = 1.6;

// Newton iteration budget
static const int iter_max = 30;
// Backtracking line-search budget per Newton step
static const int backtrack_max = 8;
static const double backtrack_factor= 0.5;

/* ********************************************** 
 *  QUAD9
********************************************** */

// Evaluate S(xi, eta) = sum_i N_i(xi, eta) * x_i for the 9 quad9 nodes.
static inline EiVector3d evaluate_surface_quad9(const double xi, const double eta,
        const std::array<EiVector3d, ElementNodeCount::QUAD9>& nodes) {
    const std::array<double, ElementNodeCount::QUAD9> N = compute_shape_quad9(xi, eta);
    EiVector3d S = EiVector3d::Zero();
    for (int i = 0; i < ElementNodeCount::QUAD9; ++i) {
        S += N[i] * nodes[i];
    }
    return S;
}

static inline void quad9_node_xi_eta(Eigen::Vector2d out[9]) {
    // Parametric (xi, eta) of each QUAD9 node, in shape-function index order
    out[0] = Eigen::Vector2d(-1.0, -1.0); // Bottom left corner
    out[1] = Eigen::Vector2d( 1.0, -1.0); // Bottom right corner
    out[2] = Eigen::Vector2d( 1.0,  1.0); // Top right corner
    out[3] = Eigen::Vector2d(-1.0,  1.0); // Top left corner
    out[4] = Eigen::Vector2d( 0.0, -1.0); // Bottom mid
    out[5] = Eigen::Vector2d( 1.0,  0.0); // Right mid
    out[6] = Eigen::Vector2d( 0.0,  1.0); // Top mid
    out[7] = Eigen::Vector2d(-1.0,  0.0); // Left mid
    out[8] = Eigen::Vector2d( 0.0,  0.0); // Centre
}

// 8-triangle fan around index 8 (centre node)
static const int quad_sub_tris[8][3] = {
    {0, 4, 8}, {4, 1, 8},
    {1, 5, 8}, {5, 2, 8},
    {2, 6, 8}, {6, 3, 8},
    {3, 7, 8}, {7, 0, 8}
};


double intersect_quad9(const Ray& ray,
    const std::array<EiVector3d, ElementNodeCount::QUAD9>& nodes,
    EiVector3d& surface_normals_out,
    Eigen::Vector2d& xi_eta_out) {

    // Element-scale residual tolerance
    const double diagonal = find_element_diagonal(&nodes[0], ElementNodeCount::QUAD9);
    double res_tol_local = eps_res_rel * diagonal; // Scaled tolerance for residuals based on the element diagonal
    if (res_tol_local < eps_res_abs) res_tol_local = eps_res_abs;

    // Parametric coordinates of every node, used to recover (xi, eta) from sub-triangle barycentrics
    Eigen::Vector2d xi_eta_table[9];
    quad9_node_xi_eta(xi_eta_table);

    // Best hit found so far across all seeds
    double best_t = std::numeric_limits<double>::infinity();
    Eigen::Vector2d best_xi_eta(0.0, 0.0);
    bool have_hit = false;

    // Iterate over sub-triangle fan, run MT to get seeds, then Newton
    for (int s = 0; s < 8; ++s) {
        const EiVector3d nodes_0 = nodes[quad_sub_tris[s][0]];
        const EiVector3d nodes_1 = nodes[quad_sub_tris[s][1]];
        const EiVector3d nodes_2 = nodes[quad_sub_tris[s][2]];

        const EiVector3d edge1 = nodes_1 - nodes_0;
        const EiVector3d edge2 = nodes_2 - nodes_0;

        // Moller-Trumbore TRI3 intersection
        const EiVector3d pvec = cross_rowwise_vec3(ray.direction, edge2);
        const double det = edge1.dot(pvec);
        if (std::fabs(det) < eps_sub_det) continue;
        const double inv_det = 1.0 / det;

        const EiVector3d tvec = ray.origin - nodes_0;
        const double u_sub = tvec.dot(pvec) * inv_det;
        if (u_sub < -eps_sub_bary || u_sub > 1.0 + eps_sub_bary) continue;

        const EiVector3d qvec = cross_rowwise_vec3(tvec, edge1);
        const double v_sub = ray.direction.dot(qvec) * inv_det;
        if (v_sub < -eps_sub_bary || (u_sub + v_sub) > 1.0 + eps_sub_bary) continue;

        const double t_sub = edge2.dot(qvec) * inv_det;
        if (t_sub <= eps_t_min) continue;

        // Initial guess in (xi, eta), via linear interpolation of the sub-triangle vertices' parametric coordinates
        const double w_sub = 1.0 - u_sub - v_sub;
        Eigen::Vector2d xi_eta_guess =
              w_sub * xi_eta_table[quad_sub_tris[s][0]]
            + u_sub * xi_eta_table[quad_sub_tris[s][1]]
            + v_sub * xi_eta_table[quad_sub_tris[s][2]];

        // Skip seeds that cannot improve on the current best
        if (t_sub >= best_t) continue;

        // Damped Newton on F(t, xi, eta) = O + t*D - S(xi, eta)
        double t = t_sub;
        double xi = xi_eta_guess.x();
        double eta = xi_eta_guess.y();

        EiVector3d S = evaluate_surface_quad9(xi, eta, nodes);
        Eigen::Vector3d residual = (ray.origin + t * ray.direction).transpose() - S.transpose();
        double res_norm = residual.norm();
        bool converged = false;

        for (int it = 0; it < iter_max; ++it) {
            if (res_norm < res_tol_local) {
                converged = true;
                break;
            }

            Eigen::Matrix<double, 3, 2> J = get_face_Jacobian_quad9(xi, eta, nodes);

            Eigen::Matrix3d M;
            M.col(0) = ray.direction.transpose();
            M.col(1) = -J.col(0);
            M.col(2) = -J.col(1);

            Eigen::Vector3d delta;
            if (!solve_3x3(M, -residual, delta)) break; // Singular

            // Backtracking line search on ||F|| to prevent divergence near folded surfaces
            double step = 1.0;
            bool step_ok = false;
            for (int bt = 0; bt < backtrack_max; ++bt) {
                const double t_n = t + step * delta(0);
                const double xi_n = xi + step * delta(1);
                const double eta_n = eta + step * delta(2);

                if (std::fabs(xi_n)  > xi_eta_trust ||
                    std::fabs(eta_n) > xi_eta_trust) {
                    step *= backtrack_factor;
                    continue;
                }

                const EiVector3d S_n = evaluate_surface_quad9(xi_n, eta_n, nodes);
                const Eigen::Vector3d res_n = (ray.origin + t_n * ray.direction).transpose() - S_n.transpose();
                const double n_n = res_n.norm();

                if (n_n < res_norm) {
                    t = t_n;
                    xi = xi_n;
                    eta = eta_n;
                    residual = res_n;
                    res_norm = n_n;
                    step_ok = true;
                    break;
                }
                step *= backtrack_factor;
            }
            if (!step_ok) break; // No progress on this seed
        }
        // Last-chance acceptance after exhausting iterations
        if (!converged && res_norm < res_tol_local) converged = true;
        if (!converged) continue;

        // Final acceptance gates
        if (t <= eps_t_min) continue;
        if (t < ray.t_min || t > ray.t_max) continue;
        if (std::fabs(xi)  > 1.0 + eps_param_accept) continue;
        if (std::fabs(eta) > 1.0 + eps_param_accept) continue;

        if (t < best_t) {
            best_t = t;
            best_xi_eta = Eigen::Vector2d(xi, eta);
            have_hit = true;
        }
    }

    if (!have_hit) return std::numeric_limits<double>::infinity();

    // Geometric normal at the converged hit, from the true quadratic Jacobian
    Eigen::Matrix<double, 3, 2> J_hit = get_face_Jacobian_quad9(best_xi_eta.x(), best_xi_eta.y(), nodes);
    EiVector3d normal = (J_hit.col(0).cross(J_hit.col(1))).transpose();

    surface_normals_out = normal;
    xi_eta_out = best_xi_eta;
    return best_t;
}

IntersectionOutput intersect_bvh_quad9(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_quad_count) {

    // Default "no intersection" output
    IntersectionOutput negative_output{
        Eigen::ArrayXXd(bvh_node_quad_count, NODE_COORDINATES),
        EiVectorD3d::Zero(bvh_node_quad_count, NODE_COORDINATES),
        Eigen::Vector<double, Eigen::Dynamic>::Constant(
            bvh_node_quad_count, 1, std::numeric_limits<double>::infinity())
    };

    EiVectorD3d geometric_normals = EiVectorD3d::Zero(bvh_node_quad_count, 3);
    Eigen::ArrayXXd t_values(bvh_node_quad_count, 1);
    Eigen::ArrayXXd xi_values(bvh_node_quad_count, 1);
    Eigen::ArrayXXd eta_values(bvh_node_quad_count, 1);

    for (unsigned int element_idx = 0; element_idx < bvh_node_quad_count; ++element_idx) {
        std::array<EiVector3d, ElementNodeCount::QUAD9> quad9_node_coords; // Single QUAD9 as an array of its constituent nodal coordinates in the EiVector3d format for Jacobian calculations
        get_face_data_vector(element_idx, node_coords, ElementNodeCount::QUAD9, &quad9_node_coords[0]);
        EiVector3d n_tmp;
        Eigen::Vector2d xe_tmp;
        const double t = intersect_quad9(ray, quad9_node_coords, n_tmp, xe_tmp);

        t_values  (element_idx, 0) = t;
        xi_values (element_idx, 0) = xe_tmp.x();
        eta_values(element_idx, 0) = xe_tmp.y();
        if (std::isfinite(t)) {
            for (int k = 0; k < 3; ++k){
                geometric_normals(element_idx, k) = n_tmp(k);
            }
        }
    }

    // Mask out anything outside the ray segment
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask =
        (t_values > 0.0) && (t_values >= ray.t_min) && (t_values <= ray.t_max);
    if (!valid_mask.any()) return negative_output;

    for (Eigen::Index i = 0; i < t_values.rows(); ++i) {
        for (Eigen::Index j = 0; j < t_values.cols(); ++j) {
            if (!valid_mask(i, j)) {
                t_values(i, j) = std::numeric_limits<double>::infinity();
            }
        }
    }

    // Pack output: (u, v, 0) with u = (xi+1)/2, v = (eta+1)/2
    Eigen::ArrayXXd quad_coords = Eigen::ArrayXXd::Zero(bvh_node_quad_count, NODE_COORDINATES);
    quad_coords.col(0) = 0.5 * (xi_values.col(0)  + 1.0);
    quad_coords.col(1) = 0.5 * (eta_values.col(0) + 1.0);
    // Column 2 left at 0

    return IntersectionOutput{ quad_coords, geometric_normals, t_values };
}

/* ********************************************** 
 *  QUAD8
********************************************** */
// Same structure as QUAD9, but:
// - Shape-function set has 8 entries (no centre node),
// - We synthesise a proxy centre at (xi, eta) = (0, 0) from the average of
//   the 4 mid-edge nodes (closer to the true serendipity surface centre
//   than the corner average), purely for the sub-triangle fan seed step.
// - Newton residual still uses the true 8-node surface S(xi, eta)

static inline EiVector3d evaluate_surface_quad8(const double xi, const double eta,
        const std::array<EiVector3d, ElementNodeCount::QUAD8>& nodes) {

    const std::array<double, ElementNodeCount::QUAD8> N = compute_shape_quad8(xi, eta);
    EiVector3d S = EiVector3d::Zero();
    for (int i = 0; i < ElementNodeCount::QUAD8; ++i) {
        S += N[i] * nodes[i];
    }
    return S;
}

double intersect_quad8(const Ray& ray,
    const std::array<EiVector3d, ElementNodeCount::QUAD8>& nodes,
    EiVector3d& surface_normals_out,
    Eigen::Vector2d& xi_eta_out) {

    const double diagonal = find_element_diagonal(&nodes[0], ElementNodeCount::QUAD8);
    double res_tol_local = eps_res_rel * diagonal;
    if (res_tol_local < eps_res_abs) res_tol_local = eps_res_abs;

    // Build the 9-entry proxy: 8 quad nodes + synthesised centre at index 8
    // The synthesised centre is only used for the sub-triangle fan; the Newton residual uses the real 8-node surface
    EiVector3d proxy_nodes[9];
    Eigen::Vector2d xi_eta_table[9];
    // Copy nodes to proxy nodes to differentiate between real data and the vector where we added computed 9-th centre node
    for (int i = 0; i < 8; ++i){
        proxy_nodes[i] = nodes[i];
    } 
    proxy_nodes[8] = 0.25 * (nodes[4] + nodes[5] + nodes[6] + nodes[7]);

    xi_eta_table[0] = Eigen::Vector2d(-1.0, -1.0);
    xi_eta_table[1] = Eigen::Vector2d( 1.0, -1.0);
    xi_eta_table[2] = Eigen::Vector2d( 1.0,  1.0);
    xi_eta_table[3] = Eigen::Vector2d(-1.0,  1.0);
    xi_eta_table[4] = Eigen::Vector2d( 0.0, -1.0);
    xi_eta_table[5] = Eigen::Vector2d( 1.0,  0.0);
    xi_eta_table[6] = Eigen::Vector2d( 0.0,  1.0);
    xi_eta_table[7] = Eigen::Vector2d(-1.0,  0.0);
    xi_eta_table[8] = Eigen::Vector2d( 0.0,  0.0);

    double best_t = std::numeric_limits<double>::infinity();
    Eigen::Vector2d best_xi_eta(0.0, 0.0);
    bool have_hit = false;

    for (int s = 0; s < 8; ++s) {
        const EiVector3d nodes_0 = proxy_nodes[quad_sub_tris[s][0]];
        const EiVector3d nodes_1 = proxy_nodes[quad_sub_tris[s][1]];
        const EiVector3d nodes_2 = proxy_nodes[quad_sub_tris[s][2]];

        const EiVector3d edge1 = nodes_1 - nodes_0;
        const EiVector3d edge2 = nodes_2 - nodes_0;

        const EiVector3d pvec = cross_rowwise_vec3(ray.direction, edge2);
        const double det = edge1.dot(pvec);
        if (std::fabs(det) < eps_sub_det) continue;
        const double inv_det = 1.0 / det;

        const EiVector3d tvec = ray.origin - nodes_0;
        const double u_sub = tvec.dot(pvec) * inv_det;
        if (u_sub < -eps_sub_bary || u_sub > 1.0 + eps_sub_bary) continue;

        const EiVector3d qvec = cross_rowwise_vec3(tvec, edge1);
        const double v_sub = ray.direction.dot(qvec) * inv_det;
        if (v_sub < -eps_sub_bary || (u_sub + v_sub) > 1.0 + eps_sub_bary) continue;

        const double t_sub = edge2.dot(qvec) * inv_det;
        if (t_sub <= eps_t_min) continue;

        const double w_sub = 1.0 - u_sub - v_sub;
        Eigen::Vector2d xi_eta_guess =
              w_sub * xi_eta_table[quad_sub_tris[s][0]]
            + u_sub * xi_eta_table[quad_sub_tris[s][1]]
            + v_sub * xi_eta_table[quad_sub_tris[s][2]];

        if (t_sub >= best_t) continue;

        // Newton on the true quad8 surface
        double t = t_sub;
        double xi = xi_eta_guess.x();
        double eta = xi_eta_guess.y();

        EiVector3d S = evaluate_surface_quad8(xi, eta, nodes);
        Eigen::Vector3d residual = (ray.origin + t * ray.direction).transpose() - S.transpose();
        double res_norm = residual.norm();
        bool converged = false;

        for (int it = 0; it < iter_max; ++it) {
            if (res_norm < res_tol_local) {
                converged = true;
                break;
            }

            Eigen::Matrix<double, 3, 2> J = get_face_Jacobian_quad8(xi, eta, nodes);

            Eigen::Matrix3d M;
            M.col(0) =  ray.direction.transpose();
            M.col(1) = -J.col(0);
            M.col(2) = -J.col(1);

            Eigen::Vector3d delta;
            if (!solve_3x3(M, -residual, delta)) break;

            // Backtracking line search on ||F|| to prevent divergence near folded surfaces
            double step = 1.0;
            bool   step_ok = false;
            for (int bt = 0; bt < backtrack_max; ++bt) {
                const double t_n = t + step * delta(0);
                const double xi_n = xi + step * delta(1);
                const double eta_n = eta + step * delta(2);

                if (std::fabs(xi_n)  > xi_eta_trust ||
                    std::fabs(eta_n) > xi_eta_trust) {
                    step *= backtrack_factor;
                    continue;
                }

                const EiVector3d S_n = evaluate_surface_quad8(xi_n, eta_n, nodes);
                const Eigen::Vector3d res_n = (ray.origin + t_n * ray.direction).transpose() - S_n.transpose();
                const double n_n = res_n.norm();

                if (n_n < res_norm) {
                    t = t_n;
                    xi = xi_n;
                    eta = eta_n;
                    residual = res_n;
                    res_norm = n_n;
                    step_ok = true;
                    break;
                }
                step *= backtrack_factor;
            }
            if (!step_ok) break;
        }
        if (!converged && res_norm < res_tol_local) converged = true;
        if (!converged) continue;

        if (t <= eps_t_min) continue;
        if (t < ray.t_min || t > ray.t_max) continue;
        if (std::fabs(xi)  > 1.0 + eps_param_accept) continue;
        if (std::fabs(eta) > 1.0 + eps_param_accept) continue;

        if (t < best_t) {
            best_t      = t;
            best_xi_eta = Eigen::Vector2d(xi, eta);
            have_hit    = true;
        }
    }

    if (!have_hit) return std::numeric_limits<double>::infinity();

    Eigen::Matrix<double, 3, 2> J_hit = get_face_Jacobian_quad8(best_xi_eta.x(), best_xi_eta.y(), nodes);
    EiVector3d normal = (J_hit.col(0).cross(J_hit.col(1))).transpose();

    surface_normals_out = normal;
    xi_eta_out = best_xi_eta;
    return best_t;
}

IntersectionOutput intersect_bvh_quad8(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_quad_count) {

    IntersectionOutput negative_output{
        Eigen::ArrayXXd(bvh_node_quad_count, NODE_COORDINATES),
        EiVectorD3d::Zero(bvh_node_quad_count, NODE_COORDINATES),
        Eigen::Vector<double, Eigen::Dynamic>::Constant(bvh_node_quad_count, 1, std::numeric_limits<double>::infinity())};

    EiVectorD3d geometric_normals = EiVectorD3d::Zero(bvh_node_quad_count, 3);
    Eigen::ArrayXXd t_values(bvh_node_quad_count, 1);
    Eigen::ArrayXXd xi_values(bvh_node_quad_count, 1);
    Eigen::ArrayXXd eta_values(bvh_node_quad_count, 1);

    for (unsigned int element_idx = 0; element_idx < bvh_node_quad_count; ++element_idx) {
        std::array<EiVector3d, ElementNodeCount::QUAD8> quad8_node_coords; // Single QUAD8 as an array of its constituent nodal coordinates in the EiVector3d format for Jacobian calculations
        get_face_data_vector(element_idx, node_coords, ElementNodeCount::QUAD8, &quad8_node_coords[0]);

        EiVector3d n_tmp;
        Eigen::Vector2d xe_tmp;
        const double t = intersect_quad8(ray, quad8_node_coords, n_tmp, xe_tmp);

        t_values  (element_idx, 0) = t;
        xi_values (element_idx, 0) = xe_tmp.x();
        eta_values(element_idx, 0) = xe_tmp.y();
        if (std::isfinite(t)) {
            for (int k = 0; k < 3; ++k){
                geometric_normals(element_idx, k) = n_tmp(k);
            } 
        }
    }

    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask =
        (t_values > 0.0) && (t_values >= ray.t_min) && (t_values <= ray.t_max);
    if (!valid_mask.any()) return negative_output;

    for (Eigen::Index i = 0; i < t_values.rows(); ++i) {
        for (Eigen::Index j = 0; j < t_values.cols(); ++j) {
            if (!valid_mask(i, j)) {
                t_values(i, j) = std::numeric_limits<double>::infinity();
            }
        }
    }

    Eigen::ArrayXXd quad_coords = Eigen::ArrayXXd::Zero(bvh_node_quad_count, NODE_COORDINATES);
    quad_coords.col(0) = 0.5 * (xi_values.col(0)  + 1.0);
    quad_coords.col(1) = 0.5 * (eta_values.col(0) + 1.0);

    return IntersectionOutput{ quad_coords, geometric_normals, t_values };
}

/* ********************************************** 
 *  TRI6
********************************************** */

// Mapping for sub-triangulation (indices within the 6-node nodes vector)
// Quadratic layout: 0,1,2 are corners; 3,4,5 are midpoints of (0-1), (1-2), (2-0)
int sub_tris[4][3] = {
    {0, 3, 5}, // Bottom-left
    {3, 1, 4}, // Bottom-right
    {5, 4, 2}, // Top
    {3, 4, 5}  // Center
};

// Barycentric coordinate offsets for the sub-triangles to map back to (g, h)
// These represent the (g, h) coordinates of the nodes in nodes
Eigen::Vector2d nodes_gh[6] = {
    {0,0}, {1,0}, {0,1}, {0.5,0}, {0.5,0.5}, {0,0.5}
};

double intersect_tri6(const Ray &ray,
    const std::array<EiVector3d, ElementNodeCount::TRI6> nodes,
    EiVector3d &surface_normals_out,
    Eigen::Vector2d &uv) {
    /* 
    Function to find intersection between one TRI6 triangle and a ray.

    Parameters
    ----------
    ray : const Ray
        Ray with which the intersection is found
    nodes : const std::array<EiVector3d, 6>
        Triangle node coordinates
    surface_normals_out : EiVector3d
        Empty vector to which the normal at the intersection point should be loaded
    uv : Eigen::Vector2d
        Empty vector to which the triangle-relative UV coordinates at the intersection point should be loaded

    Returns
    -------
    double
        The distance t from the ray origin to the intersection point 
    */
    
    // Set precision parameters
    const double eps_init_guess1 = 1e-10;
    const double eps_init_guess2 = 0.1;
    const double eps_t = 1e-5;

    const double eps_sol1 = 1e-7;
    const double eps_sol2 = 1e-8;
    const double eps_sol3 = 1e-10;

    // Find initial guess based on linear triangle intersection
    double min_t = std::numeric_limits<double>::infinity();
    bool intersect = false;

    double best_sub_t = std::numeric_limits<double>::infinity();
    Eigen::Vector2d best_gh_guess(0.33, 0.33);
    bool found_guess = false;

    for (int s = 0; s < 4; ++s) {
        EiVector3d nodes_0 = nodes[sub_tris[s][0]];
        EiVector3d nodes_1 = nodes[sub_tris[s][1]];
        EiVector3d nodes_2 = nodes[sub_tris[s][2]];

        EiVector3d edge1 = nodes_1 - nodes_0;
        EiVector3d edge2 = nodes_2 - nodes_0;

        EiVector3d pvec =
            (cross_rowwise(ray.direction, edge2));

        double det = edge1.dot(pvec);
        if (fabs(det) < eps_init_guess1) continue;

        double invDet = 1.0 / det;

        EiVector3d tvec = ray.origin - nodes_0;
        double u_sub = tvec.dot(pvec) * invDet;
        if (u_sub < 0 - eps_init_guess2 || u_sub > 1 + eps_init_guess2) continue;

        EiVector3d qvec =
            (cross_rowwise(tvec, edge1));

        double v_sub = ray.direction.dot(qvec) * invDet;
        if (v_sub < 0 - eps_init_guess2 || (u_sub + v_sub) > 1 + eps_init_guess2) continue;

        double t_sub = edge2.dot(qvec) * invDet;

        if (t_sub > eps_t && t_sub < best_sub_t) {
            best_sub_t = t_sub;

            double w_sub = 1.0 - u_sub - v_sub;
            best_gh_guess =
                w_sub * nodes_gh[sub_tris[s][0]] +
                u_sub * nodes_gh[sub_tris[s][1]] +
                v_sub * nodes_gh[sub_tris[s][2]];

            found_guess = true;
        }
    }

    // Search for the intersection if the initial guess is found
    if (found_guess) {
        Eigen::Vector2d gh = best_gh_guess;
        double t = best_sub_t;

        for (int iter = 0; iter < iter_max; ++iter) {
            Eigen::VectorXd N = compute_shape_tri6(gh.x(), gh.y());
    
            EiVector3d P = EiVector3d::Zero();
            for (int i = 0; i < 6; ++i)
                P += N[i] * nodes[i];
    
            EiVector3d residual = ray.origin + t * ray.direction - P;
    
            if (residual.norm() < eps_sol1) {
                if (gh.x() >= 0 - eps_sol2 &&
                    gh.y() >= 0 - eps_sol2 &&
                    (gh.x() + gh.y()) <= 1 + eps_sol2)
                    {
                        if (t < min_t && t > eps_t) {
                            min_t = t;
        
                            Eigen::Matrix<double, 3, 2> J =
                                get_face_Jacobian_tri6(gh.x(), gh.y(), nodes);
        
                            EiVector3d normal =
                                (J.col(0).cross(J.col(1))).transpose();
        
                            surface_normals_out = normal;
                            uv = gh;
                            intersect = true;
                        }
                    }
                break;
            }
    
            Eigen::Matrix<double, 3, 2> J =
                get_face_Jacobian_tri6(gh.x(), gh.y(), nodes);
    
            Eigen::Matrix3d M;
            M.col(0) = ray.direction.transpose();
            M.col(1) = -J.col(0);
            M.col(2) = -J.col(1);
    
            if (std::abs(M.determinant()) < eps_sol3) break;
    
            Eigen::Vector3d delta = M.inverse() * (-residual.transpose());
    
            t += delta.x();
            gh.x() += delta.y();
            gh.y() += delta.z();
    
            if (gh.x() < -0.5 || gh.y() < -0.5 || (gh.x() + gh.y()) > 1.5)
                break;
        }
    }

    return intersect ? min_t : std::numeric_limits<double>::infinity();
}

IntersectionOutput intersect_bvh_tri6(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_triangle_count) {
    /* 
    Function to find intersection between given TRI6 triangles and a ray.

    Parameters
    ----------
    ray : const Ray
        Ray with which the intersection is found
    node_coords : const std::vector<double>
        Triangle node coordinates
    bvh_node_triangle_count : const unsigned int
        Number of the given triangles

    Returns
    -------
    IntersectionOutput
        Intersection result containing information about intersections of each triangle with the ray 
    */

    // Define default negative output if there is no intersection
    IntersectionOutput negative_output{
        Eigen::ArrayXXd(bvh_node_triangle_count, 3),
        EiVectorD3d::Zero(bvh_node_triangle_count, 3),
        Eigen::Vector<double, Eigen::Dynamic>::Constant(bvh_node_triangle_count, 1, std::numeric_limits<double>::infinity())
    };

    // Iterate through all the triangles and find an intersection of each triangle with the ray

    EiVectorD3d geometric_normals(bvh_node_triangle_count, 3);
    Eigen::ArrayXXd t_values(bvh_node_triangle_count, 1);
    Eigen::ArrayXXd barycentric_u(bvh_node_triangle_count, 1);
    Eigen::ArrayXXd barycentric_v(bvh_node_triangle_count, 1);

    for (int triangle_idx = 0; triangle_idx < bvh_node_triangle_count; triangle_idx++) {
        std::array<EiVector3d, ElementNodeCount::TRI6> tri6_node_coords; // Single TRI6 as an array of its constituent nodal coordinates in the EiVector3d format for Jacobian calculations
        get_face_data_vector(triangle_idx, node_coords, ElementNodeCount::TRI6, &tri6_node_coords[0]);

        EiVector3d n_tmp;
        Eigen::Vector2d uv_tmp;
        double t = intersect_tri6(ray, tri6_node_coords, n_tmp, uv_tmp);

        // Convert the intersection results to acceptable format
        for (int i = 0; i < 3; ++i) {
            geometric_normals(triangle_idx, i) = n_tmp(i);
        }
        
        // Store the results
        t_values(triangle_idx) = t;
        barycentric_u(triangle_idx) = uv_tmp.x();
        barycentric_v(triangle_idx) = uv_tmp.y();
        
    }

    // Mask inappropriate values based on t_values
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask;
    valid_mask = (t_values > 0.0); // t=0.0 means no intersection with the triangle
    if (!valid_mask.any()) {
        //std::cout << "Condition 1 triggered" << std::endl;
        return negative_output; // No intersection -> return infinity
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
    Eigen::ArrayXXd barycentric_coordinates(bvh_node_triangle_count, 3);
    barycentric_coordinates.col(0) = barycentric_u;
    barycentric_coordinates.col(1) = barycentric_v;
    barycentric_coordinates.col(2) = 1.0 - barycentric_u - barycentric_v; // barycentric_w

    return IntersectionOutput{ barycentric_coordinates, geometric_normals, t_values };

}

/* ********************************************** 
 *  AABB
********************************************** */

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

    // Temporary debug because it often indicates something went wrong with secondary rays
    
    if (std::isnan(t_min) || std::isnan(t_max)) {
    std::cerr << "NaN slab: origin=" << ray.origin.transpose()
              << " dir=" << ray.direction.transpose()
              << " min=" << AABB.corner_min[0] << "," << AABB.corner_min[1] << "," << AABB.corner_min[2]
              << " max=" << AABB.corner_max[0] << "," << AABB.corner_max[1] << "," << AABB.corner_max[2]
              << "\n";
    }
    
    // t_min < t_max - Ray which just touches a corner, edge, or face of the AABB will be considered non-intersecting
    // t_min <= t_max - Rays which touch the box boundary are considered intersecting. A bit of a degenerate case, but decided to include it here, hence more relaxed inequality.
    // t_min < ray.t_max - Clip to ray segment
    return t_min <= t_max && t_max > 0.0 && t_min < ray.t_max; // False => No overlap => Ray does not intersect the AABB.
}

/* ********************************************** 
 *  BLAS and TLAS
********************************************** */

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
        int node_id = stack.back();
        //std::cout << "Intersecting BLAS node ID: " << node_id << std::endl;
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
                intersection_record.normal_surface = out_intersection.geometric_normals.row(min_row_idx);
                intersection_record.ray_material_ptr = mesh_bvh.ray_material_ptr;
                 // Uncomment the below 2 lines if deciding to go for switch-based dispatch in return_ray_color
                //intersection_record.material = mesh_bvh.material;
                //intersection_record.object_type = mesh_bvh.object_type;

                // Data for refractive indices
                intersection_record.hit_blas_idx = mesh_bvh.blas_idx;
                intersection_record.hit_blas_priority = mesh_bvh.priority;
                intersection_record.refractive_index = mesh_bvh.refractive_index;
                intersection_record.thickness = mesh_bvh.thickness;
                
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

bool intersect_TLAS(const Ray& ray,
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
                // Note: Comment out the below check if MAX_ELEMENTS_PER_LEAF = 1; in build_TLAS because then TLAS node AABB = BLAS AABB, so this check is unnecessary and we can also remove AABB from BLAS struct
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
     // To avoid having to evaluate this in the return_ray_color loop directly
     if (out_intersection_record.t == std::numeric_limits<double>::infinity()) {
        return false; // No hit
     }
     else {
        return true;
    }
}
        
           
// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD header files
#include <iostream>
#include <limits>

// ray tracer header files
#include "rtrayintersection_extracted.h"
#include "rtelemconstants.h"
#include "rtmathutils.h"
// #include "rtbvh.h"
// #include "rtcolorsampling.h"
#include "rtshapefuncs.h"
// #include "rtmaterials.h"
#include "rteigentypes.h"

// ================================================================================
//  Precision parameters for QUAD8 and QUAD9
// ================================================================================

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

// ================================================================================
//  QUAD9
// ================================================================================

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


// ================================================================================
//  QUAD8
// ================================================================================
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
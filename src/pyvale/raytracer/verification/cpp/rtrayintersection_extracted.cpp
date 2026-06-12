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
           
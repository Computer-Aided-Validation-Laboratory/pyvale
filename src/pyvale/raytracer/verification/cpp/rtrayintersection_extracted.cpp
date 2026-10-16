#include <iostream>
#include <limits>
#include <array>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cassert>
#include <memory>
#include <optional>
#include <cmath>
#include <iomanip>

#include "rtrayintersection_extracted.h"
#include "rtelemconstants.h"
#include "rtmathutils.h"
#include "rtshapefuncs.h"
#include "rteigentypes.h"

//  Precision parameters for Newton solver and initial guess
// ================================================================================

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
// static const double xi_eta_trust = 1.6;

// Newton iteration budget
static const int iter_max = 30;
static const double offset_perc = 0.00; // offset_perc*100% of element size

// Damping parameter for damped interpolation toward the center for initial guess
// It preserves the direction from center_uv toward the barycentric estimate, but limits the distance to alpha*100%.
// Alpha decreases, the point moves closer to the centre from the standard barycentric interpolation
// alpha = 1.0 -> standard barycentric interpolation
// alpha = 0.5 -> halfway toward center
// alpha = 0.0 -> always center
static const double alpha = 0.90;


// nodes[i] - Physical space, R3, Actual element node positions
// proj[i] - Projected space, R2, Node positions in the plane perpendicular to the ray
struct InitialGuess
{
    bool valid = false;
    Eigen::Vector2d gh;
    double t;
};

inline double edgeFunction(
    const EiVector2d& a,
    const EiVector2d& b,
    const EiVector2d& p)
{
    return (p.x() - a.x()) * (b.y() - a.y())
         - (p.y() - a.y()) * (b.x() - a.x());
}

bool barycentric_test(
    EiVector2d p,
    EiVector2d a,
    EiVector2d b,
    EiVector2d c,
    EiVector3d& bc)
{
    EiVector2d v0=b-a;
    EiVector2d v1=c-a;
    EiVector2d v2=p-a;

    double d00=v0.dot(v0);
    double d01=v0.dot(v1);
    double d11=v1.dot(v1);
    double d20=v2.dot(v0);
    double d21=v2.dot(v1);

    double denom=d00*d11-d01*d01;
    if(std::abs(denom)<1e-12) return false;

    double v=(d11*d20-d01*d21)/denom;
    double w=(d00*d21-d01*d20)/denom;
    double u=1.0-v-w;

    bc={u,v,w};

    return u>=0 && v>=0 && w>=0;
}

// Reference coordinates and edge connectivity for QUAD8 and QUAD9
static const Eigen::Vector2d uv_nodes_quad8[8] =
{
    {-1.0,-1.0},
    { 1.0,-1.0},
    { 1.0, 1.0},
    {-1.0, 1.0},
    { 0.0,-1.0},
    { 1.0, 0.0},
    { 0.0, 1.0},
    {-1.0, 0.0}
};
static const Eigen::Vector2d uv_nodes_quad9[9] =
{
    {-1.0,-1.0},
    { 1.0,-1.0},
    { 1.0, 1.0},
    {-1.0, 1.0},
    { 0.0,-1.0},
    { 1.0, 0.0},
    { 0.0, 1.0},
    {-1.0, 0.0},
    { 0.0, 0.0}
};
static const int edge_nodes_quad[4][3] =
{
    {0,4,1},
    {1,5,2},
    {2,6,3},
    {3,7,0}
};

// Reference coordinates and edge connectivity for TRI6
EiVector2d uv_nodes[6] = {
    {0.0,0.0},
    {1.0,0.0},
    {0.0,1.0},
    {0.5,0.0},
    {0.5,0.5},
    {0.0,0.5}
};
const int edge_nodes[3][3] =
{
    {0,3,1},
    {1,4,2},
    {2,5,0}
};


// Reference coordinates and edge connectivity for TRI6
static const Eigen::Vector2d uv_nodes_tri[6] = {
    {0.0,0.0},
    {1.0,0.0},
    {0.0,1.0},
    {0.5,0.0},
    {0.5,0.5},
    {0.0,0.5}
};
static const int edge_nodes_tri[3][3] =
{
    {0,3,1},
    {1,4,2},
    {2,5,0}
};

// Evaluate S(xi, eta) = sum_i N_i(xi, eta) * x_i for the TRI6, QUAD8, and QUAD9.
template<ElementNodeCount element_node_count>
static inline EiVector3d evaluate_surface(
    const double xi,
    const double eta,
    const std::array<EiVector3d, element_node_count>& nodes)
{
    static constexpr int NODES_PER_ELEMENT = static_cast<int>(element_node_count);

    std::array<double, NODES_PER_ELEMENT> N;

    if constexpr (element_node_count == ElementNodeCount::QUAD8) {
        N = shapefuncs::compute_shape_quad8(xi, eta);
    }
    else if constexpr (element_node_count == ElementNodeCount::QUAD9) {
        N = shapefuncs::compute_shape_quad9(xi, eta);
    }
    else if constexpr (element_node_count == ElementNodeCount::TRI6) {
        N = shapefuncs::compute_shape_tri6(xi, eta);
    }
    else {
        static_assert(
            element_node_count == ElementNodeCount::QUAD8 ||
            element_node_count == ElementNodeCount::QUAD9 ||
            element_node_count == ElementNodeCount::TRI6,
            "Unsupported ElementNodeCount. Supported values are "
            "QUAD8, QUAD9, and TRI6."
        );
    }

    EiVector3d S = EiVector3d::Zero();
    for (std::size_t i = 0; i < NODES_PER_ELEMENT; ++i) {
        S += N[i] * nodes[i];
    }

    return S;
}

double compute_winding_tri(const EiVector2d* proj)
{
    double winding = edgeFunction(proj[0], proj[1], proj[2]);
    return winding;
}

double compute_winding_quad(const EiVector2d* proj)
{
    double winding = 0.0;

    for(int i=0;i<4;i++)
    {
        int j=(i+1)&3;
        winding -= proj[i].x()*proj[j].y() - proj[j].x()*proj[i].y();
    }

    return winding;
}

void boundary_limit_quad(Eigen::Vector2d& gh)
{
    gh.x() = std::max(-1.0,std::min(1.0,gh.x()));

    gh.y() = std::max(-1.0,std::min(1.0,gh.y()));
}

// void boundary_limit_quad(Eigen::Vector2d& gh)
// {
//     double tol = 0.5;
//     gh.x() = std::max(-1.0 - tol, std::min(1.0 + tol, gh.x()));
//     gh.y() = std::max(-1.0 - tol, std::min(1.0 + tol, gh.y()));
// }


void boundary_limit_tri(Eigen::Vector2d& gh)
{
    gh = gh.cwiseMax(0.0).cwiseMin(1.0);
    if (gh.x() + gh.y() > 1.0)
    {
        double s = gh.x() + gh.y();
        gh /= s;
    }
}
static void remove_duplicate_hull_points(
    std::vector<EiVector2d>& hull,
    std::vector<Eigen::Vector2d>& hull_uv,
    double tol)
{
    if (hull.size() < 2)
        return;

    const double tol2 = tol * tol;

    std::vector<EiVector2d> new_hull;
    std::vector<Eigen::Vector2d> new_hull_uv;

    new_hull.reserve(hull.size());
    new_hull_uv.reserve(hull_uv.size());

    new_hull.push_back(hull[0]);
    new_hull_uv.push_back(hull_uv[0]);

    // Remove consecutive duplicates
    for (size_t i = 1; i < hull.size(); ++i)
    {
        if ((hull[i] - new_hull.back()).squaredNorm() > tol2)
        {
            new_hull.push_back(hull[i]);
            new_hull_uv.push_back(hull_uv[i]);
        }
    }

    if (new_hull.size() > 1 &&
        (new_hull.front() - new_hull.back()).squaredNorm() <= tol2)
    {
        new_hull.pop_back();
        new_hull_uv.pop_back();
    }

    hull.swap(new_hull);
    hull_uv.swap(new_hull_uv);
}


//  TRI6, QUAD8, QUAD9 adaptive hulls and initial guess for Newton solver
// ================================================================================

template<ElementNodeCount element_node_count>
InitialGuess compute_initial_guess_newton_solver(
    const Ray& ray,
    const std::array<EiVector3d, element_node_count>& nodes)
{
    /*
    Initial guess based on the projected adaptive hull:
        1. Project the curved element into the plane normal to the ray
        2. Approximate its boundary by a adaptive hull
        3. Triangulate adaptive hull with a fan.
        4. Find which fan triangle contains the projected ray
        5. Compute barycentric coordinates in that fan triangle
        6. Apply those same barycentric coordinates to the corresponding points in the reference triangle
        7. Calculate t from projection on ray direction
    */

    static constexpr int NODES_PER_ELEMENT = static_cast<int>(element_node_count);
    static const Eigen::Vector2d* uv_nodes_ptr;
    static const int (*edge_nodes_ptr)[3];
    double (*compute_winding)(const EiVector2d*);
    void (*boundary_limit)(Eigen::Vector2d&);
    int edge_number; // number of edges in an element
    if constexpr (element_node_count == ElementNodeCount::QUAD8) {
        uv_nodes_ptr = uv_nodes_quad8;
        edge_nodes_ptr = edge_nodes_quad;
        compute_winding = compute_winding_quad;
        boundary_limit = boundary_limit_quad;
        edge_number = 4;
    }
    else if constexpr (element_node_count == ElementNodeCount::QUAD9) {
        uv_nodes_ptr = uv_nodes_quad9;
        edge_nodes_ptr = edge_nodes_quad;
        compute_winding = compute_winding_quad;
        boundary_limit = boundary_limit_quad;
        edge_number = 4;
    }
    else if constexpr (element_node_count == ElementNodeCount::TRI6) {
        uv_nodes_ptr = uv_nodes_tri;
        edge_nodes_ptr = edge_nodes_tri;
        compute_winding = compute_winding_tri;
        boundary_limit = boundary_limit_tri;
        edge_number = 3;
    }
    else {
        static_assert(
            element_node_count == ElementNodeCount::QUAD8 ||
            element_node_count == ElementNodeCount::QUAD9 ||
            element_node_count == ElementNodeCount::TRI6,
            "Unsupported ElementNodeCount. Supported values are "
            "QUAD8, QUAD9, and TRI6."
        );
    }

    InitialGuess result;

    // Build ray coordinate system, i.e. plane perpendicular to the ray
    EiVector3d ez = ray.direction.normalized();

    EiVector3d tmp =
        (std::abs(ez.x()) < 0.9)
            ? EiVector3d(1.0,0.0,0.0)
            : EiVector3d(0.0,1.0,0.0);

    EiVector3d ex = ez.cross(tmp).normalized();
    EiVector3d ey = ez.cross(ex);


    // Project nodes onto plane perpendicular to ray
    EiVector2d proj[NODES_PER_ELEMENT];

    for(int i=0;i<NODES_PER_ELEMENT;i++)
    {
        EiVector3d d = nodes[i]-ray.origin;
        proj[i] = EiVector2d(d.dot(ex), d.dot(ey));
    }

    // Determine element orientation
    double winding = compute_winding(proj);
    
    const double diagonal = find_element_diagonal(&nodes[0], element_node_count);
    const double offset = offset_perc * diagonal; // offset_perc*100% of element size

    // Construct adaptive hull
    std::vector<EiVector2d> hull;
    std::vector<Eigen::Vector2d> hull_uv;

    hull.reserve(edge_number*3);
    hull_uv.reserve(edge_number*3);

    for(int e=0;e<edge_number;e++)
    {
        int a=edge_nodes_ptr[e][0];
        int m=edge_nodes_ptr[e][1];
        int b=edge_nodes_ptr[e][2];

        EiVector2d A=proj[a];
        EiVector2d M=proj[m];
        EiVector2d B=proj[b];

        Eigen::Vector2d uvA=uv_nodes_ptr[a];
        Eigen::Vector2d uvM=uv_nodes_ptr[m];
        Eigen::Vector2d uvB=uv_nodes_ptr[b];

        double ef=edgeFunction(A,B,M);

        if (winding < 0.0)
            ef = -ef;

        bool use_midpoint = (ef >= 0.0);

        EiVector2d support;
        Eigen::Vector2d support_uv;

        if(use_midpoint) // Concave edge
        {
            // midpoint lies inside
            support=M;
            support_uv=uvM;
        }
        else // Convex edge
        {
            // quadratic Bézier control point
            support = 2.0*M - 0.5*(A+B);

            // Buffer for edge points
            // double buffer = 1.0;
            EiVector2d D = M - 0.5*(A + B);
            double D_norm = D.norm();
            if (D_norm > 1e-12)
            {
                D *= offset / D_norm;   // D now has magnitude = 5% of diagonal
            }
            
            A += D;
            B += D;
            // A = A + buffer*D;
            // B = B + buffer*D;
            // support = M + buffer*D;

            support_uv = uvM;
        }

        hull.push_back(A);
        hull_uv.push_back(uvA);

        hull.push_back(support);
        hull_uv.push_back(support_uv);

        hull.push_back(B);
        hull_uv.push_back(uvB);
    }

    remove_duplicate_hull_points(hull, hull_uv, 1e-10 * std::max(diagonal, 1.0));

    // Compute centroid
    EiVector2d center(0.0,0.0);
    for(auto const& p : hull)
        center += p;
    center /= double(hull.size());

    Eigen::Vector2d center_uv(0.0,0.0);
    for(auto const& uv : hull_uv)
        center_uv += uv;
    center_uv /= double(hull_uv.size());

    // Search the fan for intersection with the projected ray
    EiVector2d P(0.0,0.0);

    for(size_t i=0;i<hull.size();i++)
    {
        size_t j=(i+1)%hull.size();

        EiVector3d bc;

        if(!barycentric_test(P, center, hull[i], hull[j], bc))
            continue;

        // Standard barycentric interpolation for linear triangle
        // result.gh =
        //       bc.x()*center_uv
        //     + bc.y()*hull_uv[i]
        //     + bc.z()*hull_uv[j];

        // Damped interpolation toward the center
        // It preserves the direction from center_uv toward the barycentric estimate, but limits the distance to alpha*100%.
        // Alpha decreases, the point moves closer to the centre from the standard barycentric interpolation
        // alpha = 1.0 -> standard barycentric interpolation
        // alpha = 0.5 -> halfway toward center
        // alpha = 0.0 -> always center

        Eigen::Vector2d gh_bary =
             bc.x() * center_uv
           + bc.y() * hull_uv[i]
           + bc.z() * hull_uv[j];
        
        result.gh =
              center_uv
            + alpha * (gh_bary - center_uv);

        boundary_limit(result.gh);

        // t from projection on ray direction

        EiVector3d X =
            evaluate_surface<element_node_count>(result.gh.x(),result.gh.y(), nodes);

        result.t = (X - ray.origin).dot(ray.direction);

        result.valid = true;
        return result;
    }

    return result;
}
bool boundary_check_quad(Eigen::Vector2d gh) 
{
    return std::fabs(gh.x()) <= 1.0 + eps_param_accept &&
           std::fabs(gh.y()) <= 1.0 + eps_param_accept;
}

bool boundary_check_tri(Eigen::Vector2d gh) 
{
    return gh.x() >= 0 - eps_param_accept &&
           gh.y() >= 0 - eps_param_accept &&
           (gh.x() + gh.y()) <= 1 + eps_param_accept;
}

// ================================================================================
//  TRI6, QUAD8, QUAD9 element intersection using Newton solver
// ================================================================================

template<ElementNodeCount element_node_count>
double intersect_newton_solver(const Ray& ray,
    const std::array<EiVector3d, element_node_count>& nodes,
    EiVector3d& surface_normals_out,
    Eigen::Vector2d& xi_eta_out,
    int* iterations_out) {

    Eigen::Matrix<double, 3, 2> (*get_face_Jacobian_ptr)(double, double, 
            const std::array<EiVector3d, element_node_count>&);
    bool (*boundary_check)(Eigen::Vector2d);
    void (*boundary_limit_ptr)(Eigen::Vector2d&);
    if constexpr (element_node_count == ElementNodeCount::QUAD8) {
        get_face_Jacobian_ptr = shapefuncs::get_face_Jacobian_quad8;
        boundary_check = boundary_check_quad;
        boundary_limit_ptr = boundary_limit_quad;
    }
    else if constexpr (element_node_count == ElementNodeCount::QUAD9) {
        get_face_Jacobian_ptr = shapefuncs::get_face_Jacobian_quad9;
        boundary_check = boundary_check_quad;
        boundary_limit_ptr = boundary_limit_quad;
    }
    else if constexpr (element_node_count == ElementNodeCount::TRI6) {
        get_face_Jacobian_ptr = shapefuncs::get_face_Jacobian_tri6;
        boundary_check = boundary_check_tri;
        boundary_limit_ptr = boundary_limit_tri;
    }
    else {
        static_assert(
            element_node_count == ElementNodeCount::QUAD8 ||
            element_node_count == ElementNodeCount::QUAD9 ||
            element_node_count == ElementNodeCount::TRI6,
            "Unsupported ElementNodeCount. Supported values are "
            "QUAD8, QUAD9, and TRI6."
        );
    }
    
    const double diagonal = find_element_diagonal(&nodes[0], element_node_count);
    double res_tol_local = eps_res_rel * diagonal;
    if (res_tol_local < eps_res_abs) res_tol_local = eps_res_abs;

    // Find initial guess and confirm that it is valid
    InitialGuess guess = compute_initial_guess_newton_solver<element_node_count>(ray, nodes);
    if (!guess.valid) {
        if (iterations_out) *iterations_out = 0;
        return std::numeric_limits<double>::infinity();
    }

    // Search for the intersection if the initial guess is found
    double min_t = std::numeric_limits<double>::infinity();
    bool converged = false;
    Eigen::Vector2d gh = guess.gh;
    double t = guess.t;
    int iter_used = 0;

    // Uncomment this and comment out Newton solver if just initial guesses are required
    // converged = true;
    // min_t = t;
    // Eigen::Matrix<double, 3, 2> J = get_face_Jacobian_ptr(gh.x(), gh.y(), nodes);
    // EiVector3d normal =
    //     (J.col(0).cross(J.col(1))).transpose();
    // surface_normals_out = normal;
    // xi_eta_out = gh;
    
    for (int iter = 0; iter < iter_max; ++iter) {

        EiVector3d P =
            evaluate_surface<element_node_count>(gh.x(),gh.y(), nodes);

        Eigen::Matrix<double, 3, 2> J = get_face_Jacobian_ptr(gh.x(), gh.y(), nodes);

        EiVector3d residual = ray.origin + t * ray.direction - P;

        if (residual.norm() < res_tol_local) {
            iter_used = iter + 1;
            if (boundary_check(gh))
                {
                    if (t < min_t && t > eps_t_min && 
                        t >= ray.t_min && t <= ray.t_max) {
                        min_t = t;
    
                        EiVector3d normal =
                            (J.col(0).cross(J.col(1))).transpose();
    
                        surface_normals_out = normal;
                        xi_eta_out = gh;
                        converged = true;
                    }
                }
            break;
        }

        Eigen::Matrix3d M;
        M.col(0) = ray.direction.transpose();
        M.col(1) = -J.col(0);
        M.col(2) = -J.col(1);

        if (std::abs(M.determinant()) < eps_sub_det) break;

        Eigen::Vector3d delta = M.inverse() * (-residual.transpose());

        t += delta.x();
        gh.x() += delta.y();
        gh.y() += delta.z();

        boundary_limit_ptr(gh);

        iter_used = iter + 1;

        // if (gh.x() < -0.5 || gh.y() < -0.5 || (gh.x() + gh.y()) > 1.5)
        //     break;
    }

    if (iterations_out) *iterations_out = iter_used;
    return converged ? min_t : std::numeric_limits<double>::infinity();
}


double intersect_tri6(const Ray &ray,
    const std::array<EiVector3d, ElementNodeCount::TRI6> nodes,
    EiVector3d &surface_normals_out,
    Eigen::Vector2d &uv,
    int* iterations_out) {
    return intersect_newton_solver<ElementNodeCount::TRI6>(ray, nodes, surface_normals_out, uv, iterations_out);
}

double intersect_quad8(const Ray& ray,
    const std::array<EiVector3d, ElementNodeCount::QUAD8>& nodes,
    EiVector3d& surface_normals_out,
    Eigen::Vector2d& xi_eta_out,
    int* iterations_out) {
    return intersect_newton_solver<ElementNodeCount::QUAD8>(ray, nodes, surface_normals_out, xi_eta_out, iterations_out);
}

double intersect_quad9(const Ray& ray,
    const std::array<EiVector3d, ElementNodeCount::QUAD9>& nodes,
    EiVector3d& surface_normals_out,
    Eigen::Vector2d& xi_eta_out,
    int* iterations_out) {
    return intersect_newton_solver<ElementNodeCount::QUAD9>(ray, nodes, surface_normals_out, xi_eta_out, iterations_out);
}

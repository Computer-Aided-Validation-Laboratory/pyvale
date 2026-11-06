// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2026 The Computer Aided Validation Team
// ================================================================================
//
// Benchmark case 2: quadratic elements with a Newton solver against sub-tessellated
// linear triangles for curved surfaces.
//
// This benchmark measures the resulting cost-accuracy
// trade-off. For two analytic geometries (a sphere and a cylinder) it performs a refinement 
// sweep over:
//
//   * quadratic_tri  - TRI6 faces intersected with the Newton solver;
//   * quadratic_quad - QUAD9 faces intersected with the Newton solver;
//   * linear_native  - the same triangulation as the quadratic mesh with the
//                      midside nodes dropped (flat facets, on-surface nodes);
//   * linear_subdiv  - each quadratic element split into k x k linear triangles
//                      whose corner nodes are sampled on the quadratic surface.
//
// Every arm scores the same frozen ray bundles against the analytic surface, so
// accuracy is both the signed radial error (|p| - r for the sphere,
// sqrt(x^2 + z^2) - r for the cylinder skin) and the surface-normal error (the
// angle in degrees between the hit normal and the analytic normal). Cost is
// reported as rays per second (single threaded), together with wall time and primitive counts.
//
// The TRI6/QUAD8/QUAD9 Newton solvers are reused from ``cpp/rtrayintersection_modes``
// (the same mode-aware code used by benchmark case 1, which performs counting),
// and the linear arm uses a local Möller-Trumbore triangle test so that it can be
// measured with identical bookkeeping.
//
// Output layout (self-contained under ``verif_bench/bench/bench2``)::
//
//   bench2_stats.csv       per geometry/arm/level/bundle measurement
//   bench2_summary.csv     per geometry/arm/level aggregates (the Pareto points)
//   bench2_vs_analytic.csv per-element geometric error of each mesh vs the surface
//
// The corresponding paper figures are produced by
// ``paper_bench2_quadratic_vs_linear.py``.
//
// Build (run from ``verif_bench/``)::
//
//   g++ -O2 -std=c++20 -fopenmp -o bench2_quadratic_vs_linear
//       bench2_quadratic_vs_linear.cpp cpp/rtrayintersection_modes.cpp
//
// Run (run from ``verif_bench/``)::
//
//   ./bench2_quadratic_vs_linear
// ================================================================================

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#include "../../common_cpp/Eigen/Dense"

#include "cpp/rteigentypes.h"
#include "cpp/rtelemconstants.h"
#include "cpp/rtshapefuncs.h"
#include "cpp/rtray.h"
#include "cpp/rtrayintersection_modes.h"

#include "verif_cpp/verifconstants.h"


// Self-contained output root
static const std::string bench_dir_name = "bench";
static const std::string bench_subdir_name = "bench2";
static const std::string data_dir_name = "data/curved";

// --------------------------------------------------------------------------------
// Geometry description
// --------------------------------------------------------------------------------

enum class Geometry
{
    Sphere,
    Cylinder
};

// Analytic surface parameters. These mirror ``data/curved/sph.geo`` and
// ``cyl.geo``.
struct Surface
{
    Geometry kind = Geometry::Sphere;
    double radius = 0.5;    // sphere radius / cylinder radius
    double y_min = 0.0;     // cylinder cap positions in y
    double y_max = 1.0;
};

// Signed radial error of a hit point against the analytic surface. Returns the
// perpendicular distance to the curved skin; for the cylinder the flat end caps
// sit exactly at |error| = 0 too because a cap hit lies on the boundary.
static double radial_error(const Surface& s, const EiVector3d& p)
{
    if (s.kind == Geometry::Sphere) {
        return p.norm() - s.radius;
    }
    // Cylinder: axis is y, curved skin is sqrt(x^2 + z^2) = r.
    return std::sqrt(p(0) * p(0) + p(2) * p(2)) - s.radius;
}

// OLD APPROACH
// Outward unit normal of the analytic surface at a point. For the cylinder the
// end caps are flat, so a point whose radial distance is far from the radius is
// treated as a cap and given the axial normal.
// static EiVector3d analytic_normal(const Surface& s, const EiVector3d& p)
// {
//     if (s.kind == Geometry::Sphere) {
//         const double r = p.norm();
//         return (r > 0.0) ? (p / r) : EiVector3d(0.0, 0.0, 1.0);
//     }

//     const double rad = std::sqrt(p(0) * p(0) + p(2) * p(2));
//     if (rad < 0.5 * s.radius) {
//         // On an end cap: normal is axial.
//         return EiVector3d(0.0, (p(1) >= 0.5 * (s.y_min + s.y_max)) ? 1.0 : -1.0,
//                           0.0);
//     }
//     if (rad > 0.0) {
//         return EiVector3d(p(0) / rad, 0.0, p(2) / rad);
//     }
//     return EiVector3d(1.0, 0.0, 0.0);
// }



// Outward unit normal of the analytic surface at a point.
// The distances to candidate surface components are computed and the nearest one is selected.
static EiVector3d analytic_normal(const Surface& s, const EiVector3d& p)
{
    if (s.kind == Geometry::Sphere) {
        const double r = p.norm();
        return (r > 0.0) ? (p / r) : EiVector3d(0.0, 0.0, 1.0);
    }

    const double x = p(0);
    const double y = p(1);
    const double z = p(2);

    const double rad = std::sqrt(x * x + z * z);

    // Candidate 1: normal of the curved cylindrical side.
    EiVector3d side_normal(1.0, 0.0, 0.0);

    if (rad > 1e-15) {
        side_normal = EiVector3d(x / rad, 0.0, z / rad);
    }

    // Distance to the curved side.
    // The side exists only between the two cap planes.
    const double clamped_y = std::clamp(y, s.y_min, s.y_max);
    const double side_y_error = y - clamped_y;
    const double side_rad_error = rad - s.radius;

    const double side_distance = std::sqrt(
        side_rad_error * side_rad_error +
        side_y_error * side_y_error);

    // Candidate 2: lower cap
    const double lower_rad = std::min(rad, s.radius);
    const double lower_rad_error = rad - lower_rad;
    const double lower_y_error = y - s.y_min;

    const double lower_distance = std::sqrt(
        lower_rad_error * lower_rad_error +
        lower_y_error * lower_y_error);

    // Candidate 3: upper cap
    const double upper_rad = std::min(rad, s.radius);
    const double upper_rad_error = rad - upper_rad;
    const double upper_y_error = y - s.y_max;

    const double upper_distance = std::sqrt(
        upper_rad_error * upper_rad_error +
        upper_y_error * upper_y_error);

    // Choose the nearest surface component.
    // At an edge, the cap normal takes precedence.
    if (lower_distance <= upper_distance &&
        lower_distance <= side_distance) {
        return EiVector3d(0.0, -1.0, 0.0);
    }

    if (upper_distance <= side_distance) {
        return EiVector3d(0.0, 1.0, 0.0);
    }

    return side_normal;
}



// Angle in degrees between two directions, clamped for numerical safety. The
// sign of the normal is discarded because a plane normal and its negation
// describe the same plane.
static double normal_angle_deg(const EiVector3d& a, const EiVector3d& b)
{
    const double na = a.norm();
    const double nb = b.norm();
    if (na <= 0.0 || nb <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double c = std::abs(a.dot(b)) / (na * nb);
    c = std::min(1.0, std::max(0.0, c));
    return std::acos(c) * 180.0 / 3.14159265358979323846;
}

// --------------------------------------------------------------------------------
// Mesh representation
// --------------------------------------------------------------------------------

// A generated mesh from ``data/curved`` is a pure surface mesh: every cell is a
// triangle (TRI3/TRI6) or a quad (QUAD4/QUAD9).
enum class CellKind
{
    TRI3,
    TRI6,
    QUAD4,
    QUAD9
};

struct Cell
{
    CellKind kind = CellKind::TRI3;
    // Corner and midside node indices, ordered as the VTK record (corners
    // first, then midside, then the quad centre for QUAD9).
    std::array<int, 9> node {};
    int node_count = 0;
};

struct Mesh
{
    std::vector<EiVector3d> points;
    std::vector<Cell> cells;

    std::size_t count_kind(CellKind k) const
    {
        std::size_t n = 0;
        for (const Cell& c : cells) {
            if (c.kind == k) ++n;
        }
        return n;
    }
};

// --------------------------------------------------------------------------------
// Minimal VTK reader
// --------------------------------------------------------------------------------

// Reads POINTS / CELLS / CELL_TYPES from a ASCII VTK file. Cell types are
// mapped from the VTK codes produced by Gmsh: 5 = TRI3, 22 = TRI6, 9 = QUAD4,
// 24 and 28 = QUAD9. Vertex records (type 1) are skipped.
static bool read_vtk(const std::string& path, Mesh& mesh)
{
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Could not open mesh: " << path << '\n';
        return false;
    }

    std::string line;
    std::size_t n_points = 0;
    std::size_t n_cells = 0;
    bool got_points = false;
    bool got_cells = false;

    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string keyword;
        ss >> keyword;

        if (keyword == "POINTS") {
            ss >> n_points;
            mesh.points.resize(n_points);
            for (std::size_t i = 0; i < n_points; ++i) {
                double x = 0.0, y = 0.0, z = 0.0;
                in >> x >> y >> z;
                mesh.points[i] = EiVector3d(x, y, z);
            }
            got_points = true;
        }
        else if (keyword == "CELLS") {
            ss >> n_cells;
            mesh.cells.reserve(n_cells);
            std::vector<std::vector<int>> records;
            records.reserve(n_cells);
            for (std::size_t i = 0; i < n_cells; ++i) {
                int width = 0;
                in >> width;
                std::vector<int> ids(static_cast<std::size_t>(width));
                for (int k = 0; k < width; ++k) {
                    in >> ids[static_cast<std::size_t>(k)];
                }
                records.push_back(std::move(ids));
            }
            // CELL_TYPES follows immediately; read it here so we can attach the
            // kind to each record.
            std::string tkey;
            while (in >> tkey) {
                if (tkey == "CELL_TYPES") break;
            }
            in >> n_cells;
            for (std::size_t i = 0; i < n_cells; ++i) {
                int code = 0;
                in >> code;
                const std::vector<int>& ids = records[i];

                Cell cell;
                switch (code) {
                    case 5:  // TRI3
                        if (ids.size() < 3) continue;
                        cell.kind = CellKind::TRI3;
                        cell.node_count = 3;
                        for (int k = 0; k < 3; ++k) cell.node[k] = ids[k];
                        break;
                    case 22: // TRI6
                        if (ids.size() < 6) continue;
                        cell.kind = CellKind::TRI6;
                        cell.node_count = 6;
                        for (int k = 0; k < 6; ++k) cell.node[k] = ids[k];
                        break;
                    case 9:  // QUAD4
                        if (ids.size() < 4) continue;
                        cell.kind = CellKind::QUAD4;
                        cell.node_count = 4;
                        for (int k = 0; k < 4; ++k) cell.node[k] = ids[k];
                        break;
                    case 24: // QUAD9
                    case 28: // QUAD9 (same element, alternate VTK code)
                        if (ids.size() < 9) continue;
                        cell.kind = CellKind::QUAD9;
                        cell.node_count = 9;
                        for (int k = 0; k < 9; ++k) cell.node[k] = ids[k];
                        break;
                    default: // type 1 vertex / unsupported
                        continue;
                }
                mesh.cells.push_back(cell);
            }
            got_cells = true;
        }
    }

    if (!got_points || !got_cells) {
        std::cerr << "Malformed VTK mesh: " << path << '\n';
        return false;
    }
    return true;
}

// --------------------------------------------------------------------------------
// Quadratic surface evaluation (shared with the Newton solver)
// --------------------------------------------------------------------------------

// Reference coordinates of the quadratic element nodes, in the same order the VTK
// records use (corners first, then midside).
static const Eigen::Vector2d uv_tri6[6] = {
    {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0},
    {0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5}
};
static const Eigen::Vector2d uv_quad9[9] = {
    {-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0},
    {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0},
    {0.0, 0.0}
};

// Isoparametric surface point S(xi, eta) for a quadratic element.
template<ElementNodeCount E>
static EiVector3d surface_point(double xi, double eta, const NodeArray<E>& nodes)
{
    static constexpr int N = static_cast<int>(E);
    std::array<double, N> shape;
    if constexpr (E == ElementNodeCount::TRI6) {
        shape = shapefuncs::compute_shape_tri6(xi, eta);
    }
    else if constexpr (E == ElementNodeCount::QUAD8) {
        shape = shapefuncs::compute_shape_quad8(xi, eta);
    }
    else {
        shape = shapefuncs::compute_shape_quad9(xi, eta);
    }

    EiVector3d S = EiVector3d::Zero();
    for (int i = 0; i < N; ++i) {
        S += shape[i] * nodes[i];
    }
    return S;
}

static EiVector3d surface_point_quad9(double xi, double eta,
                                      const NodeArray<ElementNodeCount::QUAD9>& n9)
{
    return surface_point<ElementNodeCount::QUAD9>(xi, eta, n9);
}

static EiVector3d surface_point_tri6(double g, double h,
                                     const NodeArray<ElementNodeCount::TRI6>& n6)
{
    return surface_point<ElementNodeCount::TRI6>(g, h, n6);
}

// --------------------------------------------------------------------------------
// Two primitive representations used by the arms
// --------------------------------------------------------------------------------

// A linear triangle (the linear arms, and the sub-tessellation output).
struct Tri3
{
    EiVector3d a, b, c;
};

// A quadratic element (the quadratic arms).
struct QuadElement
{
    bool is_tri = true;                       // TRI6 when true, QUAD9 when false
    NodeArray<ElementNodeCount::TRI6> tri6;
    NodeArray<ElementNodeCount::QUAD9> quad9;
};

// Möller-Trumbore intersection
static bool intersect_tri3(const Ray& ray, const Tri3& tri, double& t_out)
{
    const EiVector3d e1 = tri.b - tri.a;
    const EiVector3d e2 = tri.c - tri.a;
    const EiVector3d pvec = ray.direction.cross(e2);
    const double det = e1.dot(pvec);

    if (std::abs(det) < 1e-14) return false;

    const double inv_det = 1.0 / det;
    const EiVector3d tvec = ray.origin - tri.a;
    const double u = tvec.dot(pvec) * inv_det;
    if (u < -1e-9 || u > 1.0 + 1e-9) return false;

    const EiVector3d qvec = tvec.cross(e1);
    const double v = ray.direction.dot(qvec) * inv_det;
    if (v < -1e-9 || u + v > 1.0 + 1e-9) return false;

    const double t = e2.dot(qvec) * inv_det;
    if (t <= 1e-7) return false;

    t_out = t;
    return true;
}

// Build a TRI3 from three node indices of a mesh.
static Tri3 make_tri3(const Mesh& mesh, int i, int j, int k)
{
    Tri3 tri;
    tri.a = mesh.points[static_cast<std::size_t>(i)];
    tri.b = mesh.points[static_cast<std::size_t>(j)];
    tri.c = mesh.points[static_cast<std::size_t>(k)];
    return tri;
}

// --------------------------------------------------------------------------------
// Sub-tessellation of a quadratic element into k x k linear triangles
// --------------------------------------------------------------------------------

// Sample the parametric domain of a quadratic cell on a k x k lattice and split
// each lattice quad into two triangles. Corner nodes are evaluated on the exact
// quadratic surface, so the sub-tessellation converges to the true surface as k increases.
static void subdivide_cell(const Mesh& mesh, const Cell& cell, int k,
                           std::vector<Tri3>& out)
{
    if (cell.kind == CellKind::TRI6) {
        NodeArray<ElementNodeCount::TRI6> nodes;
        for (int i = 0; i < 6; ++i) {
            nodes[i] = mesh.points[static_cast<std::size_t>(cell.node[i])];
        }
        // Barycentric lattice over the reference triangle.
        std::vector<EiVector3d> grid(static_cast<std::size_t>((k + 1) * (k + 2) / 2));
        auto idx = [&](int i, int j) -> std::size_t {
            return static_cast<std::size_t>(i * (k + 1) - (i * (i - 1)) / 2 + j);
        };
        for (int i = 0; i <= k; ++i) {
            for (int j = 0; j <= k - i; ++j) {
                const double g = static_cast<double>(i) / k;
                const double h = static_cast<double>(j) / k;
                grid[idx(i, j)] = surface_point_tri6(g, h, nodes);
            }
        }
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k - i; ++j) {
                const EiVector3d& p00 = grid[idx(i, j)];
                const EiVector3d& p10 = grid[idx(i + 1, j)];
                const EiVector3d& p01 = grid[idx(i, j + 1)];
                out.push_back({p00, p10, p01});
                if (j < k - i - 1) {
                    const EiVector3d& p11 = grid[idx(i + 1, j + 1)];
                    out.push_back({p10, p11, p01});
                }
            }
        }
    }
    else if (cell.kind == CellKind::QUAD9) {
        NodeArray<ElementNodeCount::QUAD9> nodes;
        for (int i = 0; i < 9; ++i) {
            nodes[i] = mesh.points[static_cast<std::size_t>(cell.node[i])];
        }
        // Uniform lattice on [-1, 1]^2.
        std::vector<EiVector3d> grid(static_cast<std::size_t>((k + 1) * (k + 1)));
        for (int j = 0; j <= k; ++j) {
            const double eta = -1.0 + 2.0 * j / k;
            for (int i = 0; i <= k; ++i) {
                const double xi = -1.0 + 2.0 * i / k;
                grid[static_cast<std::size_t>(j * (k + 1) + i)] =
                    surface_point_quad9(xi, eta, nodes);
            }
        }
        for (int j = 0; j < k; ++j) {
            for (int i = 0; i < k; ++i) {
                const std::size_t p00 = static_cast<std::size_t>(j * (k + 1) + i);
                const std::size_t p10 = static_cast<std::size_t>(j * (k + 1) + i + 1);
                const std::size_t p01 = static_cast<std::size_t>((j + 1) * (k + 1) + i);
                const std::size_t p11 = static_cast<std::size_t>((j + 1) * (k + 1) + i + 1);
                out.push_back({grid[p00], grid[p10], grid[p11]});
                out.push_back({grid[p00], grid[p11], grid[p01]});
            }
        }
    }
    else {
        // TRI3 / QUAD4: corner-only linearisation (k = 1).
        if (cell.kind == CellKind::TRI3) {
            out.push_back(make_tri3(mesh, cell.node[0], cell.node[1], cell.node[2]));
        }
        else {
            out.push_back(make_tri3(mesh, cell.node[0], cell.node[1], cell.node[2]));
            out.push_back(make_tri3(mesh, cell.node[0], cell.node[2], cell.node[3]));
        }
    }
}

// --------------------------------------------------------------------------------
// Options and measurement
// --------------------------------------------------------------------------------

struct Options
{
    int grid = 96;          // camera grid size per geometry
    int targeted_per_cell = 4; // targeted rays per element (per side for quads)
    int reps = 3;           // timing repetitions, keep the fastest
    std::vector<int> subdiv_k {1, 2, 4, 8};
    std::string geometry_filter; // comma separated: sphere,cylinder
    std::string arm_filter;      // comma separated arm labels
    bool quick = false;
    int view_grid = 320;    // render resolution of the camera-view figure (0=off)
};

struct Stats
{
    long rays = 0;
    long hits = 0;
    long missed = 0;

    double sum_radial = 0.0;
    double sum_radial_sq = 0.0;
    std::vector<double> abs_errors; // trimmed stats

    // Surface-normal error at the hit, in degrees (absolute angle to the
    // analytic normal). Accumulated only for hits where a usable normal exists.
    long normal_samples = 0;
    double sum_normal = 0.0;
    double sum_normal_sq = 0.0;
    std::vector<double> normal_errors; // trimmed stats

    long primitive_tests = 0;
    long solver_runs = 0;
    long solver_iters = 0;

    double wall_s = 0.0;

    // Records one hit's radial and normal error against the analytic surface.
    void record_hit(const Surface& s, const EiVector3d& p,
                    const EiVector3d& n_hit)
    {
        const double err = radial_error(s, p);
        sum_radial += err;
        sum_radial_sq += err * err;
        abs_errors.push_back(std::abs(err));

        const double nerr = normal_angle_deg(n_hit, analytic_normal(s, p));
        if (!std::isnan(nerr)) {
            ++normal_samples;
            sum_normal += nerr;
            sum_normal_sq += nerr * nerr;
            normal_errors.push_back(nerr);
        }
    }
};

static double trimmed_max(std::vector<double>& values, double trim)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t keep =
        static_cast<std::size_t>((1.0 - trim) * static_cast<double>(values.size()));
    const std::size_t upto = std::min(values.size(), std::max<std::size_t>(keep, 1));
    return values[upto - 1];
}

// --------------------------------------------------------------------------------
// Ray bundles (frozen, shared across every arm)
// --------------------------------------------------------------------------------

// The pinhole camera used for the camera bundle, kept as an explicit object so
// the exact basis can be exported and reused to draw/render the same view.
struct CameraView
{
    EiVector3d origin = EiVector3d::Zero();
    EiVector3d centre = EiVector3d::Zero();   // point the camera looks at
    EiVector3d view {0.0, 0.0, -1.0};         // forward (unit)
    EiVector3d right {1.0, 0.0, 0.0};         // screen x (unit)
    EiVector3d up {0.0, 1.0, 0.0};            // screen y (unit)
    double extent = 0.7;                      // half-width of the screen plane
    int grid = 96;                            // samples per side
};

// Build the camera aimed at the geometry from +z. The screen spans
// [-extent, extent]^2 on the plane through ``centre`` normal to ``view``.
static CameraView make_camera(const Surface& s, int grid)
{
    CameraView cam;
    cam.extent = 1.4 * s.radius;
    cam.grid = grid;

    if (s.kind == Geometry::Cylinder) {
        cam.centre = EiVector3d(0.0, 0.5 * (s.y_min + s.y_max), 0.0);
    }
    cam.origin = EiVector3d(0.0, 0.0, 3.0 * s.radius + 1.0);

    // Build an orthonormal screen basis perpendicular to the view direction.
    cam.view = (cam.centre - cam.origin).normalized();
    EiVector3d up(0.0, 1.0, 0.0);
    if (std::abs(cam.view.dot(up)) > 0.95) up = EiVector3d(1.0, 0.0, 0.0);
    cam.right = cam.view.cross(up).normalized();
    cam.up = cam.right.cross(cam.view).normalized();
    return cam;
}

// A camera-style bundle: a square grid of pinhole rays aimed at the geometry.
// A realistic fraction of these miss.
static std::vector<Ray> build_camera_bundle(const CameraView& cam)
{
    const int grid = cam.grid;
    std::vector<Ray> rays;
    rays.reserve(static_cast<std::size_t>(grid) * grid);
    for (int j = 0; j < grid; ++j) {
        for (int i = 0; i < grid; ++i) {
            const double a =
                (i / double(std::max(grid - 1, 1)) - 0.5) * 2.0 * cam.extent;
            const double b =
                (j / double(std::max(grid - 1, 1)) - 0.5) * 2.0 * cam.extent;
            const EiVector3d target = cam.centre + a * cam.right + b * cam.up;
            rays.emplace_back(cam.origin, (target - cam.origin).normalized());
        }
    }
    return rays;
}

// Targeted rays aimed at sampled points of the analytic surface, so every ray is
// guaranteed to produce a hit for the accuracy statistics.
static std::vector<Ray> build_targeted_bundle(const Surface& s,
                                              const std::vector<Cell>& cells,
                                              const Mesh& mesh,
                                              int per_cell)
{
    const EiVector3d camera(0.0, 0.0, 3.0 * s.radius + 1.0);
    std::vector<Ray> rays;

    for (const Cell& cell : cells) {
        if (cell.kind == CellKind::TRI6) {
            NodeArray<ElementNodeCount::TRI6> nodes;
            for (int i = 0; i < 6; ++i) {
                nodes[i] = mesh.points[static_cast<std::size_t>(cell.node[i])];
            }
            for (int i = 0; i <= per_cell; ++i) {
                for (int j = 0; j <= per_cell - i; ++j) {
                    const double g = (per_cell > 0) ? i / double(per_cell) : 0.0;
                    const double h = (per_cell > 0) ? j / double(per_cell) : 0.0;
                    // Keep off the exact boundary so grazing hits are avoided.
                    const double gg = 0.02 + 0.96 * g;
                    const double hh = 0.02 + 0.96 * h;
                    if (gg + hh > 0.98) continue;
                    const EiVector3d p = surface_point_tri6(gg, hh, nodes);
                    rays.emplace_back(camera, (p - camera).normalized());
                }
            }
        }
        else if (cell.kind == CellKind::QUAD9) {
            NodeArray<ElementNodeCount::QUAD9> nodes;
            for (int i = 0; i < 9; ++i) {
                nodes[i] = mesh.points[static_cast<std::size_t>(cell.node[i])];
            }
            for (int i = 0; i <= per_cell; ++i) {
                for (int j = 0; j <= per_cell; ++j) {
                    const double xi = -1.0 + 2.0 * i / double(per_cell);
                    const double eta = -1.0 + 2.0 * j / double(per_cell);
                    const EiVector3d p = surface_point_quad9(
                        0.9 * xi, 0.9 * eta, nodes);
                    rays.emplace_back(camera, (p - camera).normalized());
                }
            }
        }
    }
    return rays;
}

// --------------------------------------------------------------------------------
// Intersection over one bundle, with cost accounting
// --------------------------------------------------------------------------------

// Quadratic arm: intersect every face with the matching Newton solver, tracking
// the total number of primitive tests. A brute-force sweep of every face.
template<ElementNodeCount E>
static Stats intersect_quadratic(const std::vector<Ray>& rays,
                                 const std::vector<QuadElement>& elements,
                                 const Surface& s)
{
    Stats st;
    st.rays = static_cast<long>(rays.size());

    for (const Ray& ray : rays) {
        double best_t = std::numeric_limits<double>::infinity();
        EiVector3d best_p = EiVector3d::Zero();
        EiVector3d best_n = EiVector3d::Zero();
        bool found = false;

        SolverCost cost;
        for (const QuadElement& e : elements) {
            st.primitive_tests += 1;

            IntersectResult r;
            if constexpr (E == ElementNodeCount::TRI6) {
                r = intersect_tri6_mode(ray, e.tri6, GuessMode::AdaptiveHull,
                                        &cost);
            }
            else {
                r = intersect_quad9_mode(ray, e.quad9, GuessMode::AdaptiveHull,
                                         &cost);
            }

            if (r.converged && r.t > 1e-7 && r.t < best_t) {
                best_t = r.t;
                best_p = ray.origin + r.t * ray.direction;
                best_n = r.normal;
                found = true;
            }
        }

        st.solver_runs += cost.runs;
        st.solver_iters += cost.iters;

        if (found) {
            st.hits += 1;
            st.record_hit(s, best_p, best_n);
        }
        else {
            st.missed += 1;
        }
    }
    return st;
}

// Linear arm: brute-force Möller-Trumbore over every sub-triangle.
static Stats intersect_linear(const std::vector<Ray>& rays,
                              const std::vector<Tri3>& tris,
                              const Surface& s)
{
    Stats st;
    st.rays = static_cast<long>(rays.size());

    for (const Ray& ray : rays) {
        double best_t = std::numeric_limits<double>::infinity();
        EiVector3d best_n = EiVector3d::Zero();
        bool found = false;

        for (const Tri3& tri : tris) {
            st.primitive_tests += 1;
            double t = 0.0;
            if (intersect_tri3(ray, tri, t) && t < best_t) {
                best_t = t;
                // Flat-facet normal of the winning triangle.
                best_n = (tri.b - tri.a).cross(tri.c - tri.a);
                found = true;
            }
        }

        if (found) {
            st.hits += 1;
            const EiVector3d p = ray.origin + best_t * ray.direction;
            st.record_hit(s, p, best_n);
        }
        else {
            st.missed += 1;
        }
    }
    return st;
}

// --------------------------------------------------------------------------------
// Camera export and view render
// --------------------------------------------------------------------------------

// Extracts the faces of one quadratic family from a mesh, as the Newton solver
// expects them.
static std::vector<QuadElement> quad_elements_of(const Mesh& mesh, CellKind want)
{
    std::vector<QuadElement> elements;
    for (const Cell& c : mesh.cells) {
        if (c.kind != want) continue;
        QuadElement e;
        if (want == CellKind::TRI6) {
            e.is_tri = true;
            for (int i = 0; i < 6; ++i) {
                e.tri6[i] = mesh.points[static_cast<std::size_t>(c.node[i])];
            }
        }
        else {
            e.is_tri = false;
            for (int i = 0; i < 9; ++i) {
                e.quad9[i] = mesh.points[static_cast<std::size_t>(c.node[i])];
            }
        }
        elements.push_back(e);
    }
    return elements;
}

// Writes the camera basis so the figure generator can redraw the same view.
static void write_camera_csv(std::ofstream& csv, const std::string& geometry,
                             const CameraView& cam)
{
    auto put = [&](const EiVector3d& v) {
        csv << ',' << v(0) << ',' << v(1) << ',' << v(2);
    };
    csv << geometry;
    put(cam.origin);
    put(cam.centre);
    put(cam.view);
    put(cam.right);
    put(cam.up);
    csv << ',' << cam.extent << ',' << cam.grid << '\n';
}

// Renders the camera bundle through the quadratic mesh and writes a PPM.
// This is the view the benchmark camera actually sees: the rays and the Newton
// intersection are exactly those timed in the camera arm.
template<ElementNodeCount E>
static long render_camera_view(const std::vector<Ray>& rays,
                               const std::vector<QuadElement>& elements,
                               const CameraView& cam,
                               const std::string& path)
{
    const int g = cam.grid;
    const std::size_t n = static_cast<std::size_t>(g) * g;
    // Near-white background
    std::vector<unsigned char> img(n * 3, 250);

    const EiVector3d light = -cam.view; // headlight, along the view direction
    // Shade in the blue.
    const double base_r = 0.231, base_g = 0.431, base_b = 0.647; // #3b6ea5
    long hits = 0;

    for (std::size_t k = 0; k < n && k < rays.size(); ++k) {
        const Ray& ray = rays[k];
        double best_t = std::numeric_limits<double>::infinity();
        EiVector3d best_n = EiVector3d::Zero();
        bool found = false;

        SolverCost cost;
        for (const QuadElement& e : elements) {
            IntersectResult hit;
            if constexpr (E == ElementNodeCount::TRI6) {
                hit = intersect_tri6_mode(ray, e.tri6, GuessMode::AdaptiveHull,
                                          &cost);
            }
            else {
                hit = intersect_quad9_mode(ray, e.quad9, GuessMode::AdaptiveHull,
                                           &cost);
            }
            if (hit.converged && hit.t > 1e-7 && hit.t < best_t) {
                best_t = hit.t;
                best_n = hit.normal;
                found = true;
            }
        }
        if (!found) continue;
        ++hits;

        const double nn = best_n.norm();
        double ndl = (nn > 0.0) ? std::abs(best_n.dot(light)) / nn : 0.0;
        ndl = std::min(1.0, std::max(0.0, ndl));
        // Headlight shading with a little ambient, tinted in the figure's blue.
        const double shade = 0.12 + 0.88 * std::pow(ndl, 0.8);
        const double tone = 0.30 + 0.70 * shade;

        // Screen row: j = 0 is the bottom of the camera basis, PPM stores the
        // top row first.
        const std::size_t j = k / static_cast<std::size_t>(g);
        const std::size_t i = k % static_cast<std::size_t>(g);
        const std::size_t row = static_cast<std::size_t>(g - 1) - j;
        const std::size_t idx = (row * static_cast<std::size_t>(g) + i) * 3;
        img[idx + 0] = static_cast<unsigned char>(255.0 * base_r * tone);
        img[idx + 1] = static_cast<unsigned char>(255.0 * base_g * tone);
        img[idx + 2] = static_cast<unsigned char>(255.0 * base_b * tone);
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Could not write camera view: " << path << '\n';
        return hits;
    }
    out << "P6\n" << g << ' ' << g << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.data()),
              static_cast<std::streamsize>(img.size()));
    out.close();
    return hits;
}

// --------------------------------------------------------------------------------
// Per-element geometric error of a mesh against the analytic surface
// --------------------------------------------------------------------------------

// Mean and max radial error of the element midpoints against the analytic
// surface.
static void mesh_geometric_error(const Mesh& mesh, const Surface& s,
                                 double& mean_err, double& max_err)
{
    double sum = 0.0;
    double worst = 0.0;
    long n = 0;

    // For the cylinder only the curved skin cells are scored.
    const bool cylinder = (s.kind == Geometry::Cylinder);

    for (const Cell& cell : mesh.cells) {
        EiVector3d centre = EiVector3d::Zero();
        int count = 0;
        for (int i = 0; i < cell.node_count; ++i) {
            centre += mesh.points[static_cast<std::size_t>(cell.node[i])];
            ++count;
        }
        centre /= double(count);

        if (cylinder) {
            const double on_skin =
                std::abs(std::sqrt(centre(0) * centre(0) + centre(2) * centre(2)) -
                         s.radius);
            // A cap cell's centroid sits well inside radius => skip it
            if (on_skin > 0.25 * s.radius) continue;
        }

        const double err = std::abs(radial_error(s, centre));
        sum += err;
        worst = std::max(worst, err);
        ++n;
    }

    mean_err = (n > 0) ? sum / double(n) : 0.0;
    max_err = worst;
}

// --------------------------------------------------------------------------------
// Arm driver
// --------------------------------------------------------------------------------

struct ArmRow
{
    std::string geometry;
    std::string arm;
    std::string level;
    int subdiv_k = 1;
    long elements = 0;
    long primitives = 0;
    std::string bundle;
    Stats stats;
};

static void write_row(std::ofstream& csv, const ArmRow& row)
{
    const double mean_radial =
        (row.stats.hits > 0) ? row.stats.sum_radial / row.stats.hits : 0.0;
    const double rms_radial = (row.stats.hits > 0)
        ? std::sqrt(row.stats.sum_radial_sq / row.stats.hits)
        : 0.0;
    std::vector<double> errors = row.stats.abs_errors;
    const double trimmed = trimmed_max(errors, 0.01);
    const double max_err = errors.empty() ? 0.0 : errors.back();
    const double coverage = (row.stats.rays > 0)
        ? double(row.stats.hits) / double(row.stats.rays)
        : 0.0;
    const double rays_per_s = (row.stats.wall_s > 0.0)
        ? double(row.stats.rays) / row.stats.wall_s
        : 0.0;
    const double mean_tests = (row.stats.rays > 0)
        ? double(row.stats.primitive_tests) / double(row.stats.rays)
        : 0.0;
    const double mean_iters = (row.stats.solver_runs > 0)
        ? double(row.stats.solver_iters) / double(row.stats.solver_runs)
        : 0.0;

    const double mean_normal = (row.stats.normal_samples > 0)
        ? row.stats.sum_normal / double(row.stats.normal_samples)
        : 0.0;
    const double rms_normal = (row.stats.normal_samples > 0)
        ? std::sqrt(row.stats.sum_normal_sq / double(row.stats.normal_samples))
        : 0.0;
    std::vector<double> normal_errs = row.stats.normal_errors;
    const double trimmed_normal = trimmed_max(normal_errs, 0.01);
    const double max_normal = normal_errs.empty() ? 0.0 : normal_errs.back();

    csv << row.geometry << ',' << row.arm << ',' << row.level << ','
        << row.subdiv_k << ',' << row.elements << ',' << row.primitives << ','
        << row.bundle << ','
        << row.stats.rays << ',' << row.stats.hits << ','
        << row.stats.missed << ','
        << std::setprecision(9) << coverage << ','
        << mean_radial << ',' << rms_radial << ','
        << trimmed << ',' << max_err << ','
        << mean_normal << ',' << rms_normal << ','
        << trimmed_normal << ',' << max_normal << ','
        << mean_tests << ',' << row.stats.solver_runs << ','
        << mean_iters << ','
        << row.stats.wall_s << ',' << rays_per_s << '\n';
}

// Time one intersection pass over a bundle, keeping the fastest of ``reps``.
template<typename Fn>
static double time_pass(int reps, Fn&& fn)
{
    double best = std::numeric_limits<double>::infinity();
    for (int rep = 0; rep < reps; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

// Runs the quadratic arm for one element family at one mesh level.
static void run_quadratic_level(const Mesh& mesh, const Surface& s,
                                const std::string& geometry_name,
                                const std::string& arm,
                                const std::string& level,
                                const std::vector<Ray>& camera,
                                const std::vector<Ray>& targeted,
                                const Options& opts,
                                std::ofstream& stats_csv)
{
    // Split the mesh faces of the requested family into elements.
    const CellKind want =
        (arm == "quadratic_tri") ? CellKind::TRI6 : CellKind::QUAD9;
    const std::vector<QuadElement> elements = quad_elements_of(mesh, want);

    if (elements.empty()) return;

    const long primitives = static_cast<long>(elements.size());

    auto emit = [&](const std::string& bundle, const std::vector<Ray>& rays) {
        Stats st;
        if (want == CellKind::TRI6) {
            st = intersect_quadratic<ElementNodeCount::TRI6>(rays, elements, s);
        }
        else {
            st = intersect_quadratic<ElementNodeCount::QUAD9>(rays, elements, s);
        }
        double wall = 0.0;
        if (want == CellKind::TRI6) {
            wall = time_pass(opts.reps, [&] {
                (void)intersect_quadratic<ElementNodeCount::TRI6>(
                    rays, elements, s);
            });
        }
        else {
            wall = time_pass(opts.reps, [&] {
                (void)intersect_quadratic<ElementNodeCount::QUAD9>(
                    rays, elements, s);
            });
        }
        st.wall_s = wall;

        ArmRow row;
        row.geometry = geometry_name;
        row.arm = arm;
        row.level = level;
        row.subdiv_k = 1;
        row.elements = static_cast<long>(elements.size());
        row.primitives = primitives;
        row.bundle = bundle;
        row.stats = st;
        write_row(stats_csv, row);
    };

    emit("camera", camera);
    emit("targeted", targeted);
}

// Runs a linear arm from a pre-built list of sub-triangles.
static void run_linear_level(const std::vector<Tri3>& tris,
                             const Surface& s,
                             const std::string& geometry_name,
                             const std::string& arm,
                             const std::string& level,
                             int subdiv_k,
                             long elements,
                             const std::vector<Ray>& camera,
                             const std::vector<Ray>& targeted,
                             const Options& opts,
                             std::ofstream& stats_csv)
{
    if (tris.empty()) return;

    auto emit = [&](const std::string& bundle, const std::vector<Ray>& rays) {
        Stats st = intersect_linear(rays, tris, s);
        st.wall_s = time_pass(opts.reps, [&] {
            (void)intersect_linear(rays, tris, s);
        });

        ArmRow row;
        row.geometry = geometry_name;
        row.arm = arm;
        row.level = level;
        row.subdiv_k = subdiv_k;
        row.elements = elements;
        row.primitives = static_cast<long>(tris.size());
        row.bundle = bundle;
        row.stats = st;
        write_row(stats_csv, row);
    };

    emit("camera", camera);
    emit("targeted", targeted);
}

// --------------------------------------------------------------------------------
// Geometry driver
// --------------------------------------------------------------------------------

static std::string geometry_name(Geometry g)
{
    return (g == Geometry::Sphere) ? "sphere" : "cylinder";
}

// Mesh files for the requested geometry. Quadratic arms use the quadratic mesh
// (one file per element family), linear_native uses the generated linear mesh,
// and linear_subdiv uses the quadratic mesh plus in-code sub-tessellation.
struct GeometryMeshes
{
    std::string tri_quad;   // TRI6 fine mesh
    std::string tri_coarse; // TRI6 coarse mesh
    std::string quad_quad;  // QUAD9 fine mesh
    std::string quad_coarse;// QUAD9 coarse mesh
    std::string lin;        // TRI3 mesh
};

static GeometryMeshes meshes_for(Geometry g)
{
    if (g == Geometry::Sphere) {
        return {
            data_dir_name + "/curved_sph_tri_fine.vtk",
            data_dir_name + "/curved_sph_tri_coarse.vtk",
            data_dir_name + "/curved_sph_quad_fine.vtk",
            std::string(),  // no coarse QUAD9 sphere mesh generated
            data_dir_name + "/curved_sph_lin.vtk",
        };
    }
    return {
        data_dir_name + "/curved_cyl_tri.vtk",
        std::string(),  // no coarse TRI6 cylinder mesh generated
        data_dir_name + "/curved_cyl_quad_fine.vtk",
        data_dir_name + "/curved_cyl_quad_coarse.vtk",
        data_dir_name + "/curved_cyl_lin.vtk",
    };
}

// The camera and targeted bundles are frozen here once per geometry and reused by every arm.
static void run_geometry(Geometry geo, const Surface& s, const Options& opts,
                         std::ofstream& stats_csv, std::ofstream& analytic_csv,
                         std::ofstream& camera_csv)
{
    const std::string gname = geometry_name(geo);
    const GeometryMeshes files = meshes_for(geo);

    Mesh mesh_tri;
    Mesh mesh_quad;
    Mesh mesh_lin;
    if (!read_vtk(files.tri_quad, mesh_tri)) return;
    if (!read_vtk(files.quad_quad, mesh_quad)) return;
    if (!read_vtk(files.lin, mesh_lin)) return;

    // Coarse quadratic meshes are optional, read them when present.
    Mesh mesh_tri_coarse;
    Mesh mesh_quad_coarse;
    const bool have_tri_coarse =
        !files.tri_coarse.empty() && read_vtk(files.tri_coarse, mesh_tri_coarse);
    const bool have_quad_coarse =
        !files.quad_coarse.empty() && read_vtk(files.quad_coarse, mesh_quad_coarse);

    const int cam_grid = opts.quick ? std::max(opts.grid / 3, 16) : opts.grid;
    const CameraView cam = make_camera(s, cam_grid);
    const std::vector<Ray> camera = build_camera_bundle(cam);
    const std::vector<Ray> targeted =
        build_targeted_bundle(s, mesh_tri.cells, mesh_tri,
                              opts.quick ? 2 : opts.targeted_per_cell);

    std::cout << "Geometry " << gname
              << ": camera rays " << camera.size()
              << ", targeted rays " << targeted.size() << '\n';

    // Export the camera basis and render the view it actually sees. The render
    // uses the fine quadratic mesh of the family the headline table reports
    // (TRI6 for the sphere, QUAD9 for the cylinder), so the image is the
    // benchmark's own intersection result. The figure is sampled more finely than the benchmark bundle 
    // for figure quality.
    write_camera_csv(camera_csv, gname, cam);
    if (opts.view_grid > 0) {
        CameraView vcam = cam;
        vcam.grid = opts.view_grid;
        const std::vector<Ray> view_rays = build_camera_bundle(vcam);
        const std::string view_path =
            bench_dir_name + "/" + bench_subdir_name +
            "/bench2_camera_view_" + gname + ".ppm";
        long view_hits = 0;
        if (geo == Geometry::Sphere) {
            view_hits = render_camera_view<ElementNodeCount::TRI6>(
                view_rays, quad_elements_of(mesh_tri, CellKind::TRI6), vcam,
                view_path);
        }
        else {
            view_hits = render_camera_view<ElementNodeCount::QUAD9>(
                view_rays, quad_elements_of(mesh_quad, CellKind::QUAD9), vcam,
                view_path);
        }
        std::cout << "  camera view (" << opts.view_grid << "x"
                  << opts.view_grid << "): " << view_hits << '/'
                  << view_rays.size() << " pixels hit -> " << view_path << '\n';
    }

    // Per-element geometric error vs the analytic surface for the table.
    auto write_analytic = [&](const std::string& label, const Mesh& mesh) {
        double mean_e = 0.0;
        double max_e = 0.0;
        mesh_geometric_error(mesh, s, mean_e, max_e);
        analytic_csv << gname << ',' << label << ','
                     << mesh.cells.size() << ','
                     << std::setprecision(9) << mean_e << ',' << max_e << '\n';
    };
    write_analytic("quadratic_tri_fine", mesh_tri);
    if (have_tri_coarse) write_analytic("quadratic_tri_coarse", mesh_tri_coarse);
    write_analytic("quadratic_quad_fine", mesh_quad);
    if (have_quad_coarse) write_analytic("quadratic_quad_coarse", mesh_quad_coarse);
    write_analytic("linear_native", mesh_lin);

    // ---- Quadratic arms -------------------------------------------------------
    run_quadratic_level(mesh_tri, s, gname, "quadratic_tri", "fine",
                        camera, targeted, opts, stats_csv);
    if (have_tri_coarse) {
        run_quadratic_level(mesh_tri_coarse, s, gname, "quadratic_tri", "coarse",
                            camera, targeted, opts, stats_csv);
    }
    run_quadratic_level(mesh_quad, s, gname, "quadratic_quad", "fine",
                        camera, targeted, opts, stats_csv);
    if (have_quad_coarse) {
        run_quadratic_level(mesh_quad_coarse, s, gname, "quadratic_quad", "coarse",
                            camera, targeted, opts, stats_csv);
    }

    // ---- Linear-native arm ----------------------------------------------------
    // The generated linear mesh matches the fine quadratic triangulation (midside
    // nodes dropped), so it is the native counterpart of the TRI6 arm only.
    {
        std::vector<Tri3> tris;
        for (const Cell& c : mesh_lin.cells) {
            if (c.kind == CellKind::TRI3) {
                tris.push_back(make_tri3(mesh_lin, c.node[0], c.node[1], c.node[2]));
            }
            else if (c.kind == CellKind::QUAD4) {
                tris.push_back(make_tri3(mesh_lin, c.node[0], c.node[1], c.node[2]));
                tris.push_back(make_tri3(mesh_lin, c.node[0], c.node[2], c.node[3]));
            }
        }
        run_linear_level(tris, s, gname, "linear_native", "native_tri6", 1,
                         static_cast<long>(mesh_lin.cells.size()),
                         camera, targeted, opts, stats_csv);
    }

    // ---- Linear sub-tessellation arm -----------------------------------------
    // Each quadratic family gets its own sub-tessellation ladder. 
    // The two ladders are reported separately (level tag carries
    // the source family).
    auto subdiv_ladder = [&](const Mesh& mesh, CellKind want,
                             const std::string& level_prefix) {
        std::vector<Cell> sources;
        for (const Cell& c : mesh.cells) {
            if (c.kind == want) sources.push_back(c);
        }
        if (sources.empty()) return;

        for (const int k : opts.subdiv_k) {
            std::vector<Tri3> tris;
            for (const Cell& c : sources) subdivide_cell(mesh, c, k, tris);

            // Geometric error of the flat-facet linearisation (triangle
            // centroids vs the analytic surface), this is the sagitta for the plot.
            {
                double sum = 0.0;
                double worst = 0.0;
                for (const Tri3& tri : tris) {
                    const EiVector3d centre = (tri.a + tri.b + tri.c) / 3.0;
                    const double err = std::abs(radial_error(s, centre));
                    sum += err;
                    worst = std::max(worst, err);
                }
                const double mean_e = tris.empty() ? 0.0 : sum / double(tris.size());
                analytic_csv << gname << ",linear_subdiv_" << level_prefix
                             << "k" << k << ',' << tris.size() << ','
                             << std::setprecision(9) << mean_e << ',' << worst
                             << '\n';
            }

            const std::string level = level_prefix + "k" + std::to_string(k);
            run_linear_level(tris, s, gname, "linear_subdiv", level, k,
                             static_cast<long>(sources.size()),
                             camera, targeted, opts, stats_csv);
        }
    };

    subdiv_ladder(mesh_tri, CellKind::TRI6, "tri6_");
    subdiv_ladder(mesh_quad, CellKind::QUAD9, "quad9_");
    if (have_tri_coarse) {
        subdiv_ladder(mesh_tri_coarse, CellKind::TRI6, "tri6coarse_");
    }
    if (have_quad_coarse) {
        subdiv_ladder(mesh_quad_coarse, CellKind::QUAD9, "quad9coarse_");
    }
}

// --------------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------------

static bool list_contains(const std::string& list, const std::string& token)
{
    if (list.empty()) return true;
    std::stringstream ss(list);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item == token) return true;
    }
    return false;
}

static void parse_args(int argc, char** argv, Options& opts)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](void) -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
        };

        if (arg == "--grid") {
            const std::string v = value();
            if (!v.empty()) opts.grid = std::stoi(v);
        }
        else if (arg == "--targeted") {
            const std::string v = value();
            if (!v.empty()) opts.targeted_per_cell = std::stoi(v);
        }
        else if (arg == "--reps") {
            const std::string v = value();
            if (!v.empty()) opts.reps = std::stoi(v);
        }
        else if (arg == "--subdiv") {
            const std::string v = value();
            if (!v.empty()) {
                opts.subdiv_k.clear();
                std::stringstream ss(v);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    opts.subdiv_k.push_back(std::stoi(item));
                }
            }
        }
        else if (arg == "--geometries") {
            opts.geometry_filter = value();
        }
        else if (arg == "--arms") {
            opts.arm_filter = value();
        }
        else if (arg == "--quick") {
            opts.quick = true;
        }
        else if (arg == "--view-grid") {
            const std::string v = value();
            if (!v.empty()) opts.view_grid = std::stoi(v);
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: bench2_quadratic_vs_linear [options]\n"
                << "  --grid N          camera grid size per geometry (default 96)\n"
                << "  --targeted N      targeted rays per element side (default 4)\n"
                << "  --reps N          timing repetitions, keep fastest (default 3)\n"
                << "  --subdiv a,b,c    sub-tessellation ladder (default 1,2,4,8)\n"
                << "  --geometries a,b  sphere,cylinder\n"
                << "  --arms a,b        quadratic_tri,quadratic_quad,"
                   "linear_native,linear_subdiv\n"
                << "  --view-grid N     camera-view render resolution, 0 = off "
                   "(default 320)\n"
                << "  --quick           fast smoke run\n";
            std::exit(0);
        }
    }
}

int main(int argc, char** argv)
{
    Options opts;
    parse_args(argc, argv, opts);

    const std::string out_dir = bench_dir_name + "/" + bench_subdir_name;
    std::filesystem::create_directories(out_dir);

    std::ofstream stats_csv(out_dir + "/bench2_stats.csv");
    std::ofstream analytic_csv(out_dir + "/bench2_vs_analytic.csv");
    std::ofstream camera_csv(out_dir + "/bench2_camera.csv");
    if (!stats_csv || !analytic_csv || !camera_csv) {
        std::cerr << "Failed to open output CSV files under " << out_dir << '\n';
        return 1;
    }

    stats_csv << "geometry,arm,level,subdiv_k,elements,primitives,bundle,"
                 "rays,hits,missed,coverage,mean_radial_err,rms_radial_err,"
                 "trimmed_max_radial_err,max_radial_err,"
                 "mean_normal_err_deg,rms_normal_err_deg,"
                 "trimmed_max_normal_err_deg,max_normal_err_deg,"
                 "mean_primitive_tests,"
                 "solver_runs,mean_iters_per_run,wall_time_s,rays_per_s\n";
    analytic_csv << "geometry,mesh,cells,mean_radial_err,max_radial_err\n";
    camera_csv << "geometry,"
                  "origin_x,origin_y,origin_z,"
                  "centre_x,centre_y,centre_z,"
                  "view_x,view_y,view_z,"
                  "right_x,right_y,right_z,"
                  "up_x,up_y,up_z,extent,grid\n";

    std::cout << "Benchmark case 2: quadratic vs sub-tessellated linear\n"
              << "  camera grid = " << opts.grid
              << ", targeted/cell = " << opts.targeted_per_cell
              << ", reps = " << opts.reps << "\n  subdiv ladder =";
    for (const int k : opts.subdiv_k) std::cout << ' ' << k;
    std::cout << "\n";

    Surface sphere;
    sphere.kind = Geometry::Sphere;
    sphere.radius = 0.5;

    Surface cylinder;
    cylinder.kind = Geometry::Cylinder;
    cylinder.radius = 0.5;
    cylinder.y_min = 0.0;
    cylinder.y_max = 1.0;

    if (list_contains(opts.geometry_filter, "sphere")) {
        run_geometry(Geometry::Sphere, sphere, opts, stats_csv, analytic_csv,
                     camera_csv);
    }
    if (list_contains(opts.geometry_filter, "cylinder")) {
        run_geometry(Geometry::Cylinder, cylinder, opts, stats_csv, analytic_csv,
                     camera_csv);
    }

    std::cout << "Complete! Wrote outputs to " << out_dir << '\n';
    return 0;
}

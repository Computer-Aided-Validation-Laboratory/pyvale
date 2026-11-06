// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================
//
// Benchmark case 1: adaptive-hull initial guess for the Newton solver.
//
// This benchmark reuses the distorted high-order element geometries of verification
// case 1 but is a self-contained run file, it does not modify or depend on the
// case 1 outputs. The aim is to quantify both halves of the adaptive-hull
// trade-off so that the one-off cost of building a hull can be justified:
//
//   1. Fewer Newton solver runs. The projected triangle-fan test rejects rays
//      that do not pass through the element before any Newton iteration is
//      performed.
//   2. Fewer Newton iterations per solve. The barycentric seed taken from the
//      fan is closer to the solution than the element centre, so the solver
//      converges in fewer iterations.
//
// The intersection is evaluated in three modes over the same rays:
//
//   * adaptive_hull           - production behaviour: triangle-fan test plus
//                               damped barycentric seed.
//   * fan_reject_center_seed  - triangle-fan test but the element-centre seed.
//                               Isolates the seed contribution.
//   * center_seed             - no hull at all: element-centre seed. This is
//                               the no-adaptive-hull baseline.
//
// Two ray bundles are used per element and frame:
//
//   * hit bundle      - rays constructed from a structured reference grid,
//                       like case 1. Every ray intersects the element; used
//                       for iterations-per-solve and accuracy.
//   * coverage bundle - a square grid of camera rays spanning the element's
//                       bounding circle, so a realistic fraction misses. Used
//                       for solver runs avoided and for the timing study.
//
// Output layout (self-contained under ``verif_bench/bench/bench1``)::
//
//   adaptive_hull_stats.csv            per case/frame/bundle/mode measurement
//   adaptive_hull_summary.csv          per element/mode totals
//   adaptive_hull_iter_hist.csv        iteration histograms, per element/mode
//   adaptive_hull_conservativeness.csv hull rejection vs centre-seed hits
//
// The corresponding paper figures are produced by ``paper_bench1_adaptive_hull.py``.
//
// Build (run from ``verif_bench/``)::
//
// Example build (run from verif_bench/):
//     g++ -O2 -std=c++20 -fopenmp -o bench1_adaptive_hull
//         bench1_adaptive_hull.cpp cpp/rtrayintersection_modes.cpp
//
// Run (run from ``verif_bench/``)::
//
//   ./bench1_adaptive_hull
// ================================================================================

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../common_cpp/Eigen/Dense"

#include "cpp/rteigentypes.h"
#include "cpp/rtshapefuncs.h"
#include "cpp/rtrayintersection_modes.h"

#include "verif_cpp/meshio.h"
#include "verif_cpp/verif.h"
#include "verif_cpp/verifconstants.h"
#include "verif_cpp/orchestration.h"


// Self-contained output root.
static const std::string bench_dir_name = "bench";
static const std::string bench_subdir_name = "bench1";

struct Options
{
    std::size_t hit_grid = 150;
    std::size_t coverage_grid = 100;
    double coverage_scale = 1.5;
    int reps = 3;
    std::string case_filter;    // comma separated case names, empty = all
    std::string element_filter; // comma separated element labels, empty = all
};

// One row of the per-bundle measurement.
struct ModeRow
{
    GuessMode mode = GuessMode::AdaptiveHull;
    long rays = 0;
    long runs = 0;
    long rejected = 0;
    long converged = 0;
    long in_domain = 0;
    long iters = 0;
    long max_iters = 0;
    long hull_points = 0;
    long projected_nodes = 0;
    long edge_tests = 0;
    double wall_s = 0.0;
    double max_abs_err_xi = 0.0;
    double max_abs_err_eta = 0.0;
    long wrong_root = 0;
};

// Totals accumulated over every case and frame, per (element, mode).
struct Totals
{
    long hit_rays = 0;
    long hit_runs = 0;
    long hit_iters = 0;
    long hit_max_iters = 0;
    long cov_rays = 0;
    long cov_runs = 0;
    long cov_rejected = 0;
    long cov_iters = 0;
    long hull_points = 0;
    long projected_nodes = 0;
    long edge_tests = 0;
    double wall_s = 0.0;
};


// --------------------------------------------------------------------------------
// Small helpers
// --------------------------------------------------------------------------------

static const char* element_label(ElementNodeCount e)
{
    switch (e) {
        case ElementNodeCount::TRI3:  return "tri3";
        case ElementNodeCount::TRI6:  return "tri6";
        case ElementNodeCount::QUAD4: return "quad4";
        case ElementNodeCount::QUAD8: return "quad8";
        case ElementNodeCount::QUAD9: return "quad9";
    }
    return "unknown";
}

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

template<ElementNodeCount E>
static NodeArray<E> frame_nodes(const SimData& sim_data, std::size_t frame_idx)
{
    NodeArray<E> nodes;
    const Eigen::Map<const Eigen::VectorXi>& elem = sim_data.connect.elem(0);

    for (std::size_t nn = 0; nn < static_cast<std::size_t>(E); ++nn) {
        const std::size_t node_idx = static_cast<std::size_t>(elem[nn]);
        nodes[nn] = sim_data.coords.mat.row(node_idx);

        if (sim_data.field) {
            const Field& field = *sim_data.field;
            nodes[nn](0) += field.at(static_cast<int>(frame_idx),
                                     static_cast<int>(node_idx), 0);
            nodes[nn](1) += field.at(static_cast<int>(frame_idx),
                                     static_cast<int>(node_idx), 1);
            nodes[nn](2) += field.at(static_cast<int>(frame_idx),
                                     static_cast<int>(node_idx), 2);
        }
    }
    return nodes;
}

// Isoparametric surface point S(xi, eta).
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

template<ElementNodeCount E>
static IntersectResult intersect_mode(const Ray& ray,
                                      const NodeArray<E>& nodes,
                                      GuessMode mode,
                                      SolverCost* cost)
{
    if constexpr (E == ElementNodeCount::TRI6) {
        return intersect_tri6_mode(ray, nodes, mode, cost);
    }
    else if constexpr (E == ElementNodeCount::QUAD8) {
        return intersect_quad8_mode(ray, nodes, mode, cost);
    }
    else {
        return intersect_quad9_mode(ray, nodes, mode, cost);
    }
}

template<ElementNodeCount E>
static bool in_domain(const IntersectResult& r)
{
    return isInParametricDomain<E>(r.uv.x(), r.uv.y());
}


// --------------------------------------------------------------------------------
// Ray bundles
// --------------------------------------------------------------------------------

// Structured reference grid, as in case 1. Every ray is aimed at a known
// surface point, so all rays hit the element.
template<ElementNodeCount E>
static void build_hit_bundle(const NodeArray<E>& nodes,
                             const EiVector3d& camera_center,
                             std::size_t grid_num,
                             std::vector<Ray>& rays,
                             std::vector<double>& xi_true,
                             std::vector<double>& eta_true)
{
    std::vector<SamplePoint> samples;
    appendStructuredSamples<E>(samples, grid_num);

    rays.clear();
    xi_true.clear();
    eta_true.clear();
    rays.reserve(samples.size());
    xi_true.reserve(samples.size());
    eta_true.reserve(samples.size());

    for (const SamplePoint& s : samples) {
        const EiVector3d P = surface_point<E>(s.xi_true, s.eta_true, nodes);
        const EiVector3d d = (P - camera_center).normalized();
        rays.emplace_back(camera_center, d);
        xi_true.push_back(s.xi_true);
        eta_true.push_back(s.eta_true);
    }
}

// Square grid of camera rays spanning the element's bounding circle in the
// element plane, so a fraction of the rays misses the element.
template<ElementNodeCount E>
static void build_coverage_bundle(const NodeArray<E>& nodes,
                                  const EiVector3d& camera_center,
                                  std::size_t grid_num,
                                  double coverage_scale,
                                  std::vector<Ray>& rays)
{
    static constexpr int N = static_cast<int>(E);

    EiVector3d centroid = EiVector3d::Zero();
    for (int i = 0; i < N; ++i) {
        centroid += nodes[i];
    }
    centroid /= double(N);

    double radius = 0.0;
    for (int i = 0; i < N; ++i) {
        radius = std::max(radius, (nodes[i] - centroid).norm());
    }

    // Element plane from two corner edge vectors.
    const EiVector3d edge_a = nodes[1] - nodes[0];
    const EiVector3d edge_b =
        (E == ElementNodeCount::TRI6) ? (nodes[2] - nodes[0])
                                      : (nodes[3] - nodes[0]);
    EiVector3d normal = edge_a.cross(edge_b);
    if (normal.norm() < 1e-14) {
        normal = EiVector3d(0.0, 0.0, 1.0);
    }
    normal.normalize();

    const EiVector3d reference =
        (std::abs(normal.z()) < 0.9) ? EiVector3d(0.0, 0.0, 1.0)
                                     : EiVector3d(1.0, 0.0, 0.0);
    const EiVector3d e1 = reference.cross(normal).normalized();
    const EiVector3d e2 = normal.cross(e1);

    const double half = coverage_scale * radius;

    rays.clear();
    rays.reserve(grid_num * grid_num);
    for (std::size_t j = 0; j < grid_num; ++j) {
        const double b = (grid_num == 1)
            ? 0.0
            : (2.0 * double(j) / double(grid_num - 1) - 1.0) * half;
        for (std::size_t i = 0; i < grid_num; ++i) {
            const double a = (grid_num == 1)
                ? 0.0
                : (2.0 * double(i) / double(grid_num - 1) - 1.0) * half;
            const EiVector3d target = centroid + a * e1 + b * e2;
            rays.emplace_back(camera_center,
                              (target - camera_center).normalized());
        }
    }
}


// --------------------------------------------------------------------------------
// Measurement
// --------------------------------------------------------------------------------

template<ElementNodeCount E>
static bool hit(const IntersectResult& r)
{
    return r.converged && in_domain<E>(r);
}

template<ElementNodeCount E>
static ModeRow measure_bundle(const std::vector<Ray>& rays,
                             const std::vector<double>* xi_true,
                             const std::vector<double>* eta_true,
                             const NodeArray<E>& nodes,
                             GuessMode mode,
                             int reps,
                             std::vector<long>& hist)
{
    ModeRow row;
    row.mode = mode;
    row.rays = static_cast<long>(rays.size());
    hist.assign(NEWTON_ITER_MAX + 2, 0L);

    SolverCost cost;
    for (std::size_t i = 0; i < rays.size(); ++i) {
        const IntersectResult r = intersect_mode<E>(rays[i], nodes, mode, &cost);

        if (r.ran_newton) {
            const long it = static_cast<long>(r.iters);
            if (it >= 0 && it < static_cast<long>(hist.size())) {
                hist[it] += 1;
            }
            row.iters += it;
            row.max_iters = std::max(row.max_iters, it);
        }

        if (r.converged) {
            row.converged += 1;
            if (in_domain<E>(r)) {
                row.in_domain += 1;
                if (xi_true != nullptr && eta_true != nullptr) {
                    const double ex = std::abs(r.uv.x() - (*xi_true)[i]);
                    const double ey = std::abs(r.uv.y() - (*eta_true)[i]);
                    row.max_abs_err_xi = std::max(row.max_abs_err_xi, ex);
                    row.max_abs_err_eta = std::max(row.max_abs_err_eta, ey);
                    if (std::hypot(ex, ey) > 1.0e-6) {
                        row.wrong_root += 1;
                    }
                }
            }
        }
    }

    row.runs = cost.runs;
    row.rejected = cost.rejected;
    row.hull_points = cost.hull_points;
    row.projected_nodes = cost.projected_nodes;
    row.edge_tests = cost.edge_tests;

    // Timing: repeat the ray loop and keep the fastest pass.
    double best = std::numeric_limits<double>::infinity();
    for (int rep = 0; rep < reps; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        for (const Ray& ray : rays) {
            (void)intersect_mode<E>(ray, nodes, mode, nullptr);
        }
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    row.wall_s = best;

    return row;
}

// This is to answer the question:
// Does the hull fan test ever reject a ray that the centre seed resolves? If
// so the hull would be producing false negatives.
// Variables:
// hull_hits	|H|	rays the AdaptiveHull mode resolves to a valid root
// center_hits	|C|	rays the CentreSeed mode resolves (no hull, so Newton runs on every ray)
// hull_miss_center_hit	|C \ H|	hull false negatives — the hull rejected a ray that the centre seed resolves
// hull_hit_center_miss	|H \ C|	hull rescues — the hull resolves a ray the centre seed does not
template<ElementNodeCount E>
static void measure_conservativeness(const std::vector<Ray>& rays,
                                     const NodeArray<E>& nodes,
                                     long& hull_hits,
                                     long& center_hits,
                                     long& hull_miss_center_hit,
                                     long& hull_hit_center_miss)
{
    hull_hits = 0;
    center_hits = 0;
    hull_miss_center_hit = 0;
    hull_hit_center_miss = 0;

    for (const Ray& ray : rays) {
        const IntersectResult a = intersect_mode<E>(
            ray, nodes, GuessMode::AdaptiveHull, nullptr);
        const IntersectResult b = intersect_mode<E>(
            ray, nodes, GuessMode::CenterSeed, nullptr);

        const bool hull_hit = hit<E>(a);
        const bool centre_hit = hit<E>(b);

        if (hull_hit) hull_hits += 1;
        if (centre_hit) center_hits += 1;
        if (!hull_hit && centre_hit) hull_miss_center_hit += 1;
        if (hull_hit && !centre_hit) hull_hit_center_miss += 1;
    }
}


// --------------------------------------------------------------------------------
// Output
// --------------------------------------------------------------------------------

static void write_stats_row(std::ofstream& csv,
                            const DistortCase& spec,
                            const std::string& element,
                            std::size_t frame,
                            const std::string& bundle,
                            const ModeRow& row)
{
    csv << spec.case_name << ',' << element << ',' << frame << ','
        << bundle << ',' << guess_mode_label(row.mode) << ','
        << row.rays << ',' << row.runs << ',' << row.rejected << ','
        << row.converged << ',' << row.in_domain << ','
        << row.iters << ',' << row.max_iters << ','
        << row.hull_points << ',' << row.projected_nodes << ','
        << row.edge_tests << ','
        << std::setprecision(9) << row.wall_s << ','
        << row.max_abs_err_xi << ',' << row.max_abs_err_eta << ','
        << row.wrong_root << '\n';
}

static void accumulate(std::map<std::string, Totals>& totals,
                       std::map<std::string, std::vector<long>>& hist_map,
                       const std::string& element,
                       const std::string& bundle,
                       const ModeRow& row,
                       const std::vector<long>& hist)
{
    const std::string mode = guess_mode_label(row.mode);
    Totals& t = totals[element + "|" + mode];

    t.hull_points += row.hull_points;
    t.projected_nodes += row.projected_nodes;
    t.edge_tests += row.edge_tests;
    t.wall_s += row.wall_s;

    if (bundle == "hit") {
        t.hit_rays += row.rays;
        t.hit_runs += row.runs;
        t.hit_iters += row.iters;
        t.hit_max_iters = std::max(t.hit_max_iters, row.max_iters);
    }
    else {
        t.cov_rays += row.rays;
        t.cov_runs += row.runs;
        t.cov_rejected += row.rejected;
        t.cov_iters += row.iters;
    }

    std::vector<long>& h = hist_map[element + "|" + bundle + "|" + mode];
    if (h.empty()) h.assign(hist.size(), 0L);
    for (std::size_t i = 0; i < hist.size(); ++i) {
        h[i] += hist[i];
    }
}


// --------------------------------------------------------------------------------
// Case driver
// --------------------------------------------------------------------------------

template<ElementNodeCount E>
static void run_case(const DistortCase& spec,
                     const SimData& sim_data,
                     const Options& opts,
                     std::ofstream& stats_csv,
                     std::ofstream& conserv_csv,
                     std::map<std::string, Totals>& totals,
                     std::map<std::string, std::vector<long>>& hist_map)
{
    const std::string element = element_label(spec.nodes_per_element);
    const std::size_t time_steps = sim_data.field
        ? static_cast<std::size_t>(sim_data.field->getTimeN())
        : 1;

    const GuessMode modes[3] = {
        GuessMode::AdaptiveHull,
        GuessMode::FanRejectCenterSeed,
        GuessMode::CenterSeed,
    };

    for (std::size_t frame = 0; frame < time_steps; ++frame) {
        const NodeArray<E> nodes = frame_nodes<E>(sim_data, frame);

        std::vector<Ray> hit_rays;
        std::vector<double> xi_true;
        std::vector<double> eta_true;
        build_hit_bundle<E>(nodes, spec.camera_center, opts.hit_grid,
                            hit_rays, xi_true, eta_true);

        std::vector<Ray> cov_rays;
        build_coverage_bundle<E>(nodes, spec.camera_center,
                                 opts.coverage_grid, opts.coverage_scale,
                                 cov_rays);

        for (const GuessMode mode : modes) {
            std::vector<long> hit_hist;
            const ModeRow hit_row = measure_bundle<E>(
                hit_rays, &xi_true, &eta_true, nodes, mode, opts.reps, hit_hist);
            write_stats_row(stats_csv, spec, element, frame, "hit", hit_row);
            accumulate(totals, hist_map, element, "hit", hit_row, hit_hist);

            std::vector<long> cov_hist;
            const ModeRow cov_row = measure_bundle<E>(
                cov_rays, nullptr, nullptr, nodes, mode, opts.reps, cov_hist);
            write_stats_row(stats_csv, spec, element, frame, "coverage", cov_row);
            accumulate(totals, hist_map, element, "coverage", cov_row, cov_hist);
        }

        long hull_hits = 0;
        long centre_hits = 0;
        long hull_miss_centre_hit = 0;
        long hull_hit_centre_miss = 0;
        measure_conservativeness<E>(cov_rays, nodes, hull_hits, centre_hits,
                                    hull_miss_centre_hit, hull_hit_centre_miss);

        conserv_csv << spec.case_name << ',' << element << ',' << frame << ','
                    << cov_rays.size() << ',' << hull_hits << ','
                    << centre_hits << ',' << hull_miss_centre_hit << ','
                    << hull_hit_centre_miss << '\n';
    }

    std::cout << "  " << spec.case_name << " " << element
              << " (" << time_steps << " frames) done\n";
}


// --------------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------------

static void parse_args(int argc, char** argv, Options& opts)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](void) -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
        };

        if (arg == "--grid") {
            const std::string v = value();
            if (!v.empty()) opts.hit_grid = std::stoul(v);
        }
        else if (arg == "--coverage") {
            const std::string v = value();
            if (!v.empty()) opts.coverage_grid = std::stoul(v);
        }
        else if (arg == "--coverage-scale") {
            const std::string v = value();
            if (!v.empty()) opts.coverage_scale = std::stod(v);
        }
        else if (arg == "--reps") {
            const std::string v = value();
            if (!v.empty()) opts.reps = std::stoi(v);
        }
        else if (arg == "--cases") {
            opts.case_filter = value();
        }
        else if (arg == "--elements") {
            opts.element_filter = value();
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: bench1_adaptive_hull [options]\n"
                << "  --grid N            structured reference grid size (default 150)\n"
                << "  --coverage N        coverage grid size (default 100)\n"
                << "  --coverage-scale F  coverage half-width in element radii (default 1.5)\n"
                << "  --reps N            timing repetitions, keep fastest (default 3)\n"
                << "  --cases a,b         restrict to these distortion cases\n"
                << "  --elements tri6,quad8,quad9\n";
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

    std::ofstream stats_csv(bench_subdir_name.empty()
        ? std::string()
        : bench_dir_name + "/" + bench_subdir_name + "/adaptive_hull_stats.csv");
    std::ofstream conserv_csv(bench_subdir_name.empty()
        ? std::string()
        : bench_dir_name + "/" + bench_subdir_name + "/adaptive_hull_conservativeness.csv");

    if (!stats_csv || !conserv_csv) {
        std::cerr << "Failed to open output CSV files under " << bench_subdir_name
                  << '\n';
        return 1;
    }

    stats_csv << "case,element,frame,bundle,mode,rays,runs,rejected,"
                 "converged,in_domain,total_iters,max_iters,hull_points,"
                 "projected_nodes,edge_tests,wall_time_s,max_abs_err_xi,"
                 "max_abs_err_eta,wrong_root\n";

    conserv_csv << "case,element,frame,coverage_rays,hull_hits,center_hits,"
                   "hull_miss_center_hit,hull_hit_center_miss\n";

    std::cout << "Benchmark case 1: adaptive-hull initial guess\n"
              << "  hit grid = " << opts.hit_grid
              << ", coverage grid = " << opts.coverage_grid
              << ", coverage scale = " << opts.coverage_scale
              << ", reps = " << opts.reps << "\n";

    std::map<std::string, Totals> totals;
    std::map<std::string, std::vector<long>> hist_map;

    for (const DistortCase& spec : distort_cases) {
        if (!list_contains(opts.case_filter, spec.case_name)) continue;
        if (!list_contains(opts.element_filter,
                           element_label(spec.nodes_per_element))) continue;

        const SimData sim_data = loadData(spec.data_dir);

        switch (spec.nodes_per_element) {
            case ElementNodeCount::TRI6:
                run_case<ElementNodeCount::TRI6>(spec, sim_data, opts,
                    stats_csv, conserv_csv, totals, hist_map);
                break;
            case ElementNodeCount::QUAD8:
                run_case<ElementNodeCount::QUAD8>(spec, sim_data, opts,
                    stats_csv, conserv_csv, totals, hist_map);
                break;
            case ElementNodeCount::QUAD9:
                run_case<ElementNodeCount::QUAD9>(spec, sim_data, opts,
                    stats_csv, conserv_csv, totals, hist_map);
                break;
            default:
                break;
        }
    }

    // Aggregated per (element, mode) totals.
    const std::string out_dir_path = bench_dir_name + "/" + bench_subdir_name;
    std::ofstream summary_csv(out_dir_path + "/adaptive_hull_summary.csv");
    summary_csv << "element,mode,hit_rays,hit_runs,hit_mean_iters,hit_max_iters,"
                << "coverage_rays,coverage_runs,coverage_rejected,"
                << "coverage_total_iters,hull_points,projected_nodes,"
                << "edge_tests,wall_time_s\n";

    for (const auto& [key, t] : totals) {
        const std::size_t bar = key.find('|');
        const std::string element = key.substr(0, bar);
        const std::string mode = key.substr(bar + 1);
        const double mean_iters = (t.hit_runs > 0)
            ? double(t.hit_iters) / double(t.hit_runs)
            : 0.0;
        summary_csv << element << ',' << mode << ','
                    << t.hit_rays << ',' << t.hit_runs << ','
                    << std::setprecision(9) << mean_iters << ','
                    << t.hit_max_iters << ','
                    << t.cov_rays << ',' << t.cov_runs << ','
                    << t.cov_rejected << ',' << t.cov_iters << ','
                    << t.hull_points << ',' << t.projected_nodes << ','
                    << t.edge_tests << ',' << t.wall_s << '\n';
    }

    // Iteration histograms.
    std::ofstream hist_csv(out_dir_path + "/adaptive_hull_iter_hist.csv");
    hist_csv << "element,bundle,mode,iters,count\n";
    for (const auto& [key, counts] : hist_map) {
        std::stringstream ss(key);
        std::string element, bundle, mode;
        std::getline(ss, element, '|');
        std::getline(ss, bundle, '|');
        std::getline(ss, mode, '|');
        for (std::size_t i = 0; i < counts.size(); ++i) {
            if (counts[i] == 0) continue;
            hist_csv << element << ',' << bundle << ',' << mode << ','
                     << i << ',' << counts[i] << '\n';
        }
    }

    std::cout << "Complete! Wrote outputs to " << out_dir_path << "\n";
    return 0;
}

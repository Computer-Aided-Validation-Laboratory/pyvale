// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================
//
// Verification-only variant of the higher-order ray-element intersection used
// by the raytracer. It reuses the same Newton-Raphson solver and the same
// projected adaptive hull, but exposes the initial-guess strategy as a runtime
// switch so that the study can separate the two benefits of the hull:
//
//   * AdaptiveHull            - projected adaptive hull + triangle-fan test,
//                               damped barycentric initial guess. This is the
//                               production behaviour.
//   * FanRejectCenterSeed     - projected adaptive hull + triangle-fan test,
//                               element-centre initial guess. Isolates the
//                               rejection benefit from the seeding benefit.
//   * CenterSeed              - no hull at all: element-centre initial guess.
//                               This is the no-adaptive-hull baseline.
//
// The file is a self-contained copy.
// ================================================================================

#pragma once

// STD header files
#include <array>
#include <vector>

// ray tracer header files
#include "rteigentypes.h"
#include "rtray.h"
#include "rthitrecord.h"
#include "rtelemconstants.h"
#include "rtshapefuncs.h"

enum class GuessMode
{
    AdaptiveHull,
    FanRejectCenterSeed,
    CenterSeed
};

// Newton iteration budget used by the solver.
inline constexpr int NEWTON_ITER_MAX = 30;

const char* guess_mode_label(GuessMode mode);

// Accumulated operation counts for one intersection call. Passing nullptr
// disables counting.
struct SolverCost
{
    // Adaptive-hull construction.
    long projected_nodes = 0; // node projections onto the ray plane
    long edge_tests = 0;      // edge convexity tests
    long hull_points = 0;     // support points kept in the hull polygon

    // Newton-Raphson.
    long runs = 0;     // calls that entered the iterative loop
    long rejected = 0; // calls rejected by the hull fan test (no Newton)
    long iters = 0;    // iterations accumulated over all runs
};

struct IntersectResult
{
    double t = 0.0;
    EiVector3d normal = EiVector3d::Zero();
    Eigen::Vector2d uv = Eigen::Vector2d::Zero();
    int iters = 0;
    bool ran_newton = false;
    bool rejected = false;
    bool converged = false;
};

IntersectResult intersect_tri6_mode(
    const Ray& ray,
    const std::array<EiVector3d, ElementNodeCount::TRI6>& nodes,
    GuessMode mode,
    SolverCost* cost);

IntersectResult intersect_quad8_mode(
    const Ray& ray,
    const std::array<EiVector3d, ElementNodeCount::QUAD8>& nodes,
    GuessMode mode,
    SolverCost* cost);

IntersectResult intersect_quad9_mode(
    const Ray& ray,
    const std::array<EiVector3d, ElementNodeCount::QUAD9>& nodes,
    GuessMode mode,
    SolverCost* cost);

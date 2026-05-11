// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once

// STD header files
#include <vector>

// raytracer header files
#include "rteigentypes.h"
#include "rtray.h"
#include "rtbvh.h"

// Struct to store ray data in the stack-based shader
struct RayState{
    Ray ray;
    EiVector3d accumulated_color; // Accumulated multipliers (albedo, Fresnel terms, etc.)
    int depth;
};

inline EiVector3d ray_blue_sky(const Ray& ray){
    double a = 0.5 * (ray.direction(1) + 1.0);
    static EiVector3d white, blue;
    white << 1.0, 1.0, 1.0;
    blue << 0.5, 0.7, 1.0;
    return (1.0 - a) * white + a * blue;
}

void ray_diffuse(const RayState& current_state,
    const HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color);

void ray_specular(const RayState& current_state,
    const HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color);

void ray_refractive(const RayState& current_state,
    const HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color);

void ray_unlit(const RayState& current_state,
    const HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color);

void ray_undefined(const RayState& current_state,
    const HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color);
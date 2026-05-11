// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================
#pragma once 
// STD header files
#include <iostream>
#include <limits>
#include <array>

// ray tracer header files
#include "rtelemconstants.h"

inline std::array<double, ElementNodeCount::QUAD4> compute_shape_quad4(const double u, const double v){
    // Weights for bilinear interpolation
    std::array<double, ElementNodeCount::QUAD4> N;
    N[0] = (1.0 - u) * (1.0 - v);
    N[1] = u * (1.0 - v);
    N[2] = u * v;
    N[3] = (1.0 - u) * v;
    return N;
}

inline std::array<double, ElementNodeCount::QUAD8> compute_shape_quad8(const double xi, const double eta){
    // xi and eta must be within [-1,1]
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
    return N;
}

inline std::array<double, ElementNodeCount::QUAD9> compute_shape_quad9(const double xi, const double eta){
    // xi and eta must be within [-1,1]
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
    return N;
}

// Quadratic triangle (TRI6) shape functions (g, h)
// r = 1 - g - h
static inline Eigen::VectorXd get_face_N(double g, double h) {
    /* 
    Function to compute the shape functions for a TRI6 triangle at a given point in barycentric coordinates.

    Parameters
    ----------
    g : double
        First barycentric coordinate
    h : double
        Second barycentric coordinate

    Returns
    -------
    Eigen::VectorXd
        Vector of shape function values, each entry corresponds to one of the 6 nodes of a TRI6 triangle
    */
    double r = 1.0 - g - h;
    Eigen::VectorXd N(6);
    N << r*(2*r-1), g*(2*g-1), h*(2*h-1), 4*g*r, 4*g*h, 4*h*r;
    return N;
}
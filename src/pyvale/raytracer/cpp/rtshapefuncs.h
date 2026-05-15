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


static inline Eigen::Matrix<double, 3, 2> get_face_Jacobian_quad4(const double u, const double v, 
    const std::vector<double> node_coords) {
    // May have to multiply these by 1/4
    std::array<double, ElementNodeCount::QUAD4> dNdu;

    dNdu[0] = -1.0 + v;
    dNdu[1] = 1.0 - v;
    dNdu[2] = v;
    dNdu[3] = -v;

    std::array<double, ElementNodeCount::QUAD4> dNdv;
    dNdv[0] = -1.0 + u;
    dNdv[1] = -u;
    dNdv[2] = u;
    dNdv[3] = 1.0 - u;

    Eigen::Matrix<double, 3, 2> J = Eigen::Matrix<double, 3, 2>::Zero();
    for (int i = 0; i < ElementNodeCount::QUAD4; ++i) {
        EiVector3d node_point;
        node_point << node_coords[i * NODE_COORDINATES], node_coords[i * NODE_COORDINATES + 1], node_coords[i * NODE_COORDINATES + 2];
        J.col(0) += node_point * dNdu[i];
        J.col(1) += node_point * dNdv[i];
    }
    return J;
}

static inline Eigen::Matrix<double, 3, 2> get_face_Jacobian_quad8(double xi, double eta, 
    const std::vector<double> node_coords) {
    
    std::array<double, ElementNodeCount::QUAD8> dNdxi;
    std::array<double, ElementNodeCount::QUAD8> dNdeta;

    // Pre-compute squares and intermediate terms
    const double xi2 = xi * xi;
    const double eta2 = eta * eta;

    // Derivatives with respect to xi
    dNdxi[0] = 0.25 * (eta - 1.0) * (-eta - 2.0 * xi);
    dNdxi[1] = 0.25 * (eta - 1.0) * (eta - 2.0 * xi);
    dNdxi[2] = 0.25 * (eta + 1.0) * (eta + 2.0 * xi);
    dNdxi[3] = 0.25 * (eta + 1.0) * (-eta + 2.0 * xi);
    dNdxi[4] = xi * (eta - 1.0);
    dNdxi[5] = 0.5 * (1.0 - eta2);
    dNdxi[6] = -xi * (eta + 1.0);
    dNdxi[7] = -0.5 * (1.0 - eta2);

    // Derivatives with respect to eta
    dNdeta[0] = 0.25 * (xi - 1.0) * (-2.0 * eta - xi);
    dNdeta[1] = 0.25 * (xi + 1.0) * (2.0 * eta - xi);
    dNdeta[2] = 0.25 * (xi + 1.0) * (2.0 * eta + xi);
    dNdeta[3] = 0.25 * (xi - 1.0) * (-2.0 * eta + xi);
    dNdeta[4] = -0.5 * (1.0 - xi2);
    dNdeta[5] = -eta * (xi + 1.0);
    dNdeta[6] = 0.5 * (1.0 - xi2);
    dNdeta[7] = eta * (xi - 1.0);

    Eigen::Matrix<double, 3, 2> J = Eigen::Matrix<double, 3, 2>::Zero();
    for (int i = 0; i < ElementNodeCount::QUAD8; ++i) {
        EiVector3d node_point;
        node_point << node_coords[i * NODE_COORDINATES], node_coords[i * NODE_COORDINATES + 1], node_coords[i * NODE_COORDINATES + 2];
        J.col(0) += node_point * dNdxi[i];
        J.col(1) += node_point * dNdeta[i];
    }
    return J;
}

static inline Eigen::Matrix<double, 3, 2> get_face_Jacobian_quad9(double xi, double eta, 
    const std::vector<double> node_coords) {
    
    std::array<double, ElementNodeCount::QUAD9> dNdxi;
    std::array<double, ElementNodeCount::QUAD9> dNdeta;

    // Pre-compute squares and intermediate terms
    const double xi2 = xi * xi;
    const double eta2 = eta * eta;
    const double two_xi = 2.0 * xi;
    const double two_eta = 2.0 * eta;

    // Derivatives with respect to xi
    dNdxi[0] = 0.25 * eta * (eta - 1.0) * (two_xi - 1.0);
    dNdxi[1] = 0.25 * eta * (eta - 1.0) * (two_xi + 1.0);
    dNdxi[2] = 0.25 * eta * (eta + 1.0) * (two_xi + 1.0);
    dNdxi[3] = 0.25 * eta * (eta + 1.0) * (two_xi - 1.0);
    dNdxi[4] = -xi * eta * (eta - 1.0);
    dNdxi[5] = 0.5 * (1.0 - eta2) * (two_xi + 1.0);
    dNdxi[6] = -xi * eta * (eta + 1.0);
    dNdxi[7] = 0.5 * (1.0 - eta2) * (two_xi - 1.0);
    dNdxi[8] = -two_xi * (1.0 - eta2);

    // Derivatives with respect to eta
    dNdeta[0] = 0.25 * xi * (xi - 1.0) * (two_eta - 1.0);
    dNdeta[1] = 0.25 * xi * (xi + 1.0) * (two_eta - 1.0);
    dNdeta[2] = 0.25 * xi * (xi + 1.0) * (two_eta + 1.0);
    dNdeta[3] = 0.25 * xi * (xi - 1.0) * (two_eta + 1.0);
    dNdeta[4] = 0.5 * (1.0 - xi2) * (two_eta - 1.0);
    dNdeta[5] = -eta * xi * (xi + 1.0);
    dNdeta[6] = 0.5 * (1.0 - xi2) * (two_eta + 1.0);
    dNdeta[7] = -eta * xi * (xi - 1.0);
    dNdeta[8] = -two_eta * (1.0 - xi2);

    Eigen::Matrix<double, 3, 2> J = Eigen::Matrix<double, 3, 2>::Zero();
    for (int i = 0; i < ElementNodeCount::QUAD9; ++i) {
         EiVector3d node_point;
        node_point << node_coords[i * NODE_COORDINATES], node_coords[i * NODE_COORDINATES + 1], node_coords[i * NODE_COORDINATES + 2];
        J.col(0) += node_point * dNdxi[i];
        J.col(1) += node_point * dNdeta[i];
    }
    return J;
}
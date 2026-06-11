// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef RTSHAPEFUNCS_H
#define RTSHAPEFUNCS_H

// STD header files
#include <iostream>
#include <limits>
#include <array>

// ray tracer header files
#include "rtelemconstants.h"

namespace shapefuncs{

    // ================================================================================
    // Shape functions
    // ================================================================================

    /**
     * @brief Evaluates the shape functions for QUAD4 at a given point given its interpolation coordinates from the ray intersection.
     * 
     * Returns the shape weights for the bilinear interpolation based on the passed u,v values.
     * 
     * @param[in] u (const double) First interpolation coordinate. Note that u is in the [0, 1] range, NOT [-1,1] which is customary for FEM.
     * @param[in] v (const double) Second interolation coordinate; also in the [0,1] range.
     * 
     * @return (std::array<double, ElementNodeCount::QUAD4) Array with the shape weight coefficients for all 4 nodes of QUAD4.
     */
    inline std::array<double, ElementNodeCount::QUAD4> compute_shape_quad4(const double u, const double v){
        // Weights for bilinear interpolation
        std::array<double, ElementNodeCount::QUAD4> N;
        // If u, v in [0,1], which they should be
        N[0] = (1.0 - u) * (1.0 - v);
        N[1] = u * (1.0 - v);
        N[2] = u * v;
        N[3] = (1.0 - u) * v;
        // If u, v in [-1,1]
        //N[0] = 0.25 * (1.0 - u) * (1.0 - v);
        //N[1] = 0.25 * (1.0 + u) * (1.0 - v);
        //N[2] = 0.25 * (1.0 + u) * (1.0 + v);
        //N[3] = 0.25 * (1.0 - u) * (1.0 + v);
        return N;
    }

    /**
     * @brief Evaluates the shape functions for QUAD8 at a given point given its interpolation coordinates from the ray intersection.
     * 
     * Returns the shape weights for the bilinear interpolation based on the passed xi, eta values.
     * 
     * @param[in] xi (const double) First interpolation coordinate. Note xi is in the [-1, 1] range.
     * @param[in] eta (const double) Second interolation coordinate; also in the [-1,1] range.
     * 
     * @return (std::array<double, ElementNodeCount::QUAD8) Array with the shape weight coefficients for all 8 nodes of QUAD8.
     */
    inline std::array<double, ElementNodeCount::QUAD8> compute_shape_quad8(const double xi, const double eta){
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

    /**
     * @brief Evaluates the shape functions for QUAD9 at a given point given its interpolation coordinates from the ray intersection.
     * 
     * Returns the shape weights for the bilinear interpolation based on the passed xi, eta values.
     * 
     * @param[in] xi (const double) First interpolation coordinate. Note xi is in the [-1, 1] range.
     * @param[in] eta (const double) Second interolation coordinate; also in the [-1,1] range.
     * 
     * @return (std::array<double, ElementNodeCount::QUAD8) Array with the shape weight coefficients for all 9 nodes of QUAD9.
     */
    inline std::array<double, ElementNodeCount::QUAD9> compute_shape_quad9(const double xi, const double eta){
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

    /**
     * @brief Computes the shape functions for a TRI6 triangle at a given point in barycentric coordinates.
     * 
     * Returns the shape weights for the bilinear interpolation based on the passed g, h values.
     * 
     * @param[in] g (const double) First barycentric coordinate. Note g is in the [-1, 1] range.
     * @param[in] h (const double) Second barycentric coordinate; also h the [-1,1] range.
     * 
     * @return (Eigen::VectorXd) Vector of shape function values, each entry corresponds to one of the 6 nodes of a TRI6 triangle
     */
    static inline Eigen::VectorXd compute_shape_tri6(double g, double h) {
        double r = 1.0 - g - h;
        Eigen::VectorXd N(6);
        N << r*(2*r-1), g*(2*g-1), h*(2*h-1), 4*g*r, 4*g*h, 4*h*r;
        return N;
    }

    // ================================================================================
    // Jacobians (for intersections and shading normals)
    // ================================================================================

     /**
     * @brief Evaluates the Jacobian matrix for QUAD4.
     * 
     * Calculates the derivatives of the shape functions based on the interpolation coordinates from the ray-element intersection,
     * then multiplies the derivative by the corresponding nodal coordinates and sums those to find [dN/du, dN/dv].
     * 
     * @param[in] u (const double) First interpolation coordinate. Note that u is in the [0, 1] range, NOT [-1,1] which is customary for FEM.
     * @param[in] v (const double) Second interolation coordinate; also in the [0,1] range.
     * @param[in] node_coords(std::array<EiVector3d, ElementNodeCount::QUAD4>) Array of 3D Eigen vectors (x,y,z) corresponding to coordinates of each node comprising a single quad.
     * 
     * @return (Eigen::Matrix<double, 3, 2>) Jacobian matrix storing derivatives of (x,y,z) coordinates corresponding to dN/du, dN/dv.
     */
    static inline Eigen::Matrix<double, 3, 2> get_face_Jacobian_quad4(const double u, const double v, 
        const std::array<EiVector3d, ElementNodeCount::QUAD4>& node_coords) {
        std::array<double, ElementNodeCount::QUAD4> dNdu;
        // If u, v in [0,1], which they should be
        dNdu[0] = (-1.0 + v);
        dNdu[1] = (1.0 - v);
        dNdu[2] = v * 0.25;
        dNdu[3] = -v * 0.25;

        std::array<double, ElementNodeCount::QUAD4> dNdv;
        dNdv[0] = -1.0 + u;
        dNdv[1] = -u;
        dNdv[2] = u;
        dNdv[3] = 1.0 - u;

        Eigen::Matrix<double, 3, 2> J = Eigen::Matrix<double, 3, 2>::Zero();
        for (int i = 0; i < ElementNodeCount::QUAD4; ++i) {
            J.col(0) += node_coords[i] * dNdu[i];
            J.col(1) += node_coords[i] * dNdv[i];
        }
        return J;
    }

    /**
     * @brief Evaluates the Jacobian matrix for QUAD8.
     * 
     * Calculates the derivatives of the shape functions based on the interpolation coordinates from the ray-element intersection,
     * then multiplies the derivative by the corresponding nodal coordinates and sums those to find [dN/du, dN/dv].
     * 
     * @param[in] xi (const double) First interpolation coordinate. Note xi is in the [-1, 1] range.
     * @param[in] eta (const double) Second interolation coordinate; also in the [-1,1] range.
     * @param[in] node_coords(std::array<EiVector3d, ElementNodeCount::QUAD8>) Array of 3D Eigen vectors (x,y,z) corresponding to coordinates of each node comprising a single quad.
     * 
     * @return (Eigen::Matrix<double, 3, 2>) Jacobian matrix storing derivatives of (x,y,z) coordinates corresponding to dN/du, dN/dv.
     */
    static inline Eigen::Matrix<double, 3, 2> get_face_Jacobian_quad8(double xi, double eta, 
        const std::array<EiVector3d, ElementNodeCount::QUAD8>& node_coords) {
        
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
            J.col(0) += node_coords[i] * dNdxi[i];
            J.col(1) += node_coords[i] * dNdeta[i];
        }
        return J;
    }

    /**
     * @brief Evaluates the Jacobian matrix for QUAD9.
     * 
     * Calculates the derivatives of the shape functions based on the interpolation coordinates from the ray-element intersection,
     * then multiplies the derivative by the corresponding nodal coordinates and sums those to find [dN/du, dN/dv].
     * 
     * @param[in] xi (const double) First interpolation coordinate. Note xi is in the [-1, 1] range.
     * @param[in] eta (const double) Second interolation coordinate; also in the [-1,1] range.
     * @param[in] node_coords(std::array<EiVector3d, ElementNodeCount::QUAD9>) Array of 3D Eigen vectors (x,y,z) corresponding to coordinates of each node comprising a single quad.
     * 
     * @return (Eigen::Matrix<double, 3, 2>) Jacobian matrix storing derivatives of (x,y,z) coordinates corresponding to dN/du, dN/dv.
     */
    static inline Eigen::Matrix<double, 3, 2> get_face_Jacobian_quad9(double xi, double eta, 
        const std::array<EiVector3d, ElementNodeCount::QUAD9>& node_coords) {
        
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
            J.col(0) += node_coords[i] * dNdxi[i];
            J.col(1) += node_coords[i] * dNdeta[i];
        }
        return J;
    }

    /**
     * @brief Computes the Jacobian matrix of a TRI6 triangle at a given point in barycentric coordinates.
     * 
     * Calculates the derivatives of the shape functions based on the interpolation coordinates from the ray-element intersection,
     * then multiplies the derivative by the corresponding nodal coordinates and sums those to find [dN/du, dN/dv].
     * 
     * @param[in] g (const double) First barycentric coordinate. Note g is in the [-1, 1] range.
     * @param[in] h (const double) Second barycentric coordinate; also h the [-1,1] range.
     * @param[in] node_coords(std::array<EiVector3d, ElementNodeCount::TRI6>) Array of 3D Eigen vectors (x,y,z) corresponding to coordinates of each node comprising a single triangle.
     * 
     * @return (Eigen::Matrix<double, 3, 2>) Jacobian matrix storing derivatives of (x,y,z) coordinates corresponding to dN/du, dN/dv.
     */
    static Eigen::Matrix<double, 3, 2> get_face_Jacobian_tri6(double g, double h, 
        const std::array<EiVector3d, ElementNodeCount::TRI6>& nodes) {
        double r = 1.0 - g - h;
        // Derivatives of N wrt g and h
        double dNdu[6] = { -(4*r-1), 4*g-1, 0, 4*(r-g), 4*h, -4*h };
        double dNdv[6] = { -(4*r-1), 0, 4*h-1, -4*g, 4*g, 4*(r-h) };

        Eigen::Matrix<double, 3, 2> J = Eigen::Matrix<double, 3, 2>::Zero();
        for (int i = 0; i < ElementNodeCount::TRI6; ++i) {
            J.col(0) += nodes[i] * dNdu[i];
            J.col(1) += nodes[i] * dNdv[i];
        }
        return J;
    }
}

#endif // RTSHAPEFUNCS_H
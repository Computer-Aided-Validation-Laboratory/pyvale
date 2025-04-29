#include <iostream>
#include <Eigen/Dense>
// #include <functional>

// Shape functions for a 4-node quadrilateral
Eigen::Vector4d shapeFunctions(double xi, double eta) {
    Eigen::Vector4d N;
    N(0) = 0.25 * (1 - xi) * (1 - eta);
    N(1) = 0.25 * (1 + xi) * (1 - eta);
    N(2) = 0.25 * (1 + xi) * (1 + eta);
    N(3) = 0.25 * (1 - xi) * (1 + eta);
    return N;
}

// Compute deformed surface point
Eigen::Vector3d deformedSurface(double xi, double eta,
                                const Eigen::Matrix<double, 4, 3>& nodes,
                                const Eigen::Matrix<double, 4, 3>& displacements) {
    Eigen::Vector4d N = shapeFunctions(xi, eta);
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    for (int i = 0; i < 4; ++i) {
        point += N(i) * (nodes.row(i) + displacements.row(i));
    }
    return point;
}

// Newton-Raphson to solve the nonlinear system
bool findIntersection(const Eigen::Matrix<double, 4, 3>& nodes,
                      const Eigen::Matrix<double, 4, 3>& displacements,
                      const Eigen::Vector3d& r0,
                      const Eigen::Vector3d& d,
                      Eigen::Vector3d& intersection) {
    Eigen::Vector3d vars(0.0, 0.0, 0.5); // Initial guess: xi, eta, t

    const double tol = 1e-8;
    const int maxIter = 50;

    for (int iter = 0; iter < maxIter; ++iter) {
        double xi = vars(0), eta = vars(1), t = vars(2);

        // Compute residual
        Eigen::Vector3d surface = deformedSurface(xi, eta, nodes, displacements);
        Eigen::Vector3d line = r0 + t * d;
        Eigen::Vector3d F = surface - line;

        if (F.norm() < tol) {
            intersection = surface;
            return true;
        }

        // Numerical Jacobian
        const double h = 1e-6;
        Eigen::Matrix3d J;
        for (int i = 0; i < 3; ++i) {
            Eigen::Vector3d delta = Eigen::Vector3d::Zero();
            delta(i) = h;
            Eigen::Vector3d F1 = deformedSurface(xi + delta(0), eta + delta(1), nodes, displacements)
                                 - (r0 + (t + delta(2)) * d);
            J.col(i) = (F1 - F) / h;
        }

        // Solve for update
        Eigen::Vector3d deltaVars = J.fullPivLu().solve(-F);
        vars += deltaVars;

        if (deltaVars.norm() < tol) {
            intersection = deformedSurface(vars(0), vars(1), nodes, displacements);
            return true;
        }
    }
    return false;
}

int main() {
    // Define quad nodes
    Eigen::Matrix<double, 4, 3> nodes;
    nodes << -2, -2, 0,
             2, -2, 0,
             2, 2, 0,
             -2, 2, 0;

    // Displacements
    Eigen::Matrix<double, 4, 3> displacements;
    displacements << 0, 0, -2,
                     0, 0, -2,
                     0, 0, 2,
                     0, 0, 2;

    // Ray definition
    Eigen::Vector3d r0(0.5, 0.5, 1.0);
    Eigen::Vector3d d(0.0, 0.0, -1.0);

    Eigen::Vector3d intersection;
    if (findIntersection(nodes, displacements, r0, d, intersection)) {
        std::cout << "Intersection found at: " << intersection.transpose() << std::endl;
    } else {
        std::cout << "No intersection found." << std::endl;
    }

    return 0;
}

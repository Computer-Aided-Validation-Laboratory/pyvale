#ifndef SHAPE_H
#define SHAPE_H

#include "hittable.h"
// #include "../materials/material.h"
#include <Eigen/Dense>



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

class ShapeQuad: public Hittable {
    public:
        ShapeQuad(Eigen::Matrix<double, Eigen::Dynamic, 3> nodes, Eigen::Matrix<double, Eigen::Dynamic, 3> displacements, shared_ptr<Material> mat ) :
            nodes(nodes), displacements(displacements), mp(mat) {};

        virtual Eigen::VectorXd shapeFunctions(double xi, double eta) const = 0;

        // Compute deformed surface point
        Eigen::Vector3d deformedSurface(double xi, double eta) const {
            Eigen::VectorXd N = shapeFunctions(xi, eta);
            Eigen::Vector3d point = Eigen::Vector3d::Zero();
            for (int i = 0; i < N.size(); ++i) {
                point += N(i) * (nodes.row(i) + displacements.row(i));
            }
            return point;
        }

        // Newton-Raphson to solve the nonlinear system
        bool findIntersection(const Eigen::Vector3d& r0,
                            const Eigen::Vector3d& d,
                            Eigen::Vector3d& intersection) const {
            Eigen::Vector3d vars(0.0, 0.0, 0); // Initial guess: xi, eta, t

            const double tol = 1e-8;
            const int maxIter = 50;

            for (int iter = 0; iter < maxIter; ++iter) {
                double xi = vars(0), eta = vars(1), t = vars(2);

                // Compute residual
                Eigen::Vector3d surface = deformedSurface(xi, eta);
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
                    Eigen::Vector3d F1 = deformedSurface(xi + delta(0), eta + delta(1))
                                        - (r0 + (t + delta(2)) * d);
                    J.col(i) = (F1 - F) / h;
                }

                // Solve for update
                Eigen::Vector3d deltaVars = J.fullPivLu().solve(-F);
                vars += deltaVars;

                if (deltaVars.norm() < tol) {
                    intersection = deformedSurface(vars(0), vars(1));
                    return true;
                }
            }
            return false;
        }

        bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const override
        {
            Eigen::Vector3d r0;
            r0 << r.get_origin().x(), r.get_origin().y(), r.get_origin().z();
            Eigen::Vector3d d;
            d << r.get_direction().x(), r.get_direction().y(), r.get_direction().z();
            Eigen::Vector3d intersection;
            if (findIntersection(r0, d, intersection)) {
                // hit.distance = ;
                hit.position = vec3(intersection[0], intersection[1], intersection[2]);
                hit.material_ptr = mp;
                // hit.set_face_normal(r, outward_normal)
            }
            return true;
        }

        // virtual bool bounding_box(double dis_min, double dis_max, aabb& output_box) const{
        //     // The bounding box must have non-zero width in each dimension, so pad the Z
        //     // dimension a small amount.
        //     output_box = aabb(point3(x0,y0, k-0.0001), point3(x1, y1, k+0.0001));
        //     return true;
        // }

    protected:
        // Define quad nodes
        Eigen::Matrix<double, Eigen::Dynamic, 3> nodes;

        // Displacements
        Eigen::Matrix<double, Eigen::Dynamic, 3> displacements;

        shared_ptr<Material> mp;
};

class ShapeQuadLin : public ShapeQuad {
    public:
        ShapeQuadLin(Eigen::Matrix<double, 4, 3> nodes, Eigen::Matrix<double, 4, 3> displacements, shared_ptr<Material> mat ) :
            ShapeQuad( nodes, displacements, mat) {};

        Eigen::VectorXd shapeFunctions(double xi, double eta) const override
        {
            Eigen::Vector4d N;
            N(0) = 0.25 * (1 - xi) * (1 - eta);
            N(1) = 0.25 * (1 + xi) * (1 - eta);
            N(2) = 0.25 * (1 + xi) * (1 + eta);
            N(3) = 0.25 * (1 - xi) * (1 + eta);
            return N;        
        }

        // Simple get the nodes that are the smallest and largest in the 3 directions
        bool bounding_box(double dis_min, double dis_max, aabb& output_box) const override
        {
            Eigen::Matrix<double, 4, 3> locations = nodes + displacements;

            Eigen::Vector3d min_vals = locations.colwise().minCoeff(); // [min_x, min_y, min_z]
            Eigen::Vector3d max_vals = locations.colwise().maxCoeff(); // [max_x, max_y, max_z]

            point3 mins = vec3(min_vals[0], min_vals[1], min_vals[2]);
            point3 maxs = vec3(max_vals[0], max_vals[1], max_vals[2]);

            // point3 mins = vec3(9999, 9999, 9999);
            // point3 maxs = vec3(-999, -999, -999);
            
            // for (int i = 0; i < nodes.size(); ++i) {
            //     Eigen::Vector3<double> loc_e = nodes.row(i) + displacements.row(i);
                
            //     if (loc.x() < mins.x()) {
            //         mins = vec3(loc.x(), mins.y(), mins.z());
            //     }
            //     if (loc.x() > maxs.x()) {
            //         maxs = vec3(loc.x(), maxs.y(), maxs.z());
            //     }
            //     if (loc.y() < mins.y()) {
            //         mins = vec3(mins.x(), loc.y(), mins.z());
            //     }
            //     if (loc.y() > maxs.y()) {
            //         maxs = vec3(maxs.x(), loc.y(), maxs.z());
            //     }
            //     if (loc.z() < mins.z()) {
            //         mins = vec3(mins.x(), mins.y(), loc.z());
            //     }
            //     if (loc.z() > maxs.z()) {
            //         maxs = vec3(maxs.x(), maxs.y(), loc.z());
            //     }
            // }

            output_box = aabb(mins, maxs);
            return true;
        }


};


#endif
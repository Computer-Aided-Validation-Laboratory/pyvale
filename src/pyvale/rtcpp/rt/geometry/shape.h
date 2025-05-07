#ifndef SHAPE_H
#define SHAPE_H

#include "hittable.h"
// #include "../materials/material.h"
#include <Eigen/Dense>
#include <cassert>
#include <cmath>


// Base class of shapes with shape function displacements
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
                            Eigen::Vector3d& intersection,
                            Eigen::Vector3d& variables // xi, eta, t
                        ) const {
            Eigen::Vector3d vars(0.0, 0.0, 0.0); // Initial guess: xi, eta, t

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
                    variables = vars;
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
                    variables = vars;
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
            Eigen::Vector3d vars;
            if (findIntersection(r0, d, intersection, vars)) {
                bool outside_bounds = false;
                if (vars(0) < -1.0 || vars[0] > 1.0) {outside_bounds = true;}
                if (vars[1] < -1.0 || vars[1] > 1.0) {outside_bounds = true;}
                if (vars[2] < dis_min || vars[2] > dis_max) {outside_bounds = true;}
                if (outside_bounds){
                    return false;
                }
                
                // vars are => (xi, eta, t)
                double u = (vars[0] + 1) / 2.0;
                double v = (vars[1] + 1) / 2.0;
                hit.distance = vars[2];
                hit.position = vec3(intersection[0], intersection[1], intersection[2]);
                hit.material_ptr = mp;
                hit.normal = vec3(0.0, 1.0, 0.0);
                // hit.set_face_normal(r, surface_normal(vars[0], vars[1]));
                hit.u = u;
                hit.v = v;
                return true;
            }
            return false;
        }

        virtual Eigen::VectorXd dN_dxi(double xi, double eta) const = 0;
        virtual Eigen::VectorXd dN_deta(double xi, double eta) const = 0;
        
        // The surface normal at a given local coordinate
        Eigen::Vector3d surface_normal(double xi, double eta) const
        {
            Eigen::Matrix<double, Eigen::Dynamic, 3> coords = nodes + displacements;
            Eigen::Vector3d dxdxi = dN_dxi(xi, eta).transpose() * coords;
            Eigen::Vector3d dxdeta = dN_deta(xi,eta).transpose() * coords;

            Eigen::Vector3d cross = dxdxi.cross(dxdeta);
            Eigen::Vector3d cross_normalized = cross.normalized();

            return cross_normalized; 


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

// Linear quadrangles (4 points)
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

            output_box = aabb(mins, maxs);
            return true;
        }

        Eigen::VectorXd dN_dxi(double xi, double eta) const override
        {
            Eigen::Vector4d N;
            N(0) = -0.25 * (1 - eta);
            N(1) =  0.25 * (1 - eta);
            N(2) =  0.25 * (1 + eta);
            N(3) = -0.25 * (1 + eta);
            return N;
        }

        Eigen::VectorXd dN_deta(double xi, double eta) const override
        {
            Eigen::Vector4d N;
            N(0) = -0.25 * (1 - xi);
            N(1) = -0.25 * (1 + xi);
            N(2) =  0.25 * (1 + xi);
            N(3) =  0.25 * (1 - xi);
            return N;
        }
};

// Quadratic quadrangles (4 corner points and 4 mid-side nodes, see VTK 8-node quad for node ordering details)
class ShapeQuadQuad : public ShapeQuad {
    public:
    ShapeQuadQuad(Eigen::Matrix<double, 8, 3> nodes, Eigen::Matrix<double, 8, 3> displacements, shared_ptr<Material> mat ) :
            ShapeQuad( nodes, displacements, mat) {};

        Eigen::VectorXd shapeFunctions(double xi, double eta) const override
        {
            Eigen::VectorXd N(8);
            N(0) = 0.25 * (1 - xi) * (1 - eta) * (-1 - xi - eta);
            N(1) = 0.25 * (1 + xi) * (1 - eta) * (-1 + xi - eta);
            N(2) = 0.25 * (1 + xi) * (1 + eta) * (-1 + xi + eta);
            N(3) = 0.25 * (1 - xi) * (1 + eta) * (-1 - xi + eta);
            N(4) = 0.5  * (1 - pow(xi, 2))* (1 - eta);
            N(5) = 0.5  * (1 + xi)             * (1 - pow(eta, 2));
            N(6) = 0.5  * (1 - pow(xi, 2))* (1 + eta);
            N(7) = 0.5  * (1 - xi)             * (1 - pow(eta, 2));
            assert(N.size() == 8);
            return N;        
        }

        // Simple get the nodes that are the smallest and largest in the 3 directions
        bool bounding_box(double dis_min, double dis_max, aabb& output_box) const override
        {
            Eigen::Matrix<double, 8, 3> locations = nodes + displacements;

            Eigen::Vector3d min_vals = locations.colwise().minCoeff(); // [min_x, min_y, min_z]
            Eigen::Vector3d max_vals = locations.colwise().maxCoeff(); // [max_x, max_y, max_z]

            point3 mins = vec3(min_vals[0], min_vals[1], min_vals[2]);
            point3 maxs = vec3(max_vals[0], max_vals[1], max_vals[2]);

            output_box = aabb(mins, maxs);
            return true;
        }

        Eigen::VectorXd dN_dxi(double xi, double eta) const override
        {
            Eigen::VectorXd N;
            N(0) =  0.25 * (1 - eta) * (-1 - 2 * xi - eta);
            N(1) =  0.25 * (1 - eta) * (-1 + 2 * xi - eta);
            N(2) =  0.25 * (1 + eta) * (-1 + 2 * xi + eta);
            N(3) =  0.25 * (1 + eta) * (-1 - 2 * xi + eta);
            N(4) =  -xi  * (1 - eta);
            N(5) =   0.5 * (1 - pow(eta, 2));
            N(6) =   -xi * (1 + eta);
            N(7) =  -0.5 * (1 - pow(eta, 2));
            return N;
        }

        Eigen::VectorXd dN_deta(double xi, double eta) const override
        {
            Eigen::VectorXd N;
            N(0) = 0.25 * (1 - xi) * (-1 - xi - 2 * eta);
            N(1) = 0.25 * (1 + xi) * (-1 + xi - 2 * eta);
            N(2) = 0.25 * (1 + xi) * (-1 + xi + 2 * eta);
            N(3) = 0.25 * (1 - xi) * (-1 - xi + 2 * eta);
            N(4) = -0.5 * (1 - pow(xi, 2));
            N(5) = -eta * (1 + xi);
            N(6) =  0.5 * (1 - pow(xi, 2));
            N(7) = -eta * (1 - xi);
            return N;
        }
};

#endif
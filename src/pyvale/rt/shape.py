import numpy as np
from scipy.optimize import root, minimize

from pyvale.rt.hittable import Hittable, HitRecord
from pyvale.rt.material import Material
from pyvale.rt.ray import Ray
from pyvale.rt.interval import Interval
from pyvale.rt.aabb import AABB

# Shape function elements.

# Define shape functions for a 4-node quadrilateral element
# def shape_functions(xi, eta):
#     N = np.array([
#         0.25 * (1 - xi) * (1 - eta),
#         0.25 * (1 + xi) * (1 - eta),
#         0.25 * (1 + xi) * (1 + eta),
#         0.25 * (1 - xi) * (1 + eta)
#     ])
#     return N

# # Compute the position on the deformed surface given (xi, eta)
# def deformed_surface(xi, eta, nodes, displacements):
#     N = shape_functions(xi, eta)
#     return np.dot(N, nodes + displacements)

# # Residual function: difference between point on surface and point on line
# def residual(vars, nodes, displacements, r0, d):
#     xi, eta, t = vars
#     surface_point = deformed_surface(xi, eta, nodes, displacements)
#     line_point = r0 + t * d
#     return surface_point - line_point

if __name__ == "__main()__":
    # Define node coordinates (quad)
    nodes = np.array([
        [-1, -1, 0],   # Node 1
        [1, -1, 0],   # Node 2
        [1, 1, 0],   # Node 3
        [-1, 1, 0]    # Node 4
    ])

    # Define displacements at each node
    displacements = np.array([
        [0, 0, -1],       # Node 1
        [0, 0, -1],     # Node 2
        [0, 0, 2],     # Node 3
        [0, 0, 1]      # Node 4
    ])

    # Define a line (ray) with origin and direction
    r0 = np.array([0.5, 0.5, 1.0])  # Starting point of the ray
    d = np.array([0.0, 0.0, -1.0])  # Direction

    # Initial guess
    initial_guess = [0, 0, 0]

    # Solve the nonlinear system
    solution = root(residual, initial_guess, args=(nodes, displacements, r0, d))

    # Extract solution
    intersection_found = solution.success
    intersection_point = None
    if intersection_found:
        xi_sol, eta_sol, t_sol = solution.x
        intersection_point = deformed_surface(xi_sol, eta_sol, nodes, displacements)

    print(intersection_found, solution.x, intersection_point)

class ShapeFunctionShape(Hittable):
    # Compute the position on the deformed surface given (xi, eta)
    def deformed_surface(self, xi, eta):
        N = self.shape_functions(xi, eta)
        return np.dot(N, self.nodes + self.displacements)

    # Residual function: difference between point on surface and point on line
    def residual(self, vars, r0, d):
        xi, eta, t = vars
        surface_point = self.deformed_surface(xi, eta)
        line_point = r0 + t * d
        return surface_point - line_point

# A triangle 
class ShapeLinQuad(ShapeFunctionShape):
    def __init__(self, nodes: np.ndarray, displacements: np.ndarray, mat: Material):
        assert len(nodes) == 4
        assert len(displacements) == 4
        self.nodes = nodes
        self.displacements = displacements
        self.mat = mat
        self.set_bounding_box()
    
    def set_bounding_box(self):
        # Linear shape, so the extreme points will be nodes themselves
        points = self.nodes + self.displacements
        min_values = points.min(axis=0)
        max_values = points.max(axis=0)
        x_inter = Interval.from_floats(min_values[0], max_values[0])
        y_inter = Interval.from_floats(min_values[1], max_values[1])
        z_inter = Interval.from_floats(min_values[2], max_values[2])

        self.bbox = AABB(x_inter, y_inter, z_inter)

    # Define shape functions for a 4-node quadrilateral element
    def shape_functions(self, xi, eta):
        N = np.array([
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta)
        ])
        return N

    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        initial_guess = [0, 0, 5] #in xi, eta, t

        r0 = r.origin
        d = r.direction

        # Solve the nonlinear system
        solution = root(self.residual, initial_guess, args=(r0, d))

        # Extract solution
        intersection_found = solution.success
        intersection_point = None
        if intersection_found:
            xi_sol, eta_sol, t_sol = solution.x
            intersection_point = self.deformed_surface(xi_sol, eta_sol)

            # Invalid solution
            if t_sol < 0 or abs(xi_sol) > 1 or abs(eta_sol) > 1:
                return None 

            # Fix the normal to reality.
            outward_normal = self.surface_normal(xi_sol, eta_sol)
            u = (xi_sol + 1) / 2
            v = (eta_sol + 1) / 2
            rec: HitRecord = HitRecord(p = intersection_point, t= solution.x[2], mat = self.mat, r=r, outward_normal=outward_normal, u=u, v = v)
            return rec
        else:
            return None
    
    # Differentiate shape functions w.r.t. xi
    def dN_dxi(self, xi, eta):
        return np.array([
            -0.25 * (1 - eta),
             0.25 * (1 - eta),
             0.25 * (1 + eta),
            -0.25 * (1 + eta)
        ])

    # Differentiate shape functions w.r.t. eta
    def dN_deta(self, xi, eta):
        return np.array([
            -0.25 * (1 - xi),
            -0.25 * (1 + xi),
             0.25 * (1 + xi),
             0.25 * (1 - xi)
        ])

    def surface_normal(self, xi, eta):
        coords = self.nodes + self.displacements

        dxdxi = np.sum(self.dN_dxi(xi, eta)[:, np.newaxis] * coords, axis=0)
        dxdeta = np.sum(self.dN_deta(xi, eta)[:, np.newaxis] * coords, axis=0)

        normal = np.cross(dxdxi, dxdeta)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 0 else normal


class ShapeQuadQuad(ShapeFunctionShape):
    def __init__(self, nodes: np.ndarray, displacements: np.ndarray, mat: Material):
        assert len(nodes) == 8
        assert len(displacements) == 8
        self.nodes = nodes
        self.displacements = displacements
        self.mat = mat
        self.set_bounding_box()
    
    def set_bounding_box(self):
        # Quadratic, so extreme points might not be nodes. Minimize to find.

        # Choose a non-symetric starting point to guess
        initial_guess = [0.1, 0.1]
        bounds = [(-1, 1), (-1, 1)]

        # maximizations are a minimization of the negative function
        min_x = minimize(lambda x: self.deformed_surface(x[0], x[1])[0], initial_guess, bounds=bounds)
        max_x = minimize(lambda x: -self.deformed_surface(x[0], x[1])[0], initial_guess, bounds=bounds)
        min_y = minimize(lambda x: self.deformed_surface(x[0], x[1])[1], initial_guess, bounds=bounds)
        max_y = minimize(lambda x: -self.deformed_surface(x[0], x[1])[1], initial_guess, bounds=bounds)
        min_z = minimize(lambda x: self.deformed_surface(x[0], x[1])[2], initial_guess, bounds=bounds)
        max_z = minimize(lambda x: -self.deformed_surface(x[0], x[1])[2], initial_guess, bounds=bounds)

        # negate the max again to get the true max value.
        x_inter = Interval.from_floats(min_x.fun, -max_x.fun)
        y_inter = Interval.from_floats(min_y.fun, -max_y.fun)
        z_inter = Interval.from_floats(min_z.fun, -max_z.fun)

        self.bbox = AABB(x_inter, y_inter, z_inter)
        print(self.bbox)
    


    # Define shape functions for a 4-node quadrilateral element
    def shape_functions(self, xi, eta):
        N = np.array([
            0.25 * (1 - xi) * (1 - eta) * (-1 - xi - eta),
            0.25 * (1 + xi) * (1 - eta) * (-1 + xi - eta),
            0.25 * (1 + xi) * (1 + eta) * (-1 + xi + eta),
            0.25 * (1 - xi) * (1 + eta) * (-1 - xi + eta),
            0.5  * (1 - xi ** 2) * (1 - eta),
            0.5  * (1 + xi)      * (1 - eta ** 2),
            0.5  * (1 - xi ** 2) * (1 + eta),
            0.5  * (1 - xi)      * (1 - eta ** 2)
        ])
        return N

    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        initial_guess = [0, 0, 0] #in xi, eta, t

        r0 = r.origin
        d = r.direction

        # Solve the nonlinear system
        solution = root(self.residual, initial_guess, args=(r0, d))

        # Extract solution
        intersection_found = solution.success
        intersection_point = None
        if intersection_found:
            xi_sol, eta_sol, t_sol = solution.x
            intersection_point = self.deformed_surface(xi_sol, eta_sol)

            # Invalid solution
            if t_sol < 0 or abs(xi_sol) > 1 or abs(eta_sol) > 1:
                return None 

            # Fix the normal to reality.
            outward_normal = self.surface_normal(xi_sol, eta_sol)
            u = (xi_sol + 1) / 2
            v = (eta_sol + 1) / 2
            rec: HitRecord = HitRecord(p = intersection_point, t= solution.x[2], mat = self.mat, r=r, outward_normal=outward_normal, u=u, v=v)
            return rec
        else:
            return None


    def dN_dxi(self, xi, eta):
        return np.array([
            0.25 * (1 - eta) * (-1 - 2 * xi - eta),
            0.25 * (1 - eta) * (-1 + 2 * xi - eta),
            0.25 * (1 + eta) * (-1 + 2 * xi + eta),
            0.25 * (1 + eta) * (-1 - 2 * xi + eta),
            -xi * (1 - eta),
            0.5 * (1 - eta**2),
            -xi * (1 + eta),
            -0.5 * (1 - eta**2)
        ])

    def dN_deta(self, xi, eta):
        return np.array([
            0.25 * (1 - xi) * (-1 - xi - 2 * eta),
            0.25 * (1 + xi) * (-1 + xi - 2 * eta),
            0.25 * (1 + xi) * (-1 + xi + 2 * eta),
            0.25 * (1 - xi) * (-1 - xi + 2 * eta),
            -0.5 * (1 - xi**2),
            -eta * (1 + xi),
            0.5 * (1 - xi**2),
            -eta * (1 - xi)
        ])

    def surface_normal(self, xi, eta):
        coords = self.nodes + self.displacements

        dxdxi = np.sum(self.dN_dxi(xi, eta)[:, np.newaxis] * coords, axis=0)
        dxdeta = np.sum(self.dN_deta(xi, eta)[:, np.newaxis] * coords, axis=0)

        normal = np.cross(dxdxi, dxdeta)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 0 else normal
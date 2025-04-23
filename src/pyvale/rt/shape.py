import numpy as np
from scipy.optimize import root

from pyvale.rt.hittable import Hittable, HitRecord
from pyvale.rt.material import Material
from pyvale.rt.ray import Ray
from pyvale.rt.interval import Interval
from pyvale.rt.aabb import AABB

# Define shape functions for a 4-node quadrilateral element
def shape_functions(xi, eta):
    N = np.array([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta)
    ])
    return N

# Compute the position on the deformed surface given (xi, eta)
def deformed_surface(xi, eta, nodes, displacements):
    N = shape_functions(xi, eta)
    return np.dot(N, nodes + displacements)

# Residual function: difference between point on surface and point on line
def residual(vars, nodes, displacements, r0, d):
    xi, eta, t = vars
    surface_point = deformed_surface(xi, eta, nodes, displacements)
    line_point = r0 + t * d
    return surface_point - line_point

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


# A triangle 
class ShapeQuad(Hittable):
    def __init__(self, nodes: np.ndarray, displacements: np.ndarray, mat: Material):
        self.nodes = nodes
        self.displacements = displacements
        self.mat = mat
        self.bbox = AABB(Interval.from_floats(-1, 1), Interval.from_floats(-1, 1), Interval.from_floats(-1, 1))

    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        initial_guess = [0, 0, 5] #in xi, eta, t

        r0 = r.origin
        d = r.direction

        # Solve the nonlinear system
        solution = root(residual, initial_guess, args=(self.nodes, self.displacements, r0, d))

        # Extract solution
        intersection_found = solution.success
        intersection_point = None
        if intersection_found:
            xi_sol, eta_sol, t_sol = solution.x
            intersection_point = deformed_surface(xi_sol, eta_sol, self.nodes, self.displacements)

            # Invalid solution
            if t_sol < 0 or abs(xi_sol) > 1 or abs(eta_sol) > 1:
                return None 

            rec: HitRecord = HitRecord(p = intersection_point, t= solution.x[2], mat = self.mat, r=r, outward_normal=np.array([0, -1, 0]))
            return rec
        else:
            return None


    
    # def is_interior(self, a: float, b: float) -> Tuple[float, float]:
    #     if a > 0 and b > 0 and a+b < 1:
    #         return (a,b)
    #     else:
    #         return None

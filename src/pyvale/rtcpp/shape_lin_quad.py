import numpy as np
from scipy.optimize import root

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

# Define node coordinates (quad)
nodes = np.array([
    [-2, -2, 0],   # Node 1
    [2, -2, 0],   # Node 2
    [2, 2, 0],   # Node 3
    [-2, 2, 0]    # Node 4
])

# Define displacements at each node
displacements = np.array([
    [0, 0, -2],       # Node 1
    [0, 0, 2],     # Node 2
    [0, 0, 2],     # Node 3
    [0, 0, -2]      # Node 4
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

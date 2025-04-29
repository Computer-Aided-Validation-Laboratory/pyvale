import math
import numpy as np
from pytest import approx

from pyvale.rt.shape import ShapeLinQuad, ShapeQuadQuad
from ray import Ray
from interval import Interval
from material import Lambertian

nodes_lin = np.array([
    [-1, 0, -1],   # Node 1
    [1, 0, -1],   # Node 2
    [1, 0, 1],   # Node 3
    [-1, 0, 1]    # Node 4
])

nodes_quad = np.array([
    [-1, 0, -1],   # Node 1
    [1, 0, -1],   # Node 2
    [1, 0, 1],   # Node 3
    [-1, 0, 1],    # Node 4
    [0, 0, -1], # 5
    [1, 0, 0], #6
    [0, 0, 1],  #7
    [-1, 0, 0]  #8
])

# arbitrary color for the material
red   = Lambertian.from_colour(np.array([.65, .05, .05]))

# Perpendicular ray
ray = Ray(np.array([0.5, 3, 0.5]), np.array([0, -1, 0]))
r_int = Interval.from_floats(0, math.inf)


# A perpendicular ray hits a flat square directly underneath it
def test_perp_collision():

    # Define displacements at each node
    displacements = np.array([
        [0, 0, 0],       # Node 1
        [0, 0, 0],     # Node 2
        [0, 0, 0],     # Node 3
        [0, 0, 0]      # Node 4
    ])

    square = ShapeLinQuad(nodes_lin, displacements, red)

    # Calculate the intersection of object and ray
    record = square.hit(ray, r_int)

    assert record is not None

    assert record.p == approx(np.array([0.5, 0, 0.5]))

    assert record.u == approx(0.75)
    assert record.v == approx(0.75)

    assert record.t == approx(3)

# Same for quadquad
def test_perp_collision_quadquad():

    # Define displacements at each node
    displacements = np.array([
        [0, 0, 0],       # Node 1
        [0, 0, 0],     # Node 2
        [0, 0, 0],     # Node 3
        [0, 0, 0],      # Node 4
        [0, 0, 0],  # 5
        [0, 0, 0],  # 6
        [0, 0, 0],  # 7
        [0, 0, 0],  #8
    ])

    square = ShapeQuadQuad(nodes_quad, displacements, red)

    record = square.hit(ray, r_int)

    assert record is not None

    assert record.p == approx(np.array([0.5, 0, 0.5]))

    assert record.u == approx(0.75)
    assert record.v == approx(0.75)



# A ray outside of eta, xi between -1 and 1 will not collide (If the square was infinitely long, it would)
def test_perp_outside():
    displacements = np.array([
        [0, 0, 0],       # Node 1
        [0, 0, 0],     # Node 2
        [0, 0, 0],     # Node 3
        [0, 0, 0]      # Node 4
    ])

    square = ShapeLinQuad(nodes_lin, displacements, red)
    ray = Ray(np.array([1.5, 3, 0.5]), np.array([0, -1, 0]))
    record = square.hit(ray, r_int)

    assert record is None

# A linear displacement
def test_lin_displacement():
    displacements = np.array([
        [0, 0, 0],       # Node 1
        [0, 0, 0],     # Node 2
        [0, 1, 0],     # Node 3
        [0, 0, 0]      # Node 4
    ])

    square = ShapeLinQuad(nodes_lin, displacements, red)
    ray = Ray(np.array([1, 3, 0.5]), np.array([0, -1, 0]))
    record = square.hit(ray, r_int)

    assert record.p == approx(np.array([1, 0.75, 0.5]))


# An known analytic result of a ray with a quadratic
def test_quad_collision():
    displacements = np.array([
        [0, 0, 0],       # Node 1
        [0, 0, 0],     # Node 2
        [0, 1, 0],     # Node 3
        [0, 0, 0],      # Node 4
        [0, 0, 0],  # 5
        [0, 0, 0],  # 6
        [0, 0, 0],  # 7
        [0, 0, 0],  #8
    ])

    square = ShapeQuadQuad(nodes_quad, displacements, red)
    ray = Ray(np.array([1, 3, 0.5]), np.array([0, -1, 0]))

    record = square.hit(ray, r_int)
    assert record.p == approx(np.array([1, 0.375, 0.5]))


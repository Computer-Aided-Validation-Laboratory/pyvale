import numpy as np
from pyvale.rtcpp.rt import *
# from .scene import Scene

scene = Scene()

red   = Diffuse(solid_color(.65, .05, .05))
white = Diffuse(solid_color(.73, .73, .73))
green = Diffuse(solid_color(.12, .45, .15))
light = Diffuse_light(solid_color(13, 13, 13))
glass = Refractive(1.5)

nodes = np.array([
    [200, 300, 100],   # Node 1
    [250, 300, 200],   # Node 2
    [250, 350, 200],   # Node 3
    [200, 350, 200]    # Node 4
])
        # [-5, 0, -5],   # Node 1
        # [5, 0, -5],   # Node 2
        # [5, 0, 5],   # Node 3
        # [-5, 0, 5]    # Node 4
# Define displacements at each node
displacements = np.array([
    [0, 0, 0],       # Node 1
    [0, 0, 0],     # Node 2
    [0, 0, 0],     # Node 3
    [0, 0, 0]      # Node 4
])
scene.add(ShapeQuadLin(nodes, displacements, red))


nodes = np.array([
    [200, 300, 200],   # Node 1
    [250, 300, 200],   # Node 2
    [250, 350, 200],   # Node 3
    [200, 350, 200],    # Node 4
    [225, 300, 200],
    [250, 325, 200],
    [225, 350, 200],
    [200, 325, 200]
])
        # [-5, 0, -5],   # Node 1
        # [5, 0, -5],   # Node 2
        # [5, 0, 5],   # Node 3
        # [-5, 0, 5]    # Node 4
# Define displacements at each node
displacements = np.array([
    [0, 0, 0],       # Node 1
    [0, 0, 0],     # Node 2
    [0, 0, 0],     # Node 3
    [0, 0, 0],      # Node 4
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])
# scene.add(ShapeQuadQuad(nodes, displacements, green))

scene.add(Plane_yz(0, 555, 0, 555, 555, green))
scene.add(Plane_yz(0, 555, 0, 555, 0, red))
scene.add(Plane_xz(213, 343, 227, 332, 554, light) , importance_sampled = True)
scene.add(Plane_xz(0, 555, 0, 555, 0, white))
scene.add(Plane_xz(0, 555, 0, 555, 555, white))
scene.add(Plane_xy(0, 555, 0, 555, 555, white))
scene.add(Sphere(point3(278, 100, 250), 100, green))

# world = HittableList(BVH_Node(world._objects))

scene.add_Camera(lookfrom = point3(278, 278, -800),
				  lookat = point3(278,278,0),
				  screen_width = 200, 
				  screen_height = 200,
				  field_of_view = 40,
				  focus_distance  = 10.0,
				  aperture  = 0.01)


img = scene.render(samples_per_pixel = 20, max_depth = 5)

img.show()

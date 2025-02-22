import numpy as np
from hittable_list import HittableList
from bvh import BVH_Node
from material import Lambertian, Material
from sphere import Sphere
from camera import Camera
from tri import Tri

world: HittableList = HittableList()

material_2: Material = Lambertian(np.array([0.4, 0.2, 0.1]))
world.add(Sphere(np.array([0,0,-1]), 0.5, material_2))

ground_mat: Material = Lambertian(np.array([0.5, 0.5, 0.5]))
world.add(Sphere(np.array([0,-100.5,-1]), 100, ground_mat))

world = HittableList(BVH_Node(world._objects, 0, len(world._objects)))

cam: Camera = Camera()
cam.image_width = 500
cam.image_height = 400
cam.samples_per_pixel = 5
cam.max_depth = 5

cam.render(world)
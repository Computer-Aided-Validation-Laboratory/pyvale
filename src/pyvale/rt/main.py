import numpy as np
from hittable_list import HittableList
from pyvale.rt.material import Lambertian, Material
from sphere import Sphere
from camera import Camera

world: HittableList = HittableList()

material_2: Material = Lambertian(np.array([0.4, 0.2, 0.1]))
world.objects.append(Sphere(np.array([0,0,-1]), 0.5, material_2))

ground_mat: Material = Lambertian(np.array([0.5, 0.5, 0.5]))
world.objects.append(Sphere(np.array([0,-100.5,-1]), 100, ground_mat))

cam: Camera = Camera()
# cam.aspect_ratio = 16.0/9.0
cam.image_width = 300
cam.image_height = 200
cam.samples_per_pixel = 100
cam.max_depth = 5

cam.render(world)
import numpy as np
from hittable_list import HittableList
from sphere import Sphere
from camera import Camera

world: HittableList = HittableList()

world.objects.append(Sphere(np.array([0,0,-1]), 0.5))
world.objects.append(Sphere(np.array([0,-100.5,-1]), 100))

cam: Camera = Camera()
# cam.aspect_ratio = 16.0/9.0
cam.image_width = 400
cam.image_height = 300
cam.samples_per_pixel = 100
cam.max_depth = 5

cam.render(world)
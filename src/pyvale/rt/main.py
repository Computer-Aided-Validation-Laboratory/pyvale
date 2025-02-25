import numpy as np
from hittable_list import HittableList
from bvh import BVH_Node
from material import Lambertian, Material
from sphere import Sphere
from camera import Camera
from quad import Disk, Quad, Tri, Disk

def sphere2():
    world: HittableList = HittableList()

    material_2: Material = Lambertian(np.array([0.4, 0.2, 0.1]))
    world.add(Sphere(np.array([0,0,-1]), 0.5, material_2))

    ground_mat: Material = Lambertian(np.array([0.5, 0.5, 0.5]))
    world.add(Sphere(np.array([0,-100.5,-1]), 100, ground_mat))

    world = HittableList(BVH_Node(world._objects))

    cam: Camera = Camera()
    cam.image_width = 50
    cam.image_height = 40
    cam.samples_per_pixel = 5
    cam.max_depth = 5

    cam.render(world)


def quads():
    world: HittableList = HittableList()

    # Materials
    left_red     = Lambertian(np.array([1.0, 0.2, 0.2]))
    back_green   = Lambertian(np.array([0.2, 1.0, 0.2]))
    right_blue   = Lambertian(np.array([0.2, 0.2, 1.0]))
    upper_orange = Lambertian(np.array([1.0, 0.5, 0.0]))
    lower_teal   = Lambertian(np.array([0.2, 0.8, 0.8]))

    # quads
    world.add(Quad(np.array([-3,-2, 5]), np.array([0, 0,-4]), np.array([0, 4, 0]), left_red))
    world.add(Quad(np.array([-2,-2, 0]), np.array([4, 0, 0]), np.array([0, 4, 0]), back_green))
    world.add(Quad(np.array([ 3,-2, 1]), np.array([0, 0, 4]), np.array([0, 4, 0]), right_blue))
    world.add(Quad(np.array([-2, 3, 1]), np.array([4, 0, 0]), np.array([0, 0, 4]), upper_orange))
    world.add(Quad(np.array([-2,-3, 5]), np.array([4, 0, 0]), np.array([0, 0,-4]), lower_teal))

    # world = HittableList(BVH_Node(world._objects))
    
    cam: Camera = Camera()
    cam.image_width = 50
    cam.image_height = 40
    cam.samples_per_pixel = 5
    cam.max_depth = 5
    cam.vfov = 80
    cam.look_from = np.array([0,0,9])
    cam.look_at = np.array([0,0,0])

    cam.render(world)

def tris():
    world: HittableList = HittableList()

    # Materials
    left_red     = Lambertian(np.array([1.0, 0.2, 0.2]))
    back_green   = Lambertian(np.array([0.2, 1.0, 0.2]))
    right_blue   = Lambertian(np.array([0.2, 0.2, 1.0]))
    upper_orange = Lambertian(np.array([1.0, 0.5, 0.0]))
    lower_teal   = Lambertian(np.array([0.2, 0.8, 0.8]))

    # quads
    world.add(Tri(np.array([-3,-2, 5]), np.array([0, 0,-4]), np.array([0, 4, 0]), left_red))
    world.add(Tri(np.array([-3,2, 0]), np.array([0, 0, 4]), np.array([0, -4, 0]), back_green))

    # world.add(Tri(np.array([-2,-2, 0]), np.array([4, 0, 0]), np.array([0, 4, 0]), back_green))
    # world.add(Tri(np.array([ 3,-2, 1]), np.array([0, 0, 4]), np.array([0, 4, 0]), right_blue))
    # world.add(Tri(np.array([-2, 3, 1]), np.array([4, 0, 0]), np.array([0, 0, 4]), upper_orange))
    # world.add(Disk(np.array([-2,-3, 5]), np.array([4, 0, 0]), np.array([0, 0,-4]), lower_teal, 2))

    # world = HittableList(BVH_Node(world._objects))
    
    cam: Camera = Camera()
    cam.image_width = 200
    cam.image_height = 150
    cam.samples_per_pixel = 5
    cam.max_depth = 5
    cam.vfov = 80
    cam.look_from = np.array([0,0,9])
    cam.look_at = np.array([0,0,0])

    cam.render(world)



if __name__ == "__main__":
    tris()
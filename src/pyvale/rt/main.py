import numpy as np
from pyvale.rt.hittable_list import HittableList
from pyvale.rt.bvh import BVH_Node
from pyvale.rt.material import Lambertian, Material
from pyvale.rt.material import DiffuseLight
from pyvale.rt.sphere import Sphere
from pyvale.rt.camera import Camera
from pyvale.rt.quad import Disk, Quad, Tri, Disk

np.random.default_rng(1)

def sphere2():
    world: HittableList = HittableList()

    material_2: Material = Lambertian.from_colour(np.array([0.4, 0.2, 0.1]))
    world.add(Sphere(np.array([0,0,-1]), 0.5, material_2))

    ground_mat: Material = Lambertian.from_colour(np.array([0.5, 0.5, 0.5]))
    world.add(Sphere(np.array([0,-100.5,-1]), 100, ground_mat))

    world = HittableList(BVH_Node(world._objects))

    cam: Camera = Camera()
    cam.image_width = 500
    cam.image_height = 400
    cam.samples_per_pixel = 1
    cam.max_depth = 5
    cam.background = np.array([0.70, 0.80, 1.00])

    cam.render(world)


def quads():
    world: HittableList = HittableList()

    # Materials
    left_red     = Lambertian.from_colour(np.array([1.0, 0.2, 0.2]))
    back_green   = Lambertian.from_colour(np.array([0.2, 1.0, 0.2]))
    right_blue   = Lambertian.from_colour(np.array([0.2, 0.2, 1.0]))
    upper_orange = Lambertian.from_colour(np.array([1.0, 0.5, 0.0]))
    lower_teal   = Lambertian.from_colour(np.array([0.2, 0.8, 0.8]))

    # quads
    world.add(Quad(np.array([-3,-2, 5]), np.array([0, 0,-4]), np.array([0, 4, 0]), left_red))
    world.add(Quad(np.array([-2,-2, 0]), np.array([4, 0, 0]), np.array([0, 4, 0]), back_green))
    world.add(Quad(np.array([ 3,-2, 1]), np.array([0, 0, 4]), np.array([0, 4, 0]), right_blue))
    world.add(Quad(np.array([-2, 3, 1]), np.array([4, 0, 0]), np.array([0, 0, 4]), upper_orange))
    world.add(Quad(np.array([-2,-3, 5]), np.array([4, 0, 0]), np.array([0, 0,-4]), lower_teal))

    world = HittableList(BVH_Node(world._objects))
    
    cam: Camera = Camera()
    cam.image_width = 500
    cam.image_height = 400
    cam.samples_per_pixel = 5
    cam.max_depth = 5
    cam.vfov = 80
    cam.look_from = np.array([0,0,9])
    cam.look_at = np.array([0,0,0])
    cam.background = np.array([0.70, 0.80, 1.00])

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

def simple_light():
    world: HittableList = HittableList()

    material_2: Material = Lambertian.from_colour(np.array([0.4, 0.2, 0.1]))
    world.add(Sphere(np.array([0,0,-1]), 0.5, material_2))

    ground_mat: Material = Lambertian.from_colour(np.array([0.5, 0.5, 0.5]))
    world.add(Sphere(np.array([0,-100.5,-1]), 100, ground_mat))

    difflight = DiffuseLight.from_colour(np.array([4,4,4]))
    world.add(Sphere(np.array([0,7,0]), 2, difflight))
    world.add(Quad(np.array([3,1,-2]), np.array([2,0,0]), np.array([0,2,0]), difflight))

    world = HittableList(BVH_Node(world._objects))

    cam: Camera = Camera()
    cam.image_width = 500
    cam.image_height = 400
    cam.samples_per_pixel = 50
    cam.max_depth = 5
    cam.background = np.array([0, 0, 0])

    cam.vfov = 20
    cam.look_from = np.array([26,3,6])
    cam.look_at = np.array([0,2,0])
    cam.v_up = np.array([0,1,0])

    cam.render(world)

def cornell_box():
    world: HittableList = HittableList()

    red   = Lambertian.from_colour(np.array([.65, .05, .05]))
    white = Lambertian.from_colour(np.array([.73, .73, .73]))
    green = Lambertian.from_colour(np.array([.12, .45, .15]))
    light = DiffuseLight.from_colour(np.array([15, 15, 15]))

    world.add(Quad(np.array([555,0,0]), np.array([0,555,0]), np.array([0,0,555]), green))
    world.add(Quad(np.array([0,0,0]), np.array([0,555,0]), np.array([0,0,555]), red))
    world.add(Quad(np.array([343, 554, 332]), np.array([-130,0,0]), np.array([0,0,-105]), light))
    world.add(Quad(np.array([0,0,0]), np.array([555,0,0]), np.array([0,0,555]), white))
    world.add(Quad(np.array([555,555,555]), np.array([-555,0,0]), np.array([0,0,-555]), white))
    world.add(Quad(np.array([0,0,555]), np.array([555,0,0]), np.array([0,555,0]), white))

    world = HittableList(BVH_Node(world._objects))

    cam: Camera = Camera()
    cam.image_width = 60
    cam.image_height = 60
    cam.samples_per_pixel = 50
    cam.max_depth = 20
    cam.background = np.array([0, 0, 0])

    cam.vfov = 40
    cam.look_from = np.array([278, 278, -800])
    cam.look_at = np.array([278, 278, 0])
    cam.v_up = np.array([0,1,0])

    cam.render(world)

if __name__ == "__main__":
    simple_light()
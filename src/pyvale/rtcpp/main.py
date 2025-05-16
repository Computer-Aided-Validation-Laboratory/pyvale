import numpy as np
# Assuming you have built the module from cmake, and now have a pyray.cpython-311...so file in the same directory as this file
import pyray

# Check module imports okay
print(pyray)


scene = pyray.Hittable_list()
lights = pyray.Hittable_list()


point3 = pyray.vec3
color = pyray.vec3


red   = pyray.Lambertian(color(.65, .05, .05))
white = pyray.Lambertian(color(.73, .73, .73))
green = pyray.Lambertian(color(.12, .45, .15))
light = pyray.Diffuse_light(color(13, 13, 13))
# glass = pyray.Refractive(1.5)

# scene.add(Plane_yz(0, 555, 0, 555, 555, green))
# scene.add(Plane_yz(0, 555, 0, 555, 0, red))
# scene.add(Plane_xz(213, 343, 227, 332, 554, light) , importance_sampled = True)
# scene.add(Plane_xz(0, 555, 0, 555, 0, white))
# scene.add(Plane_xz(0, 555, 0, 555, 555, white))
# scene.add(Plane_xy(0, 555, 0, 555, 555, white))

camera = pyray.Camera(pyray.vec3(0, 0, 5),
                    pyray.vec3(0, 0, 0),
                    40.0,
                    float(200.0 / 200.0),
                    0.01,
                    10.0,
                    pyray.vec3(0.,1.,0.),
                    600,
                    200,
                    color(0.6, 0.6,0.6))

sphere = pyray.Sphere(point3(0, -152, 0), 150, white)
scene.add(sphere)

l = pyray.Quad(point3(0, 5, 0), pyray.vec3(1,0,0), pyray.vec3(0,0,1), red)
lights.add(l)

camera.render(scene, lights)


# box1 = Box(point3(265, 0, 295), point3(430, 330, 460), white)
# scene.add(box1)
# scene.add(Sphere(vec3(160, 165/2, +65+185/2), 165/2, Refractive(1.5)))

# scene.add(Tri(point3(10, 10, 10), vec3(500,0,300), vec3(0,500,300), green))



# scene.add_Camera(lookfrom = point3(278, 278, -800),
# 				  lookat = point3(278,278,0),
# 				  screen_width = 200, 
# 				  screen_height = 200,
# 				  field_of_view = 40,
# 				  focus_distance  = 10.0,
# 				  aperture  = 0.01)


img = scene.render(samples_per_pixel = 100, max_depth = 5)

img.save("box.png")

img.show()

